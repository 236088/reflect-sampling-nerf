"""
ReflectSamplingNeRF Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch import nn
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss, pred_normal_loss, orientation_loss, interlevel_loss, distortion_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, LogSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    NormalsRenderer,
    SemanticRenderer,
)
from nerfstudio.utils import colormaps, colors, misc

from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from reflect_sampling_nerf.reflect_sampling_nerf_field import ReflectSamplingNeRFPropField, ReflectSamplingNeRFEnvironmentField, ReflectSamplingNeRFNerfField
from reflect_sampling_nerf.reflect_sampling_nerf_components import IntegratedSHEncoding, PolyhedronFFEncoding, ReciprocalSampler

@dataclass
class ReflectSamplingNeRFModelConfig(ModelConfig):
    """ReflectSamplingNeRF Model Configuration.

    Add your custom model config parameters here.
    https://github.com/google-research/multinerf/blob/main/configs/blender_refnerf.gin
    """

    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 64
    """Number of samples in fine field evaluation"""
    
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss": 1.0,
        "ref_rgb_loss": 1.0,
        "pred_normal_loss": 3e-4,
        "orientation_loss": 1e-1,
        "smooth_normal_loss": 1e-2,
        "interlevel_loss": 1.0,
        "distortion_loss": 1e-2,
        # "existance_loss": 1e-5,
        })

    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""
    _target: Type = field(default_factory=lambda: ReflectSamplingNeRFModel)


class ReflectSamplingNeRFModel(Model):
    """ReflectSamplingNeRF Model."""

    config: ReflectSamplingNeRFModelConfig

    def __init__(
        self,
        config: ReflectSamplingNeRFModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        assert config.collider_params is not None, "MipNeRF model requires bounding box collider parameters."
        super().__init__(config=config, **kwargs)
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = PolyhedronFFEncoding(
            num_frequencies=10, min_freq_exp=0.0, max_freq_exp=10.0, include_input=True
        )
        direction_encoding = IntegratedSHEncoding()
        spatial_distortion = SceneContraction()

        self.prop = ReflectSamplingNeRFPropField(
            position_encoding=position_encoding,
            spatial_distortion=spatial_distortion
        )

        self.env = ReflectSamplingNeRFEnvironmentField(
            position_encoding=position_encoding
        )

        self.field = ReflectSamplingNeRFNerfField(
            position_encoding=position_encoding, 
            direction_encoding=direction_encoding,
            spatial_distortion=spatial_distortion
        )

        # samplers
        self.sampler_reciprocal = ReciprocalSampler(num_samples=self.config.num_coarse_samples, tan=4)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)
        self.far = 2**10
        self.near = 1.0/16
        
        # renderers
        self.background_color = colors.WHITE
        self.renderer_rgb = RGBRenderer(background_color=self.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_roughness = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["proposal_networks"] = list(self.prop.parameters())
        param_groups["environment_networks"] = list(self.env.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups
       
        
    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        # ray_bundle.nears = torch.ones_like(ray_bundle.nears)*self.near
        ray_bundle.fars = torch.ones_like(ray_bundle.fars)*self.far
        ray_samples_reciprocal = self.sampler_reciprocal(ray_bundle)

        # First pass:
        density_outputs_prop, embedding_coarse = self.prop.get_density(ray_samples_reciprocal, True)
        weights_coarse = ray_samples_reciprocal.get_weights(density_outputs_prop)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_reciprocal)
        #
        background_color = self.background_color.to(weights_coarse.device)
               
        

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_reciprocal, weights_coarse)

        # Second pass:
        density_outputs, embedding_fine = self.field.get_density(ray_samples_pdf, True)
        weights_fine = ray_samples_pdf.get_weights(density_outputs)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        
        
        raw_normals_outputs = self.field.get_pred_normals(embedding_fine)
        pred_normals_outputs = nn.functional.normalize(raw_normals_outputs, dim=-1)
        
        normals_outputs = self.field.get_normals() if self.training else pred_normals_outputs
        reflections_outputs, n_dot_d_outputs = self.field.get_reflection(ray_samples_pdf.frustums.directions, pred_normals_outputs)
            
        diff_outputs = self.field.get_diff(embedding_fine) 
        tint_outputs = self.field.get_tint(embedding_fine)
        
        # low_outputs = self.field.get_low(embedding_fine, True)
        roughness_outputs = self.field.get_roughness(embedding_fine)
        mid_outputs = self.field.get_mid(reflections_outputs, n_dot_d_outputs, roughness_outputs, embedding_fine, True)
        
        outputs = diff_outputs + tint_outputs*mid_outputs
        rgb_fine = self.renderer_rgb(outputs, weights_fine, background_color=background_color)
        rgb_fine = torch.clip(rgb_fine, 0.0, 1.0)


        diff = self.renderer_rgb(diff_outputs, weights_fine.detach(), background_color=background_color)
        tint = self.renderer_rgb(tint_outputs, weights_fine.detach())
        
        pred_normals = self.renderer_normals(raw_normals_outputs, weights_fine.detach())
        n_dot_d = torch.sum(pred_normals*ray_bundle.directions, dim=-1, keepdim=True)

        roughness = torch.exp(-roughness_outputs)
        roughness = 1-self.renderer_roughness(roughness, weights_fine.detach())
        
        mask = torch.logical_and(accumulation_fine>1e-2, n_dot_d<0).reshape(-1)
        print(mask[mask].shape, mask[(accumulation_fine>1e-2).reshape(-1)].shape, mask[(n_dot_d<0).reshape(-1)].shape)

        q=torch.tensor([0.1,0.3,0.5,0.7,0.9])
        print("roughness :",torch.quantile(roughness[mask,:].cpu(), q=q).detach().numpy())
        print("diffuse   :", torch.quantile(torch.sum((diff[mask,:]).cpu(), dim=-1), q=q).detach().numpy())
        print("tint      :", torch.quantile(torch.sum((tint[mask,:]).cpu(), dim=-1), q=q).detach().numpy())

        origins = ray_bundle.origins + depth_fine*ray_bundle.directions
        reflections = ray_bundle.directions - 2*n_dot_d.detach()*pred_normals.detach()
        reflections = torch.nn.functional.normalize(reflections, dim=-1)
        sqradius = 2*torch.abs(n_dot_d.detach())*roughness**2
        # sqradius = sqradius.detach()
        
        
        outputs = {
            "rgb_fine": rgb_fine,
            "reflect_fine": background_color.expand(rgb_fine.shape)*(1.0-accumulation_fine),
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "weights_list": [weights_coarse, weights_fine],
            "ray_samples_list": [ray_samples_reciprocal, ray_samples_pdf],
            "diff_outputs": diff_outputs,
            "tint_outputs": tint_outputs,
            "roughness_outputs": 1-torch.exp(-roughness_outputs),
            "pred_normals_outputs": pred_normals_outputs,
            "normals_outputs": normals_outputs,
            "directions": ray_bundle.directions,
            "diff":diff.detach(),
            "tint":tint.detach(),
            "roughness":roughness.detach(),
            "pred_normals":pred_normals.detach(),
            "sqradius":sqradius,
            "mask":mask,
            "reflect": torch.zeros_like(rgb_fine),
            "accumulation_reflect": torch.zeros_like(accumulation_fine),
        }
        if not mask.any():
            return outputs

        '''        
        roughness to pixelarea as spherical gaussian lobe
        g(|x-mean|, sigma^2) = exp(-|x-mean|^2/(2*sigma^2))
        sigma^2 = 1/sharpness
        sharpness = 2/roughness^2/(4*|direction*normal|)
        sigma^2 = roughness^2*2*|direction*normal|
        sigma as radius
        '''
        
        '''
        normal_ray_bundle = RayBundle(
            origins=origins[mask, :],
            directions=pred_normals[mask, :],
            pixel_area=torch.pi*2*torch.ones_like(sqradius[mask, :]),
            nears=torch.ones_like(ray_bundle.nears[mask, :])*self.near,
            fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
        )
        normal_background_color = self.env.get_env(pred_normals[mask, :], 2*torch.ones_like(sqradius[mask, :]))

        ray_samples_normal_reciprocal = self.sampler_reciprocal(normal_ray_bundle)
        density_outputs_normal_coarse, embedding_normal_coarse = self.prop.get_density(ray_samples_normal_reciprocal)
        weights_normal_coarse = ray_samples_normal_reciprocal.get_weights(density_outputs_normal_coarse)
      
        ray_samples_normal_pdf = self.sampler_pdf(normal_ray_bundle, ray_samples_normal_reciprocal, weights_normal_coarse)

        density_outputs_normal_fine, embedding_normal_fine = self.field.get_density(ray_samples_normal_pdf)
        weights_normal_fine = ray_samples_normal_pdf.get_weights(density_outputs_normal_fine)
        
        pred_normals_outputs_normal_fine = self.field.get_pred_normals(embedding_normal_fine)
        reflections_outputs_normal_fine, n_dot_d_outputs_normal_fine = self.field.get_reflection(ray_samples_normal_pdf.frustums.directions, pred_normals_outputs_normal_fine)
            
        diff_outputs_normal_fine = self.field.get_diff(embedding_normal_fine) 
        tint_outputs_normal_fine = self.field.get_tint(embedding_normal_fine)
        
        # low_outputs_normal_fine = self.field.get_low(embedding_normal_fine, True)
        roughness_outputs_normal_fine = self.field.get_roughness(embedding_normal_fine)
        mid_outputs_normal_fine = self.field.get_mid(ray_samples_normal_pdf.frustums.directions, n_dot_d_outputs_normal_fine, roughness_outputs_normal_fine.detach(), embedding_normal_fine, True)
        
        outputs_normal_fine = diff_outputs_normal_fine + tint_outputs_normal_fine.detach()*mid_outputs_normal_fine
        normal_fine = self.renderer_rgb(outputs_normal_fine.detach(), weights_normal_fine, normal_background_color)
        '''
        
        
        reflect_ray_bundle = RayBundle(
            origins=origins[mask, :].detach(),
            directions=reflections[mask, :].detach(),
            pixel_area=torch.pi*sqradius[mask, :],
            nears=torch.ones_like(ray_bundle.nears[mask, :])*self.near,
            fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
        )
        ref_background_color = self.env.get_env(reflections[mask, :].detach(), sqradius[mask, :])

        ray_samples_reflect_reciprocal = self.sampler_reciprocal(reflect_ray_bundle)
        
        density_outputs_ref_prop, embedding_reflect_coarse = self.prop.get_density(ray_samples_reflect_reciprocal)
        weights_reflect_coarse = ray_samples_reflect_reciprocal.get_weights(density_outputs_ref_prop)
      
        ray_samples_reflect_pdf = self.sampler_pdf(reflect_ray_bundle, ray_samples_reflect_reciprocal, weights_reflect_coarse)

        density_outputs_ref, embedding_reflect_fine = self.field.get_density(ray_samples_reflect_pdf)
        weights_reflect_fine = ray_samples_reflect_pdf.get_weights(density_outputs_ref)
        
        raw_normals_outputs_ref = self.field.get_pred_normals(embedding_reflect_fine)
        pred_normals_outputs_ref = nn.functional.normalize(raw_normals_outputs_ref, dim=-1)
        reflections_outputs_ref, n_dot_d_outputs_ref = self.field.get_reflection(ray_samples_reflect_pdf.frustums.directions, pred_normals_outputs_ref)
        
        diff_outputs_ref = self.field.get_diff(embedding_reflect_fine) 
        tint_outputs_ref = self.field.get_tint(embedding_reflect_fine)
        
        # low_outputs_ref = self.field.get_low(embedding_reflect_fine, True)
        roughness_outputs_ref = self.field.get_roughness(embedding_reflect_fine)
        mid_outputs_ref = self.field.get_mid(reflections_outputs_ref, n_dot_d_outputs_ref, roughness_outputs_ref, embedding_reflect_fine, True)
        
        outputs_ref = diff_outputs_ref + tint_outputs_ref*mid_outputs_ref
        reflect_fine = self.renderer_rgb(outputs_ref, weights_reflect_fine, background_color=ref_background_color)
                
        outputs["reflect_fine"][mask, :] = diff[mask, :] + tint[mask, :]*reflect_fine
        outputs["reflect_fine"][mask, :] = torch.clip(outputs["reflect_fine"][mask, :], 0.0, 1.0)
        
        accumulation_reflect = self.renderer_accumulation(weights_reflect_fine)
        outputs["accumulation_reflect"][mask, :] = accumulation_reflect
        print("accumulation :",torch.quantile(outputs["accumulation_reflect"][mask, :].cpu(), q=q).detach().numpy())
        # print(torch.sum(accumulation_reflect>1-torch.exp(-torch.pi*sqradius[mask])))

        depth_reflect = self.renderer_depth(weights_reflect_fine, ray_samples_reflect_pdf)
        print("depth        :",torch.quantile(depth_reflect.cpu(), q=q).detach().numpy())
        print("acc/depth    :",torch.quantile(accumulation_reflect.cpu()/depth_reflect.cpu(), q=q).detach().numpy())
        print("env :",torch.quantile(torch.sum(ref_background_color.cpu(),dim=-1), q=q).detach().numpy())
        
        outputs["reflect"][mask, :] = reflect_fine
        
        
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)

        pred_fine, image_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        pred_reflect_fine, image_reflect_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["reflect_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        
        rgb_loss = self.rgb_loss(image_fine, pred_fine)
        ref_rgb_loss = self.rgb_loss(image_reflect_fine, pred_reflect_fine)

        pred_normal_loss_value = torch.sum(pred_normal_loss(outputs["weights_list"][-1].detach(), outputs["normals_outputs"], outputs["pred_normals_outputs"]))
        orientation_loss_value = torch.sum(orientation_loss(outputs["weights_list"][-1].detach(), outputs["pred_normals_outputs"], outputs["directions"]))
        smooth_normal_loss = torch.sum(outputs["weights_list"][-1].detach()*(outputs["pred_normals"][...,None,:] - outputs["pred_normals_outputs"])**2)
        
        interlevel_loss_value = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
        distortion_loss_value = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        
        # existance_loss_value = torch.sum(self.bce_loss(outputs["accumulation_reflect"], outputs["accumulation_reflect"]))
        
        print(ref_rgb_loss.item(), "<" if ref_rgb_loss.item()<rgb_loss.item() else ">", rgb_loss.item())
        print(pred_normal_loss_value.item(), orientation_loss_value.item())
        print(interlevel_loss_value.item(), distortion_loss_value.item())

        if rgb_loss.isnan().any() or ref_rgb_loss.isnan().any() or pred_normal_loss_value.isnan().any() or orientation_loss_value.isnan().any():
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        loss_dict = {
            "rgb_loss": rgb_loss,
            "ref_rgb_loss": ref_rgb_loss,
            "pred_normal_loss": pred_normal_loss_value,
            "orientation_loss": orientation_loss_value,
            "smooth_normal_loss": smooth_normal_loss,
            "interlevel_loss": interlevel_loss_value,
            "distortion_loss": distortion_loss_value,
            # "existance_loss": existance_loss_value,
            }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_fine = outputs["reflect_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=0, max=1)
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
