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

    num_prop_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 32
    """Number of samples in fine field evaluation"""
    
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss": 1.0,
        "separate_rgb_loss": 1.0,
        "ref_rgb_loss": 1.0,
        "pred_normal_loss": 3e-4,
        "orientation_loss": 1e-1,
        "smooth_normal_loss": 1e-3,
        "interlevel_loss": 1.0,
        "distortion_loss": 1e-2,
        "cauchy_loss": 1e-4,
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
            num_frequencies=12, min_freq_exp=-1.0, max_freq_exp=10.0, include_input=True
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
            
        self.far = 2**10
        self.near = 1.0/16
        self.alpha = 0.0 if self.training else 1.0

        # samplers
        self.sampler_reciprocal = ReciprocalSampler(num_samples=self.config.num_prop_samples, tan=4)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # renderers
        self.background_color = colors.WHITE
        self.renderer_rgb = RGBRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_roughness = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()

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

        # First pass:
        ray_samples_reciprocal = self.sampler_reciprocal(ray_bundle)
        density_outputs_prop, _ = self.prop.get_density(ray_samples_reciprocal, True)
        weights_prop = ray_samples_reciprocal.get_weights(density_outputs_prop)
        accumulation_prop = self.renderer_accumulation(weights_prop)
        depth_prop = self.renderer_depth(weights_prop, ray_samples_reciprocal)
        
        # back ground
        background_color = self.background_color.to(weights_prop.device)
               
        
        # Second pass:
        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_reciprocal, weights_prop)
        density_outputs, embedding = self.field.get_density(ray_samples_pdf, True)
        weights = ray_samples_pdf.get_weights(density_outputs)
        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples_pdf)
        
        
        raw_normals_outputs = self.field.get_pred_normals(embedding)
        pred_normals_outputs = nn.functional.normalize(raw_normals_outputs, dim=-1)
        
        normals_outputs = self.field.get_normals() if self.training else pred_normals_outputs
        reflections_outputs, n_dot_d_outputs = self.field.get_reflection(ray_samples_pdf.frustums.directions, pred_normals_outputs)
            
        diff_outputs = self.field.get_diff(embedding) 
        tint_outputs = self.field.get_tint(embedding)
        
        # low_outputs = self.field.get_low(embedding, True)
        roughness_outputs = self.field.get_roughness(embedding)
        low_outputs = self.field.get_low(reflections_outputs, n_dot_d_outputs, roughness_outputs, embedding)
        
        rgb_outputs = diff_outputs + tint_outputs*low_outputs
        rgb = torch.clip(self.renderer_rgb(rgb_outputs, weights, background_color=background_color), 0.0, 1.0)

        diff = self.renderer_rgb(diff_outputs, weights.detach(), background_color=background_color)
        tint = self.renderer_rgb(tint_outputs, weights.detach())
        low = self.renderer_rgb(low_outputs, weights.detach())
        
        separate_rgb = torch.clip(diff + tint*low, 0.0, 1.0)
        
        pred_normals = self.renderer_normals(raw_normals_outputs, weights.detach())
        n_dot_d = torch.sum(pred_normals*ray_bundle.directions, dim=-1, keepdim=True)

        roughness = 1-torch.exp(-self.renderer_roughness(roughness_outputs, weights.detach()))
        
        mask = torch.logical_and(accumulation>1e-2, n_dot_d<0).reshape(-1)


        origins = ray_bundle.origins + depth*ray_bundle.directions
        reflections = ray_bundle.directions - 2*n_dot_d.detach()*pred_normals.detach()
        reflections = torch.nn.functional.normalize(reflections, dim=-1)
        sqradius = 2*torch.abs(n_dot_d.detach())*roughness**2
        # sqradius = sqradius.detach()
        
        
        outputs = {
            "rgb": rgb,
            "separate_rgb": separate_rgb,
            "ref_rgb": background_color.expand(rgb.shape)*(1.0-accumulation),
            "accumulation_prop": accumulation_prop,
            "accumulation": accumulation,
            "depth_prop": depth_prop,
            "depth": depth,
            "density_list": [density_outputs_prop, density_outputs],
            "weights_list": [weights_prop, weights],
            "ray_samples_list": [ray_samples_reciprocal, ray_samples_pdf],
            "diff_outputs": diff_outputs,
            "tint_outputs": tint_outputs,
            "roughness_outputs": self.field.get_roughness(embedding, activation=nn.Sigmoid()),
            "pred_normals_outputs": pred_normals_outputs,
            "normals_outputs": normals_outputs,
            "directions": ray_bundle.directions,
            "diff":diff,
            "tint":tint,
            "roughness":roughness,
            "pred_normals":pred_normals,
            "sqradius":sqradius,
            "mask":mask,
            "reflect": torch.zeros_like(rgb),
            "accumulation_ref": torch.zeros_like(accumulation),
        }
        if not mask.any():
            return outputs

        # if self.training:
        #     self.alpha = 1 - (1 - 1e-4)*(1 - self.alpha)
        '''        
        roughness to pixelarea as spherical gaussian lobe
        g(|x-mean|, sigma^2) = exp(-|x-mean|^2/(2*sigma^2))
        sigma^2 = 1/sharpness
        sharpness = 2/roughness^2/(4*|direction*normal|)
        sigma^2 = roughness^2*2*|direction*normal|
        sigma as radius
        '''
        
        
        reflect_ray_bundle = RayBundle(
            origins=origins[mask, :].detach(),
            directions=reflections[mask, :].detach(),
            pixel_area=torch.pi*sqradius[mask, :],
            nears=torch.ones_like(ray_bundle.nears[mask, :])*self.near,
            fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
        )
        ref_background_color = self.env.get_env(reflections[mask, :].detach(), sqradius[mask, :])

        ray_samples_ref_reciprocal = self.sampler_reciprocal(reflect_ray_bundle)
        density_outputs_ref_prop, _ = self.prop.get_density(ray_samples_ref_reciprocal)
        weights_ref_prop = ray_samples_ref_reciprocal.get_weights(density_outputs_ref_prop)
        
        ray_samples_ref_pdf = self.sampler_pdf(reflect_ray_bundle, ray_samples_ref_reciprocal, weights_ref_prop)
        density_outputs_ref, embedding_ref = self.field.get_density(ray_samples_ref_pdf)
        weights_ref = ray_samples_ref_pdf.get_weights(density_outputs_ref)
        accumulation_ref = self.renderer_accumulation(weights_ref)
        depth_ref = self.renderer_depth(weights_ref, ray_samples_ref_pdf)
                
        raw_normals_outputs_ref = self.field.get_pred_normals(embedding_ref)
        pred_normals_outputs_ref = nn.functional.normalize(raw_normals_outputs_ref, dim=-1)
        reflections_outputs_ref, n_dot_d_outputs_ref = self.field.get_reflection(ray_samples_ref_pdf.frustums.directions, pred_normals_outputs_ref)
        
        diff_outputs_ref = self.field.get_diff(embedding_ref) 
        tint_outputs_ref = self.field.get_tint(embedding_ref)
        
        roughness_outputs_ref = self.field.get_roughness(embedding_ref)
        low_outputs_ref = self.field.get_low(reflections_outputs_ref, n_dot_d_outputs_ref, roughness_outputs_ref, embedding_ref)
        
        outputs_ref = diff_outputs_ref + tint_outputs_ref*low_outputs_ref
        reflect = self.renderer_rgb(outputs_ref.detach(), weights_ref.detach(), background_color=ref_background_color)
                
        outputs["ref_rgb"][mask, :] = torch.clip(diff[mask, :] + tint[mask, :]*reflect, 0.0, 1.0)
        
        outputs["reflect"][mask, :] = reflect                
        
        # print debug       
        print(mask[mask].shape, mask[(accumulation>1e-2).reshape(-1)].shape, mask[(n_dot_d<0).reshape(-1)].shape)

        q=torch.tensor([0.1,0.3,0.5,0.7,0.9])
        print("roughness :",torch.quantile(roughness[mask,:].cpu(), q=q).detach().numpy())
        print("diffuse   :", torch.quantile(torch.sum((diff[mask,:]).cpu(), dim=-1), q=q).detach().numpy())
        print("tint      :", torch.quantile(torch.sum((tint[mask,:]).cpu(), dim=-1), q=q).detach().numpy())
        print("env :",torch.quantile(torch.sum(ref_background_color.cpu(),dim=-1), q=q).detach().numpy())
        
        print("accumulation :",torch.quantile(accumulation_ref.cpu(), q=q).detach().numpy())
        # print(torch.sum(accumulation_ref>1-torch.exp(-torch.pi*sqradius[mask])))

        print("depth        :",torch.quantile(depth_ref.cpu(), q=q).detach().numpy())
        print("acc/depth    :",torch.quantile(accumulation_ref.cpu()/depth_ref.cpu(), q=q).detach().numpy())

        
        
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)

        pred, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        pred_separate, image_separate = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["separate_rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        pred_ref, image_ref = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["ref_rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        
        rgb_loss_ = self.rgb_loss(image, pred)
        separate_rgb_loss_ = self.rgb_loss(image_separate, pred_separate)
        ref_rgb_loss_ = self.rgb_loss(image_ref, pred_ref)

        pred_normal_loss_ = torch.sum(pred_normal_loss(outputs["weights_list"][-1].detach(), outputs["normals_outputs"], outputs["pred_normals_outputs"]))
        orientation_loss_ = torch.sum(orientation_loss(outputs["weights_list"][-1].detach(), outputs["pred_normals_outputs"], outputs["directions"]))
        smooth_normal_loss = torch.sum(outputs["weights_list"][-1].detach()*(outputs["pred_normals"][...,None,:] - outputs["pred_normals_outputs"])**2)
        
        interlevel_loss_ = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
        distortion_loss_ = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        cauchy_loss_ = torch.sum(torch.log(1+outputs["density_list"][-1]**2))
        
        print(ref_rgb_loss_.item(), "<" if ref_rgb_loss_.item()<rgb_loss_.item() else ">", rgb_loss_.item())
        print(pred_normal_loss_.item(), orientation_loss_.item())
        print(interlevel_loss_.item(), distortion_loss_.item())

        if rgb_loss_.isnan().any() or ref_rgb_loss_.isnan().any() or pred_normal_loss_.isnan().any() or orientation_loss_.isnan().any():
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        loss_dict = {
            "rgb_loss": rgb_loss_,
            "ref_rgb_loss": ref_rgb_loss_,
            "separate_rgb_loss": separate_rgb_loss_,
            "pred_normal_loss": pred_normal_loss_,
            "orientation_loss": orientation_loss_,
            "smooth_normal_loss": smooth_normal_loss,
            "interlevel_loss": interlevel_loss_,
            "distortion_loss": distortion_loss_,
            "cauchy_loss": cauchy_loss_,
            }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_prop"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["ref_rgb"]
        acc_prop = colormaps.apply_colormap(outputs["accumulation_prop"])
        acc = colormaps.apply_colormap(outputs["accumulation"])

        assert self.config.collider_params is not None
        depth_prop = colormaps.apply_depth_colormap(
            outputs["depth_prop"],
            accumulation=outputs["accumulation_prop"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc_prop, acc], dim=1)
        combined_depth = torch.cat([depth_prop, depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_prop = torch.moveaxis(rgb_prop, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        rgb_prop = torch.clip(rgb_prop, min=0, max=1)
        rgb = torch.clip(rgb, min=0, max=1)

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        assert isinstance(ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
