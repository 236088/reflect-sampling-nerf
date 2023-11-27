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
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, UniformLinDispPiecewiseSampler
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
from reflect_sampling_nerf.reflect_sampling_nerf_components import IntegratedSHEncoding, PolyhedronFFEncoding

@dataclass
class ReflectSamplingNeRFModelConfig(ModelConfig):
    """ReflectSamplingNeRF Model Configuration.

    Add your custom model config parameters here.
    https://github.com/google-research/multinerf/blob/main/configs/blender_refnerf.gin
    """

    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 32
    """Number of samples in fine field evaluation"""
    
    num_reflect_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_reflect_importance_samples: int = 32
    """Number of samples in fine field evaluation"""
    
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "loss_fine": 1.0,
        "loss_reflect_fine": 1.0,
        "pred_normal_loss_value": 3e-4,
        "orientation_loss_value": 1e-1,
        "interlevel_loss_value": 1.0,
        "distortion_loss_value": 1e-2,
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
            num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
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
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)
        self.sampler_lindisp = UniformLinDispPiecewiseSampler(num_samples=self.config.num_reflect_coarse_samples)
        self.far = 2**16
        self.near = 1.0/16
        
        # renderers
        self.background_color = colors.WHITE
        self.renderer_rgb = RGBRenderer(background_color=self.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_roughness = SemanticRenderer()
        self.renderer_factor = RGBRenderer()
        self.renderer_reflect = RGBRenderer()

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
        param_groups["fields"] = list(self.prop.parameters()) + list(self.env.parameters()) + list(self.field.parameters())
        return param_groups
       
        
    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")


        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        density_outputs_coarse, embedding_coarse = self.prop.get_density(ray_samples_uniform, True)
        weights_coarse = ray_samples_uniform.get_weights(density_outputs_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
               
        

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        density_outputs_fine, embedding_fine = self.field.get_density(ray_samples_pdf, True)
        weights_fine = ray_samples_pdf.get_weights(density_outputs_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        
        
        pred_normals_outputs_fine = self.field.get_pred_normals(embedding_fine)
        normals_outputs_fine = self.field.get_normals() if self.training else pred_normals_outputs_fine
        reflections_outputs_fine, n_dot_d_outputs_fine = self.field.get_reflection(ray_samples_pdf.frustums.directions, pred_normals_outputs_fine)
            
        diff_outputs_fine = self.field.get_diff(embedding_fine) 
        tint_outputs_fine = self.field.get_tint(embedding_fine)
        
        # low_outputs_fine = self.field.get_low(embedding_fine, True)
        roughness_outputs_fine = self.field.get_roughness(embedding_fine, activation=nn.Softplus())
        mid_outputs_fine = self.field.get_mid(reflections_outputs_fine, n_dot_d_outputs_fine, roughness_outputs_fine, embedding_fine, True)
        
        outputs_fine = diff_outputs_fine + tint_outputs_fine*mid_outputs_fine
        rgb_fine = self.renderer_rgb(outputs_fine, weights_fine)
        rgb_fine = torch.clip(rgb_fine, 0.0, 1.0)


        diff_fine = self.renderer_rgb(diff_outputs_fine, weights_fine)
        diff_fine = diff_fine.detach()
        tint_fine = self.renderer_factor(tint_outputs_fine, weights_fine)
        tint_fine = tint_fine.detach()
        
        pred_normals_fine = self.renderer_normals(pred_normals_outputs_fine, weights_fine)
        pred_normals_fine = pred_normals_fine.detach()
        n_dot_d = torch.sum(pred_normals_fine*ray_bundle.directions, dim=-1, keepdim=True)
        n_dot_d = n_dot_d.detach()

        roughness_outputs = self.field.get_roughness(embedding_fine, activation=lambda x: nn.functional.sigmoid(-x))
        roughness = 1 - self.renderer_roughness(roughness_outputs, weights_fine)
        roughness = roughness.detach()
        # roughness_fine = self.renderer_roughness(roughness_outputs_fine, weights_fine)
        
        mask = torch.logical_and(accumulation_fine>1e-2, n_dot_d<0).reshape(-1)
        print(mask[mask].shape, mask[(accumulation_fine>1e-2).reshape(-1)].shape, mask[(n_dot_d<0).reshape(-1)].shape)
        
        
        outputs = {
            "rgb_fine": rgb_fine,
            "reflect_fine": self.background_color.to(rgb_fine.device).expand(rgb_fine.shape)*(1.0-accumulation_fine),
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "weights_list": [weights_coarse, weights_fine],
            "ray_samples_list": [ray_samples_uniform, ray_samples_pdf],
            "pred_normals_fine": pred_normals_outputs_fine,
            "normals_fine": normals_outputs_fine,
            "directions": ray_bundle.directions,
            "diff":diff_fine,
            "tint":tint_fine,
            "roughness":roughness,
            "mask":mask,
        }
        if not mask.any():
            return outputs

        q=torch.tensor([0.1,0.3,0.5,0.7,0.9])
        print("roughness :",torch.quantile(roughness[mask,:].cpu(), q=q).detach().numpy())
        print("diffuse   :", torch.quantile(torch.sum((diff_fine[mask,:]).cpu(), dim=-1), q=q).detach().numpy())
        print("tint      :", torch.quantile(torch.sum((tint_fine[mask,:]).cpu(), dim=-1), q=q).detach().numpy())

        origins = ray_bundle.origins[mask, :] + depth_fine[mask, :]*ray_bundle.directions[mask, :]
        origins=origins.detach()
        reflections = ray_bundle.directions[mask, :] - 2*n_dot_d[mask, :]*pred_normals_fine[mask, :]
        reflections = torch.nn.functional.normalize(reflections, dim=-1)
        # reflections = reflections.detach()
        sqradius = 2*torch.abs(n_dot_d[mask, :])*roughness[mask, :]**2
        sqradius = sqradius.detach()

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
            origins=origins,
            directions=pred_normals_fine[mask, :],
            pixel_area=torch.pi*2*torch.ones_like(sqradius),
            nears=torch.rand_like(ray_bundle.nears[mask, :])*self.near,
            fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
        )
        normal_background_color = self.env.get_env(pred_normals_fine[mask, :], 2*torch.ones_like(sqradius))

        ray_samples_normal_lindisp = self.sampler_lindisp(normal_ray_bundle)
        density_outputs_normal_coarse, embedding_normal_coarse = self.prop.get_density(ray_samples_normal_lindisp)
        weights_normal_coarse = ray_samples_normal_lindisp.get_weights(density_outputs_normal_coarse)
      
        ray_samples_normal_pdf = self.sampler_pdf(normal_ray_bundle, ray_samples_normal_lindisp, weights_normal_coarse)

        density_outputs_normal_fine, embedding_normal_fine = self.field.get_density(ray_samples_normal_pdf)
        weights_normal_fine = ray_samples_normal_pdf.get_weights(density_outputs_normal_fine)
        
        pred_normals_outputs_normal_fine = self.field.get_pred_normals(embedding_normal_fine)
        reflections_outputs_normal_fine, n_dot_d_outputs_normal_fine = self.field.get_reflection(ray_samples_normal_pdf.frustums.directions, pred_normals_outputs_normal_fine)
            
        diff_outputs_normal_fine = self.field.get_diff(embedding_normal_fine) 
        tint_outputs_normal_fine = self.field.get_tint(embedding_normal_fine)
        
        # low_outputs_normal_fine = self.field.get_low(embedding_normal_fine, True)
        roughness_outputs_normal_fine = self.field.get_roughness(embedding_normal_fine, activation=nn.Softplus())
        mid_outputs_normal_fine = self.field.get_mid(reflections_outputs_normal_fine, n_dot_d_outputs_normal_fine, roughness_outputs_normal_fine, embedding_normal_fine, True)
        
        outputs_normal_fine = diff_outputs_normal_fine.detach() + tint_outputs_normal_fine.detach()*mid_outputs_normal_fine
        normal_fine = self.renderer_reflect(outputs_normal_fine, weights_normal_fine.detach(), normal_background_color)
        '''
        
        
        reflect_ray_bundle = RayBundle(
            origins=origins,
            directions=reflections,
            pixel_area=torch.pi*sqradius,
            nears=torch.rand_like(ray_bundle.nears[mask, :])*self.near,
            fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
        )
        reflect_background_color = self.env.get_env(reflections, sqradius)

        ray_samples_reflect_lindisp = self.sampler_lindisp(reflect_ray_bundle)
        
        density_outputs_reflect_coarse, embedding_reflect_coarse = self.prop.get_density(ray_samples_reflect_lindisp)
        weights_reflect_coarse = ray_samples_reflect_lindisp.get_weights(density_outputs_reflect_coarse)
      
        ray_samples_reflect_pdf = self.sampler_pdf(reflect_ray_bundle, ray_samples_reflect_lindisp, weights_reflect_coarse)

        density_outputs_reflect_fine, embedding_reflect_fine = self.field.get_density(ray_samples_reflect_pdf)
        weights_reflect_fine = ray_samples_reflect_pdf.get_weights(density_outputs_reflect_fine)
        
        pred_normals_outputs_reflect_fine = self.field.get_pred_normals(embedding_reflect_fine)
        reflections_outputs_reflect_fine, n_dot_d_outputs_reflect_fine = self.field.get_reflection(ray_samples_reflect_pdf.frustums.directions, pred_normals_outputs_reflect_fine)
        
        diff_outputs_reflect_fine = self.field.get_diff(embedding_reflect_fine) 
        tint_outputs_reflect_fine = self.field.get_tint(embedding_reflect_fine)
        
        # low_outputs_reflect_fine = self.field.get_low(embedding_reflect_fine, True)
        roughness_outputs_reflect_fine = self.field.get_roughness(embedding_reflect_fine, activation=nn.Softplus())
        mid_outputs_reflect_fine = self.field.get_mid(reflections_outputs_reflect_fine, n_dot_d_outputs_reflect_fine, roughness_outputs_reflect_fine.detach(), embedding_reflect_fine, True)
        
        outputs_reflect_fine = diff_outputs_reflect_fine.detach() + tint_outputs_reflect_fine.detach()*mid_outputs_reflect_fine
        reflect_fine = self.renderer_reflect(outputs_reflect_fine, weights_reflect_fine, background_color=reflect_background_color)
        
        
        
        outputs["reflect_fine"][mask, :] = diff_fine[mask, :] + tint_fine[mask, :]*reflect_fine
        outputs["reflect_fine"][mask, :] = torch.clip(outputs["reflect_fine"][mask, :], 0.0, 1.0)
        
        accumulation_reflect = self.renderer_accumulation(weights_reflect_fine)
        print("accumulation :",torch.quantile(accumulation_reflect.cpu(), q=q).detach().numpy())
        
        depth_reflect = self.renderer_depth(weights_reflect_fine, ray_samples_reflect_pdf)
        print("depth        :",torch.quantile(depth_reflect.cpu(), q=q).detach().numpy())
        
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
        
        loss_fine = self.rgb_loss(image_fine, pred_fine)
        loss_reflect_fine = self.rgb_loss(image_reflect_fine, pred_reflect_fine)

        pred_normal_loss_value = torch.sum(pred_normal_loss(outputs["weights_list"][-1].detach(), outputs["normals_fine"], outputs["pred_normals_fine"]))
        orientation_loss_value = torch.sum(orientation_loss(outputs["weights_list"][-1].detach(), outputs["pred_normals_fine"], outputs["directions"]))
        
        interlevel_loss_value = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
        distortion_loss_value = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        
        print(loss_reflect_fine.item(), "<" if loss_reflect_fine.item()<loss_fine.item() else ">", loss_fine.item())
        print(pred_normal_loss_value.item(), orientation_loss_value.item())
        print(interlevel_loss_value.item(), distortion_loss_value.item())

        if loss_reflect_fine.isnan().any() and pred_normal_loss_value.isnan().any() and orientation_loss_value.isnan().any():
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        loss_dict = {
            "loss_fine": loss_fine,
            "loss_reflect_fine": loss_reflect_fine,
            "pred_normal_loss_value": pred_normal_loss_value,
            "orientation_loss_value": orientation_loss_value,
            "interlevel_loss_value": interlevel_loss_value,
            "distortion_loss_value": distortion_loss_value,
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
