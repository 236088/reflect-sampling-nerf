"""
ReflectSamplingNeRF Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.model_components.losses import MSELoss
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
from reflect_sampling_nerf.reflect_sampling_nerf_field import ReflectSamplingNeRFNerfField
from reflect_sampling_nerf.reflect_sampling_nerf_components import ReciprocalSampler, IntegratedSHEncoding

@dataclass
class ReflectSamplingNeRFModelConfig(ModelConfig):
    """ReflectSamplingNeRF Model Configuration.

    Add your custom model config parameters here.
    https://github.com/google-research/multinerf/blob/main/configs/blender_refnerf.gin
    """

    num_coarse_samples: int = 128
    """Number of samples in rgb_coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in rgb_fine field evaluation"""
    
    num_reflect_coarse_samples: int = 64
    """Number of samples in rgb_coarse field evaluation"""
    num_reflect_importance_samples: int = 64
    """Number of samples in rgb_fine field evaluation"""
    
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "loss_coarse": 1e-2,
        "loss_fine": 1e-2,
        "loss_reflect_coarse": 1.0,
        "loss_reflect_fine": 1.0,
        "loss_reflect_interpolation": 1.0,
        "predicted_normal_loss_coarse": 3e-5,
        "predicted_normal_loss_fine": 3e-4,
        "orientation_loss_coarse": 1e-2,
        "orientation_loss_fine": 1e-1,
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
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = IntegratedSHEncoding()

        self.field = ReflectSamplingNeRFNerfField(
            position_encoding=position_encoding, 
            direction_encoding=direction_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)
        self.sampler_reciprocal = ReciprocalSampler(num_samples=self.config.num_reflect_coarse_samples, tan=0.25)
        self.sampler_reflect_pdf = PDFSampler(num_samples=self.config.num_reflect_importance_samples, include_original=False)
        self.far = 2**8
        self.near = 1.0/32
        
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
        param_groups["fields"] = list(self.field.parameters())
        return param_groups
       
        
    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")


        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        mean_coarse, cov_coarse = self.field.get_blob(ray_samples_uniform)
        mean_coarse, cov_coarse = self.field.contract(mean_coarse, cov_coarse)
        density_outputs_coarse, embedding_coarse = self.field.get_density(mean_coarse, cov_coarse, True)
        weights_coarse = ray_samples_uniform.get_weights(density_outputs_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        pred_normals_outputs_coarse = self.field.get_pred_normals(embedding_coarse)
        normals_outputs_coarse = self.field.get_normals() if self.training else pred_normals_outputs_coarse
        
        reflections_coarse, n_dot_d_coarse = self.field.get_reflection(ray_samples_uniform.frustums.directions, pred_normals_outputs_coarse)
        
        # diff_outputs_coarse = self.field.get_diff(embedding_coarse)
        # tint_outputs_coarse = self.field.get_tint(embedding_coarse)
        
        # roughness_outputs_coarse = self.field.get_roughness(embedding_coarse)
        outputs_coarse = self.field.get_mid(ray_samples_uniform.frustums.directions, embedding_coarse, True)
        rgb_coarse = self.renderer_rgb(outputs_coarse.detach(), weights_coarse)
        rgb_coarse = torch.clip(rgb_coarse, 0.0, 1.0)
        
        # diff_coarse = self.renderer_rgb(diff_outputs_coarse, weights_coarse)
        # tint_coarse = self.renderer_factor(tint_outputs_coarse, weights_coarse)
        
        

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        mean_fine, cov_fine = self.field.get_blob(ray_samples_pdf)
        mean_fine, cov_fine = self.field.contract(mean_fine, cov_fine)
        density_outputs_fine, embedding_fine = self.field.get_density(mean_fine, cov_fine, True)
        weights_fine = ray_samples_pdf.get_weights(density_outputs_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        
        
        pred_normals_outputs_fine = self.field.get_pred_normals(embedding_fine)
        normals_outputs_fine = self.field.get_normals() if self.training else pred_normals_outputs_fine
        reflections_fine, n_dot_d_fine = self.field.get_reflection(ray_samples_pdf.frustums.directions, pred_normals_outputs_fine)
        
        roughness_outputs_fine = self.field.get_roughness(embedding_fine)
        
        outputs_fine = self.field.get_mid(ray_samples_pdf.frustums.directions, embedding_fine, True)
                
        rgb_fine = self.renderer_rgb(outputs_fine.detach(), weights_fine)
        rgb_fine = torch.clip(rgb_fine, 0.0, 1.0)

        diff_outputs_fine = self.field.get_diff(embedding_fine) 
        tint_outputs_fine = self.field.get_tint(embedding_fine)
        diff_fine = self.renderer_rgb(diff_outputs_fine, weights_fine)
        tint_fine = self.renderer_factor(tint_outputs_fine, weights_fine)
        
        pred_normals_fine = self.renderer_normals(pred_normals_outputs_fine, weights_fine)
        n_dot_d = torch.sum(pred_normals_fine*ray_bundle.directions, dim=-1, keepdim=True)

        roughness_fine = self.renderer_roughness(roughness_outputs_fine, weights_fine)
        
        mask = torch.logical_and(accumulation_fine>1e-2, n_dot_d<0).reshape(-1)
        print(mask[mask].shape, mask[(accumulation_fine>1e-2).reshape(-1)].shape, mask[(n_dot_d<0).reshape(-1)].shape)
        
        
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "reflect_coarse": self.background_color.to(rgb_fine.device).expand(rgb_fine.shape)*(1.0-accumulation_fine),
            "reflect_fine": self.background_color.to(rgb_fine.device).expand(rgb_fine.shape)*(1.0-accumulation_fine),
            "reflect_interpolation": self.background_color.to(rgb_fine.device).expand(rgb_fine.shape)*(1.0-accumulation_fine),
            "accumulation_coarse": accumulation_coarse.detach(),
            "accumulation_fine": accumulation_fine.detach(),
            "depth_coarse": depth_coarse.detach(),
            "depth_fine": depth_fine.detach(),
            "weights_coarse": weights_coarse.detach(),
            "weights_fine": weights_fine.detach(),
            "pred_normals_coarse": pred_normals_outputs_coarse,
            "pred_normals_fine": pred_normals_outputs_fine,
            "normals_coarse": normals_outputs_coarse.detach(),
            "normals_fine": normals_outputs_fine.detach(),
            "n_dot_d_coarse": n_dot_d_coarse,
            "n_dot_d_fine": n_dot_d_fine,
            "diff":diff_fine,
            "tint":tint_fine,
            "roughness":roughness_fine,
            "mask":mask,
        }
        if not mask.any():
            return outputs

        q=torch.tensor([0.1,0.3,0.5,0.7,0.9])
        print("roughness :",torch.quantile(roughness_fine[mask,:].cpu(), q=q).detach().numpy())
        print("diffuse   :", torch.quantile(torch.sum((diff_fine[mask,:]).cpu(), dim=-1), q=q).detach().numpy())
        print("tint      :", torch.quantile(torch.sum((tint_fine[mask,:]).cpu(), dim=-1), q=q).detach().numpy())

        origins = ray_bundle.origins[mask, :] + depth_fine[mask, :]*ray_bundle.directions[mask, :]
        origins=origins.detach()
        reflections = ray_bundle.directions[mask, :] - 2*n_dot_d[mask, :]*pred_normals_fine[mask, :]
        reflections = torch.nn.functional.normalize(reflections, dim=-1)
        reflections = reflections.detach()

        '''        
        roughness to pixelarea as spherical gaussian lobe
        g(|x-mean|, sigma^2) = exp(-|x-mean|^2/(2*sigma^2))
        sigma^2 = 1/sharpness
        sharpness = 2/roughness^2/(4*|direction*normal|)
        sigma^2 = roughness^2*2*|direction*normal|
        sigma as radius
        '''

        
        sqradius = 2*torch.abs(n_dot_d[mask, :].detach())*roughness_fine[mask, :]**2
        reflect_ray_bundle = RayBundle(
            origins=origins,
            directions=reflections,
            pixel_area=torch.pi*sqradius,
            nears=torch.zeros_like(ray_bundle.nears[mask, :])*self.near,
            fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
        )
        background_color = self.field.get_inf_color(reflections, sqradius)

        ray_samples_reciprocal = self.sampler_reciprocal(reflect_ray_bundle)
        mean_reflect_coarse, cov_reflect_coarse = self.field.get_blob(ray_samples_reciprocal)
        mean_reflect_coarse, cov_reflect_coarse = self.field.contract(mean_reflect_coarse, cov_reflect_coarse)
        density_outputs_reflect_coarse, embedding_reflect_coarse = self.field.get_density(mean_reflect_coarse, cov_reflect_coarse)
        
        delta_density_reflect_coarse = ray_samples_reciprocal.deltas * density_outputs_reflect_coarse
        transmittance_reflect_coarse = torch.cumsum(delta_density_reflect_coarse[..., :-1, :], dim=-2)
        transmittance_reflect_coarse = torch.cat(
            [torch.zeros((*transmittance_reflect_coarse.shape[:1], 1, 1), device=density_outputs_reflect_coarse.device), transmittance_reflect_coarse], dim=-2
        )
        transmittance_reflect_coarse = torch.exp(-transmittance_reflect_coarse)  # [..., "num_samples"]
        weights_reflect_coarse = torch.nan_to_num((1 - torch.exp(-delta_density_reflect_coarse)) * transmittance_reflect_coarse)
        
        outputs["reflect_depth_coarse"] = self.renderer_depth(weights_reflect_coarse, ray_samples_reciprocal)
        
        # roughness_outputs_reflect_coarse = self.field.get_roughness(embedding_reflect_coarse)
        outputs_reflect_coarse = self.field.get_mid(ray_samples_reciprocal.frustums.directions, embedding_reflect_coarse, True)
        
        # diff_outputs_reflect_coarse = self.field.get_diff(embedding_reflect_coarse)
        # tint_outputs_reflect_coarse = self.field.get_tint(embedding_reflect_coarse)
        
        reflect_coarse = self.renderer_reflect(outputs_reflect_coarse, weights_reflect_coarse, background_color=background_color)
        # reflect_coarse = torch.clip(reflect_coarse, 0.0, 1.0)
        outputs["reflect_coarse"][mask, :] = diff_fine[mask, :] + tint_fine[mask, :] * reflect_coarse
        outputs["reflect_coarse"][mask, :] = torch.clip(outputs["reflect_coarse"][mask, :], 0.0, 1.0)
            
        if self.training:
            sqradius=torch.zeros_like(roughness_fine[mask, :])
            reflect_ray_bundle = RayBundle(
                origins=origins,
                directions=reflections,
                pixel_area=sqradius,
                nears=torch.zeros_like(ray_bundle.nears[mask, :])*self.near,
                fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
            )
            background_color = self.field.get_inf_color(reflections, sqradius)

            ray_samples_reciprocal = self.sampler_reciprocal(reflect_ray_bundle)
            mean_reflect_smooth, cov_reflect_smooth = self.field.get_blob(ray_samples_reciprocal)
            mean_reflect_smooth, cov_reflect_smooth = self.field.contract(mean_reflect_smooth, cov_reflect_smooth)
            density_outputs_reflect_smooth, embedding_reflect_smooth = self.field.get_density(mean_reflect_smooth, cov_reflect_smooth)
            weights_reflect_smooth = ray_samples_reciprocal.get_weights(density_outputs_reflect_smooth)
            
            # roughness_outputs_reflect_smooth = self.field.get_roughness(embedding_reflect_smooth)
            outputs_smooth = self.field.get_mid(ray_samples_reciprocal.frustums.directions, embedding_reflect_smooth, True)
            
            reflect_smooth = self.renderer_reflect(outputs_smooth, weights_reflect_smooth, background_color=background_color)
            
            sqradius=2*torch.abs(n_dot_d[mask, :].detach())*torch.ones_like(roughness_fine[mask, :])
            reflect_ray_bundle = RayBundle(
                origins=origins,
                directions=reflections,
                pixel_area=torch.pi*sqradius,
                nears=torch.zeros_like(ray_bundle.nears[mask, :])*self.near,
                fars=torch.ones_like(ray_bundle.fars[mask, :])*self.far
            )
            background_color = self.field.get_inf_color(reflections, sqradius)

            ray_samples_reciprocal = self.sampler_reciprocal(reflect_ray_bundle)
            mean_reflect_rough, cov_reflect_rough = self.field.get_blob(ray_samples_reciprocal)
            mean_reflect_rough, cov_reflect_rough = self.field.contract(mean_reflect_rough, cov_reflect_rough)
            density_outputs_reflect_rough, embedding_reflect_rough = self.field.get_density(mean_reflect_rough, cov_reflect_rough)
            weights_reflect_rough = ray_samples_reciprocal.get_weights(density_outputs_reflect_rough)
            
            # roughness_outputs_reflect_rough = self.field.get_roughness(embedding_reflect_rough)
            outputs_rough = self.field.get_mid(ray_samples_reciprocal.frustums.directions, embedding_reflect_rough, True)
            
            reflect_rough = self.renderer_reflect(outputs_rough, weights_reflect_rough, background_color=background_color)
            reflect_interpolation = (1-roughness_fine[mask, :])*reflect_smooth + roughness_fine[mask, :]*reflect_rough
            outputs["reflect_interpolation"][mask, :] = diff_fine[mask, :] + tint_fine[mask, :] * reflect_interpolation
            
            
        ray_samples_reflect_pdf = self.sampler_reflect_pdf(reflect_ray_bundle, ray_samples_reciprocal, weights_reflect_coarse)

        mean_reflect_fine, cov_reflect_fine = self.field.get_blob(ray_samples_reflect_pdf)
        mean_reflect_fine, cov_reflect_fine = self.field.contract(mean_reflect_fine, cov_reflect_fine)
        density_outputs_reflect_fine, embedding_reflect_fine = self.field.get_density(mean_reflect_fine, cov_reflect_fine)
        
        delta_density_reflect_fine = ray_samples_reflect_pdf.deltas * density_outputs_reflect_fine
        transmittance_reflect_fine = torch.cumsum(delta_density_reflect_fine[..., :-1, :], dim=-2)
        transmittance_reflect_fine = torch.cat(
            [torch.zeros((*transmittance_reflect_fine.shape[:1], 1, 1), device=density_outputs_reflect_fine.device), transmittance_reflect_fine], dim=-2
        )
        transmittance_reflect_fine = torch.exp(-transmittance_reflect_fine)  # [..., "num_samples"]
        weights_reflect_fine = torch.nan_to_num((1 - torch.exp(-delta_density_reflect_fine)) * transmittance_reflect_fine)
        
        outputs["depth_reflect_fine"] = self.renderer_depth(weights_reflect_fine, ray_samples_reflect_pdf)
        print("depth :",torch.quantile(outputs["depth_reflect_fine"].cpu(), q=q).detach().numpy())
        
        
        # roughness_outputs_reflect_fine = self.field.get_roughness(embedding_reflect_fine)
        outputs_reflect_fine = self.field.get_mid(ray_samples_reflect_pdf.frustums.directions, embedding_reflect_fine, True)
        
        # diff_outputs_reflect_fine = self.field.get_diff(embedding_reflect_fine)
        # tint_outputs_reflect_fine = self.field.get_tint(embedding_reflect_fine)
        
        reflect_fine = self.renderer_reflect(outputs_reflect_fine, weights_reflect_fine, background_color=background_color)
        # reflect_fine = torch.clip(reflect_fine, 0.0, 1.0)
        outputs["reflect_fine"][mask, :] = diff_fine[mask, :] + tint_fine[mask, :] * reflect_fine
        outputs["reflect_fine"][mask, :] = torch.clip(outputs["reflect_fine"][mask, :], 0.0, 1.0)
        
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)

        pred_coarse, image_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        pred_fine, image_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        pred_reflect_coarse, image_reflect_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["reflect_coarse"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        pred_reflect_fine, image_reflect_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["reflect_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        
        loss_coarse = self.rgb_loss(image_coarse, pred_coarse)
        loss_fine = self.rgb_loss(image_fine, pred_fine)
        loss_reflect_coarse = self.rgb_loss(image_reflect_coarse, pred_reflect_coarse)
        loss_reflect_fine = self.rgb_loss(image_reflect_fine, pred_reflect_fine)
        loss_reflect_interpolation = torch.zeros_like(loss_reflect_coarse)
        
        if self.training:
            pred_reflect_interpolation, image_reflect_interpolation = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["reflect_interpolation"],
                pred_accumulation=outputs["accumulation_fine"],
                gt_image=image,
            )
            loss_reflect_interpolation = self.rgb_loss(image_reflect_interpolation, pred_reflect_interpolation)
        predicted_normal_loss_coarse = torch.sum(outputs["weights_coarse"]*torch.sum((outputs["normals_coarse"]-outputs["pred_normals_coarse"])**2, dim=-1, keepdim=True))
        predicted_normal_loss_fine = torch.sum(outputs["weights_fine"]*torch.sum((outputs["normals_fine"]-outputs["pred_normals_fine"])**2, dim=-1, keepdim=True))

        orientation_loss_coarse = torch.sum(outputs["weights_coarse"]*torch.max(torch.zeros_like(outputs["n_dot_d_coarse"]),outputs["n_dot_d_coarse"])**2)
        orientation_loss_fine = torch.sum(outputs["weights_fine"]*torch.max(torch.zeros_like(outputs["n_dot_d_fine"]),outputs["n_dot_d_fine"])**2)
        
        print(loss_reflect_fine.item(), loss_fine.item())
        print(predicted_normal_loss_fine.item(), orientation_loss_fine.item())

        if loss_reflect_fine.isnan().any() and predicted_normal_loss_fine.isnan().any() and orientation_loss_fine.isnan().any():
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        loss_dict = {
            "loss_coarse": loss_coarse,
            "loss_fine": loss_fine,
            "loss_reflect_coarse": loss_reflect_coarse,
            "loss_reflect_fine": loss_reflect_fine,
            "loss_reflect_interpolation": loss_reflect_interpolation,
            "predicted_normal_loss_coarse": predicted_normal_loss_coarse,
            "predicted_normal_loss_fine": predicted_normal_loss_fine,
            "orientation_loss_coarse": orientation_loss_coarse,
            "orientation_loss_fine": orientation_loss_fine,
            }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["low_coarse"]
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

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=0, max=1)
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
