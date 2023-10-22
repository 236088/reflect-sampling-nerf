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
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
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
from reflect_sampling_nerf.reflect_sampling_nerf_components import ReciprocalSampler

@dataclass
class ReflectSamplingNeRFModelConfig(ModelConfig):
    """ReflectSamplingNeRF Model Configuration.

    Add your custom model config parameters here.
    https://github.com/google-research/multinerf/blob/main/configs/blender_refnerf.gin
    """

    num_coarse_samples: int = 128
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    
    num_reflect_coarse_samples: int = 128
    """Number of samples in coarse field evaluation"""
    num_reflect_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss_coarse": 1.0,
        "rgb_loss_fine_coarse": 1.0,
        "rgb_loss_fine": 1.0,
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
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field = ReflectSamplingNeRFNerfField(
            position_encoding=position_encoding, 
            direction_encoding=direction_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)
        self.sampler_reciprocal = ReciprocalSampler(num_samples=self.config.num_reflect_coarse_samples)
        self.sampler_reflect_pdf = PDFSampler(num_samples=self.config.num_reflect_importance_samples, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color="white")
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_roughness = SemanticRenderer()
        self.renderer_factor = RGBRenderer()
        self.renderer_reflect = RGBRenderer(background_color="last_sample")

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
        
        diff_coarse = self.renderer_rgb(self.field.get_diff(embedding_coarse), weights_coarse)
        tint_coarse = self.renderer_factor(self.field.get_tint(embedding_coarse), weights_coarse)
        # roughness_coarse = self.renderer_roughness(self.field.get_roughness(embedding_coarse), weights_coarse)
        # embedding_coarse = self.field.get_bottleneck(embedding_coarse)
        # low_coarse = self.renderer_rgb(self.field.get_low(ray_samples_uniform.frustums.directions, embedding_coarse), weights_coarse)
        
        # rgb_coarse = self.field.get_padding(diff_coarse + tint_coarse * low_coarse)

        pred_normals_outputs_coarse = self.field.get_pred_normals(embedding_coarse)
        if self.training:
            normals_outputs_coarse = self.field.get_normals()
        else:
            normals_outputs_coarse = pred_normals_outputs_coarse
        n_dot_d_coarse = torch.sum(pred_normals_outputs_coarse*ray_samples_uniform.frustums.directions, dim=-1, keepdim=True)




        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        mean_fine, cov_fine = self.field.get_blob(ray_samples_pdf)
        mean_fine, cov_fine = self.field.contract(mean_fine, cov_fine)
        density_outputs_fine, embedding_fine = self.field.get_density(mean_fine, cov_fine, True)
        weights_fine = ray_samples_pdf.get_weights(density_outputs_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        
        diff_fine = self.renderer_rgb(self.field.get_diff(embedding_fine), weights_fine)
        tint_fine = self.renderer_factor(self.field.get_tint(embedding_fine), weights_fine)
        roughness_fine = self.renderer_roughness(self.field.get_roughness(embedding_fine), weights_fine)
        # embedding_fine = self.field.get_bottleneck(embedding_fine)
        # low_fine = self.renderer_rgb(self.field.get_low(ray_samples_pdf.frustums.directions, embedding_fine), weights_fine)
        
        # rgb_fine = self.field.get_padding(diff_fine + tint_fine*low_fine)
        
        mask = (accumulation_fine>1e-4).reshape(-1)

        pred_normals_outputs_fine = self.field.get_pred_normals(embedding_fine)
        if self.training:
            normals_outputs_fine = self.field.get_normals()
        else:
            normals_outputs_fine = pred_normals_outputs_fine
        n_dot_d_fine = torch.sum(pred_normals_outputs_fine*ray_samples_pdf.frustums.directions, dim=-1, keepdim=True)
        pred_normals_fine = self.renderer_normals(pred_normals_outputs_fine, weights_fine)

        print("roughness :",torch.quantile(roughness_fine.cpu(), q=torch.linspace(0,1,5)).detach().numpy())
        print("diff+tint :", torch.quantile(torch.sum((diff_fine+tint_fine).cpu(), dim=-1), q=torch.linspace(0,1,5)).detach().numpy())
        
        
        outputs = {
            "rgb_coarse": diff_coarse,
            "rgb_fine_coarse": diff_fine,
            "rgb_fine": diff_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "weights_coarse": weights_coarse.detach(),
            "weights_fine": weights_fine.detach(),
            "pred_normals_coarse":pred_normals_outputs_coarse,
            "pred_normals_fine":pred_normals_outputs_fine,
            "normals_coarse": normals_outputs_coarse.detach(),
            "normals_fine": normals_outputs_fine.detach(),
            "n_dot_d_coarse": n_dot_d_coarse,
            "n_dot_d_fine": n_dot_d_fine,
        }
        if not mask.any():
            return outputs
        
        print(mask[mask].shape)


        origins = ray_bundle.origins[mask, :] + depth_fine[mask, :]*ray_bundle.directions[mask, :]
        origins = origins.detach()
        n_dot_d = torch.sum(pred_normals_fine[mask, :]*ray_bundle.directions[mask, :], dim=-1, keepdim=True)
        reflections = ray_bundle.directions[mask, :] - 2*n_dot_d*pred_normals_fine[mask, :]
        reflections = torch.nn.functional.normalize(reflections)

        '''
        roughness to pixelarea as spherical gaussian lobe
        g(|x-mean|, sigma^2) = exp(-|x-mean|^2/(2*sigma^2))
        sigma^2 = 1/sharpness
        sharpness = 2/roughness^2/(4*|direction*normal|)
        sigma^2 = roughness^2*2*|direction*normal|
        '''
        reflect_ray_bundle = RayBundle(
            origins=origins,
            directions=reflections,
            pixel_area=torch.abs(n_dot_d)*2*torch.pi*roughness_fine[mask, :]**2,
            nears=torch.zeros_like(ray_bundle.nears[mask, :]),
            fars=torch.ones_like(ray_bundle.fars[mask, :])*2**8
        )

        ray_samples_reciprocal = self.sampler_reciprocal(reflect_ray_bundle)

        mean_reflect_coarse, cov_reflect_coarse = self.field.get_blob(ray_samples_reciprocal)
        mean_reflect_coarse, cov_reflect_coarse = self.field.contract(mean_reflect_coarse, cov_reflect_coarse)
        density_outputs_reflect_coarse, embedding_reflect_coarse = self.field.get_reflect_density(mean_reflect_coarse, cov_reflect_coarse)
        weights_reflect_coarse = ray_samples_reciprocal.get_weights(density_outputs_reflect_coarse)
        
        # roughness_reflect_outputs_coarse = self.field.get_roughness(embedding_reflect_coarse)
        bottleneck_reflect_coarse = self.field.get_bottleneck(embedding_reflect_coarse)
        low_reflect_outputs_coarse = self.field.get_low(ray_samples_reciprocal.frustums.directions, bottleneck_reflect_coarse)

        # pred_normals_outputs_reflect_coarse = self.field.get_pred_normals(embedding_reflect_coarse)
        # n_dot_d_reflect_coarse = torch.sum(pred_normals_outputs_reflect_coarse*ray_samples_reciprocal.frustums.directions, dim=-1, keepdim=True)

        # reflect_coarse = self.renderer_reflect(low_reflect_outputs_coarse, weights_reflect_coarse)
        
        diff_reflect_outputs_coarse = self.field.get_diff(embedding_reflect_coarse)
        tint_reflect_outputs_coarse = self.field.get_tint(embedding_reflect_coarse)
        reflect_coarse = self.renderer_reflect(diff_reflect_outputs_coarse + tint_reflect_outputs_coarse * low_reflect_outputs_coarse, weights_reflect_coarse)
        
        rgb_coarse = outputs["rgb_coarse"]
        rgb_coarse[mask, :] = rgb_coarse[mask, :] + tint_coarse[mask, :] * reflect_coarse
        outputs["rgb_coarse"] = self.field.get_padding(rgb_coarse)
        
        rgb_fine_coarse = outputs["rgb_fine_coarse"]
        rgb_fine_coarse[mask, :] = rgb_fine_coarse[mask, :] + tint_fine[mask, :] * reflect_coarse
        outputs["rgb_fine_coarse"] = self.field.get_padding(rgb_fine_coarse)



        ray_samples_recflect_pdf = self.sampler_reflect_pdf(reflect_ray_bundle, ray_samples_reciprocal, weights_reflect_coarse)

        mean_reflect_fine, cov_reflect_fine = self.field.get_blob(ray_samples_recflect_pdf)
        mean_reflect_fine, cov_reflect_fine = self.field.contract(mean_reflect_fine, cov_reflect_fine)
        density_outputs_reflect_fine, embedding_reflect_fine = self.field.get_reflect_density(mean_reflect_fine, cov_reflect_fine)
        weights_reflect_fine = ray_samples_recflect_pdf.get_weights(density_outputs_reflect_fine)
        
        # roughness_reflect_outputs_fine = self.field.get_roughness(embedding_reflect_fine)
        bottleneck_reflect_fine = self.field.get_bottleneck(embedding_reflect_fine)
        low_reflect_outputs_fine = self.field.get_low(ray_samples_recflect_pdf.frustums.directions, bottleneck_reflect_fine)

        # pred_normals_outputs_reflect_fine = self.field.get_pred_normals(embedding_reflect_fine)
        # n_dot_d_reflect_fine = torch.sum(pred_normals_outputs_reflect_fine*ray_samples_recflect_pdf.frustums.directions, dim=-1, keepdim=True)
        
        # reflect_fine = self.renderer_reflect(low_reflect_outputs_fine, weights_reflect_fine)
        
        diff_reflect_outputs_fine = self.field.get_diff(embedding_reflect_fine)
        tint_reflect_outputs_fine = self.field.get_tint(embedding_reflect_fine)
        reflect_fine = self.renderer_reflect(diff_reflect_outputs_fine + tint_reflect_outputs_fine*low_reflect_outputs_fine, weights_reflect_fine)
        reflect_fine = self.field.get_padding(reflect_fine)
        
        rgb_fine = outputs["rgb_fine"]
        rgb_fine[mask, :] = rgb_fine[mask, :] + tint_fine[mask, :] * reflect_fine
        outputs["rgb_fine"] = self.field.get_padding(rgb_fine)

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_coarse, image_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        pred_fine_coarse, image_fine_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine_coarse"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        pred_fine, image_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        
        rgb_loss_coarse = self.rgb_loss(image_coarse, pred_coarse)
        rgb_loss_fine_coarse = self.rgb_loss(image_fine_coarse, pred_fine_coarse)
        rgb_loss_fine = self.rgb_loss(image_fine, pred_fine)

        predicted_normal_loss_coarse = torch.sum(outputs["weights_coarse"]*torch.sum((outputs["normals_coarse"]-outputs["pred_normals_coarse"])**2, dim=-1, keepdim=True))
        predicted_normal_loss_fine = torch.sum(outputs["weights_fine"]*torch.sum((outputs["normals_fine"]-outputs["pred_normals_fine"])**2, dim=-1, keepdim=True))

        orientation_loss_coarse = torch.sum(outputs["weights_coarse"]*torch.max(torch.zeros_like(outputs["n_dot_d_coarse"]),outputs["n_dot_d_coarse"])**2)
        orientation_loss_fine = torch.sum(outputs["weights_fine"]*torch.max(torch.zeros_like(outputs["n_dot_d_fine"]),outputs["n_dot_d_fine"])**2)
        
        print(rgb_loss_fine.item()/rgb_loss_fine_coarse.item(), rgb_loss_fine_coarse.item()/rgb_loss_coarse.item(), rgb_loss_coarse.item())
        print(predicted_normal_loss_fine.item(), orientation_loss_fine.item())

        if rgb_loss_fine.isnan().any()  and predicted_normal_loss_fine.isnan().any() and orientation_loss_fine.isnan().any():
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine_coarse": rgb_loss_fine_coarse,
            "rgb_loss_fine": rgb_loss_fine,
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
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
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
