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
from nerfstudio.field_components.encodings import NeRFEncoding
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
    
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss_coarse": 1.0,
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
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0
        )

        self.field = ReflectSamplingNeRFNerfField(
            position_encoding=position_encoding, 
            direction_encoding=direction_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
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
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")


        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        density_outputs_coarse, embedding_coarse = self.field.get_density(ray_samples_uniform, True)
        weights_coarse = ray_samples_uniform.get_weights(density_outputs_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        pred_normals_outputs_coarse = self.field.get_pred_normals(embedding_coarse)
        if self.training:
            normals_outputs_coarse = self.field.get_normals()
        else:
            normals_outputs_coarse = pred_normals_outputs_coarse

        embedding_coarse = self.field.get_mid(embedding_coarse)
        diff_outputs_coarse = self.field.get_diff(embedding_coarse)
        tint_outputs_coarse = self.field.get_tint(embedding_coarse)
        low_outputs_coarse = self.field.get_low(ray_samples_uniform, embedding_coarse)
        
        rgb_outputs_coarse = diff_outputs_coarse + tint_outputs_coarse * low_outputs_coarse
        rgb_coarse = self.renderer_rgb(rgb_outputs_coarse, weights_coarse)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        density_outputs_fine, embedding_fine = self.field.get_density(ray_samples_pdf, True)
        weights_fine = ray_samples_pdf.get_weights(density_outputs_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        pred_normals_outputs_fine = self.field.get_pred_normals(embedding_fine)
        if self.training:
            normals_outputs_fine = self.field.get_normals()
        else:
            normals_outputs_fine = pred_normals_outputs_fine
        # pred_normals_fine = self.renderer_normals(pred_normals_outputs_fine, weights_fine)

        # rougness_outputs_fine = self.field.get_roughness(embedding_fine)
        # rougness_fine = self.renderer_roughness(rougness_outputs_fine, weights_fine)
        
        embedding_fine = self.field.get_mid(embedding_fine)
        # diff_outputs_fine = self.field.get_diff(embedding_fine)
        # diff_fine = self.renderer_rgb(diff_outputs_fine, weights_fine)
        # tint_outputs_fine = self.field.get_tint(embedding_fine)
        # tint_fine = self.renderer_rgb(tint_outputs_fine, weights_fine)

        # mask = torch.where(accumulation_fine>1e-2, True, False).reshape(-1)
        # if not mask.any():
        #     print(mask.any())
        #     rgb_coarse = rgb_coarse + (1.0 - accumulation_coarse)*self.background_color.to(accumulation_coarse)
        #     rgb_coarse = self.field.get_padding(rgb_coarse)
        #     rgb_fine = rgb_coarse
        #     outputs = {
        #         "rgb_coarse": rgb_coarse,
        #         "rgb_fine": rgb_fine,
        #         "accumulation_coarse": accumulation_coarse,
        #         "accumulation_fine": accumulation_fine,
        #         "depth_coarse": depth_coarse,
        #         "depth_fine": depth_fine,
        #         "weights_coarse": weights_coarse.detach(),
        #         "weights_fine": weights_fine.detach(),
        #         "pred_normals_coarse":pred_normals_outputs_coarse,
        #         "pred_normals_fine":pred_normals_outputs_fine,
        #         "normals_coarse": normals_outputs_coarse.detach(),
        #         "normals_fine": normals_outputs_fine.detach(),
        #         "direction_coarse": ray_samples_uniform.frustums.directions,
        #         "direction_fine": ray_samples_pdf.frustums.directions,
        #     }
        #     return outputs

        # origins = ray_bundle.origins[mask,:] + depth_fine[mask,:]*ray_bundle.directions[mask,:]
        # dot_nd = torch.sum(pred_normals_fine[mask,:]*ray_bundle.directions[mask,:], dim=-1, keepdim=True)
        # reflections = ray_bundle.directions[mask,:] - 2*dot_nd*pred_normals_fine[mask,:]
        # print(torch.min(accumulation_fine).item(), torch.max(accumulation_fine).item(), reflections.shape)

        # '''
        # roughness to pixelarea as spherical gaussian lobe
        # g(|x-mean|, sigma^2) = exp(-|x-mean|^2/(2*sigma^2))
        # sigma^2 = 1/sharpness
        # sharpness = 2/roughness^2/(4*|direction*normal|)
        # sigma^2 = roughness^2*2*|direction*normal|

        # '''
        # reflect_ray_bundle = RayBundle(
        #     origins=origins,
        #     directions=reflections,
        #     pixel_area=rougness_fine[mask,:],
        #     nears=torch.zeros_like(ray_bundle.nears[mask,:]),
        #     fars=torch.ones_like(ray_bundle.fars[mask,:])
        # )

        # # low_outputs_fine = self.field.get_low(ray_samples_pdf, embedding_fine)
        # # low_fine = self.renderer_rgb(low_outputs_fine, weights_fine)
        # radius_variance = rougness_fine[mask,:]*2*torch.abs(dot_nd)
        
        # rgb_fine = diff_fine
        # rgb_fine[mask,:] = rgb_fine[mask,:] + tint_fine[mask,:]*env_fine

        low_outputs_fine = self.field.get_low(ray_samples_pdf, embedding_fine)
        rgb_fine = self.renderer_rgb(low_outputs_fine, weights_fine)

        rgb_coarse = rgb_coarse + (1.0 - accumulation_coarse)*self.background_color.to(accumulation_coarse)
        rgb_coarse = self.field.get_padding(rgb_coarse)
        rgb_fine = rgb_fine + (1.0 - accumulation_fine)*self.background_color.to(accumulation_fine)
        rgb_fine = self.field.get_padding(rgb_fine)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
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
            "direction_coarse": ray_samples_uniform.frustums.directions,
            "direction_fine": ray_samples_pdf.frustums.directions,
        }
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
        
        rgb_loss_coarse = self.rgb_loss(image_coarse, pred_coarse)
        rgb_loss_fine = self.rgb_loss(image_fine, pred_fine)

        predicted_normal_loss_coarse = torch.sum(torch.sum((outputs["normals_coarse"]-outputs["pred_normals_coarse"])**2, dim=-1, keepdim=True))
        predicted_normal_loss_fine = torch.sum(torch.sum((outputs["normals_fine"]-outputs["pred_normals_fine"])**2, dim=-1, keepdim=True))

        dot_nd_coarse = torch.sum(outputs["pred_normals_coarse"]*outputs["direction_coarse"], dim=-1, keepdim=True)
        orientation_loss_coarse = torch.sum(torch.max(torch.zeros_like(dot_nd_coarse),dot_nd_coarse)**2)
        dot_nd_fine = torch.sum(outputs["pred_normals_fine"]*outputs["direction_fine"], dim=-1, keepdim=True)
        orientation_loss_fine = torch.sum(torch.max(torch.zeros_like(dot_nd_fine),dot_nd_fine)**2)
        
        print(rgb_loss_fine.item(), torch.max(torch.sum((outputs["normals_fine"]-outputs["pred_normals_fine"])**2, dim=-1, keepdim=True)), orientation_loss_fine.item())

        if rgb_loss_fine.isnan().any() and predicted_normal_loss_fine.isnan().any() and orientation_loss_fine.isnan().any():
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
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
