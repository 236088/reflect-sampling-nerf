"""
Nerfstudio ReflectSamplingNeRF Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from reflect_sampling_nerf.reflect_sampling_nerf_datamanager import (
    ReflectSamplingNeRFDataManagerConfig,
)
from reflect_sampling_nerf.reflect_sampling_nerf_model import ReflectSamplingNeRFModelConfig
from reflect_sampling_nerf.reflect_sampling_nerf_pipeline import (
    ReflectSamplingNeRFPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


reflect_sampling_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="reflect-sampling-nerf",  # TODO: rename to your own model
        steps_per_eval_batch=100,
        steps_per_save=1000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=ReflectSamplingNeRFPipelineConfig(
            datamanager=ReflectSamplingNeRFDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=1024,
            ),
            model=ReflectSamplingNeRFModelConfig(
                eval_num_rays_per_chunk=1 << 10,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 10),
        vis="viewer",
    ),
    description="Nerfstudio reflect-sampling-nerf.",
)
