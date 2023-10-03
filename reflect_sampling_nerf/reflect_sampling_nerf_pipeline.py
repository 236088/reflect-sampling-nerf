"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from reflect_sampling_nerf.reflect_sampling_nerf_datamanager import ReflectSamplingNeRFDataManagerConfig
from reflect_sampling_nerf.reflect_sampling_nerf_model import ReflectSamplingNeRFModel, ReflectSamplingNeRFModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)


@dataclass
class ReflectSamplingNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ReflectSamplingNeRFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = ReflectSamplingNeRFDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ReflectSamplingNeRFModelConfig()
    """specifies the model config"""


class ReflectSamplingNeRFPipeline(VanillaPipeline):
    """ReflectSamplingNeRF Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: ReflectSamplingNeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                ReflectSamplingNeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    def get_train_loss_dict(self, step: int):
        if step<0:
            self.model.config.loss_coefficients["predicted_normal_loss_coarse"]=0.0
            self.model.config.loss_coefficients["predicted_normal_loss_fine"]=0.0
            self.model.config.loss_coefficients["orientation_loss_coarse"]=0.0
            self.model.config.loss_coefficients["orientation_loss_fine"]=0.0
        else:
            self.model.config.loss_coefficients["predicted_normal_loss_coarse"]=3e-5
            self.model.config.loss_coefficients["predicted_normal_loss_fine"]=3e-4
            self.model.config.loss_coefficients["orientation_loss_coarse"]=1e-2
            self.model.config.loss_coefficients["orientation_loss_fine"]=1e-1

        return super().get_train_loss_dict(step)
