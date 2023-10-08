
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import SpacedSampler

class ReciprocalSampler(SpacedSampler):
    """Sample along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x/(1+x),
            spacing_fn_inv=lambda x: x/(1-x),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )
