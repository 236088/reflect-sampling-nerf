
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union, Literal

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import SpacedSampler
from nerfstudio.field_components.encodings import Encoding

class ReciprocalSampler(SpacedSampler):
    """Sample along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        tan: float = 1.0,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x/(1/tan+x),
            spacing_fn_inv=lambda x: x/tan/(1-x),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )

class IntegratedSHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self) -> None:
        super().__init__(in_dim=3)


    def get_out_dim(self) -> int:
        return 34

    @torch.no_grad()
    def pytorch_fwd(self, directions: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """
        Returns value for each component of spherical harmonics.

        Args:
            levels: Number of spherical harmonic levels to compute.
            directions: Spherical harmonic coefficients
        """
        components = torch.zeros((*directions.shape[:-1], 34), device=directions.device)

        assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

        x = directions[..., 0]
        y = directions[..., 1]
        z = directions[..., 2]
        # l1
        components[..., 0] = 0.48860251190291992*y
        components[..., 1] = 0.48860251190291992*z
        components[..., 2] = 0.48860251190291992*x
        
        x2 = x**2
        y2 = y**2
        z2 = z**2
        xy = x*y
        xz = x*z
        yz = y*z
        x2_y2 = x2 - y2
        
        #l2
        components[..., 3] = 1.09254843059207907*xy
        components[..., 4] = 1.09254843059207907*yz
        components[..., 5] = 0.31539156525252001*(3*z2-1)
        components[..., 6] = 1.09254843059207907*xz
        components[..., 7] = 0.54627421529603953*x2_y2
        
        y2_3x2=3*x2 - y2
        x2_3y2=x2 - 3*y2

        z4=z**4
        # l4
        components[..., 8] = 2.50334294179670453*xy*x2_y2
        components[..., 9] = 1.77013076977993053*yz*y2_3x2
        components[..., 10] = 0.94617469575756001*xy*(7*z2 - 1)
        components[..., 11] = 0.66904654355728916*yz*(7*z2 - 3)
        components[..., 12] = 0.1057855469152043038*(35*z4 - 30*z2 + 3)
        components[..., 13] = 0.66904654355728916*xz*(7*z2 - 3)
        components[..., 14] = 0.473087347878780009*x2_y2*(7*z2 - 1)
        components[..., 15] = 1.77013076977993053*xz*x2_3y2
        components[..., 16] = 0.62583573544917613*(x2*x2_3y2 - y2*y2_3x2)
        
        x4=x**4
        y4=y**4
        y4_10y2x2_5x4=y4 - 10*x2*y2 + 5*x4
        x4_10x2y2_5y4=x4 - 10*x2*y2 + 5*y4
        y6_21y4x2_35y2x4_7x6=(x2 - 5*y2)*7*x4 + (21*x2 - y2)*y4
        x6_21x4y2_35x2y4_7y6=(x2 - 21*y2)*x4 + (5*x2 - y2)*7*y4
        
        # l8
        components[..., 17] = 5.83141328139863895*xy*(x2*x4-7*x4*y2+7*x2*y4-y2*y4)
        components[..., 18] = 5.83141328139863895*yz*y6_21y4x2_35y2x4_7x6
        components[..., 19] = 1.06466553211908514*xy*(15*z2-1)*(3*x4-10*x2*y2+3*y4)
        components[..., 20] = 3.44991062209810801*yz*(5*z2-1)*y4_10y2x2_5x4
        components[..., 21] = 1.91366609903732278*xy*(65*z4-26*z2+1)*x2_y2
        components[..., 22] = 1.23526615529554407*yz*(39*z4-26*z2+3)*y2_3x2
        components[..., 23] = 0.91230451686981894*xy*(143*z4*z2-143*z4+33*z2-1)
        components[..., 24] = 0.1090412458987799555*yz*(715*z4*z2-1001*z4+385*z2-35)
        components[..., 25] = 0.0090867704915649962938*(6435*z4*z4-12012*z4*z2+6930*z4-1260*z2+35)
        components[..., 26] = 0.1090412458987799555*xz*(715*z4*z2-1001*z4+385*z2-35)
        components[..., 27] = 0.456152258434909470*(143*z4*z2-143*z4+33*z2-1)*x2_y2
        components[..., 28] = 1.23526615529554407*xz*(39*z4-26*z2+3)*x2_3y2
        components[..., 29] = 0.478416524759330697*(65*z4-26*z2+1)*(x2*x2_3y2-y2*y2_3x2)
        components[..., 30] = 3.44991062209810801*xz*(5*z2-1)*x4_10x2y2_5y4
        components[..., 31] = 0.53233276605954257*(15*z2-1)*(x2*x4_10x2y2_5y4 -y2*y4_10y2x2_5x4)
        components[..., 32] = 5.83141328139863895*xz*x6_21x4y2_35x2y4_7y6
        components[..., 33] = 0.72892666017482986*(x2*x6_21x4y2_35x2y4_7y6-y2*y6_21y4x2_35y2x4_7x6)
        
        return components
    
    '''
    exp(-softplus(x)*a)=exp(-softplus(x))**a=sigmoid(-x)**a
    '''
    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        outputs = self.pytorch_fwd(in_tensor)
        return outputs