"""
ReflectSamplingNeRF Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""
from typing import Dict, Literal, Optional, Tuple, Type
from jaxtyping import Float

import torch
from torch import Tensor, nn

from nerfstudio.utils.math import conical_frustum_to_gaussian

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    RGBFieldHead,
    PredNormalsFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


class ReflectSamplingNeRFNerfField(Field):
    """ReflectSamplingNeRF Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
        normals_mlp_num_layers: int = 2,
        normals_mlp_layer_width: int = 128,
        rgb_mlp_num_layers: int = 2,
        rgb_mlp_layer_width: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        rgb_padding: float = 0.01,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.spatial_distortion = spatial_distortion
        self.rgb_padding = rgb_padding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.softplus = nn.Softplus()

        self.mlp_mid = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=1,
            layer_width=self.mlp_base.get_out_dim(),
            out_activation=None,
        )

        self.mlp_normals = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=normals_mlp_num_layers,
            layer_width=normals_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_normals = PredNormalsFieldHead(in_dim=self.mlp_normals.get_out_dim())

        self.mlp_rgb = MLP(
            in_dim=self.mlp_base.get_out_dim()+self.direction_encoding.get_out_dim(),
            num_layers=rgb_mlp_num_layers,
            layer_width=rgb_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_rgb = RGBFieldHead(self.mlp_rgb.get_out_dim())

    def get_density(
        self, ray_samples: RaySamples, require_grad:bool = False
    ) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if require_grad and self.training:
            gaussian_samples.mean.requires_grad = True
            self._sample_locations = gaussian_samples.mean
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(mlp_out)
        if require_grad and self.training:
            self._density_before_activation=density
        density = self.softplus(density)
        return density, mlp_out

    def get_mid(
        self, embedding:Tensor
    ) -> Tensor:
        return self.mlp_mid(embedding)
    
    def get_pred_normals(
        self, embedding:Tensor
    ) -> Tensor:
        mlp_out = self.mlp_normals(embedding)
        normals = self.field_output_normals(mlp_out)
        normals = nn.functional.normalize(normals)
        return normals
    
    def get_normals(self) -> Tensor:
        return super().get_normals()

    def get_outputs(
        self, ray_samples: RaySamples, embedding:Tensor
    ) -> Tensor:
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        mlp_out = self.mlp_rgb(torch.cat([encoded_dir, embedding], dim=-1))  # type: ignore
        outputs = self.field_output_rgb(mlp_out)
        outputs = (1 + self.rgb_padding*2)*outputs - self.rgb_padding
        return outputs
 
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
