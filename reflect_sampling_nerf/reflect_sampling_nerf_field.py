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
    FieldHeadNames,
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
        roughness_mlp_num_layers: int = 2,
        roughness_mlp_layer_width: int = 128,
        diff_mlp_num_layers: int = 2,
        diff_mlp_layer_width: int = 128,
        tint_mlp_num_layers: int = 2,
        tint_mlp_layer_width: int = 128,
        low_mlp_num_layers: int = 2,
        low_mlp_layer_width: int = 128,
        env_mlp_num_layers: int = 4,
        env_mlp_layer_width: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        padding: float = 0.01
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.spatial_distortion = spatial_distortion
        self.padding = padding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.softplus = nn.Softplus()

        self.mlp_normals = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=normals_mlp_num_layers,
            layer_width=normals_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_normals = PredNormalsFieldHead(in_dim=self.mlp_normals.get_out_dim())

        self.mlp_roughness = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=roughness_mlp_num_layers,
            layer_width=roughness_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_roughness = FieldHead(out_dim=1, field_head_name="roughness", in_dim=self.mlp_roughness.get_out_dim(), activation=nn.Sigmoid())


        
        self.mlp_diff = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=diff_mlp_num_layers,
            layer_width=diff_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        
        self.field_output_diff = RGBFieldHead(self.mlp_diff.get_out_dim())

        self.mlp_tint = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=tint_mlp_num_layers,
            layer_width=tint_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_tint = RGBFieldHead(self.mlp_tint.get_out_dim())
        
        self.mlp_low = MLP(
            in_dim=self.mlp_base.get_out_dim()+self.direction_encoding.get_out_dim(),
            num_layers=low_mlp_num_layers,
            layer_width=low_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_low = RGBFieldHead(self.mlp_low.get_out_dim())
        
        self.mlp_env = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=env_mlp_num_layers,
            layer_width=env_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_env = RGBFieldHead(self.mlp_env.get_out_dim())

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
    
    def get_pred_normals(
        self, embedding:Tensor
    ) -> Tensor:
        mlp_out = self.mlp_normals(embedding)
        normals = self.field_output_normals(mlp_out)
        return normals
    
    def get_normals(self) -> Tensor:
        return super().get_normals()

    def get_roughness(
        self, embedding:Tensor
    ) -> Tensor:
        mlp_out = self.mlp_roughness(embedding)
        outputs = self.field_output_roughness(mlp_out)
        return outputs



    def get_diff(
        self, embedding:Tensor
    ) -> Tensor:
        mlp_out = self.mlp_diff(embedding)
        outputs = self.field_output_diff(mlp_out)
        return outputs

    def get_tint(
        self, embedding:Tensor
    ) -> Tensor:
        mlp_out = self.mlp_tint(embedding)
        outputs = self.field_output_tint(mlp_out)
        return outputs
    
    def get_low(
        self, ray_samples: RaySamples, embedding:Tensor
    ) ->Tensor:
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        mlp_out = self.mlp_low(torch.cat([encoded_dir, embedding], dim=-1))
        outputs = self.field_output_low(mlp_out)
        outputs = self.get_padding(outputs)
        return outputs
    
    def get_env(
        self, directions: Tensor, radius_variance: Tensor
    ) ->Tensor:
        eye = torch.eye(directions.shape[-1], device=directions.device)
        null_outer_product = eye - directions[..., :, None] * directions[..., None, :]
        radius_cov_diag = radius_variance * null_outer_product[..., :, :]

        encoded_xyz = self.position_encoding(directions, radius_cov_diag)
        mlp_out = self.mlp_env(encoded_xyz)
        outputs = self.field_output_env(mlp_out)
        outputs = self.get_padding(outputs)
        return outputs
    
    def get_padding(self, inputs: Tensor):
        outputs = (1 + 2*self.padding)*inputs - self.padding
        outputs = torch.clip(outputs, 0.0, 1.0)
        return outputs
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
