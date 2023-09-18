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
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
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
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        bottle_neck_width: int = 256,
        head_mlp_num_layers: int = 8,
        head_mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.softplus = nn.Softplus()
        self.field_output_diffuse = FieldHead(out_dim=3, field_head_name="diffuse", in_dim=self.mlp_base.get_out_dim(), activation=nn.Sigmoid())
        self.field_output_tint = FieldHead(out_dim=3, field_head_name="tint", in_dim=self.mlp_base.get_out_dim(), activation=nn.Sigmoid())
        self.field_bottle_neck = FieldHead(out_dim=bottle_neck_width, field_head_name="bottle_neck", in_dim=self.mlp_base.get_out_dim())
        self.field_output_roughness = FieldHead(out_dim=1, field_head_name="rouhness", in_dim=self.mlp_base.get_out_dim(), activation=nn.Softplus())
        self.field_output_normal = PredNormalsFieldHead(in_dim=self.mlp_base.get_out_dim())


        self.mlp_head = MLP(
            in_dim=self.field_bottle_neck.get_out_dim() + self.direction_encoding.get_out_dim() + 1,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.field_output_rgb = RGBFieldHead(self.mlp_head.get_out_dim())

    def get_density(self, ray_samples: RaySamples, require_grad:bool = False) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if require_grad and self.training:
            gaussian_samples.mean.requires_grad = True
            self._sample_locations = gaussian_samples.mean
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        if require_grad and self.training:
            self._density_before_activation=density
        density = self.softplus(density)
        return density, base_mlp_out
    
    def get_diffuse(self, embedding: Tensor) -> Tensor:
        outputs = self.field_output_diffuse(embedding)
        return outputs
    
    def get_tint(self, embedding: Tensor) -> Tensor:
        outputs = self.field_output_tint(embedding)
        return outputs
    
    def get_bottle_neck(self, embedding: Tensor) -> Tensor:
        outputs = self.field_bottle_neck(embedding)
        return outputs
    
    def get_roughness(self, embedding: Tensor) -> Tensor:
        outputs = self.field_output_roughness(embedding)
        return outputs
    
    def get_pred_normals(self, embedding: Tensor) -> Tensor:
        outputs = self.field_output_normal(embedding)
        outputs = nn.functional.normalize(outputs)
        return outputs

    def get_normals(self) -> Tensor:
        return super().get_normals()
    
    def get_outputs(
        self, ray_samples: RaySamples, embedding: Tensor, normal: Tensor
    ) -> Tensor:
        diffuse = self.get_diffuse(embedding)
        tint = self.get_tint(embedding)
        bottle_neck = self.get_bottle_neck(embedding)
        roughness = self.get_roughness(embedding)
        dot_product = torch.sum(ray_samples.frustums.directions*normal, dim=-1, keepdim=True)
        reflection = ray_samples.frustums.directions - 2*normal*dot_product
        
        attenuation = torch.tensor([1, 3,3,3, 6,6,6,6,6, 10,10,10,10,10,10,10], dtype=torch.float32, device=embedding.device)
        attenuation = torch.exp(-roughness*attenuation.unsqueeze(0).unsqueeze(0))
        encoded_dir = self.direction_encoding(reflection)*attenuation

        mlp_out = self.mlp_head(torch.cat([bottle_neck, encoded_dir, dot_product], dim=-1))  # type: ignore
        outputs = self.field_output_rgb(mlp_out)
        outputs = diffuse + tint*outputs

        return outputs
 
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
