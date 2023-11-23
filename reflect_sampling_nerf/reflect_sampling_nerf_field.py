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


class ReflectSamplingNeRFPropField(Field):
    """ReflectSamplingNeRF Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 4,
        base_mlp_layer_width: int = 256,
        spatial_distortion: Optional[SpatialDistortion] = None,
        density_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.density_bias = density_bias
        
        self.softplus = nn.Softplus()
        
    

    def get_blob(
        self, ray_samples: RaySamples
    ) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        return gaussian_samples.mean, gaussian_samples.cov
    
            
    def get_density(
        self, mean:Tensor, cov:Tensor=None, requires_density_grad:bool = False
    ) -> Tuple[Tensor, Tensor]:
        if requires_density_grad and self.training:
            mean.requires_grad = True
            self._sample_locations = mean
        if cov is not None:
            encoded_xyz = self.position_encoding(mean, covs=cov)
        else:
            encoded_xyz = self.position_encoding(mean)
        mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(mlp_out)
        if requires_density_grad and self.training:
            self._density_before_activation=density
        density = self.softplus(density + self.density_bias)
        return density, mlp_out
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.

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
        head_mlp_num_layers: int = 4,
        head_mlp_layer_width: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        density_bias: float = 0.5,
        roughness_bias: float = -1.0,
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
        self.density_bias = density_bias
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        self.field_output_low = RGBFieldHead(self.mlp_base.get_out_dim())
        
        self.field_output_bottleneck = FieldHead(out_dim=self.mlp_base.get_out_dim(), field_head_name="bottleneck", in_dim=self.mlp_base.get_out_dim(), activation=None)
        
        self.mlp_mid = MLP(
            in_dim=self.direction_encoding.get_out_dim()+1+self.mlp_base.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU()
        )
        self.field_output_mid = RGBFieldHead(self.mlp_mid.get_out_dim())
                
        self.field_output_normals = PredNormalsFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        
        self.field_output_roughness = FieldHead(out_dim=1, field_head_name="roughness", in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.roughness_bias = roughness_bias

        self.field_output_diff = RGBFieldHead(self.mlp_base.get_out_dim())

        self.field_output_tint = RGBFieldHead(self.mlp_base.get_out_dim())
        
    

    def get_blob(
        self, ray_samples: RaySamples
    ) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        return gaussian_samples.mean, gaussian_samples.cov
    
        
    def get_density(
        self, mean:Tensor, cov:Tensor=None, requires_density_grad:bool = False
    ) -> Tuple[Tensor, Tensor]:
        if requires_density_grad and self.training:
            mean.requires_grad = True
            self._sample_locations = mean
        if cov is not None:
            encoded_xyz = self.position_encoding(mean, covs=cov)
        else:
            encoded_xyz = self.position_encoding(mean)
        mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(mlp_out)
        if requires_density_grad and self.training:
            self._density_before_activation=density
        density = self.softplus(density + self.density_bias)
        return density, mlp_out
    
    def get_pred_normals(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = -self.field_output_normals(embedding)
        outputs = nn.functional.normalize(outputs, dim=-1)
        return outputs
    
    def get_normals(self) -> Tensor:
        return super().get_normals().detach()

    '''exp(-softplus(x))=sigmoid(-x)'''
    def get_roughness(
        self, embedding:Tensor, activation:Optional[nn.Module]=nn.Sigmoid()
    ) -> Tensor:
        outputs = self.field_output_roughness(embedding)
        outputs = activation(outputs)
        return outputs
    
    
    def get_low(
        self, embedding:Tensor, use_bottleneck:bool=True
    ) ->Tensor:
        embedding = self.field_output_bottleneck(embedding) if use_bottleneck else embedding
        mlp_out = self.mlp_mid(torch.cat([torch.zeros(embedding.shape[:-1]+(self.direction_encoding.get_out_dim()+1,), device=embedding.device), embedding], dim=-1))
        outputs = self.field_output_mid(mlp_out)
        return outputs
    
    
    def get_mid(
        self, directions:Tensor, n_dot_d:Tensor, roughness:Tensor, embedding:Tensor, use_bottleneck:bool=True
    ) ->Tensor:
        encoded_dir = self.direction_encoding(directions, roughness)
        embedding = self.field_output_bottleneck(embedding) if use_bottleneck else embedding
        mlp_out = self.mlp_mid(torch.cat([encoded_dir, n_dot_d, embedding], dim=-1))
        outputs = self.field_output_mid(mlp_out)
        return outputs
    
    def get_diff(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = self.field_output_diff(embedding)
        return outputs

    def get_tint(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = self.field_output_tint(embedding)
        return outputs
    
    def get_reflection(self, directions:Tensor, normals:Tensor) -> Tuple[Tensor, Tensor]:
        n_dot_d = torch.sum(directions*normals, dim=-1, keepdim=True)
        reflections = directions - 2*n_dot_d*normals
        reflections = torch.nn.functional.normalize(reflections, dim=-1)
        return reflections, n_dot_d
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
    