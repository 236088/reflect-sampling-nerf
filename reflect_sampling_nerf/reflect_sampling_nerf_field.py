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
        
    
          
    def get_density(
        self, ray_samples: RaySamples, requires_density_grad:bool = False
    ) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        if requires_density_grad and self.training:
            gaussian_samples.mean.requires_grad = True
            self._sample_locations = gaussian_samples.mean
        encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(mlp_out)
        if requires_density_grad and self.training:
            self._density_before_activation=density
        density = self.softplus(density + self.density_bias)
        return density, mlp_out
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.

class ReflectSamplingNeRFEnvironmentField(Field):
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
        env_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        
        self.field_output_env = RGBFieldHead(self.mlp_base.get_out_dim())
        self.env_bias = env_bias
        
 
    '''
    lim t -> +inf
    (2-1/|x|)*x/|x| -> 2*d
    sigms_t -> 3/20*1/(t/2)^2*dd^T
    sigma_r -> r^2*3/5*1/(t/2)^2*(I-dd^T)
    cov = sigma_t + sigma_r
    J jacobian contract(x) -> 2/t + O(1/t^2)
    J*cov*J^T -> 3/20*dd^T+3/5*r^2*(I-dd^T)
    '''
    
    def get_env(
        self, directions:Tensor, sqradius:Tensor
    ) -> Tensor:
        outer = directions[...,:,None]*directions[...,None,:]
        eyes = torch.eye(directions.shape[-1], device=directions.device).expand(outer.shape)
        mean = 2*directions
        cov = 0.6*sqradius[...,None]*(eyes-outer)
        encoded_xyz = self.position_encoding(mean, covs=cov)
        mlp_out = self.mlp_base(encoded_xyz)
        outputs = self.field_output_env(mlp_out+self.env_bias)
        return outputs
       
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
        head_mlp_num_layers: int = 1,
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
                
        self.field_output_bottleneck = FieldHead(out_dim=self.mlp_base.get_out_dim(), field_head_name="bottleneck", in_dim=self.mlp_base.get_out_dim(), activation=None)
        
        self.mlp_low = MLP(
            in_dim=self.direction_encoding.get_out_dim()+1+self.mlp_base.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU()
        )
        self.field_output_low = RGBFieldHead(self.mlp_low.get_out_dim())
                
        self.field_output_normals = PredNormalsFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        
        self.field_output_roughness = FieldHead(out_dim=1, field_head_name="roughness", in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.roughness_bias = roughness_bias

        self.field_output_diff = RGBFieldHead(self.mlp_base.get_out_dim())

        self.field_output_tint = RGBFieldHead(self.mlp_base.get_out_dim())
        
 
        
    def get_density(
        self, ray_samples: RaySamples, requires_density_grad:bool = False
    ) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        if requires_density_grad and self.training:
            gaussian_samples.mean.requires_grad = True
            self._sample_locations = gaussian_samples.mean
        encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
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
        return outputs
    
    def get_normals(self) -> Tensor:
        return super().get_normals().detach()

    '''exp(-softplus(x))=sigmoid(-x)'''
    def get_roughness(
        self, embedding:Tensor, activation:Optional[nn.Module]=nn.Softplus()
    ) -> Tensor:
        outputs = self.field_output_roughness(embedding)
        outputs = activation(outputs)
        return outputs
    
    def get_low(
        self, directions:Tensor, n_dot_d:Tensor, roughness:Tensor, embedding:Tensor
    ) ->Tensor:
        encoded_dir = self.direction_encoding(directions.detach(), roughness)
        embedding = self.field_output_bottleneck(embedding)
        mlp_out = self.mlp_low(torch.cat([encoded_dir, n_dot_d.detach(), embedding], dim=-1))
        outputs = self.field_output_low(mlp_out)
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
    