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
        low_mlp_num_layers: int = 2,
        low_mlp_layer_width: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        density_bias: float = 0.5,
        padding: float = 0.01,
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
        self.density_bias = density_bias
        self.softplus = nn.Softplus()

        self.field_output_normals = PredNormalsFieldHead(in_dim=self.mlp_base.get_out_dim())

        self.field_output_roughness = FieldHead(out_dim=1, field_head_name="roughness", in_dim=self.mlp_base.get_out_dim(), activation=nn.Sigmoid())



        self.mlp_mid = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            num_layers=1,
            layer_width=self.mlp_base.get_out_dim(),
            out_activation=nn.ReLU(),
        )
        
        self.field_output_diff = RGBFieldHead(self.mlp_base.get_out_dim())

        self.field_output_tint = RGBFieldHead(self.mlp_base.get_out_dim())
        
        self.mlp_low = MLP(
            in_dim=self.direction_encoding.get_out_dim()+1+self.mlp_base.get_out_dim(),
            num_layers=low_mlp_num_layers,
            layer_width=low_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        self.field_output_low = RGBFieldHead(self.mlp_low.get_out_dim())
    

    def get_blob(
        self, ray_samples: RaySamples
    ) -> Tuple[Tensor, Tensor]:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        return gaussian_samples.mean, gaussian_samples.cov
    
    def contract(
        self, mean: Tensor, cov: Tensor, 
    ) -> Tensor:
        norm2 = torch.sum(mean**2, dim=-1, keepdim=True)
        norm = torch.sqrt(norm2)
        mean_contract = torch.where(norm>1, (2*norm-1)/norm2*mean, mean)
        
        norm = norm.unsqueeze(-1)
        norm2 = norm2.unsqueeze(-1)
        outer = mean[...,:,None]*(mean[...,None,:]/norm2)
        eyes = torch.eye(mean.shape[-1], device=mean.device).expand(outer.shape)
        # jacobian = torch.autograd.functional.jacobian(mean_contract, mean)
        jacobian = torch.where(norm>1, 2*(1-norm)/norm2*outer + (2*norm-1)/norm2*eyes, eyes)
        ''' J*cov*J.T :(J.T=J)'''
        cov_contract = torch.matmul(torch.matmul(jacobian, cov), jacobian)
        return mean_contract, cov_contract
        
    def get_density(
        self, mean:Tensor, cov:Tensor, requires_density_grad:bool = False
    ) -> Tuple[Tensor, Tensor]:
        if requires_density_grad and self.training:
            mean.requires_grad = True
            self._sample_locations = mean
        encoded_xyz = self.position_encoding(mean, covs=cov)
        mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(mlp_out)
        if requires_density_grad and self.training:
            self._density_before_activation=density
        density = self.softplus(density + self.density_bias)
        return density, mlp_out
    
    def get_pred_normals(
        self, embedding:Tensor
    ) -> Tensor:
        normals = -self.field_output_normals(embedding)
        return normals
    
    def get_normals(self) -> Tensor:
        return super().get_normals()

    def get_roughness(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = self.field_output_roughness(embedding)
        return outputs


    def get_mid(
        self, embedding: Tensor
    ) -> Tensor:
        outputs = self.mlp_mid(embedding)
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
    
    def get_low(
        self, directions:Tensor, n_dot_d: Tensor, embedding:Tensor
    ) ->Tensor:
        encoded_dir = self.direction_encoding(directions)
        mlp_out = self.mlp_low(torch.cat([encoded_dir, n_dot_d, embedding], dim=-1))
        outputs = self.field_output_low(mlp_out)
        outputs = self.get_padding(outputs)
        return outputs
    

        

    def get_padding(self, inputs: Tensor):
        outputs = (1 + 2*self.padding)*inputs - self.padding
        outputs = torch.clip(outputs, 0.0, 1.0)
        return outputs
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
