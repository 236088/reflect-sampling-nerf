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
        low_mlp_num_layers: int = 1,
        low_mlp_layer_width: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        density_bias: float = 0.5,
        density_reflect_bias: float = 0.0,
        roughness_bias: float = -1.0,
        diff_bias:float=0.0,
        tint_bias:float=1.0,
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

        self.field_output_reflect_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.density_reflect_bias = density_reflect_bias
        
        self.field_output_normals = PredNormalsFieldHead(in_dim=self.mlp_base.get_out_dim(), activation=None)

        self.field_output_roughness = FieldHead(out_dim=1, field_head_name="roughness", in_dim=self.mlp_base.get_out_dim(), activation=None)
        self.roughness_bias = roughness_bias
        

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()


        
        self.field_output_diff = RGBFieldHead(self.mlp_base.get_out_dim(), activation=None)
        self.diff_bias = diff_bias

        self.field_output_tint = RGBFieldHead(self.mlp_base.get_out_dim(), activation=None)
        self.tint_bias = tint_bias

        self.mlp_bottleneck = FieldHead(out_dim=self.mlp_base.get_out_dim(), field_head_name="bottleneck", in_dim=self.mlp_base.get_out_dim(), activation=None)
        
        self.mlp_low = MLP(
            in_dim=self.direction_encoding.get_out_dim()+1+self.mlp_bottleneck.get_out_dim(),
            num_layers=low_mlp_num_layers,
            layer_width=low_mlp_layer_width,
            out_activation=nn.ReLU()
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
        self, mean: Tensor, cov: Tensor, mask_return:bool = False
    ) -> Tensor:
        norm2 = torch.sum(mean**2, dim=-1, keepdim=True)
        norm = torch.sqrt(norm2)
        mask = norm>1
        mean_contract = torch.where(mask, (2*norm-1)/norm2*mean, mean)
        
        norm = norm.unsqueeze(-1)
        norm2 = norm2.unsqueeze(-1)
        outer = mean[...,:,None]*mean[...,None,:]/norm2
        eyes = torch.eye(mean.shape[-1], device=mean.device).expand(outer.shape)
        # jacobian = torch.autograd.functional.jacobian(mean_contract, mean)
        jacobian = torch.where(mask[...,None], ((2*norm-2)*(eyes-outer)+eyes)/norm2, eyes)
        ''' J*cov*J.T :(J.T=J)'''
        cov_contract = torch.matmul(torch.matmul(jacobian, cov), jacobian)
        for i in range(cov_contract.shape[-1]):
            cov_contract[...,i,i]=torch.nn.functional.relu(cov_contract[...,i,i])
        if mask_return:
            return mean_contract, cov_contract, mask
        else:
            return mean_contract, cov_contract
    
    def get_contract_inf(
        self, directions:Tensor, sqradius:Tensor
    ) -> Tuple[Tensor, Tensor]:
        outer = directions[...,:,None]*directions[...,None,:]
        eyes = torch.eye(directions.shape[-1], device=directions.device).expand(outer.shape)
        mean = 2*directions
        cov = 0.6*outer + 2.4*sqradius[...,None]*(eyes-outer)
        return mean, cov
        
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
        
    def get_reflect_density(
        self, mean:Tensor, cov:Tensor
    ) -> Tuple[Tensor, Tensor]:
        encoded_xyz = self.position_encoding(mean, covs=cov)
        mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_reflect_density(mlp_out)
        density = self.softplus(density + self.density_reflect_bias)
        return density, mlp_out
    
    def get_pred_normals(
        self, embedding:Tensor
    ) -> Tensor:
        normals = -self.field_output_normals(embedding)
        normals = nn.functional.normalize(normals, dim=-1)
        return normals
    
    def get_normals(self) -> Tensor:
        return super().get_normals()


    '''exp(-softplus(x))=sigmoid(-x)'''
    def get_roughness(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = self.field_output_roughness(embedding)
        outputs = self.softplus(outputs + self.roughness_bias)
        return outputs

    def get_diff(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = self.field_output_diff(embedding)
        outputs = self.softplus(outputs + self.diff_bias)
        return outputs

    def get_tint(
        self, embedding:Tensor
    ) -> Tensor:
        outputs = self.field_output_tint(embedding)
        outputs = self.softplus(outputs + self.tint_bias)
        return outputs

    def get_bottleneck(
        self, embedding: Tensor
    ) -> Tensor:
        outputs = self.mlp_bottleneck(embedding)
        return outputs
    
    def get_low(
        self, directions:Tensor, n_dot_d: Tensor, embedding:Tensor, roughness:Tensor
    ) ->Tensor:
        encoded_dir = self.direction_encoding(directions)
        for l in range(self.direction_encoding.levels):
            begin = l**2
            end = (l+1)**2
            encoded_dir[...,begin:end]*=torch.exp(-roughness*0.5*l*(l+1))
        mlp_out = self.mlp_low(torch.cat([encoded_dir, n_dot_d, embedding], dim=-1))
        outputs = self.field_output_low(mlp_out)
        return outputs
    
    
    def get_inf_color(
        self, directions:Tensor, sqradius:Tensor
    ) ->Tensor:
        mean, cov = self.get_contract_inf(directions, sqradius)
        _, embedding = self.get_density(mean, cov)
        diff = self.get_diff(embedding)
        return diff

    
    def get_reflection(self, directions:Tensor, normals:Tensor) -> Tuple[Tensor, Tensor]:
        n_dot_d = torch.sum(directions*normals, dim=-1, keepdim=True)
        reflections = directions - 2*n_dot_d*normals
        reflections = torch.nn.functional.normalize(reflections, dim=-1)
        return reflections, n_dot_d
    
    def get_normalized(self, embedding: Tensor) -> Tuple[Tensor, Tensor]:
        diff = self.get_diff(embedding)
        tint = self.get_tint(embedding)
        norm = diff + tint + 1e-2
        diff = diff/norm
        tint = tint/norm
        return diff, tint
    
    def get_padding(self, inputs: Tensor) -> Tensor:
        outputs = (1 + 2*self.padding)*inputs - self.padding
        outputs = torch.clip(inputs, 0.0, 1.0)
        return outputs
       
    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
