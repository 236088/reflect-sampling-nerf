
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union, Literal
import itertools
import math

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import SpacedSampler
from nerfstudio.field_components.encodings import Encoding
from nerfstudio.utils.math import expected_sin

'''
pulled from this contribution
https://github.com/nerfstudio-project/nerfstudio/pull/2463
'''

def columnwise_squared_l2_distance(
    x: Float[Tensor, "*M N"],
    y: Float[Tensor, "*M N"],
) -> Float[Tensor, "N N"]:
    """Compute the squared Euclidean distance between all pairs of columns.
    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        x: tensor of floats, with shape [M, N].
        y: tensor of floats, with shape [M, N].
    Returns:
        sq_dist: tensor of floats, with shape [N, N].
    """
    # Use the fact that ||x - y||^2 == ||x||^2 + ||y||^2 - 2 x^T y.
    sq_norm_x = torch.sum(x**2, 0)
    sq_norm_y = torch.sum(y**2, 0)
    sq_dist = sq_norm_x[:, None] + sq_norm_y[None, :] - 2 * x.T @ y
    return sq_dist


def _compute_tesselation_weights(v: int) -> Tensor:
    """Tesselate the vertices of a triangle by a factor of `v`.
    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        v: int, the factor of the tesselation (v==1 is a no-op to the triangle).

    Returns:
        weights: tesselated weights.
    """
    if v < 1:
        raise ValueError(f"v {v} must be >= 1")
    int_weights = []
    for i in range(v + 1):
        for j in range(v + 1 - i):
            int_weights.append((i, j, v - (i + j)))
    int_weights = torch.FloatTensor(int_weights)
    weights = int_weights / v  # Barycentric weights.
    return weights


def _tesselate_geodesic(
    vertices: Float[Tensor, "N 3"], faces: Float[Tensor, "M 3"], v: int, eps: float = 1e-4
) -> Tensor:
    """Tesselate the vertices of a geodesic polyhedron.

    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        vertices: tensor of floats, the vertex coordinates of the geodesic.
        faces: tensor of ints, the indices of the vertices of base_verts that
            constitute eachface of the polyhedra.
        v: int, the factor of the tesselation (v==1 is a no-op).
        eps: float, a small value used to determine if two vertices are the same.

    Returns:
        verts: a tensor of floats, the coordinates of the tesselated vertices.
    """
    tri_weights = _compute_tesselation_weights(v)

    verts = []
    for face in faces:
        new_verts = torch.matmul(tri_weights, vertices[face, :])
        new_verts /= torch.sqrt(torch.sum(new_verts**2, 1, keepdim=True))
        verts.append(new_verts)
    verts = torch.concatenate(verts, 0)

    sq_dist = columnwise_squared_l2_distance(verts.T, verts.T)
    assignment = torch.tensor([torch.min(torch.argwhere(d <= eps)) for d in sq_dist])
    unique = torch.unique(assignment)
    verts = verts[unique, :]
    return verts


def generate_polyhedron_basis(
    basis_shape: Literal["icosahedron", "octahedron"],
    angular_tesselation: int,
    remove_symmetries: bool = True,
    eps: float = 1e-4,
) -> Tensor:
    """Generates a 3D basis by tesselating a geometric polyhedron.
    Basis is used to construct Fourier features for positional encoding.
    See Mip-Nerf360 paper: https://arxiv.org/abs/2111.12077
    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        base_shape: string, the name of the starting polyhedron, must be either
            'icosahedron' or 'octahedron'.
        angular_tesselation: int, the number of times to tesselate the polyhedron,
            must be >= 1 (a value of 1 is a no-op to the polyhedron).
        remove_symmetries: bool, if True then remove the symmetric basis columns,
            which is usually a good idea because otherwise projections onto the basis
            will have redundant negative copies of each other.
        eps: float, a small number used to determine symmetries.

    Returns:
        basis: a matrix with shape [3, n].
    """
    if basis_shape == "icosahedron":
        a = (math.sqrt(5) + 1) / 2
        verts = torch.FloatTensor(
            [
                (-1, 0, a),
                (1, 0, a),
                (-1, 0, -a),
                (1, 0, -a),
                (0, a, 1),
                (0, a, -1),
                (0, -a, 1),
                (0, -a, -1),
                (a, 1, 0),
                (-a, 1, 0),
                (a, -1, 0),
                (-a, -1, 0),
            ]
        ) / math.sqrt(a + 2)
        faces = torch.tensor(
            [
                (0, 4, 1),
                (0, 9, 4),
                (9, 5, 4),
                (4, 5, 8),
                (4, 8, 1),
                (8, 10, 1),
                (8, 3, 10),
                (5, 3, 8),
                (5, 2, 3),
                (2, 7, 3),
                (7, 10, 3),
                (7, 6, 10),
                (7, 11, 6),
                (11, 0, 6),
                (0, 1, 6),
                (6, 1, 10),
                (9, 0, 11),
                (9, 11, 2),
                (9, 2, 5),
                (7, 2, 11),
            ]
        )
        verts = _tesselate_geodesic(verts, faces, angular_tesselation)
    elif basis_shape == "octahedron":
        verts = torch.FloatTensor([(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)])
        corners = torch.FloatTensor(list(itertools.product([-1, 1], repeat=3)))
        pairs = torch.argwhere(columnwise_squared_l2_distance(corners.T, verts.T) == 2)
        faces, _ = torch.sort(torch.reshape(pairs[:, 1], [3, -1]).T, 1)
        verts = _tesselate_geodesic(verts, faces, angular_tesselation)

    if remove_symmetries:
        # Remove elements of `verts` that are reflections of each other.
        match = columnwise_squared_l2_distance(verts.T, -verts.T) < eps
        verts = verts[torch.any(torch.triu(match), 1), :]

    basis = verts.flip(-1)
    return basis

class FFEncoding(Encoding):
    """Fourier Feature encoding. Supports integrated encodings.

    Args:
        in_dim: Input dimension of tensor
        basis: Basis matrix from which to construct the Fourier features.
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        basis: Float[Tensor, "M N"],
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
    ) -> None:
        super().__init__(in_dim)
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.register_buffer(name="b_matrix", tensor=basis)
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        assert isinstance(self.b_matrix, Tensor)
        out_dim = self.b_matrix.shape[1] * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates FF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.

        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = scaled_in_tensor @ self.b_matrix  # [..., "num_frequencies"]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device)
        scaled_inputs = scaled_inputs[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.sum((covs @ self.b_matrix) * self.b_matrix, -2)
            input_var = input_var[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs
    
class PolyhedronFFEncoding(FFEncoding):
    """Fourier Feature encoding using polyhedron basis as proposed by mip-NeRF360. Supports integrated encodings.

    Args:
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        basis_shape: Shape of polyhedron basis. Either "octahedron" or "icosahedron"
        basis_subdivisions: Number of times to tesselate the polyhedron.
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        basis_shape: Literal["octahedron", "icosahedron"] = "icosahedron",
        basis_subdivisions: int = 1,
        include_input: bool = False,
    ) -> None:
        basis_t = generate_polyhedron_basis(basis_shape, basis_subdivisions).T
        super().__init__(3, basis_t, num_frequencies, min_freq_exp, max_freq_exp, include_input)

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
        
        # xyz=x*y*z
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
    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"], roughness: Float[Tensor, "*bs 1"]=None) -> Float[Tensor, "*bs output_dim"]:
        outputs = self.pytorch_fwd(in_tensor)
        outputs[...,0:3]*=torch.exp(-roughness)
        outputs[...,3:8]*=torch.exp(-roughness*3)
        outputs[...,8:17]*=torch.exp(-roughness*10)
        outputs[...,17:34]*=torch.exp(-roughness*36)
        return outputs