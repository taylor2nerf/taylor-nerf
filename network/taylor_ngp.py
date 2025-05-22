from typing import Callable, List, Union, Optional
import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

class _TruncExp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply

class NGPRadianceField(torch.nn.Module):    
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        n_features_per_level: int = 4
    ) -> None:
        super().__init__()
        
        aabb = self._init_aabb(aabb, num_dim)
        self.register_buffer("aabb", aabb)
        
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.geo_feat_dim = geo_feat_dim
        self.n_features_per_level = n_features_per_level
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        
        self._init_regularization_buffers()
        self._init_encoding_networks()

    def _init_aabb(self, aabb, num_dim):
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
    
        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        return torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

    def _init_regularization_buffers(self):
        self._density_grad_loss = []
        self._density_hessian_loss = []
        self._latent_grad_loss = []
        self._latent_hessian_loss = []
        
        # Dimensions for gradient/hessian components
        self.density_grad_dim = 3
        self.density_hessian_dim = 9
        self.latent_grad_dim = 15 * 3
        self.latent_hessian_dim = 9 * 15

    def _init_encoding_networks(self):
        per_level_scale = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution)) / 
            (self.n_levels - 1)
        ).tolist()
        
        self.resolutions = torch.tensor(
            [self.base_resolution * (per_level_scale ** level) 
             for level in range(self.n_levels)]
        ).cuda()

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=self.num_dim,
            encoding_config={
                "otype": "Composite",
                "nested": [{
                    "n_dims_to_encode": 3,
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                }],
            },
        )
        
        # Geometry MLP
        self.geo_mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=self.num_dim,
            n_output_dims=1 + self.geo_feat_dim + self.density_grad_dim + 
                         self.density_hessian_dim + self.latent_grad_dim + 
                         self.latent_hessian_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        # Appearance MLP
        self.app_mlp = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def query_density(
        self, 
        x: torch.Tensor, 
        t_dirs: Optional[torch.Tensor] = None,
        t_dists: Optional[torch.Tensor] = None,
        return_feat: bool = False
    ):
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        
        x = self.geo_mlp(x).view(
            list(x.shape[:-1]) + [1 + self.geo_feat_dim + self.density_grad_dim + 
                                self.density_hessian_dim + self.latent_grad_dim + 
                                self.latent_hessian_dim]
        ).to(x)
        
        density_before_activation, density_grad, density_hessian, \
            base_mlp_out, base_mlp_out_grad, base_mlp_out_hessian = torch.split(
                x, [1, self.density_grad_dim, self.density_hessian_dim, 
                    self.geo_feat_dim, self.latent_grad_dim, self.latent_hessian_dim], 
                dim=-1
            )
        
        if t_dirs is not None:
            density_before_activation, base_mlp_out = self._apply_directional_derivatives(
                t_dirs, density_before_activation, density_grad, density_hessian,
                base_mlp_out, base_mlp_out_grad, base_mlp_out_hessian,
                return_feat
            )
        
        density = self.density_activation(density_before_activation) * selector[..., None]
        return (density, base_mlp_out) if return_feat else density
    
    ### taylor-nerf core code
    def _apply_directional_derivatives(
        self,
        t_dirs: torch.Tensor,
        density_before_activation: torch.Tensor,
        density_grad: torch.Tensor,
        density_hessian: torch.Tensor,
        base_mlp_out: torch.Tensor,
        base_mlp_out_grad: torch.Tensor,
        base_mlp_out_hessian: torch.Tensor,
        return_feat: bool=False
    ):
        t_dirs_3d = t_dirs.unsqueeze(-1)  # [..., 3, 1]
        t_dirs_9d = (t_dirs_3d @ t_dirs_3d.transpose(-2, -1)).flatten(start_dim=-2)  # [..., 9]
        
        density_grad = torch.einsum('...i,...i->...', density_grad, t_dirs).unsqueeze(-1)
        density_hessian = torch.einsum('...i,...i->...', 
                                      density_hessian.view(-1, 9), 
                                      t_dirs_9d).unsqueeze(-1)
        base_mlp_out_grad = torch.einsum('...ij,...j->...i',
                                        base_mlp_out_grad.view(-1, self.geo_feat_dim, 3),
                                        t_dirs)
        base_mlp_out_hessian = torch.einsum('...ij,...j->...i',
                                           base_mlp_out_hessian.view(-1, self.geo_feat_dim, 9),
                                           t_dirs_9d)
        
        if self.training and return_feat:
            zeros = torch.zeros_like(density_grad)
            self._density_grad_loss.append(F.mse_loss(density_grad, zeros))
            self._density_hessian_loss.append(F.mse_loss(density_hessian, zeros))
            self._latent_grad_loss.append(F.mse_loss(base_mlp_out_grad, zeros.expand_as(base_mlp_out_grad)))
            self._latent_hessian_loss.append(F.mse_loss(base_mlp_out_hessian, zeros.expand_as(base_mlp_out_hessian)))
        
        # Update outputs
        density_before_activation = density_before_activation + density_grad + density_hessian
        base_mlp_out = base_mlp_out + base_mlp_out_grad + base_mlp_out_hessian
        return density_before_activation, base_mlp_out

    @property
    def density_grad_loss(self):
        loss = sum(self._density_grad_loss) / len(self._density_grad_loss) if self._density_grad_loss else 0
        self._density_grad_loss = []
        return loss
    
    @property 
    def density_hessian_loss(self):
        loss = sum(self._density_hessian_loss) / len(self._density_hessian_loss) if self._density_hessian_loss else 0
        self._density_hessian_loss = []
        return loss
    
    @property
    def latent_grad_loss(self):
        loss = sum(self._latent_grad_loss) / len(self._latent_grad_loss) if self._latent_grad_loss else 0
        self._latent_grad_loss = []
        return loss
    
    @property
    def latent_hessian_loss(self):
        loss = sum(self._latent_hessian_loss) / len(self._latent_hessian_loss) if self._latent_hessian_loss else 0
        self._latent_hessian_loss = []
        return loss
        
    def _query_rgb(self, t_dirs: torch.Tensor, embedding: torch.Tensor, apply_act: bool = True):
        t_dirs = (t_dirs + 1.0) / 2.0  
        d = self.direction_encoding(t_dirs.reshape(-1, t_dirs.shape[-1]))
        h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        rgb = self.app_mlp(h).reshape(list(embedding.shape[:-1]) + [3]).to(embedding)
        return torch.sigmoid(rgb) if apply_act else rgb

    def forward(
        self,
        positions: torch.Tensor,
        t_dirs: Optional[torch.Tensor] = None,
        t_dists: Optional[torch.Tensor] = None,
    ):
        density, embedding = self.query_density(
            positions, t_dirs=t_dirs, t_dists=t_dists, return_feat=True
        )
        rgb = self._query_rgb(t_dirs, embedding=embedding)
        return rgb, density