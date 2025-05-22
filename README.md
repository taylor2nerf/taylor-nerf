# Neural Rendering with Polynomial Volume Integral Representation
## Method Overview
![Pipeline](assets/pipeline.png)
Neural Radiance Fields (NeRFs) have gained popularity by demonstrating impressive capabilities in synthesizing realistic novel views from multiple view images. It approximates the continuous integration of rays as a finite Riemann sum of the estimated colors and densities of the sampled points. Although this allows for efficient rendering, approximating divergent integrals under varying directions and interval lengths with piecewise constant features does not account for high-order variations within integration intervals, leading to ambiguous representations and limited reconstruction quality. In this paper, we propose to model the distribution of the sampled intervals with Taylor series, which can encode the length and direction information of integrals to disambiguate interval distributions and mitigate integral approximation errors in volume rendering. We introduce a learnable gradient estimator and an adaptive interval length scaling module to capture smooth high-order spatial variations, enhancing optimization stability and performance. Our proposed method allows an easy integration with existing NeRF-based rendering frameworks. Experimental results on both synthetic and real-world scenes demonstrate that our method significantly boosts the rendering quality of various NeRF models, achieving state-of-the-art performance.

## Visual Blender Result
<p align="center">
  <img src="assets/rip_materials_rgb.gif" width="45%">
  <img src="assets/rip_materials_depth.gif" width="45%">
</p>

## Only a few lines of changes were needed in key sections of code to convert various NeRF backbone into ours, such as Nerfacc OccGrid：
```diff
  -def query_density(self, x, return_feat: bool=False, t_dirs: torch.Tensor=None, t_dists: torch.Tensor=None):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
            
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        
        temp_x = (
                self.mlp_base(x.reshape(-1, 3))
                .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim + 192])
                .to(x)
            )

        density_before_activation,base_mlp_out,  density_grad, density_hessian, base_mlp_out_grad, base_mlp_out_hessian = \
            torch.split(
                temp_x, [1, self.geo_feat_dim, self.density_grad_dim, self.density_hessian_dim, self.latent_grad_dim, self.latent_hessian_dim], dim=-1)
                        
                
        if  t_dirs is not None:
            t_dirs_t = t_dirs[..., None,:]
            t_dirs_t = (t_dirs_t.permute(0, 2, 1) @ t_dirs_t).reshape(-1, 1, 9)

            base_mlp_out_grad = base_mlp_out_grad.reshape(-1, self.geo_feat_dim, 3)
            base_mlp_out_grad = (base_mlp_out_grad @ t_dirs[..., None]).reshape(x.shape[0], -1) 
                    
            base_mlp_out_hessian = base_mlp_out_hessian.reshape(-1, self.geo_feat_dim, 9, 1)
            base_mlp_out_hessian = (t_dirs_t[:, None] @ base_mlp_out_hessian).reshape(-1, 15)
                    
            density_grad = (density_grad[..., None,:] @ t_dirs[..., None]).reshape(x.shape[0], -1)            

            density_hessian = density_hessian.reshape(-1, 9, 1)
            density_hessian = (t_dirs_t @ density_hessian).reshape(-1, 1)
                
            if self.training and return_feat: 
                self.gradient_loss.append(F.mse_loss(density_grad, torch.zeros_like(density_grad)))
                self.density_hessian_loss.append(F.mse_loss(density_hessian, torch.zeros_like(density_hessian)))
                    
                self.latent_loss.append(F.mse_loss(base_mlp_out_grad, torch.zeros_like(base_mlp_out_grad)))
                self.latent_hessian_loss.append(F.mse_loss(base_mlp_out_hessian, torch.zeros_like(base_mlp_out_hessian)))

            base_mlp_out = base_mlp_out + base_mlp_out_grad + base_mlp_out_hessian
            density_before_activation = density_before_activation + density_grad + density_hessian 
            
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
            
        if return_feat:
            return density, base_mlp_out
        else:
            return density
```

## ⚙️ Setup
Our code is based on [Nerfacc](https://github.com/nerfstudio-project/nerfacc).
### Clone this repository.
```text
git clone --recursive https://github.com/taylor2nerf/taylor-nerf.git
```
### Install Environment via Anaconda (Recommended)
```text
conda create -n nerfacc python=3.10
conda activate nerfacc
pip install -r requirements.txt
```
### Compilation tiny-cuda-nn
```text
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
| Windows & Linux | cu113 | cu115 | cu116 | cu117 | cu118 |
|-----------------|-------|-------|-------|-------|-------|
| torch 1.11.0    | ✅    | ✅    |       |       |       |
| torch 1.12.0    | ✅    |       | ✅    |       |       |
| torch 1.13.0    |       |       | ✅    | ✅    |       |
| torch 2.0.0     |       |       |       | ✅    | ✅    |


### 📦 Dataset
We mainly test our method on Synthetic Blender, and Mip-360 v2 dataset. Put them under the data folder:
```text
data
└── nerf_synthetic
    └── hotdog
    └── lego
```

### 🏃 Training
We provide the script to test our code on each scene of NeRF Synthetic datasets. Just run:
```bash
bash scripts/train_ns_all.sh
```

