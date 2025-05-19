# Neural Rendering with Polynomial Volume Integral Representation
## Method Overview

Neural Radiance Fields (NeRFs) have gained popularity by demonstrating impressive capabilities in synthesizing realistic novel views from multiple view images. It approximates the continuous integration of rays as a finite Riemann sum of the estimated colors and densities of the sampled points. Although this allows for efficient rendering, approximating divergent integrals under varying directions and interval lengths with piecewise constant features does not account for high-order variations within integration intervals, leading to ambiguous representations and limited reconstruction quality. In this paper, we propose to model the distribution of the sampled intervals with Taylor series, which can encode the length and direction information of integrals to disambiguate interval distributions and mitigate integral approximation errors in volume rendering. We introduce a learnable gradient estimator and an adaptive interval length scaling module to capture smooth high-order spatial variations, enhancing optimization stability and performance. Our proposed method allows an easy integration with existing NeRF-based rendering frameworks. Experimental results on both synthetic and real-world scenes demonstrate that our method significantly boosts the rendering quality of various NeRF models, achieving state-of-the-art performance.

## ⚙️ Setup
### Install Environment via Anaconda (Recommended)
#### Clone this repository.


### 📦 Dataset
We mainly test our method on NeRF Synthetic, and MipNeRF-360 v2 dataset. Put them under the data folder:
data
└── nerf_synthetic
    └── hotdog
    └── lego

### 🏃 Training
We provide the script to test our code on each scene of NeRF Synthetic datasets. Just run:
bash scripts/train_ns_all.sh

