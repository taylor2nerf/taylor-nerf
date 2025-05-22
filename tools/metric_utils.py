import torch
from lpips import LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Optional, Union
import numpy as np

_LPIPS_NET = None

def _get_lpips_net(device: Optional[Union[str, torch.device]] = None) -> LPIPS:
    global _LPIPS_NET
    if _LPIPS_NET is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _LPIPS_NET = LPIPS(pretrained=True, net="vgg").to(device).eval()
    elif device is not None:
        _LPIPS_NET = _LPIPS_NET.to(device)
    return _LPIPS_NET

@torch.no_grad()
def calculate_lpips(
    x: torch.Tensor, 
    y: torch.Tensor, 
    device: Optional[Union[str, torch.device]] = None
) -> float:
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lpips_net = _get_lpips_net(device)
    
    def normalize(t):
        t = t.unsqueeze(0) if t.ndim == 3 else t
        return t.permute(0, 3, 1, 2) * 2 - 1
    
    x_norm = normalize(x.to(device))
    y_norm = normalize(y.to(device))
    
    return lpips_net(x_norm, y_norm).mean().item()

@torch.no_grad()
def calculate_ssim(
    x: torch.Tensor, 
    y: torch.Tensor, 
    device: Optional[Union[str, torch.device]] = None
) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ssim = SSIM(data_range=1.0).to(device)
    
    def prepare(t):
        t = t.unsqueeze(0) if t.ndim == 3 else t
        return t.permute(0, 3, 1, 2)
    
    return ssim(prepare(x.to(device)), prepare(y.to(device))).item()

@torch.no_grad()
def calculate_psnr(
    x: torch.Tensor, 
    y: torch.Tensor, 
    device: Optional[Union[str, torch.device]] = None
) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    psnr = PSNR().to(device)
    return psnr(x.to(device), y.to(device)).item()

@torch.no_grad()
def calculate_ssim_np(x: np.ndarray, y: np.ndarray) -> float:
    def prepare(arr):
        return arr.transpose(2, 0, 1) if arr.ndim == 3 else arr
    return float(structural_similarity(
        prepare(x), 
        prepare(y), 
        channel_axis=0 if x.ndim == 3 else None,
        data_range=1
    ))

@torch.no_grad()
def calculate_psnr_np(x: np.ndarray, y: np.ndarray) -> float:
    return peak_signal_noise_ratio(x, y, data_range=1)

def psnr_to_mse(psnr: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(psnr, torch.Tensor):
        psnr = torch.tensor(psnr)
    return torch.exp(-0.1 * torch.log(torch.tensor(10.0)) * psnr)

@torch.no_grad()
def calculate_avg_metric(
    psnr: Union[float, torch.Tensor],
    ssim: Union[float, torch.Tensor],
    lpips: Union[float, torch.Tensor]
) -> torch.Tensor:
    if not all(isinstance(x, torch.Tensor) for x in [psnr, ssim, lpips]):
        psnr = torch.tensor(psnr)
        ssim = torch.tensor(ssim)
        lpips = torch.tensor(lpips)
    
    mse = psnr_to_mse(psnr)
    dssim = torch.sqrt(1 - ssim)
    joint_metric = torch.stack([mse, dssim, lpips])
    return torch.exp(torch.mean(torch.log(joint_metric)))

if __name__ == "__main__":
    # Example usage
    psnr_values = [29.39, 28.01, 46.71, 25.22, 32.81, 29.09, 31.87166667]
    ssim_values = [0.912, 0.926, 0.996, 0.892, 0.968, 0.934, 0.938]
    lpips_values = [0.242, 0.075, 0.014, 0.189, 0.119, 0.168, 0.1345]
    
    for psnr, ssim, lpips in zip(psnr_values, ssim_values, lpips_values):
        print(calculate_avg_metric(psnr, ssim, lpips))