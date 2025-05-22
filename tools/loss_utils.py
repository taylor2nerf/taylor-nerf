import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss

def calc_eikonal_loss(gradients, outside=None):
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # [B,R,N]
    gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (gradient_error * (~outside).float()).mean()
    else:
        return gradient_error.mean()

def calc_curvature_loss(hessian, outside=None):
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:, 1:,:])
        count_w = self._tensor_size(x[:,:,:, 1:])
        h_tv = torch.pow((x[:,:, 1:,:] - x[:,:,:h_x - 1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:, 1:] - x[:,:,:,:w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-7
        
    def forward(self, predict, target):
        error = torch.sqrt((predict - target) ** 2 + self.eps)
        return torch.mean(error)

def fast_loss_dist(t, w):
    ut = (t[..., 1:] + t[...,:-1]) / 2
    interval = t[..., 1:] - t[...,:-1]
    return eff_distloss(w, ut, interval)

def loss_dist(t, w):
    ut = (t[..., 1:] + t[...,:-1]) / 2
    dut = torch.abs(ut[...,:, None] - ut[..., None,:])
    loss_inter = torch.sum(w * torch.sum(w[..., None,:] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w ** 2 * (t[..., 1:] - t[...,:-1]), dim=-1) / 3
    return loss_inter + loss_intra

def compute_depth_smoothness_loss(renderings, config):
    smoothness_losses = []
    loss = lambda x: torch.mean(torch.abs(x))
    bilateral_filter = lambda x: torch.exp(-torch.abs(x).mean(-1, keepdim=True))

    for rendering in renderings:
        depths = rendering['distance']

        with torch.no_grad():
            acc00 = rendering['acc'][...,:-1,:-1,None]
            weights = rendering['rgb']

        v00 = depths[...,:-1,:-1,:]
        v01 = depths[...,:-1,1:,:]
        v10 = depths[...,1:,:-1,:]

        w01 = bilateral_filter(weights[...,:-1,:-1,:] - weights[...,:-1,1:,:])
        w10 = bilateral_filter(weights[...,:-1,:-1,:] - weights[...,1:,:-1,:])
        L1 = loss(acc00 * w01 * (v00 - v01)**2)
        L2 = loss(acc00 * w10 * (v00 - v10)**2)
        smoothness_losses.append((L1 + L2) / 2)

    smoothness_losses = torch.stack(smoothness_losses)

    loss = (config.depth_smoothness_coarse_loss_mult * torch.sum(smoothness_losses[:-1]) + \
        config.depth_smoothness_loss_mult * smoothness_losses[-1])
    return loss

