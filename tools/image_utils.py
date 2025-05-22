import os
from pathlib import Path
from typing import Optional, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torch.nn import functional as F

OUTPUT_SUBDIRS = {
    "test": "rgb_test",
    "error": "rgb_error",  
    "depth": "rgb_depth",
    "depth_tiff": "rgb_depth_tiff"
}

def save_image(
    out_dir: Union[str, Path],
    image: torch.Tensor,
    filename: str,
    flag: str = "test",
    const_norm: Optional[float] = None
) -> None:
    out_dir = Path(out_dir)
    image = image.detach().cpu().numpy()
    
    if flag in {"test", "error"}:
        save_dir = out_dir / OUTPUT_SUBDIRS[flag]
        save_dir.mkdir(parents=True, exist_ok=True)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(save_dir / filename, image)
    
    elif flag == "depth":
        save_dir_depth = out_dir / OUTPUT_SUBDIRS["depth"]
        save_dir_tiff = out_dir / OUTPUT_SUBDIRS["depth_tiff"]
        save_dir_depth.mkdir(parents=True, exist_ok=True)
        save_dir_tiff.mkdir(parents=True, exist_ok=True)
        
        depth = image[..., 0]
        pseudo_color = depth_colmap(depth)
        cv2.imwrite(str(save_dir_depth / filename), pseudo_color)
        
        tiff_filename = filename.replace(".png", ".tiff")
        save_img_f32(depth * 10000, save_dir_tiff / tiff_filename)
    
    elif flag in OUTPUT_SUBDIRS:
        save_dir = out_dir / OUTPUT_SUBDIRS[flag]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if image.ndim == 3 and image.shape[-1] == 3: 
            image = F.normalize(torch.from_numpy(image), dim=-1).numpy()
        
        image = ((image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(save_dir / filename, image)

def depth_colmap(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)

def save_img_f32(data: np.ndarray, path: Union[str, Path]) -> None:
    Image.fromarray(np.nan_to_num(data).astype(np.float32)).save(path, "TIFF")