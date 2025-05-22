import torch
import random
import numpy as np
import pathlib
import argparse
import os
from typing import Tuple, Type

from .render_utils import MIPNERF360_UNBOUNDED_SCENES, NERF_SYNTHETIC_SCENES


def get_config(filename: str, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    with open(filename, encoding="utf8") as f:
        data = f.readlines()
    
    for line in data:
        line = line.strip().replace(" ", "")
        if len(line) >= 2 and "=" in line and "#" not in line:
            parameter_name, data = line.split("=")
            
            if data.isdigit() or data.startswith(("[", "(", "{")) or data[0].isdigit():
                default = eval(data)
            elif data == 'None':
                default = None
            else:
                default = data
                
            parser.add_argument(f"--{parameter_name}", default=default)
            
    return parser

def get_opt(seed: int = 42) -> Tuple[argparse.ArgumentParser, Type, float]:
    parser = argparse.ArgumentParser(description="NeRF training configuration")
    
    # Basic configuration
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        help="Which scene to use",
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/nerf_synthetic/",
        help="The root directory of the dataset",
    )
    
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="Which train split to use",
    )
    
    # Rate parameters
    rate_params = {
        "occ_rate": 1e-2,
        "dgrad_rate": 1e-4,
        "dhess_rate": 1e-4,
        "lgrad_rate": 1e-4,
        "lhess_rate": 1e-4
    }
    
    for param, default in rate_params.items():
        parser.add_argument(
            f"--{param}",
            type=float,
            default=default,
            help=f"{param.replace('_', ' ')} parameter",
        )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Which device to use",
    )
    
    args, _ = parser.parse_known_args()
    scene = args.scene
    set_random_seed(seed) 
    parser = get_config("configs/nerf_synthetic_config.py", parser)
    weight_decay = (
        1e-5 if scene in ["materials", "ficus", "drums"] else 1e-6
    )
    
    return parser, weight_decay

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    print(MIPNERF360_UNBOUNDED_SCENES)