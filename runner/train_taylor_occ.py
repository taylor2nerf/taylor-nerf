import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from nerfacc.estimators.occ_grid import OccGridEstimator
from network.taylor_ngp import NGPRadianceField
from datasets.nerf_synthetic import SubjectLoader
from tools.config_utils import get_opt
from tools.image_utils import save_image
from tools.metric_utils import calculate_lpips, calculate_psnr, calculate_ssim
from tools.render_utils import (
    render_image_with_occgrid,
    render_image_with_occgrid_test
)

class NeRFTrainer:
    def __init__(self, args, weight_decay):
        self.args = args
        self.device = args.device
        self.weight_decay = weight_decay
        self.setup_datasets()
        self.setup_models()
        self.setup_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        self.start_time = time.time()
        self.best_metrics = {"psnr": 0, "ssim": 0, "lpips": float('inf')}

    def setup_datasets(self):
        self.train_dataset = SubjectLoader(
            subject_id=self.args.scene,
            root_fp=self.args.data_root,
            split=self.args.train_split,
            num_rays=self.args.init_batch_size,
            device=self.device,
            **self.args.train_dataset_kwargs,
        )

        self.test_dataset = SubjectLoader(
            subject_id=self.args.scene,
            root_fp=self.args.data_root,
            split="test",
            num_rays=None,
            device=self.device,
            **self.args.test_dataset_kwargs,
        )

    def setup_models(self):
        self.estimator = OccGridEstimator(
            roi_aabb=self.args.aabb,
            resolution=self.args.grid_resolution,
            levels=self.args.grid_nlvl
        ).to(self.device)

        self.radiance_field = NGPRadianceField(
            aabb=self.estimator.aabbs[-1]
        ).to(self.device)

    def setup_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.radiance_field.parameters(),
            lr=1e-2,
            eps=1e-15,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[
                        self.args.max_steps // 2,
                        self.args.max_steps * 3 // 4,
                        self.args.max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )

    def train_step(self, step: int) -> bool:
        self.radiance_field.train()
        self.estimator.train()

        # Sample random ray batch
        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]
        render_bkgd, rays, pixels = data["color_bkgd"], data["rays"], data["pixels"]

        # Occupancy grid update
        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.args.render_step_size

        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=self.args.occ_rate,
        )

        # Render image
        render_result = render_image_with_occgrid(
            self.radiance_field,
            self.estimator,
            rays,
            near_plane=self.args.near_plane,
            render_step_size=self.args.render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=self.args.cone_angle,
            alpha_thre=self.args.alpha_thre,
        )
        
        if render_result[-1] == 0:  # n_rendering_samples
            return

        # Dynamic batch size adjustment
        if self.args.target_sample_batch_size > 0:
            num_rays = int(len(pixels) * (self.args.target_sample_batch_size / float(render_result[-1])))
            self.train_dataset.update_num_rays(num_rays)

        # Compute losses
        rgb = render_result[0]
        loss = F.smooth_l1_loss(rgb, pixels)
        loss += self.args.dgrad_rate * self.radiance_field.density_grad_loss
        loss += self.args.dhess_rate * self.radiance_field.density_hessian_loss
        loss += self.args.lgrad_rate * self.radiance_field.latent_grad_loss
        loss += self.args.lhess_rate * self.radiance_field.latent_hessian_loss

        # Optimization step
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()

        if step % 5000 == 0:
            elapsed_time = time.time() - self.start_time
            psnr = calculate_psnr(rgb, pixels)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss.item():.5f} | psnr={psnr:.2f} | "
                f"n_rendering_samples={render_result[-1]:d} | "
                f"max_depth={render_result[2].max():.3f} | "
            )

    def evaluate(self, step: int):
        """Evaluate model on test set and save results."""
        self.radiance_field.eval()
        self.estimator.eval()

        metrics = {"psnr": [], "lpips": [], "ssim": []}
        save_root_dir = self.create_save_directory(step)

        with torch.no_grad():
            for i in tqdm.trange(len(self.test_dataset)):
                data = self.test_dataset[i]
                # render_result = render_image_with_occgrid(
                #     self.radiance_field,
                #     self.estimator,
                #     data["rays"],
                #     near_plane=self.args.near_plane,
                #     render_step_size=self.args.render_step_size,
                #     render_bkgd=data["color_bkgd"],
                #     cone_angle=self.args.cone_angle,
                #     alpha_thre=self.args.alpha_thre,
                # )

                render_result = render_image_with_occgrid_test(
                    self.args.test_chunk_size,
                    self.radiance_field,
                    self.estimator,
                    data["rays"],
                    near_plane=self.args.near_plane,
                    render_step_size=self.args.render_step_size,
                    render_bkgd=data["color_bkgd"],
                    cone_angle=self.args.cone_angle,
                    alpha_thre=self.args.alpha_thre,
                )

                self.process_render_result(i, render_result, data["pixels"], metrics, save_root_dir)
                
        self.save_final_metrics(metrics, save_root_dir)

    def create_save_directory(self, step: int, save_root_dir:str="result") -> Path:
        save_root_dir = Path(save_root_dir) 
        save_root_dir = save_root_dir / self.args.scene
        save_root_dir.mkdir(parents=True, exist_ok=True)
        return save_root_dir

    def process_render_result(self, idx: int, render_result: Tuple, pixels: torch.Tensor, 
                            metrics: Dict, save_dir: Path):
        
        rgb, acc, depth, _ = render_result
        metrics["psnr"].append(calculate_psnr(rgb, pixels))
        metrics["lpips"].append(calculate_lpips(rgb, pixels))
        metrics["ssim"].append(calculate_ssim(rgb, pixels))
        
        # # Save outputs
        # filename = f"{idx}.png"
        # save_image(save_dir, depth, filename, flag="depth")
        # save_image(save_dir, rgb, filename)

    def save_final_metrics(self, metrics: Dict, save_dir: Path):
        psnr_avg = sum(metrics["psnr"]) / len(metrics["psnr"])
        lpips_avg = sum(metrics["lpips"]) / len(metrics["lpips"])
        ssim_avg = sum(metrics["ssim"]) / len(metrics["ssim"])

        print(f"Evaluation: PSNR={psnr_avg:.2f}, LPIPS={lpips_avg:.4f}, SSIM={ssim_avg:.4f}")

        results = {
            "psnr": psnr_avg,
            "ssim": ssim_avg,
            "lpips": lpips_avg,
            **vars(self.args)
        }

        with open(save_dir / "result.json", "w") as f:
            json.dump(results, f, indent=2)

    def train(self):        
        for step in tqdm.trange(self.args.max_steps + 1, desc=f"{self.args.scene}"):
            success = self.train_step(step)
            if step > 0 and step % self.args.max_steps == 0:
                self.evaluate(step)

def main():
    parser, weight_decay = get_opt(42)
    args = parser.parse_args()
    trainer = NeRFTrainer(args, weight_decay)
    trainer.train()

if __name__ == "__main__":
    main()