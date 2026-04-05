#!/usr/bin/env python3
"""
LPF Sigma Calibration.

Find the Gaussian blur sigma that produces average SSIM ≈ 0.7 on PMC-VQA images.

Usage:
    python scripts/calibrate_lpf.py --data_dir ./data/pmc_vqa --target_ssim 0.7
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def calibrate(data_dir: str, target_ssim: float = 0.7, n_samples: int = 200):
    from src.dataset import PMCVQADataset

    dataset = PMCVQADataset(data_dir)
    total = len(dataset)
    sample_indices = np.random.RandomState(42).choice(
        total, size=min(n_samples, total), replace=False
    )

    sigmas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    results = {s: [] for s in sigmas}

    print(f"Calibrating LPF sigma on {len(sample_indices)} images...")
    for idx in tqdm(sample_indices):
        try:
            image = dataset.load_image(idx)
        except Exception:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        for sigma in sigmas:
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            blurred_gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY) if len(blurred.shape) == 3 else blurred
            s = ssim(gray, blurred_gray)
            results[sigma].append(s)

    # Compute mean SSIM per sigma
    mean_ssim = {s: np.mean(v) for s, v in results.items() if len(v) > 0}

    print("\nSigma | Mean SSIM")
    print("-" * 25)
    for s in sigmas:
        if s in mean_ssim:
            marker = " <-- closest" if s == min(mean_ssim, key=lambda x: abs(mean_ssim[x] - target_ssim)) else ""
            print(f"  {s:4d}  |  {mean_ssim[s]:.4f}{marker}")

    # Find best sigma
    best_sigma = min(mean_ssim, key=lambda x: abs(mean_ssim[x] - target_ssim))
    print(f"\nBest sigma for SSIM ≈ {target_ssim}: {best_sigma}")
    print(f"Achieved mean SSIM: {mean_ssim[best_sigma]:.4f}")
    print(f"\n--> Update configs/experiment_config.yaml:")
    print(f"    lpf.sigma: {best_sigma}")
    print(f"    hpf.sigma: {best_sigma}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(mean_ssim.keys()), list(mean_ssim.values()), "o-", color="steelblue")
    ax.axhline(y=target_ssim, color="red", linestyle="--", label=f"Target SSIM = {target_ssim}")
    ax.axvline(x=best_sigma, color="green", linestyle="--", alpha=0.5, label=f"Best sigma = {best_sigma}")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Mean SSIM")
    ax.set_title("LPF Sigma Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(data_dir), "calibration_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")

    return best_sigma, mean_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_ssim", type=float, default=0.7)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    calibrate(args.data_dir, args.target_ssim, args.n_samples)
