#!/usr/bin/env python3
"""
LPF & HPF Sigma Calibration.

Find separate sigma values so that:
  - LPF: mean SSIM(original, LPF_image) ≈ 0.7
  - HPF: mean SSIM(original, HPF_image) ≈ 0.7

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


def compute_hpf(image: np.ndarray, sigma: float) -> np.ndarray:
    """HPF = Original - LPF + 128, clipped to [0, 255]."""
    lpf = cv2.GaussianBlur(image, (0, 0), sigma)
    hpf = image.astype(np.float32) - lpf.astype(np.float32) + 128.0
    return np.clip(hpf, 0, 255).astype(np.uint8)


def calibrate(data_dir: str, target_ssim: float = 0.7, n_samples: int = 200):
    from src.dataset import PMCVQADataset

    dataset = PMCVQADataset(data_dir)
    total = len(dataset)
    sample_indices = np.random.RandomState(42).choice(
        total, size=min(n_samples, total), replace=False
    )

    sigmas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    lpf_results = {s: [] for s in sigmas}
    hpf_results = {s: [] for s in sigmas}

    print(f"Calibrating LPF & HPF sigma on {len(sample_indices)} images...")
    for idx in tqdm(sample_indices):
        try:
            image = dataset.load_image(idx)
        except Exception:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        for sigma in sigmas:
            # LPF
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            blurred_gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY) if len(blurred.shape) == 3 else blurred
            lpf_results[sigma].append(ssim(gray, blurred_gray))

            # HPF
            hpf_img = compute_hpf(image, sigma)
            hpf_gray = cv2.cvtColor(hpf_img, cv2.COLOR_RGB2GRAY) if len(hpf_img.shape) == 3 else hpf_img
            hpf_results[sigma].append(ssim(gray, hpf_gray))

    lpf_mean = {s: np.mean(v) for s, v in lpf_results.items() if len(v) > 0}
    hpf_mean = {s: np.mean(v) for s, v in hpf_results.items() if len(v) > 0}

    # --- LPF ---
    print("\n" + "=" * 50)
    print("LPF Calibration: SSIM(original, GaussianBlur)")
    print("=" * 50)
    print("Sigma | Mean SSIM")
    print("-" * 30)
    best_lpf = min(lpf_mean, key=lambda x: abs(lpf_mean[x] - target_ssim))
    for s in sigmas:
        if s in lpf_mean:
            marker = " <-- best" if s == best_lpf else ""
            print(f"  {s:4d}  |  {lpf_mean[s]:.4f}{marker}")
    print(f"\nLPF best sigma: {best_lpf}  (SSIM = {lpf_mean[best_lpf]:.4f})")

    # --- HPF ---
    print("\n" + "=" * 50)
    print("HPF Calibration: SSIM(original, Original-LPF+128)")
    print("=" * 50)
    print("Sigma | Mean SSIM")
    print("-" * 30)
    best_hpf = min(hpf_mean, key=lambda x: abs(hpf_mean[x] - target_ssim))
    for s in sigmas:
        if s in hpf_mean:
            marker = " <-- best" if s == best_hpf else ""
            print(f"  {s:4d}  |  {hpf_mean[s]:.4f}{marker}")
    print(f"\nHPF best sigma: {best_hpf}  (SSIM = {hpf_mean[best_hpf]:.4f})")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  LPF sigma: {best_lpf}  (SSIM = {lpf_mean[best_lpf]:.4f})")
    print(f"  HPF sigma: {best_hpf}  (SSIM = {hpf_mean[best_hpf]:.4f})")

    # --- Auto-save to config ---
    config_path = os.path.join(os.path.dirname(data_dir), "..", "configs", "experiment_config.yaml")
    config_path = os.path.normpath(config_path)
    if os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for p in config.get("perturbations", []):
            if p["name"] == "lpf":
                p["sigma"] = float(best_lpf)
            elif p["name"] == "hpf":
                p["sigma"] = float(best_hpf)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"\n  configs/experiment_config.yaml 자동 업데이트 완료!")
        print(f"    lpf.sigma = {best_lpf}")
        print(f"    hpf.sigma = {best_hpf}")
    else:
        print(f"\n  [WARN] config 파일을 찾을 수 없음: {config_path}")
        print(f"  수동으로 입력하세요:")
        print(f"    lpf.sigma: {best_lpf}")
        print(f"    hpf.sigma: {best_hpf}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(lpf_mean.keys()), list(lpf_mean.values()), "o-", color="steelblue")
    ax1.axhline(y=target_ssim, color="red", linestyle="--", label=f"Target = {target_ssim}")
    ax1.axvline(x=best_lpf, color="green", linestyle="--", alpha=0.5, label=f"Best = {best_lpf}")
    ax1.set_xlabel("Sigma")
    ax1.set_ylabel("Mean SSIM")
    ax1.set_title("LPF Sigma Calibration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(hpf_mean.keys()), list(hpf_mean.values()), "o-", color="darkorange")
    ax2.axhline(y=target_ssim, color="red", linestyle="--", label=f"Target = {target_ssim}")
    ax2.axvline(x=best_hpf, color="green", linestyle="--", alpha=0.5, label=f"Best = {best_hpf}")
    ax2.set_xlabel("Sigma")
    ax2.set_ylabel("Mean SSIM")
    ax2.set_title("HPF Sigma Calibration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(data_dir), "calibration_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")

    return {"lpf_sigma": best_lpf, "hpf_sigma": best_hpf}, lpf_mean, hpf_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_ssim", type=float, default=0.7)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    calibrate(args.data_dir, args.target_ssim, args.n_samples)
