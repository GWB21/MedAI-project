#!/usr/bin/env python3
"""
Perturbation Sanity Check 시각화.
원본 + 5개 조건을 한 줄에 나란히 보여주는 예시 이미지를 생성합니다.

Usage:
    python scripts/visualize_perturbations.py --data_dir ./data/pmc_vqa --n_examples 5
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.perturbations import apply_perturbation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/pmc_vqa")
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--output", default="./results/perturbation_examples.png")
    parser.add_argument("--lpf_sigma", type=float, default=3)
    parser.add_argument("--hpf_sigma", type=float, default=25)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.dataset import PMCVQADataset

    dataset = PMCVQADataset(args.data_dir)

    # 다양한 이미지를 균등하게 선택
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_examples, len(dataset)), replace=False)

    conditions = [
        ("Original", "original", {}),
        ("Black", "black", {}),
        (f"LPF (σ={args.lpf_sigma})", "lpf", {"sigma": args.lpf_sigma}),
        (f"HPF (σ={args.hpf_sigma})", "hpf", {"sigma": args.hpf_sigma}),
        (f"Patch Shuffle ({args.patch_size}×{args.patch_size})", "patch_shuffle",
         {"patch_size": args.patch_size, "seed": args.seed}),
    ]

    n_rows = len(indices)
    n_cols = len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, data_idx in enumerate(indices):
        image = dataset.load_image(data_idx)
        item = dataset[data_idx]
        image_id = item["image_id"]

        for col_idx, (title, cond_name, kwargs) in enumerate(conditions):
            perturbed = apply_perturbation(image, cond_name, **kwargs)
            ax = axes[row_idx, col_idx]
            ax.imshow(perturbed)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(image_id, fontsize=9, rotation=0, labelpad=80, ha="right")

    fig.suptitle("Perturbation Sanity Check (PMC-VQA test_clean)", fontsize=14, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"저장 완료: {args.output}")
    print(f"  {n_rows}개 이미지 × {n_cols}개 조건")


if __name__ == "__main__":
    main()
