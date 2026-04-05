"""
Main inference loop.
Iterates over dataset samples and conditions, runs model inference, saves results.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import PMCVQADataset
from .perturbations import apply_perturbation
from .models.base_model import BaseMedVQAModel


CONDITIONS = ["original", "black", "lpf", "hpf", "patch_shuffle"]


def run_inference(
    model: BaseMedVQAModel,
    dataset: PMCVQADataset,
    conditions: List[str],
    output_dir: str,
    lpf_sigma: Optional[float] = None,
    hpf_sigma: Optional[float] = None,
    patch_size: int = 16,
    seed: int = 42,
    max_new_tokens: int = 32,
) -> pd.DataFrame:
    """
    Run inference for all conditions and save results.
    LPF and HPF use separate sigma values (each calibrated to SSIM ≈ 0.7).

    Returns:
        Combined DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  Model: {model.name} | Condition: {condition}")
        print(f"{'='*60}")

        # Select the appropriate sigma for this condition
        if condition == "lpf":
            sigma = lpf_sigma
        elif condition == "hpf":
            sigma = hpf_sigma
        else:
            sigma = None

        results = _run_single_condition(
            model=model,
            dataset=dataset,
            condition=condition,
            sigma=sigma,
            patch_size=patch_size,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        all_results.extend(results)

        # Save intermediate results per condition
        cond_df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cond_path = Path(output_dir) / f"{model.name}_{condition}_{timestamp}.csv"
        cond_df.to_csv(cond_path, index=False)
        print(f"  Saved: {cond_path}")

        acc = cond_df["correct"].mean()
        parse_fail = (1 - cond_df["parse_success"].mean()) * 100
        print(f"  Accuracy: {acc:.4f} | Parse fail: {parse_fail:.1f}%")

    # Save combined results
    combined_df = pd.DataFrame(all_results)
    combined_path = Path(output_dir) / f"{model.name}_all_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nAll results saved: {combined_path}")

    return combined_df


def _run_single_condition(
    model: BaseMedVQAModel,
    dataset: PMCVQADataset,
    condition: str,
    sigma: Optional[float],
    patch_size: int,
    seed: int,
    max_new_tokens: int = 32,
) -> List[dict]:
    results = []
    perturbation_kwargs = {}
    if condition in ("lpf", "hpf"):
        if sigma is None:
            raise ValueError(f"sigma is required for {condition}. Run calibrate_lpf.py first.")
        perturbation_kwargs["sigma"] = sigma
    if condition == "patch_shuffle":
        perturbation_kwargs["patch_size"] = patch_size
        perturbation_kwargs["seed"] = seed

    for idx in tqdm(range(len(dataset)), desc=condition):
        item = dataset[idx]
        # MedVInT uses its own prompt format with raw choices (접두사 포함)
        if hasattr(model, 'build_prompt'):
            prompt = model.build_prompt(
                item["question"], item["raw_choice_A"], item["raw_choice_B"],
                item["raw_choice_C"], item["raw_choice_D"]
            )
        else:
            prompt = dataset.get_prompt(idx)

        try:
            image = dataset.load_image(idx)
        except (FileNotFoundError, Exception) as e:
            print(f"  [SKIP] {item['image_id']}: {e}")
            continue

        # Apply perturbation
        perturbed = apply_perturbation(image, condition, **perturbation_kwargs)

        # Run inference
        try:
            output = model.inference(perturbed, prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"  [ERROR] {item['image_id']}: {e}")
            output = None

        if output is None:
            results.append({
                "image_id": item["image_id"],
                "condition": condition,
                "model": model.name,
                "question": item["question"],
                "choice_A": item["choice_A"],
                "choice_B": item["choice_B"],
                "choice_C": item["choice_C"],
                "choice_D": item["choice_D"],
                "gt_answer": item["gt_answer"],
                "pred_answer": "ERROR",
                "correct": 0,
                "parse_success": 0,
                "raw_output": "",
                "logit_A": float("nan"),
                "logit_B": float("nan"),
                "logit_C": float("nan"),
                "logit_D": float("nan"),
            })
            continue

        # Primary: generate() 텍스트 파싱
        # Logit은 CSV 기록용 (답변 결정에 사용하지 않음)
        if output.parse_success:
            pred = output.parsed_answer
        else:
            pred = "PARSE_FAIL"

        correct = 1 if pred == item["gt_answer"] else 0

        results.append({
            "image_id": item["image_id"],
            "condition": condition,
            "model": model.name,
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "gt_answer": item["gt_answer"],
            "pred_answer": pred,
            "correct": correct,
            "parse_success": int(output.parse_success),
            "raw_output": output.raw_text,
            "logit_A": output.logits.get("A", float("nan")),
            "logit_B": output.logits.get("B", float("nan")),
            "logit_C": output.logits.get("C", float("nan")),
            "logit_D": output.logits.get("D", float("nan")),
        })

    return results
