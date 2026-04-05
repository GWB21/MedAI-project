#!/usr/bin/env python3
"""
Main experiment runner.

Usage:
    python scripts/run_experiment.py --model llava_med --gpu 0
    python scripts/run_experiment.py --model huatuogpt --gpu 0
    python scripts/run_experiment.py --model medvint --gpu 0
"""

import argparse
import os
import sys

import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Medical VQA Perturbation Experiment")
    parser.add_argument("--model", required=True, choices=["llava_med", "huatuogpt", "medvint"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Run specific conditions only (default: all)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine sigma
    sigma = None
    for p in config["perturbations"]:
        if p["name"] == "lpf" and p.get("sigma") is not None:
            sigma = p["sigma"]
            break

    # Determine conditions
    conditions = args.conditions or [p["name"] for p in config["perturbations"]]

    # Check sigma requirement
    if any(c in ("lpf", "hpf") for c in conditions) and sigma is None:
        print("ERROR: sigma is not set in config. Run calibrate_lpf.py first.")
        print("       Or exclude lpf/hpf: --conditions original black patch_shuffle")
        sys.exit(1)

    # Load dataset
    from src.dataset import PMCVQADataset

    dataset = PMCVQADataset(config["dataset"]["data_dir"])
    print(f"Dataset: {len(dataset)} samples")

    # Load model
    from src.models import get_model

    print(f"Loading model: {args.model}...")
    model = get_model(args.model)
    model.load(device="cuda")
    print(f"Model loaded: {model.name} ({model.precision})")

    # Run inference
    from src.inference import run_inference

    patch_cfg = next((p for p in config["perturbations"] if p["name"] == "patch_shuffle"), {})
    df = run_inference(
        model=model,
        dataset=dataset,
        conditions=conditions,
        output_dir=config["output"]["dir"],
        sigma=sigma,
        patch_size=patch_cfg.get("patch_size", 16),
        seed=config.get("seed", 42),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cond in conditions:
        subset = df[df["condition"] == cond]
        acc = subset["correct"].mean()
        pf = (1 - subset["parse_success"].mean()) * 100
        print(f"  {cond:15s}  Acc={acc:.4f}  ParseFail={pf:.1f}%")


if __name__ == "__main__":
    main()
