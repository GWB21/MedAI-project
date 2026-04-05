#!/usr/bin/env python3
"""
Main experiment runner.

Usage:
    python scripts/run_experiment.py --model llava_v15 --gpu 0
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
    parser.add_argument("--model", required=True, choices=["llava_v15", "huatuogpt", "medvint"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Run specific conditions only (default: all)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (adjust by VRAM, default: 1)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader num_workers (adjust by CPU cores, default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max new tokens to generate (overrides config, default: 32)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine separate sigmas for LPF and HPF
    lpf_sigma = None
    hpf_sigma = None
    for p in config["perturbations"]:
        if p["name"] == "lpf" and p.get("sigma") is not None:
            lpf_sigma = p["sigma"]
        if p["name"] == "hpf" and p.get("sigma") is not None:
            hpf_sigma = p["sigma"]

    # Determine conditions
    conditions = args.conditions or [p["name"] for p in config["perturbations"]]

    # Check sigma requirements
    if "lpf" in conditions and lpf_sigma is None:
        print("ERROR: lpf.sigma is not set in config. Run calibrate_lpf.py first.")
        print("       Or exclude lpf: --conditions original black hpf patch_shuffle")
        sys.exit(1)
    if "hpf" in conditions and hpf_sigma is None:
        print("ERROR: hpf.sigma is not set in config. Run calibrate_lpf.py first.")
        print("       Or exclude hpf: --conditions original black lpf patch_shuffle")
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

    # max_new_tokens: CLI > 모델별 설정 > 전역 설정 > 기본값 32
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    else:
        model_cfg = config.get("models", {}).get(args.model, {})
        max_new_tokens = model_cfg.get(
            "max_new_tokens",
            config.get("decoding", {}).get("max_new_tokens", 32)
        )
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"batch_size: {args.batch_size} | num_workers: {args.num_workers}")

    patch_cfg = next((p for p in config["perturbations"] if p["name"] == "patch_shuffle"), {})
    df = run_inference(
        model=model,
        dataset=dataset,
        conditions=conditions,
        output_dir=config["output"]["dir"],
        lpf_sigma=lpf_sigma,
        hpf_sigma=hpf_sigma,
        patch_size=patch_cfg.get("patch_size", 16),
        seed=config.get("seed", 42),
        max_new_tokens=max_new_tokens,
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
