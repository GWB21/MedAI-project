#!/usr/bin/env python3
"""
Analyze experiment results from all models.

Usage:
    python scripts/analyze_results.py --results results/*_all_*.csv
    python scripts/analyze_results.py --results results/llava_v15_all.csv results/huatuogpt_all.csv results/medvint_all.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import (
    answer_consistency,
    answer_distribution,
    compute_metrics_for_model,
    transition_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="CSV result files")
    parser.add_argument("--output_dir", default="./results/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and combine all CSVs
    dfs = []
    for path in args.results:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"Loaded {path}: {len(df)} rows")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows")

    models = combined["model"].unique()
    conditions = ["original", "black", "lpf", "hpf", "patch_shuffle"]

    # === 1. Main accuracy table ===
    print("\n" + "=" * 70)
    print("TABLE 1: Accuracy (Model x Condition)")
    print("=" * 70)
    acc_table = {}
    for model in models:
        model_df = combined[combined["model"] == model]
        metrics = compute_metrics_for_model(model_df)
        acc_table[model] = {c: metrics[f"acc_{c}"] for c in conditions}
    acc_df = pd.DataFrame(acc_table).T
    acc_df.columns = [c.capitalize() for c in conditions]
    print(acc_df.round(4).to_string())
    acc_df.round(4).to_csv(os.path.join(args.output_dir, "accuracy_table.csv"))

    # === 2. Diagnostic metrics ===
    print("\n" + "=" * 70)
    print("TABLE 2: Diagnostic Metrics")
    print("=" * 70)
    diag_table = {}
    for model in models:
        model_df = combined[combined["model"] == model]
        metrics = compute_metrics_for_model(model_df)
        diag_table[model] = {
            "VRS": metrics["VRS"],
            "IS": metrics["IS"],
            "NSR": metrics["NSR"],
            "HPF_Drop": metrics["HPF_Drop"],
        }
    diag_df = pd.DataFrame(diag_table).T
    print(diag_df.round(4).to_string())
    diag_df.round(4).to_csv(os.path.join(args.output_dir, "diagnostic_metrics.csv"))

    # === 3. Answer distribution ===
    print("\n" + "=" * 70)
    print("TABLE 3: Answer Distribution")
    print("=" * 70)
    for model in models:
        model_df = combined[combined["model"] == model]
        dist = answer_distribution(model_df)
        print(f"\n--- {model} ---")
        print(dist.to_string())

    # === 4. Answer consistency ===
    print("\n" + "=" * 70)
    print("TABLE 4: Answer Consistency (vs Original)")
    print("=" * 70)
    cons_table = {}
    for model in models:
        model_df = combined[combined["model"] == model]
        cons_table[model] = {}
        for cond in conditions:
            if cond == "original":
                continue
            cons_table[model][cond] = answer_consistency(model_df, cond)
    cons_df = pd.DataFrame(cons_table).T
    print(cons_df.round(4).to_string())
    cons_df.round(4).to_csv(os.path.join(args.output_dir, "answer_consistency.csv"))

    # === 5. Parse failure rate ===
    print("\n" + "=" * 70)
    print("TABLE 5: Parse Failure Rate (%)")
    print("=" * 70)
    pf_table = {}
    for model in models:
        model_df = combined[combined["model"] == model]
        pf_table[model] = {}
        for cond in conditions:
            subset = model_df[model_df["condition"] == cond]
            pf_table[model][cond] = (1 - subset["parse_success"].mean()) * 100
    pf_df = pd.DataFrame(pf_table).T
    print(pf_df.round(2).to_string())

    # === 6. Transition matrices ===
    print("\n" + "=" * 70)
    print("TABLE 6: Answer Transition (Original -> Black)")
    print("=" * 70)
    for model in models:
        model_df = combined[combined["model"] == model]
        tm = transition_matrix(model_df, "original", "black")
        print(f"\n--- {model} ---")
        print(tm.to_string())

    print(f"\nAnalysis saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
