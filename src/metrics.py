"""
Diagnostic metrics for Medical VQA perturbation experiment.

VRS  - Visual Reliance Score
IS   - Image Sensitivity
NSR  - Noise Sensitivity Rate
HPF_Drop - High-Pass Filter accuracy drop
Answer Consistency - same answer as original condition
"""

import pandas as pd
import numpy as np
from typing import Dict


def accuracy(df: pd.DataFrame) -> float:
    """Compute accuracy from a dataframe with 'correct' column."""
    if len(df) == 0:
        return 0.0
    return df["correct"].mean()


def compute_metrics_for_model(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute all diagnostic metrics for a single model.
    df must contain columns: condition, correct, pred_answer, image_id
    """
    acc = {}
    for cond in ("original", "black", "lpf", "hpf", "patch_shuffle"):
        subset = df[df["condition"] == cond]
        acc[cond] = accuracy(subset) if len(subset) > 0 else np.nan

    # VRS: Visual Reliance Score = Acc(Original) - Acc(Black)
    vrs = acc["original"] - acc["black"]

    # IS: Image Sensitivity = 1 - Acc(PatchShuffle) / Acc(Original)
    is_score = 1 - (acc["patch_shuffle"] / acc["original"]) if acc["original"] > 0 else np.nan

    # NSR: Noise Sensitivity Rate = 1 - Acc(LPF) / Acc(Original)
    nsr = 1 - (acc["lpf"] / acc["original"]) if acc["original"] > 0 else np.nan

    # HPF Drop = Acc(Original) - Acc(HPF)
    hpf_drop = acc["original"] - acc["hpf"]

    return {
        "acc_original": acc["original"],
        "acc_black": acc["black"],
        "acc_lpf": acc["lpf"],
        "acc_hpf": acc["hpf"],
        "acc_patch_shuffle": acc["patch_shuffle"],
        "VRS": vrs,
        "IS": is_score,
        "NSR": nsr,
        "HPF_Drop": hpf_drop,
    }


def answer_consistency(df: pd.DataFrame, condition: str) -> float:
    """
    Fraction of samples where model gives the same answer as in original condition.
    High consistency on 'black' suggests text-prior dependence.
    """
    orig = df[df["condition"] == "original"][["image_id", "pred_answer"]].rename(
        columns={"pred_answer": "orig_pred"}
    )
    cond = df[df["condition"] == condition][["image_id", "pred_answer"]].rename(
        columns={"pred_answer": "cond_pred"}
    )
    merged = orig.merge(cond, on="image_id", how="inner")
    if len(merged) == 0:
        return np.nan
    return (merged["orig_pred"] == merged["cond_pred"]).mean()


def answer_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return answer distribution (A/B/C/D/PARSE_FAIL) per condition."""
    return (
        df.groupby(["condition", "pred_answer"])
        .size()
        .unstack(fill_value=0)
    )


def transition_matrix(df: pd.DataFrame, from_cond: str, to_cond: str) -> pd.DataFrame:
    """
    Answer transition matrix from one condition to another.
    Rows = answer in from_cond, Columns = answer in to_cond.
    """
    df_from = df[df["condition"] == from_cond][["image_id", "pred_answer"]].rename(
        columns={"pred_answer": "from_answer"}
    )
    df_to = df[df["condition"] == to_cond][["image_id", "pred_answer"]].rename(
        columns={"pred_answer": "to_answer"}
    )
    merged = df_from.merge(df_to, on="image_id", how="inner")
    return pd.crosstab(merged["from_answer"], merged["to_answer"])
