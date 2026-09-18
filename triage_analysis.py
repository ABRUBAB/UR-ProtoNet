"""Phase 6: Post-Hoc Calibration, Clinical Triage Analysis, and Abstention Sweeps.

Calibrates model confidences via Platt/Temperature Scaling, models domain shift degradation,
and computes selective classification curves (deferral/rejection) based on Dirichlet epistemic vacuity.
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, precision_recall_fscore_support

from config import cfg


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            acc_bin = np.mean(y_true[mask] == (y_prob[mask] >= 0.5))
            conf_bin = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(acc_bin - conf_bin)
    return float(ece)


def apply_temp(probs_np: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs_np, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    logit_cal = logit / max(T, 1e-4)
    return 1.0 / (1.0 + np.exp(-logit_cal))


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_f1 = -1.0
    best_thresh = 0.5
    for th in np.linspace(0.05, 0.95, 181):
        preds = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
    return float(best_thresh), float(best_f1)


def find_results_file(filename: str) -> Optional[str]:
    candidates = [
        os.path.join("results", "results", filename),
        os.path.join("results", filename),
        os.path.join("..", "results", "results", filename),
        os.path.join("..", "results", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def catalogue_errors(df: pd.DataFrame, threshold: float, name: str, T: float) -> pd.DataFrame:
    res = df.copy()
    res["pred_prob_orig"] = res["pred_prob"]
    res["pred_prob"] = apply_temp(res["pred_prob"].values, T)
    res["pred_label"] = (res["pred_prob"] >= threshold).astype(int)
    res["is_error"] = (res["pred_label"] != res["true_label"]).astype(int)
    res["confidence"] = 1.0 - res["uncertainty"]
    res["error_type"] = "TN"
    res.loc[(res["is_error"] == 0) & (res["pred_label"] == 1), "error_type"] = "TP"
    res.loc[(res["is_error"] == 1) & (res["pred_label"] == 1), "error_type"] = "FP"
    res.loc[(res["is_error"] == 1) & (res["pred_label"] == 0), "error_type"] = "FN"
    n_err = res["is_error"].sum()
    print(f"  [{name}] N={len(res):,} | Total Errors={n_err:,} (FP={(res['error_type']=='FP').sum():,}, FN={(res['error_type']=='FN').sum():,})")
    return res


def compute_abstention_curve(df: pd.DataFrame, n_points: int = 20) -> pd.DataFrame:
    coverages = np.linspace(0.1, 1.0, n_points)
    rows = []
    # Rank by certainty (ascending uncertainty = descending certainty)
    df_sorted = df.sort_values("uncertainty", ascending=True).reset_index(drop=True)
    n = len(df_sorted)

    for cov in coverages:
        k = max(int(np.ceil(cov * n)), 2)
        sub = df_sorted.iloc[:k]
        y_t = sub["true_label"].values
        y_p = (sub["pred_prob"] >= 0.5).astype(int)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        pos_mask = (y_t == 1)
        rec_pneu = np.mean(y_p[pos_mask] == 1) if pos_mask.sum() > 0 else 0.0
        rows.append({
            "coverage": cov,
            "macro_f1": f1,
            "recall_pneumonia": rec_pneu,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Clinical Triage & Uncertainty Sweep")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 6: Uncertainty Calibration & Clinical Triage Analysis")
    print("=" * 70)

    val_csv = find_results_file("cxr_val_predictions.csv")
    chex_csv = find_results_file("chexpert_predictions.csv") or find_results_file("chexpert_study_predictions.csv")

    if not val_csv:
        print("Notice: No predictions CSV found. Run evaluate.py first.")
        return

    val_df = pd.read_csv(val_csv)
    y_true_val = val_df["true_label"].values
    y_prob_val = val_df["pred_prob"].values.clip(1e-6, 1.0 - 1e-6)

    # Temperature Scaling optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits_1d = torch.tensor(np.log(y_prob_val / (1.0 - y_prob_val)), dtype=torch.float32)
    logits_2cl = torch.stack([-logits_1d, logits_1d], dim=1).to(device)
    labels_ts = torch.tensor(y_true_val, dtype=torch.long, device=device)

    log_T = nn.Parameter(torch.zeros(1, device=device))
    opt_ts = torch.optim.LBFGS([log_T], lr=0.01, max_iter=500)

    def ts_closure():
        opt_ts.zero_grad()
        T = torch.exp(log_T).clamp(min=0.5, max=5.0)
        loss = F.cross_entropy(logits_2cl / T, labels_ts)
        loss.backward()
        return loss

    opt_ts.step(ts_closure)
    T_opt = float(torch.exp(log_T).clamp(min=0.5, max=5.0).item())

    ece_before = compute_ece(y_true_val, y_prob_val)
    y_prob_cal = apply_temp(y_prob_val, T_opt)
    ece_after = compute_ece(y_true_val, y_prob_cal)
    print(f"Optimal Calibration Temperature T = {T_opt:.4f}")
    print(f"In-Domain ECE: {ece_before:.4f} -> {ece_after:.4f}")

    # Catalogue errors
    val_cal_df = catalogue_errors(val_df, 0.50, "CXR_VAL", T_opt)
    val_cal_path = os.path.join(args.output_dir, "phase6_cxr_val_calibrated.csv")
    val_cal_df.to_csv(val_cal_path, index=False)

    chex_cal_df = None
    if chex_csv:
        chex_raw_df = pd.read_csv(chex_csv)
        chex_cal_df = catalogue_errors(chex_raw_df, 0.50, "CHEXPERT", T_opt)
        chex_cal_path = os.path.join(args.output_dir, "phase6_chexpert_calibrated.csv")
        chex_cal_df.to_csv(chex_cal_path, index=False)

        # Domain shift summary
        shift_summary = [
            {
                "dataset": "CXR_VAL", "n": len(val_cal_df),
                "errors": int(val_cal_df["is_error"].sum()),
                "error_rate": float(val_cal_df["is_error"].mean()),
                "mean_unc_correct": float(val_cal_df[val_cal_df["is_error"] == 0]["uncertainty"].mean()),
                "mean_unc_error": float(val_cal_df[val_cal_df["is_error"] == 1]["uncertainty"].mean()),
            },
            {
                "dataset": "CHEXPERT", "n": len(chex_cal_df),
                "errors": int(chex_cal_df["is_error"].sum()),
                "error_rate": float(chex_cal_df["is_error"].mean()),
                "mean_unc_correct": float(chex_cal_df[chex_cal_df["is_error"] == 0]["uncertainty"].mean()),
                "mean_unc_error": float(chex_cal_df[chex_cal_df["is_error"] == 1]["uncertainty"].mean()),
            }
        ]
        shift_df = pd.DataFrame(shift_summary)
        shift_df.to_csv(os.path.join(args.output_dir, "phase6_shift_summary.csv"), index=False)
        print("\nDomain Shift Summary:")
        print(shift_df.to_string(index=False))

        # Abstention sweeps
        abst_chex = compute_abstention_curve(chex_cal_df)
        abst_chex.to_csv(os.path.join(args.output_dir, "phase6_abstention_chexpert.csv"), index=False)
        abst_val = compute_abstention_curve(val_cal_df)
        abst_val.to_csv(os.path.join(args.output_dir, "phase6_abstention_cxr_val.csv"), index=False)
        print("\nAbstention Sweeps generated successfully.")


if __name__ == "__main__":
    main()