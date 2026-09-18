"""Phase 5: Evaluation on In-Domain (CXR-Val) & Out-of-Domain (Stanford CheXpert) Test Sets.

Computes Macro-F1, AUROC, Brier score, Expected Calibration Error (ECE), and Uncertainty-to-Error AUROC
across in-domain validation and external domain shift.
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
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, roc_auc_score, brier_score_loss,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)

from config import cfg
from model import FeatureEncoder, MemoryBank, FusionGate, EvidentialHead
from data import (
    CXRDataset, load_cxr_df, load_chexpert_df, sample_support_df,
    build_support_tensors, DEVICE
)


def safe_l2_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cos_sim = torch.matmul(F.normalize(a, dim=-1), F.normalize(b, dim=-1).T)
    return (2.0 - 2.0 * cos_sim).clamp(min=0.0).sqrt()


def dirichlet_vacuity(alpha: torch.Tensor) -> torch.Tensor:
    K = alpha.size(-1)
    S = alpha.sum(dim=-1)
    return float(K) / S.clamp(min=1e-8)


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


def full_metric_summary(df: pd.DataFrame, threshold: float, prefix: str = "") -> Dict:
    y_true = df["true_label"].values
    y_prob = df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)
    y_unc = df["uncertainty"].values

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    auc = safe_auroc(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)

    err = (y_pred != y_true).astype(int)
    unc_err_auroc = safe_auroc(err, y_unc)

    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"{prefix}Macro-F1={macro_f1:.4f} | Micro-F1={micro_f1:.4f} | AUC={auc:.4f} | Brier={brier:.4f} | ECE={ece:.4f}")
    print(f"{prefix}Uncertainty->Error AUROC: {unc_err_auroc:.4f}")
    for cls_idx, (p, r, f1v, count) in enumerate(zip(prec, rec, f1c, sup)):
        name = "Normal" if cls_idx == 0 else "Pneumonia"
        print(f"  {name:>10s} | Prec={p:.3f} Recall={r:.3f} F1={f1v:.3f} N={count:,}")

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "auc": auc,
        "brier": brier,
        "ece": ece,
        "unc_err_auroc": unc_err_auroc,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_unc": y_unc,
    }


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


def find_default_checkpoint() -> str:
    candidates = [
        "checkpoints/ur_protonet_best.pt",
        "results/checkpoints/ur_protonet_best.pt",
        os.path.join("..", "results", "checkpoints", "ur_protonet_best.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="UR-ProtoNet Comprehensive Diagnostic Evaluation")
    parser.add_argument("--checkpoint", type=str, default=find_default_checkpoint())
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fresh", action="store_true", help="Run model forward pass on raw images")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5: UR-ProtoNet Evaluation (In-Domain CXR-Val & Stanford CheXpert)")
    print("=" * 70)

    val_csv = find_results_file("cxr_val_predictions.csv")
    chex_csv = find_results_file("chexpert_study_predictions.csv") or find_results_file("chexpert_predictions.csv")

    if val_csv and not args.fresh:
        print(f"Loading precomputed validation predictions from: {val_csv}")
        val_df = pd.read_csv(val_csv)
        val_threshold, val_f1 = choose_threshold(val_df["true_label"].values, val_df["pred_prob"].values)
        print(f"Optimal Decision Threshold (Max In-Domain F1): {val_threshold:.4f} (F1 = {val_f1:.4f})")
        print("\n--- [1] In-Domain CXR Validation Performance ---")
        val_stats = full_metric_summary(val_df, threshold=val_threshold, prefix="CXR-Val | ")

        if chex_csv:
            print(f"\nLoading precomputed CheXpert predictions from: {chex_csv}")
            chex_df = pd.read_csv(chex_csv)
            print("\n--- [2] Stanford CheXpert External Domain Shift Performance ---")
            chex_stats = full_metric_summary(chex_df, threshold=val_threshold, prefix="CheXpert | ")

        out_dir = "results" if os.path.isdir("results") else "."
        val_out = {"val_threshold": val_threshold, "best_val_f1": val_f1, "temperature": 1.0}
        with open(os.path.join(out_dir, "val_threshold.json"), "w") as fp:
            json.dump(val_out, fp, indent=2)
        print(f"\nSaved threshold config -> {os.path.join(out_dir, 'val_threshold.json')}")
    else:
        print("Running live evaluation...")
        print("Note: Provide raw images and checkpoints to execute live batch inference.")


if __name__ == "__main__":
    main()
