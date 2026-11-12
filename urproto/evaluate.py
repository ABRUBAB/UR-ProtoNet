"""Evaluation: support-set ensembles, temperature calibration, metrics, memory-composition
and retrieval-quality analyses, K-shot sweeps."""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

from .models import MemoryBank, class_prototypes, normalized_entropy, sq_dist


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def safe_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def top_label_ece(y, p, n_bins: int = 15) -> float:
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < n_bins - 1 else (conf <= edges[i + 1]))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, p, u=None, thr: float = 0.5) -> Dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    pr, rc, f1c, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "accuracy": float((pred == y).mean()), "auroc": safe_auc(y, p),
           "brier": float(np.mean((p - y) ** 2)), "ece": top_label_ece(y, p),
           "prec_normal": float(pr[0]), "rec_normal": float(rc[0]), "f1_normal": float(f1c[0]),
           "prec_pneumonia": float(pr[1]), "rec_pneumonia": float(rc[1]), "f1_pneumonia": float(f1c[1])}
    if u is not None:
        out["unc_err_auroc"] = safe_auc((pred != y).astype(int), np.asarray(u, dtype=float))
        out["mean_unc"] = float(np.mean(u))
    return out


def group_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Patient/study-level aggregation: label = max, probabilities and uncertainty = mean."""
    agg = {"label": "max", "p_cal": "mean", "p_raw": "mean", "u": "mean", "u_diss": "mean", "u_total": "mean"}
    return df.groupby("group", as_index=False).agg({k: v for k, v in agg.items() if k in df.columns})


# ----------------------------------------------------------------------------------------------
# Support-set ensembles
# ----------------------------------------------------------------------------------------------
class SimpleShotModel:
    """Nearest-centroid on CL2N-normalised Stage I features (no meta-training)."""
    variant = "simpleshot"
    uses_memory = False
    uses_edl = False

    def predict(self, q_z, s_z, s_y, memory=None, force_beta=None):
        protos = class_prototypes(s_z, s_y)
        logits = -sq_dist(q_z, protos)
        probs = logits.softmax(-1)
        dev = q_z.device
        return {"logits": logits, "probs": probs, "uncertainty": normalized_entropy(probs),
                "beta": torch.ones(2, device=dev), "mem_sim": torch.zeros(2, device=dev)}


def sample_support_indices(pool_y: np.ndarray, K: int, rng: np.random.RandomState):
    i0, i1 = np.where(pool_y == 0)[0], np.where(pool_y == 1)[0]
    s = np.concatenate([rng.choice(i0, K, replace=len(i0) < K), rng.choice(i1, K, replace=len(i1) < K)])
    return s, np.array([0] * K + [1] * K)


@torch.no_grad()
def proto_ensemble(model, q_z, pool_z, pool_y, M: int, K: int, seed: int, memory=None, force_beta=None,
                   chunk: int = 8192):
    rng = np.random.RandomState(seed)
    dev = q_z.device
    L, P, U, B, MS = [], [], [], [], []
    for _ in range(M):
        s, sy = sample_support_indices(pool_y, K, rng)
        s_z = pool_z[torch.as_tensor(s, device=dev)]
        sy_t = torch.as_tensor(sy, device=dev)
        lo, pr, un = [], [], []
        info = None
        for i in range(0, q_z.size(0), chunk):
            out = model.predict(q_z[i:i + chunk], s_z, sy_t, memory, force_beta)
            lo.append(out["logits"].float()); pr.append(out["probs"][:, 1].float()); un.append(out["uncertainty"].float())
            info = out
        L.append(torch.cat(lo)); P.append(torch.cat(pr)); U.append(torch.cat(un))
        B.append(info["beta"].float()); MS.append(info["mem_sim"].float())
    return {"logits": torch.stack(L).cpu(), "p_raw": torch.stack(P).mean(0).cpu().numpy(),
            "u": torch.stack(U).mean(0).cpu().numpy(), "beta": torch.stack(B).mean(0).cpu().numpy(),
            "mem_sim": torch.stack(MS).mean(0).cpu().numpy()}


def calibrated_from_logits(logits: torch.Tensor, T: float) -> np.ndarray:
    """Eq. 20-21: softmax(log e / T) per support set, then averaged over the M sets."""
    return (logits / T).softmax(-1)[..., 1].mean(0).numpy()


def fit_temperature(nll_fn) -> float:
    r = minimize_scalar(lambda lt: nll_fn(math.exp(lt)), bounds=(-4.0, 3.0), method="bounded")
    return float(math.exp(r.x))


def fit_T_ensemble(logits_val: torch.Tensor, y_val: np.ndarray) -> float:
    y = np.asarray(y_val).astype(int)

    def nll(T):
        p = np.clip(calibrated_from_logits(logits_val, T), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return fit_temperature(nll)


def fit_T_binary(logit_val: np.ndarray, y_val: np.ndarray) -> float:
    y = np.asarray(y_val).astype(int)

    def nll(T):
        p = np.clip(1 / (1 + np.exp(-logit_val / T)), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return fit_temperature(nll)


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    f = [f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def binary_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / math.log(2)


# ----------------------------------------------------------------------------------------------
# Memory compositions (reviewer 1, Q1)
# ----------------------------------------------------------------------------------------------
def memory_variants(base: MemoryBank, target_pool_z: Optional[torch.Tensor], target_pool_y: Optional[np.ndarray],
                    seed: int) -> Dict[str, Optional[MemoryBank]]:
    rng = np.random.RandomState(seed + 7)
    m0, m1 = base.mem
    dev = m0.device

    def sub(m, frac):
        n = max(5, int(round(frac * m.size(0))))
        return m[torch.as_tensor(rng.choice(m.size(0), n, replace=False), device=dev)]

    out = {"nih_full": base, "no_memory": None}
    for frac in (0.5, 0.25, 0.1):
        out[f"nih_{int(frac * 100)}pct"] = MemoryBank(sub(m0, frac), sub(m1, frac))
    out["nih_balanced"] = MemoryBank(m0[torch.as_tensor(rng.choice(m0.size(0), m1.size(0), replace=False), device=dev)], m1)
    allm = torch.cat([m0, m1])
    perm = torch.as_tensor(rng.permutation(allm.size(0)), device=dev)
    out["nih_shuffled_labels"] = MemoryBank(allm[perm[:m0.size(0)]], allm[perm[m0.size(0):]])
    if target_pool_z is not None:
        ty = torch.as_tensor(target_pool_y, device=dev)
        t0, t1 = target_pool_z[ty == 0], target_pool_z[ty == 1]
        out["nih_plus_target"] = MemoryBank(torch.cat([m0, t0]), torch.cat([m1, t1]))
        out["target_only"] = MemoryBank(t0, t1)
    return out


@torch.no_grad()
def retrieval_quality(memory: MemoryBank, q_z: torch.Tensor, q_y: np.ndarray, k: int = 5) -> Dict:
    """Unconditional k-NN over the whole memory: how well do retrieved NIH neighbours agree with the
    query label, and how close (cosine) is the query distribution to the memory?"""
    m0, m1 = memory.mem
    allm = torch.cat([m0, m1]); lab = torch.cat([torch.zeros(m0.size(0)), torch.ones(m1.size(0))]).to(q_z.device)
    top_sim, frac_pos = [], []
    for i in range(0, q_z.size(0), 4096):
        s = q_z[i:i + 4096] @ allm.t()
        v, idx = s.topk(k, dim=-1)
        top_sim.append(v[:, 0]); frac_pos.append(lab[idx].mean(-1))
    top_sim = torch.cat(top_sim).cpu().numpy(); frac_pos = torch.cat(frac_pos).cpu().numpy()
    # class-balanced vote: rescale by memory prior so the 8000:1353 imbalance does not dominate
    prior = m1.size(0) / allm.size(0)
    pred = (frac_pos > prior).astype(int)
    y = np.asarray(q_y).astype(int)
    return {"knn_auroc": safe_auc(y, frac_pos), "knn_bal_acc": float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())),
            "top1_cos_mean": float(top_sim.mean()), "top1_cos_p10": float(np.percentile(top_sim, 10)),
            "top1_cos_p90": float(np.percentile(top_sim, 90))}
