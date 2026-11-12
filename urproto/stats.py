"""Statistics used in the paper: seed-level paired tests with Holm correction, image-level paired
bootstrap, DeLong's test for correlated AUROCs, bootstrap CIs and risk-coverage analysis."""

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as st
from sklearn.metrics import f1_score


# ----------------------------------------------------------------------------------------------
# Seed-level paired comparisons
# ----------------------------------------------------------------------------------------------
def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_seed_test(a: Sequence[float], b: Sequence[float]) -> Dict:
    """a = proposed, b = baseline, paired by seed."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    t = st.ttest_rel(a, b) if n > 1 else None
    try:
        w = st.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact" if n <= 25 else "auto")
        wp = float(w.pvalue)
    except Exception:
        wp = float("nan")
    tcrit = st.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    return {"n_seeds": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(d.mean()),
            "delta_ci_lo": float(d.mean() - tcrit * sd / math.sqrt(n)) if n > 1 else float("nan"),
            "delta_ci_hi": float(d.mean() + tcrit * sd / math.sqrt(n)) if n > 1 else float("nan"),
            "t": float(t.statistic) if t is not None else float("nan"), "p_t": float(t.pvalue) if t is not None else float("nan"),
            "p_wilcoxon": wp, "cohens_dz": float(d.mean() / sd) if n > 1 and sd > 0 else float("nan"),
            "wins": int((d > 0).sum())}


# ----------------------------------------------------------------------------------------------
# DeLong (Sun & Xu, 2014 fast implementation)
# ----------------------------------------------------------------------------------------------
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_paired(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> Dict:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    y, s1, s2 = y[order], np.asarray(s1)[order], np.asarray(s2)[order]
    m = int(y.sum()); n = len(y) - m
    preds = np.vstack([s1, s2])
    k = 2
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(max(var, 1e-12))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta_auc": float(aucs[0] - aucs[1]),
            "z": float(z), "p": float(2 * st.norm.sf(abs(z)))}


# ----------------------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------------------
def macro_f1(y, p, thr=0.5):
    return f1_score(y, (np.asarray(p) >= thr).astype(int), average="macro", zero_division=0)


def bootstrap_ci(y, p, fn, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def paired_bootstrap(y, pa, pb, fn, thr_a=0.5, thr_b=0.5, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    obs = fn(y, pa, thr_a) - fn(y, pb, thr_b)
    d = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        d.append(fn(y[i], pa[i], thr_a) - fn(y[i], pb[i], thr_b))
    d = np.asarray(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(obs), "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p": float(max(p, 1.0 / n_boot))}


# ----------------------------------------------------------------------------------------------
# Selective prediction (risk-coverage)
# ----------------------------------------------------------------------------------------------
def risk_coverage(y, p, unc, thr=0.5, coverages=None, random_state=None):
    """Retain the `c` fraction of cases with the lowest uncertainty. Reports both recall
    denominators: among retained pneumonia cases, and among all pneumonia cases."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); unc = np.asarray(unc, float)
    if random_state is not None:
        unc = np.random.RandomState(random_state).rand(len(y))
    order = np.argsort(unc, kind="mergesort")
    pred = (p >= thr).astype(int)
    n, P = len(y), int(y.sum())
    if coverages is None:
        coverages = np.linspace(0.05, 1.0, 96)
    rows = []
    for c in coverages:
        k = max(1, int(math.ceil(c * n)))
        r = order[:k]
        yr, pr = y[r], pred[r]
        tp = int(((pr == 1) & (yr == 1)).sum())
        pos_r = int(yr.sum())
        deferred_pos = P - pos_r
        rows.append({"coverage": float(k / n), "risk": float((pr != yr).mean()), "accuracy": float((pr == yr).mean()),
                     "macro_f1": float(f1_score(yr, pr, average="macro", zero_division=0)) if len(np.unique(yr)) > 1 else float("nan"),
                     "recall_retained": tp / pos_r if pos_r else float("nan"),
                     "recall_all_autonomous": tp / P if P else float("nan"),
                     "pneumonia_deferred_frac": deferred_pos / P if P else float("nan"),
                     "recall_with_expert_review": (tp + deferred_pos) / P if P else float("nan"),
                     "spec_retained": float(((pr == 0) & (yr == 0)).sum() / max(1, (yr == 0).sum())),
                     "n_retained": k, "pos_retained": pos_r})
    return rows


def aurc(y, p, unc, thr=0.5):
    """Area under the risk-coverage curve (lower is better), computed over all n coverage points."""
    y = np.asarray(y).astype(int); pred = (np.asarray(p) >= thr).astype(int)
    order = np.argsort(np.asarray(unc, float), kind="mergesort")
    err = (pred[order] != y[order]).astype(float)
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risks.mean())
