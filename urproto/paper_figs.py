"""Manuscript figures (multi-panel versions of the report figures).

    python -m urproto.paper_figs --runs <job out dirs> --out <dir> [--seed 42]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from . import stats as ST
from .report import (load, group_aggregate, unc_scores, PROPOSED, BASELINES, COLORS, DS_LABEL, PRIMARY_LEVEL)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
PROTO_LABEL = {"source": "source-domain support", "target": "target-domain support"}
PANELS = [("cxr_test", "source"), ("chexpert", "source"), ("chexpert", "target")]


def _sel(P, ds, proto, seed, method=PROPOSED):
    d = P[(P.method == method) & (P.dataset == ds) & (P.protocol.isin([proto, "n/a"])) & (P.seed == seed)]
    return group_aggregate(d) if PRIMARY_LEVEL[ds] == "group" and len(d) else d


def _title(ds, proto):
    return DS_LABEL[ds] if ds.startswith("cxr") else f"{DS_LABEL[ds]}\n({PROTO_LABEL[proto]})"


def fig_training(logs, out, seed):
    lg = logs[("full", seed)]
    enc = lg[lg.phase == "encoder"] if "phase" in lg else lg
    gate = lg[lg.phase == "gate"] if "phase" in lg else lg.iloc[0:0]
    split = enc.epoch.max() + 0.5

    def shade(ax):
        if len(gate):
            ax.axvspan(split, lg.epoch.max() + 0.5, color="#E8E1F2", lw=0, zorder=0)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    ax = axes[0]; shade(ax)
    ax.plot(lg.epoch, lg.train_loss, color="#4C2A85", label="training loss")
    ax.plot(lg.epoch, lg.val_nll, color="#D1495B", label="CXR-Val NLL")
    ax.set_ylabel("Loss"); ax.legend(fontsize=8)
    ax = axes[1]; shade(ax)
    ax.plot(enc.epoch, enc.train_f1, color="#4C2A85", label="training episodes")
    ax.plot(lg.epoch, lg.val_f1, color="#D1495B", label="CXR-Val episodes")
    ax.set_ylabel("Macro-F1"); ax.legend(fontsize=8, loc="lower right")
    ax = axes[2]; shade(ax)
    ax.plot(enc.epoch, enc.train_unc, color="#2E86AB", label="mean vacuity (training)")
    ax.plot(enc.epoch, enc.val_unc, color="#2E86AB", ls="--", label="mean vacuity (CXR-Val)")
    ax.set_ylabel("Mean vacuity $u$")
    if len(gate):
        ax2 = ax.twinx()
        ax2.plot(gate.epoch, gate.train_beta, color="#7A5195", marker="o", ms=3, label=r"FusionGate $\beta$")
        ax2.set_ylabel(r"FusionGate $\beta$"); ax2.set_ylim(0, 1); ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
        ax.set_xlim(0.5, lg.epoch.max() + 0.5)
    if len(gate):
        axes[0].text(split + 0.3, axes[0].get_ylim()[1], "FusionGate\nphase", va="top", fontsize=7.5, color="#4C2A85")
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_training.png")); plt.close(fig)


def fig_confusion(P, out, seed):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, (ds, proto) in zip(axes, PANELS):
        d = _sel(P, ds, proto, seed)
        cm = confusion_matrix(d.label, (d.p_cal >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Purples")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"]); ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        unit = "patients" if PRIMARY_LEVEL[ds] == "group" and ds == "chexpert" else "images"
        ax.set_title(f"{_title(ds, proto)}\n$n$ = {cm.sum():,} {unit}", fontsize=8.5)
        ax.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_confusion.png")); plt.close(fig)


def fig_roc(P, out, seed):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, proto in zip(axes, ("source", "target")):
        for meth in [PROPOSED] + BASELINES:
            d = _sel(P, "chexpert", proto, seed, meth)
            if d.empty:
                continue
            fpr, tpr, _ = roc_curve(d.label, d.p_cal)
            ax.plot(fpr, tpr, color=COLORS[meth], lw=2 if meth == PROPOSED else 1.1,
                    label=f"{meth} ({roc_auc_score(d.label, d.p_cal):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"CheXpert ({PROTO_LABEL[proto]})", fontsize=9); ax.legend(loc="lower right", fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_roc.png")); plt.close(fig)


def fig_reliability(P, out, seed, n_bins=10):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, (ds, proto) in zip(axes, PANELS):
        d = _sel(P, ds, proto, seed)
        for col, lab, c in (("p_raw", "before temperature scaling", "#999999"),
                            ("p_cal", "after temperature scaling", "#4C2A85")):
            p, y = d[col].values, d.label.values
            edges = np.linspace(0, 1, n_bins + 1)
            idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
            keep = [i for i in range(n_bins) if (idx == i).sum() > 20]
            ax.plot([p[idx == i].mean() for i in keep], [y[idx == i].mean() for i in keep], "o-", color=c, ms=3.5, label=lab)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted pneumonia probability"); ax.set_ylabel("Observed pneumonia frequency")
        ax.set_title(_title(ds, proto), fontsize=8.5); ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_reliability.png")); plt.close(fig)


def fig_kshot(SW, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), sharex=True)
    for col, proto in enumerate(("source", "target")):
        d = SW[(SW.sweep == "kshot") & (SW.dataset == "chexpert") & (SW.protocol == proto) & (SW.level == "group")]
        d = d.assign(K=d.setting.str.replace("K=", "").astype(int))
        for meth in [PROPOSED] + [m for m in BASELINES if m in set(d.method)] + ["Proto+Memory", "Proto+EDL"]:
            g = d[d.method == meth]
            if g.empty:
                continue
            a = g.groupby("K").agg(f1=("macro_f1", "mean"), f1s=("macro_f1", "std"), auc=("auroc", "mean"), aucs=("auroc", "std"))
            for row, m in enumerate(("f1", "auc")):
                axes[row, col].errorbar(a.index, a[m], yerr=a[m + "s"], marker="o", ms=3.2, capsize=2,
                                        color=COLORS.get(meth, "#333333"), lw=2 if meth == PROPOSED else 1.0,
                                        ls="--" if meth in ("Proto+Memory", "Proto+EDL") else "-", label=meth)
        axes[0, col].set_title(f"CheXpert ({PROTO_LABEL[proto]})", fontsize=9)
        axes[1, col].set_xlabel("Support images per class ($K$)")
    axes[0, 0].set_ylabel("Macro-F1"); axes[1, 0].set_ylabel("AUROC")
    for ax in axes.ravel():
        ax.set_xticks([1, 3, 5, 10]); ax.grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_kshot.png")); plt.close(fig)


def fig_risk_coverage(P, out, primary):
    cov = np.linspace(0.05, 1.0, 96)
    styles = {primary: ("-", "#4C2A85", 2.0), "vacuity": ("-.", "#2E86AB", 1.2), "max-prob": ("--", "#555555", 1.2),
              "random": (":", "#999999", 1.2)}
    names = {primary: "vacuity + dissonance (primary)" if primary == "vacuity+dissonance" else primary,
             "vacuity": "vacuity", "max-prob": "1 - max probability", "random": "random deferral"}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True)
    for col, proto in enumerate(("source", "target")):
        for score, (ls, c, lw) in styles.items():
            R, REC = [], []
            for seed in sorted(P.seed.unique()):
                d = _sel(P, "chexpert", proto, seed)
                if d.empty:
                    continue
                sc = unc_scores(d)
                s = sc.get(score, sc["max-prob"])
                rc = ST.risk_coverage(d.label.values, d.p_cal.values, s, coverages=cov,
                                      random_state=(seed if score == "random" else None))
                R.append([r["risk"] for r in rc]); REC.append([r["recall_retained"] for r in rc])
            for row, cur in enumerate((R, REC)):
                c_ = np.asarray(cur, float); mu = np.nanmean(c_, 0); sd = np.nanstd(c_, 0)
                ax = axes[row, col]
                ax.plot(cov, mu, ls, color=c, lw=lw, label=names[score])
                lo, hi = mu - sd, mu + sd
                if row == 1:
                    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
                ax.fill_between(cov, lo, hi, color=c, alpha=0.10, lw=0)
        axes[0, col].set_title(f"CheXpert ({PROTO_LABEL[proto]})", fontsize=9)
        axes[1, col].set_xlabel("Coverage (fraction of patients decided automatically)")
        axes[1, col].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("Selective risk (error rate)"); axes[1, 0].set_ylabel("Retained pneumonia recall")
    for ax in axes.ravel():
        ax.set_xlim(0.05, 1.0); ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_risk_coverage.png")); plt.close(fig)


def fig_unc_hist(P, out, seed, primary):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, (ds, proto) in zip(axes, PANELS):
        d = _sel(P, ds, proto, seed)
        s = unc_scores(d).get(primary)
        err = (d.p_cal >= 0.5).astype(int).values != d.label.values
        bins = np.linspace(np.nanmin(s), np.nanmax(s), 40)
        ax.hist(s[~err], bins=bins, alpha=0.6, density=True, color="#4C2A85", label="correct")
        ax.hist(s[err], bins=bins, alpha=0.6, density=True, color="#D1495B", label="error")
        ax.set_xlabel("Vacuity + dissonance" if primary == "vacuity+dissonance" else primary)
        ax.set_ylabel("Density"); ax.set_title(_title(ds, proto), fontsize=8.5)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "paper_unc_hist.png")); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--primary", default="vacuity+dissonance")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    M_job, SW, R, P, logs = load(a.runs)
    fig_training(logs, a.out, a.seed)
    fig_confusion(P, a.out, a.seed)
    fig_roc(P, a.out, a.seed)
    fig_reliability(P, a.out, a.seed)
    fig_kshot(SW, a.out)
    fig_risk_coverage(P, a.out, a.primary)
    fig_unc_hist(P, a.out, a.seed, a.primary)
    print("paper figures written to", a.out)


if __name__ == "__main__":
    main()
