"""Aggregate per-seed outputs into the paper's tables (CSV + LaTeX rows) and figures.

  python -m urproto.report --runs RUN_DIR [RUN_DIR ...] --out REPORT_DIR [--stage1 DIR] [--xai DIR]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from . import stats as ST
from .evaluate import compute_metrics, group_aggregate

PROPOSED = "UR-ProtoNet"
BASELINES = ["EfficientNet-Binary", "VanillaProtoNet", "MatchingNet", "SimpleShot"]
ABLATION = ["VanillaProtoNet", "Proto+Memory", "Proto+EDL", "UR-ProtoNet (joint)", "UR-ProtoNet", "UR-ProtoNet (transductive)"]
DS_LABEL = {"cxr_val": "CXR-Val", "cxr_test": "CXR-Test", "chexpert": "CheXpert", "siim": "SIIM-FISABIO-RSNA"}
PRIMARY_LEVEL = {"cxr_val": "image", "cxr_test": "image", "chexpert": "group", "siim": "group"}
COLORS = {"UR-ProtoNet": "#4C2A85", "EfficientNet-Binary": "#D1495B", "VanillaProtoNet": "#2E86AB",
          "MatchingNet": "#EDAE49", "SimpleShot": "#66A182", "Proto+Memory": "#7FB7BE", "Proto+EDL": "#BC8034"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "axes.titlesize": 11, "legend.frameon": False})


def _cat(paths, pattern, reader=pd.read_csv):
    fs = []
    for p in paths:
        fs += glob.glob(os.path.join(p, pattern))
    return pd.concat([reader(f) for f in sorted(fs)], ignore_index=True) if fs else pd.DataFrame()


def load(runs):
    M = _cat(runs, "metrics_seed*.csv")
    SW = _cat(runs, "sweeps_seed*.csv")
    R = _cat(runs, "retrieval_seed*.csv")
    P = _cat(runs, "preds_seed*.csv.gz", reader=lambda f: pd.read_csv(f, low_memory=False, dtype={"key": str, "group": str}))
    if len(P):
        P["protocol"] = P["protocol"].fillna("n/a")  # supervised baseline has no support protocol ("n/a" -> NaN)
    logs = {}
    for p in runs:
        for f in glob.glob(os.path.join(p, "trainlog_*_seed*.csv")):
            b = os.path.basename(f)[9:-4]
            meth, seed = b.rsplit("_seed", 1)
            logs[(meth, int(seed))] = pd.read_csv(f)
    for df in (M, SW, P):
        if len(df):
            df.drop_duplicates(inplace=True)
    return M, SW, R, P, logs


def unc_scores(d: pd.DataFrame) -> dict:
    """Candidate triage scores (higher = less reliable), computed from saved predictions.

    vacuity      u = 2/S (lack of evidence)
    evidential   1 - b_pred, with beliefs b_k = E[p_k] - u/2 (lack of evidence + conflicting evidence)
    max-prob     1 - max calibrated class probability
    """
    p = d["p_cal"].values
    pred = (p >= 0.5).astype(int)
    out = {"max-prob": 1 - np.maximum(p, 1 - p)}
    if "u" in d:
        u = d["u"].values
        out["vacuity"] = u
        if "p_raw" in d:
            b1 = d["p_raw"].values - u / 2
            b0 = 1 - d["p_raw"].values - u / 2
            out["evidential"] = 1 - np.where(pred == 1, b1, b0)
    if "u_diss" in d and d["u_diss"].notna().all():
        out["dissonance"] = d["u_diss"].values
        out["vacuity+dissonance"] = d["u_total"].values
    return out


def table_uncertainty_selection(P, out, method=PROPOSED):
    """Error-detection AUROC and AURC of each candidate score; the primary score is chosen on CXR-Val only."""
    rows = []
    for (seed, ds, proto), d in P[P.method == method].groupby(["seed", "dataset", "protocol"]):
        if PRIMARY_LEVEL.get(ds) == "group":
            d = group_aggregate(d)
        y, p = d.label.values, d.p_cal.values
        err = ((p >= 0.5).astype(int) != y).astype(int)
        for name, s in unc_scores(d).items():
            rows.append({"seed": seed, "dataset": ds, "protocol": proto, "score": name,
                         "err_auroc": np.nan if err.sum() in (0, len(err)) else roc_auc_score(err, s),
                         "aurc": ST.aurc(y, p, s)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, "vacuity"
    g = t.groupby(["dataset", "protocol", "score"]).agg(err_auroc=("err_auroc", "mean"), err_auroc_sd=("err_auroc", "std"),
                                                        aurc=("aurc", "mean"), aurc_sd=("aurc", "std")).reset_index()
    val = g[(g.dataset == "cxr_val") & (g.score != "max-prob")]
    primary = val.sort_values("aurc").score.iloc[0] if len(val) else "vacuity"
    g["primary_selected_on_val"] = g.score == primary
    g.to_csv(os.path.join(out, "table_uncertainty_scores.csv"), index=False)
    return g, primary


def build_metrics(P: pd.DataFrame, M_job: pd.DataFrame) -> pd.DataFrame:
    """Recompute all metrics from per-image predictions with one rule for every method:
    threshold 0.5 on the calibrated probability; error-detection AUROC from the calibrated confidence
    (plus vacuity / evidential scores for evidential models). EfficientNet-Binary is also reported with
    the CXR-Val-tuned threshold that was used during training (row 'EfficientNet-Binary (val thr)')."""
    rows = []
    thr_eff = {}
    if len(M_job):
        e = M_job[M_job.method == "EfficientNet-Binary"]
        thr_eff = e.groupby("seed").threshold.first().to_dict()
    datasets_protocols = {"cxr_val": ["source"], "cxr_test": ["source"], "cxr_test_all": ["source"],
                          "chexpert": ["source", "target"], "siim": ["source", "target"]}
    for (seed, meth, ds, proto), d in P.groupby(["seed", "method", "dataset", "protocol"]):
        protos = datasets_protocols.get(ds, ["source"]) if proto == "n/a" else [proto]
        variants = [(meth, 0.5)]
        if meth == "EfficientNet-Binary" and seed in thr_eff:
            variants.append(("EfficientNet-Binary (val thr)", float(thr_eff[seed])))
        for level in ("image", "group"):
            dd = group_aggregate(d) if level == "group" else d
            sc = unc_scores(dd)
            for name, thr in variants:
                m = compute_metrics(dd.label.values, dd.p_cal.values, sc["max-prob"], thr)
                y = dd.label.values; err = ((dd.p_cal.values >= thr).astype(int) != y).astype(int)
                for sname in ("vacuity", "evidential", "dissonance", "vacuity+dissonance"):
                    if sname in sc and meth.startswith(("UR-ProtoNet", "Proto+EDL")):
                        m[f"err_auroc_{sname}"] = np.nan if err.sum() in (0, len(err)) else roc_auc_score(err, sc[sname])
                        m[f"mean_{sname}"] = float(np.mean(sc[sname]))
                for pr in protos:
                    rows.append(dict(m, seed=seed, method=name, dataset=ds, protocol=pr, level=level, threshold=thr))
    out = pd.DataFrame(rows)
    # carry evidential-head parameters and gate statistics from the job metrics
    if len(M_job):
        keep = [c for c in M_job.columns if c.startswith(("head_", "beta_", "memsim_"))]
        if keep:
            extra = M_job[M_job.method == "UR-ProtoNet"][["seed", "dataset", "protocol", "level"] + keep]
            out = out.merge(extra, on=["seed", "dataset", "protocol", "level"], how="left") if len(extra) else out
    return out


def ms(x, d=3):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "--"
    return f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
def table_main(M, P, out):
    rows = []
    for ds in ["cxr_val", "cxr_test", "chexpert", "siim"]:
        for proto in ["source", "target"]:
            for level in (["image", "group"] if ds in ("chexpert", "siim", "cxr_test") else ["image"]):
                d = M[(M.method == PROPOSED) & (M.dataset == ds) & (M.protocol == proto) & (M.level == level)]
                if d.empty:
                    continue
                row = {"dataset": DS_LABEL[ds], "protocol": proto, "level": level, "seeds": d.seed.nunique(), "n": int(d.n.iloc[0])}
                for m in ["macro_f1", "auroc", "brier", "ece", "unc_err_auroc", "err_auroc_vacuity", "err_auroc_evidential",
                          "rec_pneumonia", "rec_normal", "head_gamma", "head_tau", "head_u_min", "beta_normal", "beta_pneumonia"]:
                    if m in d:
                        row[m + "_mean"] = d[m].mean(); row[m + "_sd"] = d[m].std(ddof=1)
                # bootstrap CI on pooled per-image predictions of the first seed
                s0 = sorted(d.seed.unique())[0]
                pp = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto) & (P.seed == s0)]
                if len(pp):
                    if level == "group":
                        pp = group_aggregate(pp)
                    row["f1_ci_seed0"] = ST.bootstrap_ci(pp.label, pp.p_cal, ST.macro_f1, 1000)
                    row["auc_ci_seed0"] = ST.bootstrap_ci(pp.label, pp.p_cal, roc_auc_score, 1000)
                rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "table_main.csv"), index=False)
    return t


def table_compare(M, out, methods, name, metrics=("macro_f1", "auroc", "brier", "ece", "unc_err_auroc", "rec_pneumonia"), ref=PROPOSED):
    rows, tests = [], []
    for ds in ["cxr_test", "chexpert", "siim"]:
        lvl = PRIMARY_LEVEL[ds]
        for proto in ["source", "target"]:
            sub = M[(M.dataset == ds) & (M.protocol == proto) & (M.level == lvl)]
            if sub.empty:
                continue
            refd = sub[sub.method == ref].set_index("seed")
            for meth in methods:
                d = sub[sub.method == meth]
                if d.empty:
                    continue
                r = {"dataset": DS_LABEL[ds], "protocol": proto, "method": meth, "seeds": d.seed.nunique()}
                for m in metrics:
                    r[m] = ms(d[m])
                    r[m + "_mean"] = d[m].mean()
                rows.append(r)
                if meth != ref and len(refd):
                    dd = d.set_index("seed")
                    common = sorted(set(refd.index) & set(dd.index))
                    for m in [x for x in ("macro_f1", "auroc", "brier", "ece") if x in metrics]:
                        res = ST.paired_seed_test(refd.loc[common, m].values, dd.loc[common, m].values)
                        res.update({"dataset": DS_LABEL[ds], "protocol": proto, "baseline": meth, "metric": m})
                        tests.append(res)
    T = pd.DataFrame(rows)
    S = pd.DataFrame(tests)
    if len(S):
        S["p_t_holm"] = np.nan; S["p_w_holm"] = np.nan
        for _, g in S.groupby(["dataset", "protocol", "metric"]):
            S.loc[g.index, "p_t_holm"] = ST.holm(g.p_t.values)
            S.loc[g.index, "p_w_holm"] = ST.holm(g.p_wilcoxon.fillna(1).values)
    T.to_csv(os.path.join(out, f"table_{name}.csv"), index=False)
    S.to_csv(os.path.join(out, f"table_{name}_tests.csv"), index=False)
    return T, S


def table_image_level_tests(P, out, ds="chexpert", proto="source"):
    """Per seed: DeLong (AUROC) and paired bootstrap (macro-F1) of UR-ProtoNet vs each baseline."""
    rows = []
    for seed in sorted(P.seed.unique()):
        ref = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto) & (P.seed == seed)]
        if ref.empty:
            continue
        ref = group_aggregate(ref).set_index("group")
        for b in BASELINES:
            bb = P[(P.method == b) & (P.dataset == ds) & (P.seed == seed) & (P.protocol.isin([proto, "n/a"]))]
            if bb.empty:
                continue
            bb = group_aggregate(bb).set_index("group").loc[ref.index]
            dl = ST.delong_paired(ref.label.values, ref.p_cal.values, bb.p_cal.values)
            rows.append({"seed": seed, "baseline": b, **{f"delong_{k}": v for k, v in dl.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, f"table_delong_{ds}_{proto}.csv"), index=False)
    return t


def table_sweep(SW, out, sweep, name):
    d = SW[SW.sweep == sweep]
    if d.empty:
        return pd.DataFrame()
    keys = ["method", "dataset", "protocol", "level", "setting"]
    agg = d.groupby(keys).agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                              auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                              unc_err_mean=("unc_err_auroc", "mean"), ece_mean=("ece", "mean"),
                              seeds=("seed", "nunique")).reset_index()
    if "beta_mean" in d.columns and sweep == "memory":
        b = d.groupby(keys)["beta_mean"].mean().reset_index()
        agg = agg.merge(b, on=keys, how="left")
    agg.to_csv(os.path.join(out, f"table_{name}.csv"), index=False)
    return agg


def table_triage(P, out, ds="chexpert", proto="source", level="image", coverages=(1.0, 0.9, 0.8, 0.7, 0.5, 0.3)):
    rows, aurcs = [], []
    for seed in sorted(P.seed.unique()):
        d = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto) & (P.seed == seed)]
        if d.empty:
            continue
        if level == "group":
            d = group_aggregate(d)
        y, p = d.label.values, d.p_cal.values
        for score, unc in unc_scores(d).items():
            for r in ST.risk_coverage(y, p, unc, coverages=coverages):
                r.update({"seed": seed, "score": score}); rows.append(r)
            aurcs.append({"seed": seed, "score": score, "aurc": ST.aurc(y, p, unc)})
        aurcs.append({"seed": seed, "score": "random", "aurc": float((((p >= .5).astype(int)) != y).mean())})
    t = pd.DataFrame(rows)
    a = pd.DataFrame(aurcs)
    if len(t):
        t["coverage_target"] = t["coverage"].round(2)
        g = t.groupby(["score", "coverage_target"]).agg(
            n_retained=("n_retained", "mean"), macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
            risk=("risk", "mean"), recall_retained=("recall_retained", "mean"),
            recall_all_autonomous=("recall_all_autonomous", "mean"), pneumonia_deferred=("pneumonia_deferred_frac", "mean"),
            spec_retained=("spec_retained", "mean")).reset_index()
        g.to_csv(os.path.join(out, f"table_triage_{ds}_{proto}_{level}.csv"), index=False)
        a.groupby("score").aurc.agg(["mean", "std"]).reset_index().to_csv(os.path.join(out, f"table_aurc_{ds}_{proto}_{level}.csv"), index=False)
        return g, a
    return t, a


def table_shift(P, out, proto="source"):
    rows = []
    for ds in ["cxr_val", "cxr_test", "chexpert", "siim"]:
        for seed in sorted(P.seed.unique()):
            d = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto) & (P.seed == seed)]
            if d.empty:
                continue
            err = (d.p_cal >= 0.5).astype(int) != d.label
            rows.append({"dataset": DS_LABEL[ds], "seed": seed, "n": len(d), "error_rate": err.mean(),
                         "u_correct": d.u[~err].mean(), "u_error": d.u[err].mean()})
    t = pd.DataFrame(rows)
    if len(t):
        t = t.groupby("dataset", sort=False).agg(n=("n", "first"), error_rate=("error_rate", "mean"),
                                                 u_correct=("u_correct", "mean"), u_error=("u_error", "mean")).reset_index()
        t.to_csv(os.path.join(out, "table_shift.csv"), index=False)
    return t


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def fig_risk_coverage(P, out, ds="chexpert", proto="source", primary="vacuity"):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    cov = np.linspace(0.05, 1.0, 96)
    styles = {"evidential": ("-", "#7A5195"), "vacuity": ("-.", "#2E86AB"), "max-prob": ("--", "#555555"),
              "random": (":", "#999999"), "dissonance": ("-", "#D1495B"), "vacuity+dissonance": ("-", "#4C2A85")}
    order = [primary] + [s for s in ("vacuity", "max-prob", "random") if s != primary]
    for score in order:
        ls, col = styles[score]
        curves_r, curves_rec = [], []
        for seed in sorted(P.seed.unique()):
            d = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto) & (P.seed == seed)]
            if d.empty:
                continue
            y, p = d.label.values, d.p_cal.values
            sc = unc_scores(d)
            unc = sc.get(score, sc["max-prob"])
            rc = ST.risk_coverage(y, p, unc, coverages=cov, random_state=(seed if score == "random" else None))
            curves_r.append([r["risk"] for r in rc]); curves_rec.append([r["recall_retained"] for r in rc])
        if not curves_r:
            continue
        for ax, cur in ((axes[0], curves_r), (axes[1], curves_rec)):
            c = np.asarray(cur, float); mu = np.nanmean(c, 0); sd = np.nanstd(c, 0)
            ax.plot(cov, mu, ls, color=col, lw=2 if score == primary else 1.2, label=score)
            ax.fill_between(cov, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
    axes[0].set_xlabel("Coverage (fraction of cases retained)"); axes[0].set_ylabel("Selective risk (error rate)")
    axes[1].set_xlabel("Coverage"); axes[1].set_ylabel("Pneumonia recall (retained cases)")
    axes[0].legend(loc="upper left")
    for ax in axes:
        ax.set_xlim(0.05, 1.0); ax.grid(alpha=0.25)
    fig.suptitle(f"{DS_LABEL[ds]} ({proto}-domain support)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_risk_coverage_{ds}_{proto}.png")); plt.close(fig)


def fig_reliability(P, out, ds, proto="source", seed=None, n_bins=10):
    d = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto)]
    if d.empty:
        return
    seed = seed or sorted(d.seed.unique())[0]
    d = d[d.seed == seed]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for col, lab, c in (("p_raw", "before calibration", "#999999"), ("p_cal", "after calibration", "#4C2A85")):
        p, y = d[col].values, d.label.values
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        xs = [p[idx == i].mean() for i in range(n_bins) if (idx == i).sum() > 20]
        ys = [y[idx == i].mean() for i in range(n_bins) if (idx == i).sum() > 20]
        ax.plot(xs, ys, "o-", color=c, ms=3.5, label=lab)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Predicted pneumonia probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(DS_LABEL[ds]); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_reliability_{ds}.png")); plt.close(fig)


def fig_roc(P, out, ds="chexpert", proto="source", seed=None):
    d0 = P[(P.dataset == ds) & (P.protocol.isin([proto, "n/a"]))]
    if d0.empty:
        return
    seed = seed or sorted(d0.seed.unique())[0]
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    for meth in [PROPOSED] + BASELINES:
        d = d0[(d0.method == meth) & (d0.seed == seed)]
        if d.empty:
            continue
        g = group_aggregate(d) if PRIMARY_LEVEL[ds] == "group" else d
        fpr, tpr, _ = roc_curve(g.label, g.p_cal)
        auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=COLORS[meth], lw=2 if meth == PROPOSED else 1.2, label=f"{meth} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{DS_LABEL[ds]}, seed {seed}"); ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_roc_{ds}_{proto}.png")); plt.close(fig)


def fig_confusion(P, out, ds, proto="source", seed=None):
    d = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto)]
    if d.empty:
        return
    seed = seed or sorted(d.seed.unique())[0]
    d = d[d.seed == seed]
    if PRIMARY_LEVEL[ds] == "group":
        d = group_aggregate(d)
    cm = confusion_matrix(d.label, (d.p_cal >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.imshow(cm, cmap="Purples")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Pneumonia"]); ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{DS_LABEL[ds]} (n={cm.sum():,})")
    ax.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_cm_{ds}.png")); plt.close(fig)


def fig_training(logs, out, seed=None):
    keys = [k for k in logs if k[0] == "full"]
    if not keys:
        return
    seed = seed or sorted(k[1] for k in keys)[0]
    lg = logs[("full", seed)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    axes[0].plot(lg.epoch, lg.train_loss, color="#4C2A85", label="train loss")
    axes[0].plot(lg.epoch, lg.val_nll, color="#D1495B", label="val NLL")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(lg.epoch, lg.train_f1, color="#4C2A85", label="train")
    axes[1].plot(lg.epoch, lg.val_f1, color="#D1495B", label="validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1"); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_meta_training.png")); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    axes[0].plot(lg.epoch, lg.train_unc, label="train"); axes[0].plot(lg.epoch, lg.val_unc, label="validation")
    axes[0].set_ylabel("Mean vacuity $u$"); axes[0].legend()
    axes[1].plot(lg.epoch, lg.train_beta, label="train"); axes[1].plot(lg.epoch, lg.val_beta, label="validation")
    axes[1].set_ylabel(r"FusionGate $\beta$")
    axes[2].plot(lg.epoch, lg.kl_weight, color="#555555", label=r"KL weight $\lambda(t)$")
    axes[2].plot(lg.epoch, lg.train_kl, color="#D1495B", label="KL term")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_edl_dynamics.png")); plt.close(fig)


def fig_stage1(stage1_dir, out):
    f = os.path.join(stage1_dir, "stage1_log.csv")
    if not os.path.exists(f):
        return
    lg = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    axes[0].plot(lg.epoch, lg.train_loss, label="train"); axes[0].plot(lg.epoch, lg.val_loss, label="held-out patients")
    multilabel = "val_auroc_pneumonia" in lg.columns
    axes[0].set_ylabel("Multi-label BCE" if multilabel else "Weighted BCE"); axes[0].legend()
    if multilabel:
        axes[1].plot(lg.epoch, lg.val_auroc, label="mean AUROC (14 findings)")
        axes[1].plot(lg.epoch, lg.val_auroc_pneumonia, label="Pneumonia AUROC")
    else:
        axes[1].plot(lg.epoch, lg.val_auroc, label="AUROC"); axes[1].plot(lg.epoch, lg.val_bal_acc, label="balanced accuracy")
    best = int(lg.epoch[lg.val_auroc.idxmax()])
    for ax in axes:
        ax.axvline(best, color="#888888", ls="--", lw=0.8)
    axes[1].set_ylabel("Held-out NIH patients"); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_stage1.png")); plt.close(fig)


def fig_kshot(SW, out, ds="chexpert", proto="target"):
    d = SW[(SW.sweep == "kshot") & (SW.dataset == ds) & (SW.protocol == proto) & (SW.level == PRIMARY_LEVEL[ds])]
    if d.empty:
        return
    d = d.assign(K=d.setting.str.replace("K=", "").astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    for meth, g in d.groupby("method"):
        a = g.groupby("K").agg(f1=("macro_f1", "mean"), f1s=("macro_f1", "std"), auc=("auroc", "mean"), aucs=("auroc", "std"))
        for ax, col in ((axes[0], "f1"), (axes[1], "auc")):
            ax.errorbar(a.index, a[col], yerr=a[col + "s"], marker="o", ms=3.5, capsize=2,
                        color=COLORS.get(meth, "#333333"), lw=2 if meth == PROPOSED else 1.1, label=meth)
    axes[0].set_ylabel("Macro-F1"); axes[1].set_ylabel("AUROC")
    for ax in axes:
        ax.set_xlabel("Support images per class (K)"); ax.set_xticks([1, 3, 5, 10]); ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"{DS_LABEL[ds]} ({proto}-domain support)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_kshot_{ds}_{proto}.png")); plt.close(fig)


def fig_uncertainty_hist(P, out, ds="chexpert", proto="source"):
    d = P[(P.method == PROPOSED) & (P.dataset == ds) & (P.protocol == proto)]
    if d.empty:
        return
    d = d[d.seed == sorted(d.seed.unique())[0]]
    err = (d.p_cal >= 0.5).astype(int) != d.label
    fig, ax = plt.subplots(figsize=(4, 3))
    bins = np.linspace(d.u.min(), d.u.max(), 40)
    ax.hist(d.u[~err], bins=bins, alpha=0.6, density=True, color="#4C2A85", label="correct")
    ax.hist(d.u[err], bins=bins, alpha=0.6, density=True, color="#D1495B", label="error")
    ax.set_xlabel("Vacuity $u$"); ax.set_ylabel("Density"); ax.set_title(DS_LABEL[ds]); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_unc_hist_{ds}.png")); plt.close(fig)


def fig_xai(xai_dir, out):
    fs = glob.glob(os.path.join(xai_dir, "xai_seed*.npz"))
    if not fs:
        return
    d = np.load(fs[0], allow_pickle=True)
    for ds in ("cxr_test", "chexpert"):
        if f"{ds}_imgs" not in d.files:
            continue
        imgs, cam, campp, lab, prob = d[f"{ds}_imgs"], d[f"{ds}_cam"], d[f"{ds}_campp"], d[f"{ds}_label"], d[f"{ds}_prob"]
        n = len(imgs)
        fig, axes = plt.subplots(n, 3, figsize=(6.0, 2.0 * n))
        for i in range(n):
            for j, (title, over) in enumerate((("Input", None), ("Grad-CAM", cam[i]), ("Grad-CAM++", campp[i]))):
                ax = axes[i, j]
                ax.imshow(imgs[i], cmap="gray")
                if over is not None:
                    ax.imshow(over, cmap="jet", alpha=0.35)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(title, fontsize=9)
            axes[i, 0].set_ylabel(f"{'Pneumonia' if lab[i] == 1 else 'Normal'}\np={prob[i]:.2f}", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_gradcam_{ds}.png")); plt.close(fig)
        lp = d.get(f"{ds}_lime_pos") if hasattr(d, "get") else d[f"{ds}_lime_pos"]
        if lp is not None and lp.size:
            from skimage.segmentation import mark_boundaries
            la = d[f"{ds}_lime_all"]
            fig, axes = plt.subplots(n, 3, figsize=(6.0, 2.0 * n))
            for i in range(n):
                rgb = np.repeat(imgs[i][..., None], 3, -1) / 255.0
                for j, (title, img) in enumerate((("Input", rgb), ("LIME (supporting)", mark_boundaries(rgb, lp[i], color=(1, 1, 0))),
                                                  ("LIME (all regions)", mark_boundaries(rgb, la[i], color=(1, 1, 0))))):
                    axes[i, j].imshow(img); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                    if i == 0:
                        axes[i, j].set_title(title, fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_lime_{ds}.png")); plt.close(fig)
    for f in glob.glob(os.path.join(xai_dir, "umap_seed*.csv")):
        u = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        for ds, g in u.groupby("dataset"):
            axes[0].scatter(g.x, g.y, s=2, alpha=0.5, label=ds)
        for lab, g in u.groupby("label"):
            axes[1].scatter(g.x, g.y, s=2, alpha=0.5, label="Pneumonia" if lab == 1 else "Normal",
                            color="#D1495B" if lab == 1 else "#2E86AB")
        for ax, t in zip(axes, ("By dataset", "By class")):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t); ax.legend(markerscale=4, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_umap.png")); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage1", default=None)
    p.add_argument("--xai", default=None)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    M_job, SW, R, P, logs = load(a.runs)
    M = build_metrics(P, M_job)
    M.to_csv(os.path.join(a.out, "metrics_all.csv"), index=False)
    print("metrics", M.shape, "sweeps", SW.shape, "preds", P.shape, "seeds", sorted(M.seed.unique()) if len(M) else [])
    table_main(M, P, a.out)
    mets = ("macro_f1", "auroc", "brier", "ece", "unc_err_auroc", "rec_pneumonia", "rec_normal")
    table_compare(M, a.out, [PROPOSED] + BASELINES + ["EfficientNet-Binary (val thr)"], "baselines", metrics=mets)
    table_compare(M, a.out, [PROPOSED + " (transductive)"] + BASELINES + ["VanillaProtoNet (transductive)", PROPOSED],
                  "baselines_transductive", metrics=mets, ref=PROPOSED + " (transductive)")
    table_compare(M, a.out, ABLATION, "ablation",
                  metrics=("macro_f1", "auroc", "brier", "ece", "unc_err_auroc", "err_auroc_vacuity", "rec_pneumonia"))
    for proto in ("source", "target"):
        table_image_level_tests(P, a.out, "chexpert", proto)
    table_sweep(SW, a.out, "kshot", "kshot")
    table_sweep(SW, a.out, "memory", "memory")
    if len(R):
        R.groupby("dataset").mean(numeric_only=True).drop(columns=["seed"]).reset_index().to_csv(os.path.join(a.out, "table_retrieval.csv"), index=False)
    _, primary = table_uncertainty_selection(P, a.out)
    print("primary triage score (selected on CXR-Val by AURC):", primary)
    for ds in ("chexpert", "siim", "cxr_test"):
        for proto in ("source", "target"):
            for level in ("image", "group"):
                table_triage(P, a.out, ds, proto, level)
            fig_risk_coverage(P, a.out, ds, proto, primary)
            fig_kshot(SW, a.out, ds, proto)
    table_shift(P, a.out)
    for ds in ("cxr_test", "chexpert", "siim"):
        fig_reliability(P, a.out, ds); fig_confusion(P, a.out, ds); fig_roc(P, a.out, ds); fig_uncertainty_hist(P, a.out, ds)
    fig_training(logs, a.out)
    if a.stage1:
        fig_stage1(a.stage1, a.out)
    if a.xai:
        fig_xai(a.xai, a.out)
    print("report written to", a.out)


if __name__ == "__main__":
    main()
