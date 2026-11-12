"""LaTeX table bodies for the manuscript, generated from the report CSVs (no hand-typed numbers).

    python -m urproto.paper_tables --rep runs/report_final2 --out revision/tables
"""
import argparse
import os

import numpy as np
import pandas as pd

LVL = {"CXR-Val": "image", "CXR-Test": "image", "CheXpert": "group", "SIIM-FISABIO-RSNA": "group"}
DSK = {"CXR-Val": "cxr_val", "CXR-Test": "cxr_test", "CheXpert": "chexpert", "SIIM-FISABIO-RSNA": "siim"}


def pm(m, s, d=3):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def pfmt(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def signed(x, d=3):
    return f"{x:+.{d}f}"


def load_metrics(rep):
    M = pd.read_csv(os.path.join(rep, "metrics_all.csv"))
    return M


def agg(M, method, ds, proto, level, col):
    d = M[(M.method == method) & (M.dataset == ds) & (M.protocol.isin([proto, "n/a"])) & (M.level == level)][col]
    return d.mean(), d.std(), len(d)


def t_main(rep, M):
    U = pd.read_csv(os.path.join(rep, "table_uncertainty_scores.csv"))
    rows = []
    spec = [("CXR-Val", "cxr_val", "source", "image", "--", "1,550 images"),
            ("CXR-Test", "cxr_test", "source", "image", "--", "2,834 images"),
            ("CheXpert", "chexpert", "source", "group", "source", "8,856 patients"),
            ("CheXpert", "chexpert", "target", "group", "target", "8,856 patients"),
            ("SIIM-FISABIO-RSNA", "siim", "source", "group", "source", "5,454 studies"),
            ("SIIM-FISABIO-RSNA", "siim", "target", "group", "target", "5,454 studies")]
    for name, ds, proto, lvl, plab, n in spec:
        vals = [pm(*agg(M, "UR-ProtoNet", ds, proto, lvl, c)[:2]) for c in ("macro_f1", "auroc", "brier", "ece", "rec_pneumonia")]
        u = U[(U.dataset == ds) & (U.protocol == proto) & (U.score == "vacuity+dissonance")]
        vals.append(pm(u.err_auroc.iloc[0], u.err_auroc_sd.iloc[0]))
        rows.append(f"{name} & {plab} & {n} & " + " & ".join(vals) + " \\\\")
    return "\n".join(rows)


def t_compare(rep, M, ds_label, protos, methods, ref="UR-ProtoNet", tests_file="table_baselines_tests.csv"):
    S = pd.read_csv(os.path.join(rep, tests_file))
    ds, lvl = DSK[ds_label], LVL[ds_label]
    out = []
    for proto in protos:
        best = {}
        for c, hi in (("macro_f1", True), ("auroc", True), ("brier", False), ("ece", False)):
            vals = {m: agg(M, m, ds, proto, lvl, c)[0] for m in methods}
            vals = {k: v for k, v in vals.items() if not np.isnan(v)}
            best[c] = (max if hi else min)(vals, key=vals.get)
        hdr = ds_label if ds_label != "CheXpert" else f"CheXpert, {proto}-domain support"
        if ds_label.startswith("SIIM"):
            hdr = f"SIIM-FISABIO-RSNA, {proto}-domain support"
        out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{hdr}}}}} \\\\")
        for m in methods:
            cells = []
            for c in ("macro_f1", "auroc", "brier", "ece"):
                mu, sd, n = agg(M, m, ds, proto, lvl, c)
                s = pm(mu, sd)
                cells.append(f"\\textbf{{{s}}}" if best[c] == m else s)
            if m == ref:
                cells += ["--", "--"]
            else:
                for met in ("macro_f1", "auroc"):
                    r = S[(S.dataset == ds_label) & (S.protocol == proto) & (S.baseline == m) & (S.metric == met)]
                    cells.append(f"{signed(r.delta.iloc[0])} ({pfmt(r.p_t_holm.iloc[0])})" if len(r) else "--")
            out.append(f"\\quad {m} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def t_stats(rep, ds_label="CheXpert", protos=("source", "target"), tests_file="table_baselines_tests.csv"):
    S = pd.read_csv(os.path.join(rep, tests_file))
    out = []
    for proto in protos:
        for met in ("macro_f1", "auroc", "brier", "ece"):
            r = S[(S.dataset == ds_label) & (S.protocol == proto) & (S.metric == met)]
            for _, x in r.iterrows():
                out.append(f"{proto} & {met.replace('macro_f1', 'Macro-F1').replace('auroc', 'AUROC').replace('brier', 'Brier').replace('ece', 'ECE')} & "
                           f"{x.baseline} & {signed(x.delta)} [{x.delta_ci_lo:+.3f}, {x.delta_ci_hi:+.3f}] & {x.cohens_dz:+.2f} & "
                           f"{pfmt(x.p_t)} & {pfmt(x.p_t_holm)} & {pfmt(x.p_wilcoxon)} & {pfmt(x.p_w_holm)} & {int(x.wins)}/10 \\\\")
    return "\n".join(out)


def t_ablation(M):
    out = []
    for proto in ("source", "target"):
        out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{CheXpert, {proto}-domain support}}}} \\\\")
        for m, lab in (("VanillaProtoNet", "PlainProto (VanillaProtoNet)"), ("Proto+Memory", "+ memory and FusionGate"),
                       ("Proto+EDL", "+ evidential head"), ("UR-ProtoNet", "UR-ProtoNet (memory + gate + evidential)")):
            cells = [pm(*agg(M, m, "chexpert", proto, "group", c)[:2]) for c in ("macro_f1", "auroc", "brier", "ece", "unc_err_auroc")]
            out.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def t_transductive(M):
    out = []
    for ds, proto, lab in (("cxr_test", "source", "CXR-Test"), ("chexpert", "source", "CheXpert, source support"),
                           ("chexpert", "target", "CheXpert, target support"), ("siim", "source", "SIIM-FISABIO-RSNA, source support"),
                           ("siim", "target", "SIIM-FISABIO-RSNA, target support")):
        lvl = "image" if ds.startswith("cxr") else "group"
        for m, ml in (("UR-ProtoNet", "standard"), ("UR-ProtoNet (transductive)", "transductive")):
            cells = [pm(*agg(M, m, ds, proto, lvl, c)[:2]) for c in ("macro_f1", "auroc", "brier", "ece", "rec_pneumonia")]
            out.append(f"{lab if ml == 'standard' else ''} & {ml} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def t_kshot(rep):
    K = pd.read_csv(os.path.join(rep, "table_kshot.csv"))
    K = K[(K.method == "UR-ProtoNet") & (K.dataset == "chexpert") & (K.level == "group")]
    out = []
    for k in (1, 3, 5, 10):
        cells = []
        for proto in ("source", "target"):
            r = K[(K.protocol == proto) & (K.setting == f"K={k}")].iloc[0]
            cells += [pm(r.macro_f1_mean, r.macro_f1_sd), pm(r.auroc_mean, r.auroc_sd), f"{r.ece_mean:.3f}"]
        out.append(f"{k} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def t_memory(rep):
    T = pd.read_csv(os.path.join(rep, "table_memory.csv"))
    T = T[(T.dataset == "chexpert") & (T.level == "group")]
    lab = [("no_memory", "No memory ($\\beta = 1$)"), ("nih_10pct", "NIH, 10\\% of entries"), ("nih_25pct", "NIH, 25\\% of entries"),
           ("nih_50pct", "NIH, 50\\% of entries"), ("nih_balanced", "NIH, class-balanced (1,353 per class)"),
           ("nih_full", "NIH, full (default; 8,000 / 1,353)"), ("target_only", "CheXpert support pool only$^{a}$"),
           ("nih_plus_target", "NIH + CheXpert support pool$^{a}$")]
    out = []
    for key, l in lab:
        cells = []
        for proto in ("source", "target"):
            r = T[(T.protocol == proto) & (T.setting == key)].iloc[0]
            cells += [pm(r.macro_f1_mean, r.macro_f1_sd), pm(r.auroc_mean, r.auroc_sd), f"{r.ece_mean:.3f}"]
        out.append(f"{l} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def t_retrieval(rep, M):
    R = pd.read_csv(os.path.join(rep, "table_retrieval.csv")).set_index("dataset")
    out = []
    for ds, lab in (("cxr_test", "CXR-Test"), ("chexpert", "CheXpert"), ("siim", "SIIM-FISABIO-RSNA")):
        r = R.loc[ds]
        bs = [agg(M, "UR-ProtoNet", ds, "source", "image", c)[0] for c in ("beta_normal", "beta_pneumonia", "memsim_normal", "memsim_pneumonia")]
        if ds != "cxr_test":
            bt = [agg(M, "UR-ProtoNet", ds, "target", "image", c)[0] for c in ("beta_normal", "beta_pneumonia", "memsim_normal", "memsim_pneumonia")]
            tgt = f"{bt[2]:.2f} / {bt[3]:.2f} & {bt[0]:.2f} / {bt[1]:.2f}"
        else:
            tgt = "-- & --"
        out.append(f"{lab} & {r.top1_cos_mean:.3f} [{r.top1_cos_p10:.3f}, {r.top1_cos_p90:.3f}] & {r.knn_auroc:.3f} & "
                   f"{bs[2]:.2f} / {bs[3]:.2f} & {bs[0]:.2f} / {bs[1]:.2f} & {tgt} \\\\")
    return "\n".join(out)


def t_unc(rep):
    U = pd.read_csv(os.path.join(rep, "table_uncertainty_scores.csv"))
    A = {}
    for ds, proto, lvl in (("chexpert", "source", "group"), ("chexpert", "target", "group"), ("cxr_test", "source", "image")):
        f = os.path.join(rep, f"table_aurc_{ds}_{proto}_{lvl}.csv")
        if os.path.exists(f):
            a = pd.read_csv(f).set_index("score")
            A[(ds, proto)] = a.loc["random", "mean"]
    out = []
    for ds, proto, lab in (("cxr_val", "source", "CXR-Val"), ("cxr_test", "source", "CXR-Test"),
                           ("chexpert", "source", "CheXpert, source support"), ("chexpert", "target", "CheXpert, target support")):
        cells = []
        for sc in ("vacuity", "dissonance", "vacuity+dissonance", "evidential", "max-prob"):
            r = U[(U.dataset == ds) & (U.protocol == proto) & (U.score == sc)].iloc[0]
            cells.append(f"{r.err_auroc:.3f} / {r.aurc:.3f}")
        rnd = A.get((ds, proto))
        cells.append(f"-- / {rnd:.3f}" if rnd is not None else "--")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def t_triage(rep, ds="chexpert", proto="target", level="group", score="vacuity+dissonance"):
    T = pd.read_csv(os.path.join(rep, f"table_triage_{ds}_{proto}_{level}.csv"))
    T = T[T.score == score].sort_values("coverage_target", ascending=False)
    out = []
    for _, r in T.iterrows():
        out.append(f"{100 * r.coverage_target:.0f}\\% & {int(r.n_retained):,} & {r.risk:.3f} & {r.macro_f1:.3f} & "
                   f"{r.recall_retained:.3f} & {r.recall_all_autonomous:.3f} & {100 * r.pneumonia_deferred:.1f}\\% \\\\")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rep", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    M = load_metrics(a.rep)
    B = ["UR-ProtoNet", "EfficientNet-Binary", "VanillaProtoNet", "MatchingNet", "SimpleShot"]
    parts = {
        "main": t_main(a.rep, M),
        "baselines": t_compare(a.rep, M, "CXR-Test", ["source"], B) + "\n\\midrule\n" + t_compare(a.rep, M, "CheXpert", ["source", "target"], B),
        "baselines_siim": t_compare(a.rep, M, "SIIM-FISABIO-RSNA", ["source", "target"], B),
        "stats_chexpert": t_stats(a.rep),
        "ablation": t_ablation(M),
        "transductive": t_transductive(M),
        "kshot": t_kshot(a.rep),
        "memory": t_memory(a.rep),
        "retrieval": t_retrieval(a.rep, M),
        "unc": t_unc(a.rep),
        "triage_chexpert_target": t_triage(a.rep, "chexpert", "target"),
        "triage_chexpert_source": t_triage(a.rep, "chexpert", "source"),
        "triage_cxr_test": t_triage(a.rep, "cxr_test", "source", "image"),
    }
    for k, v in parts.items():
        with open(os.path.join(a.out, f"{k}.tex"), "w") as f:
            f.write(v + "\n")
        print(f"%%%%% {k}\n{v}\n")


if __name__ == "__main__":
    main()
