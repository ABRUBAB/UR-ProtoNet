"""Render report CSVs as LaTeX table bodies (pasted into the manuscript).

  python -m urproto.latex --report REPORT_DIR --out REPORT_DIR/latex
"""

import argparse
import os

import numpy as np
import pandas as pd

DS = {"CXR-Test": "CXR-Test", "CheXpert": "CheXpert", "SIIM-FISABIO-RSNA": "SIIM-FISABIO-RSNA"}


def pm(m, s, d=3):
    if pd.isna(m):
        return "--"
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}" if not pd.isna(s) else f"{m:.{d}f}"


def p_fmt(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main_table(rep):
    t = pd.read_csv(os.path.join(rep, "table_main.csv"))
    rows = []
    for _, r in t.iterrows():
        lvl = {"image": "image", "group": "patient" if r.dataset == "CheXpert" else ("study" if "SIIM" in r.dataset else "patient")}[r.level]
        rows.append(f"{r.dataset} & {r.protocol} & {lvl} & {int(r.n):,} & {pm(r.macro_f1_mean, r.macro_f1_sd)} & "
                    f"{pm(r.auroc_mean, r.auroc_sd)} & {pm(r.brier_mean, r.brier_sd)} & {pm(r.ece_mean, r.ece_sd)} & "
                    f"{pm(r.unc_err_auroc_mean, r.unc_err_auroc_sd)} \\\\")
    return "\n".join(rows)


def compare_table(rep, name, metrics=("macro_f1", "auroc", "brier", "ece", "unc_err_auroc")):
    t = pd.read_csv(os.path.join(rep, f"table_{name}.csv"))
    s = pd.read_csv(os.path.join(rep, f"table_{name}_tests.csv")) if os.path.exists(os.path.join(rep, f"table_{name}_tests.csv")) else pd.DataFrame()
    out = []
    for (ds, proto), g in t.groupby(["dataset", "protocol"], sort=False):
        out.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{ds}, {proto}-domain support}}}} \\\\")
        for _, r in g.iterrows():
            cells = [str(r[m]).replace("--", "--") for m in metrics]
            test = s[(s.dataset == ds) & (s.protocol == proto) & (s.baseline == r.method) & (s.metric == "macro_f1")] if len(s) else s
            if len(test):
                x = test.iloc[0]
                extra = f"{x.delta:+.3f} & {p_fmt(x.p_w_holm)}"
            else:
                extra = "--- & ---"
            out.append(f"{r.method} & " + " & ".join(cells) + f" & {extra} \\\\")
        out.append("\\midrule")
    return "\n".join(out[:-1])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or os.path.join(a.report, "latex")
    os.makedirs(out, exist_ok=True)
    for name, fn in (("main", lambda: main_table(a.report)), ("baselines", lambda: compare_table(a.report, "baselines")),
                     ("ablation", lambda: compare_table(a.report, "ablation"))):
        try:
            txt = fn()
            open(os.path.join(out, f"{name}.tex"), "w", encoding="utf-8").write(txt)
            print(f"--- {name}\n{txt[:1500]}")
        except Exception as e:
            print(name, "failed:", e)


if __name__ == "__main__":
    main()
