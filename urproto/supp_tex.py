"""Builds the Supplementary Material (LaTeX) from the report CSVs.

    python -m urproto.supp_tex --rep runs/report_final2 --out revision/supplementary.tex
"""
import argparse
import os

import pandas as pd

from .paper_tables import pfmt, t_compare, t_stats, t_triage, load_metrics

HEAD = r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,graphicx,amsmath,amssymb,caption}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\captionsetup{font=small}
\title{Supplementary Material\\[4pt]\large UR-ProtoNet: Evidential Retrieval-Augmented Prototypical Networks for Uncertainty-Aware Few-Shot Chest X-Ray Classification Across Domain Shifts}
\date{}
\begin{document}
\maketitle
All results are means over ten independent runs (seeds 42, 123, 777, 0, 999, 1, 7, 13, 2024, 31337) with $K = 5$ support images per class and $M = 20$ support sets, unless stated otherwise. CheXpert is evaluated per patient ($N = 8{,}856$) and SIIM-FISABIO-RSNA per study ($N = 5{,}454$). The tables are generated directly from the released prediction files by the analysis code of the public repository.
"""


def delong_table(rep):
    out = []
    for proto in ("source", "target"):
        D = pd.read_csv(os.path.join(rep, f"table_delong_chexpert_{proto}.csv"))
        base = ["EfficientNet-Binary", "VanillaProtoNet", "MatchingNet", "SimpleShot"]
        out.append(r"\multicolumn{5}{@{}l}{\emph{CheXpert, %s-domain support}} \\" % proto)
        for seed in sorted(D.seed.unique()):
            cells = []
            for b in base:
                r = D[(D.seed == seed) & (D.baseline == b)].iloc[0]
                cells.append(f"{r.delong_delta_auc:+.3f} ({pfmt(r.delong_p)})")
            out.append(f"{seed} & " + " & ".join(cells) + r" \\")
        if proto == "source":
            out.append(r"\midrule")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rep", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    M = load_metrics(a.rep)
    B = ["UR-ProtoNet", "EfficientNet-Binary", "VanillaProtoNet", "MatchingNet", "SimpleShot"]
    body = HEAD
    body += r"""
\section*{S1. Paired statistical tests on CheXpert}

\begin{table}[h]
\centering
\caption{Paired comparison of UR-ProtoNet with each baseline on CheXpert over the ten runs. $\Delta$: mean paired difference UR-ProtoNet $-$ baseline with its 95\% confidence interval; $d_z$: standardized paired effect size; $p_t$: two-sided paired $t$-test; $p_W$: exact Wilcoxon signed-rank test; Holm: adjusted over the baselines within each protocol and metric. Higher: number of runs in which UR-ProtoNet had the higher value (for Brier and ECE, lower is better). ``(val thr)'': EfficientNet-Binary with the threshold maximizing macro-F1 on CXR-Val.}
\label{tab:s_stats}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllccccccc}
\toprule
Support & Metric & Baseline & $\Delta$ [95\% CI] & $d_z$ & $p_t$ & $p_t$ (Holm) & $p_W$ & $p_W$ (Holm) & Higher \\
\midrule
""" + t_stats(a.rep) + r"""
\bottomrule
\end{tabular}
\end{table}

\clearpage
\section*{S2. Per-run DeLong tests on CheXpert}

\begin{table}[h]
\centering
\caption{Per-run comparison of patient-level ROC curves on CheXpert with DeLong's test for correlated ROC curves: AUROC difference UR-ProtoNet $-$ baseline ($p$-value).}
\label{tab:s_delong}
\small
\begin{tabular}{lcccc}
\toprule
Seed & EfficientNet-Binary & VanillaProtoNet & MatchingNet & SimpleShot \\
\midrule
""" + delong_table(a.rep) + r"""
\bottomrule
\end{tabular}
\end{table}

\clearpage
\section*{S3. Baseline comparison on SIIM-FISABIO-RSNA}

\begin{table}[h]
\centering
\caption{Comparison with the baselines on SIIM-FISABIO-RSNA (study level, mean $\pm$ standard deviation over ten runs). $\Delta$: mean paired difference UR-ProtoNet $-$ baseline, with the Holm-adjusted paired $t$-test $p$-value in parentheses. Best value per block in bold.}
\label{tab:s_siim}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Method & Macro-F1 & AUROC & Brier & ECE & $\Delta$ Macro-F1 ($p$) & $\Delta$ AUROC ($p$) \\
\midrule
""" + t_compare(a.rep, M, "SIIM-FISABIO-RSNA", ["source", "target"], B) + r"""
\bottomrule
\end{tabular}
\end{table}

\section*{S4. Selective prediction on CXR-Test}

\begin{table}[h]
\centering
\caption{Selective prediction of UR-ProtoNet on CXR-Test (image level, primary score vacuity + dissonance, mean over ten runs). Columns as in the main-text table of selective prediction. At low coverage the retained images are dominated by one class, so the macro-F1 becomes unstable; the error rate is the more informative quantity.}
\label{tab:s_triage}
\small
\begin{tabular}{ccccccc}
\toprule
Coverage & Retained & Error rate & Macro-F1 & Retained recall & Autonomous sensitivity & Pneumonia deferred \\
\midrule
""" + t_triage(a.rep, "cxr_test", "source", "image") + r"""
\bottomrule
\end{tabular}
\end{table}

\clearpage
\section*{S5. Supplementary figures}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figs/unc_hist.png}
\caption{Distribution of the primary uncertainty score (vacuity + dissonance) for correct and erroneous predictions of UR-ProtoNet (seed 42) on CXR-Test and on CheXpert (patient level) with source-domain and target-domain support images.}
\label{fig:s_unc}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figs/augmentation.png}
\caption{Examples of the Stage~III training augmentation (geometric transformations, zoom-in crop, gamma, contrast and brightness, blurring or sharpening, resolution loss and noise) applied to preprocessed CXR-Test and CheXpert radiographs.}
\label{fig:s_aug}
\end{figure}

\end{document}
"""
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(body)
    print("written", a.out)


if __name__ == "__main__":
    main()
