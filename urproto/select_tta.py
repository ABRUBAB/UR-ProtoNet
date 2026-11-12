"""Validation-only selection of the test-time-adaptation variants in tta.py (NO test data, NO retraining for
UR-ProtoNet if a saved checkpoint is given).

Protocol = jobs.job_select, extended:
  val        CXR-Val queries, source supports                      (also fits the temperature T per variant)
  pool_src   CheXpert support pool as queries, source supports      (patient level)
  pool_tgt   CheXpert support pool split by patient into support / query halves (both directions)
  prevXX     pool_src with the pool re-sampled by patient to 25 % / 75 % pneumonia  (prior-shift stress test)
  triage     AURC and error-detection AUROC of vacuity / dissonance / total / 1 - max p_cal / ensemble std
CheXpert queries and SIIM are never loaded.

Kaggle usage (same inputs as `python -m urproto.jobs select`):
  python select_tta.py --ckpt_dir <dir with full_seed42.pt ...> --seeds 42 123 --out /kaggle/working/tta
  python select_tta.py --methods full proto --seeds 42 123      # also trains VanillaProtoNet (+ same TTA) for the
                                                                # "is it only the TTA?" ablation; ~6 min per run
CPU smoke test on random embeddings:  python select_tta.py --selftest
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from .tta import TTAConfig, proto_ensemble_tta, calibrated_from_logits_prior, evidential_uncertainties, risk_coverage
from . import evaluate as EV

# em_temp is filled in with the inductive validation temperature (calibrated responsibilities) at run time.
GRID = {
    "inductive": TTAConfig(align="none"),
    "P1_shift": TTAConfig(),
    "P1_shift_it10": TTAConfig(iters=10),
    "P1_shift_prior": TTAConfig(iters=10, est_prior=True),
    "P1_shift_priorMAP": TTAConfig(iters=10, est_prior=True, decision_prior="estimated"),
    "P1_center": TTAConfig(center=True),
    "P2a_tcr_only": TTAConfig(align="none", tcr=1.0),
    "P2a_tcr0.5": TTAConfig(tcr=0.5),
    "P2a_tcr1": TTAConfig(tcr=1.0),
    "P2b_rho0.25": TTAConfig(rho=0.25),
    "P2b_rho0.5": TTAConfig(rho=0.5),
    "P2b_rho0.75": TTAConfig(rho=0.75),
    "P2b_rho0.5_top20": TTAConfig(rho=0.5, topk_frac=0.2),
    "P2_tcr1_rho0.5": TTAConfig(tcr=1.0, rho=0.5),
    "P2_tcr1_rho0.5_prior": TTAConfig(tcr=1.0, rho=0.5, iters=10, est_prior=True),
    "P2_tcr1_rho0.5_priorMAP": TTAConfig(tcr=1.0, rho=0.5, iters=10, est_prior=True, decision_prior="estimated"),
}


def fit_T(e, y):
    y = np.asarray(y).astype(int)

    def nll(T):
        p = np.clip(calibrated_from_logits_prior(e["logits"], T, e["log_prior"]), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return EV.fit_temperature(nll)


def _group_metrics(df, e, T, prefix):
    unc = evidential_uncertainties(e["logits"], T)
    p = calibrated_from_logits_prior(e["logits"], T, e["log_prior"])
    d = df[["group", "label"]].assign(p_cal=p, p_raw=e["p_raw"], u=e["u"], diss=unc["dissonance"], total=unc["total"],
                                      ens_std=unc["ens_std"])
    g = d.groupby("group", as_index=False).agg({"label": "max", "p_cal": "mean", "p_raw": "mean", "u": "mean",
                                                "diss": "mean", "total": "mean", "ens_std": "mean"})
    m = EV.compute_metrics(g.label.values, g.p_cal.values, g.u.values)
    out = {f"{prefix}_{k}": m[k] for k in ("macro_f1", "auroc", "ece", "rec_normal", "rec_pneumonia")}
    out[f"{prefix}_pi_hat"] = e["pi_hat"]
    out[f"{prefix}_shift"] = e["shift_norm"]
    err = ((g.p_cal.values >= 0.5).astype(int) != g.label.values).astype(int)
    scores = {"vac": g.u.values, "diss": g.diss.values, "total": g.total.values,
              "maxp": 1 - np.maximum(g.p_cal.values, 1 - g.p_cal.values), "ensstd": g.ens_std.values}
    for k, s in scores.items():
        out[f"{prefix}_errauc_{k}"] = EV.safe_auc(err, s)
        out[f"{prefix}_aurc_{k}"] = risk_coverage(g.label.values, g.p_cal.values, s)[2]
    return out


def _prev_subset(pool, prev, rng):
    lab = pool.groupby("group").label.max()
    pos, neg = lab.index[lab == 1].values, lab.index[lab == 0].values
    if len(pos) / (len(pos) + len(neg)) < prev:     # keep every positive patient, thin out the negatives
        neg = rng.choice(neg, min(len(neg), max(2, int(round(len(pos) * (1 - prev) / prev)))), replace=False)
    else:                                            # keep every negative patient, thin out the positives
        pos = rng.choice(pos, min(len(pos), max(2, int(round(len(neg) * prev / (1 - prev))))), replace=False)
    return np.where(pool.group.isin(set(pos) | set(neg)).values)[0]


def evaluate_variants(model, Zv, val, Zs, src, Zp, pool, seed, M, K, grid=GRID, log=print):
    """All tensors are embeddings on one device; val / src / pool are DataFrames with `label` (+ `group` for pool)."""
    dev = Zv.device
    yv, ys = val.label.values, src.label.values
    base = proto_ensemble_tta(model, Zv, Zs, ys, M, K, seed * 100 + 1, tta=TTAConfig(align="none"))
    T_base = fit_T(base, yv)
    rows = []
    for name, tta in grid.items():
        tta = dataclasses.replace(tta, em_temp=T_base)
        ev = proto_ensemble_tta(model, Zv, Zs, ys, M, K, seed * 100 + 1, tta=tta)
        T = fit_T(ev, yv)
        res = {"variant": name, "seed": seed, "T": T, "T_base": T_base, "tta": tta.tag()}
        res.update(_group_metrics(val.assign(group=np.arange(len(val))), ev, T, "val"))
        ep = proto_ensemble_tta(model, Zp, Zs, ys, M, K, seed * 100 + 3, tta=tta)
        res.update(_group_metrics(pool, ep, T, "pool_src"))
        pats = np.array(sorted(pool.group.unique()))
        half = set(np.random.RandomState(seed).choice(pats, len(pats) // 2, replace=False))
        acc = []
        for sup_mask in (pool.group.isin(half).values, ~pool.group.isin(half).values):
            qi, si = np.where(~sup_mask)[0], np.where(sup_mask)[0]
            e = proto_ensemble_tta(model, Zp[torch.as_tensor(qi, device=dev)], Zp[torch.as_tensor(si, device=dev)],
                                   pool.label.values[si], M, K, seed * 100 + 4, tta=tta)
            acc.append(_group_metrics(pool.iloc[qi], e, T, "pool_tgt"))
        res.update({k: float(np.mean([a[k] for a in acc])) for k in acc[0]})
        for prev in (0.25, 0.75):
            qi = _prev_subset(pool, prev, np.random.RandomState(seed + int(prev * 100)))
            e = proto_ensemble_tta(model, Zp[torch.as_tensor(qi, device=dev)], Zs, ys, M, K, seed * 100 + 3, tta=tta)
            res.update(_group_metrics(pool.iloc[qi], e, T, f"prev{int(prev * 100)}"))
        rows.append(res)
        log(f"SELECT-TTA {name:26s} val F1={res['val_macro_f1']:.3f} | pool_src F1={res['pool_src_macro_f1']:.3f} "
            f"AUC={res['pool_src_auroc']:.3f} | pool_tgt F1={res['pool_tgt_macro_f1']:.3f} AUC={res['pool_tgt_auroc']:.3f} | "
            f"prev25 F1={res['prev25_macro_f1']:.3f} prev75 F1={res['prev75_macro_f1']:.3f} pi75={res['prev75_pi_hat']:.2f}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=None, help="folder with full_seed{seed}.pt saved by `jobs seeds`")
    p.add_argument("--cache", default=None)
    p.add_argument("--encoder", default=None)
    p.add_argument("--out", default="/kaggle/working/tta")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 123])
    p.add_argument("--methods", nargs="*", default=["full"])
    p.add_argument("--override", default="")
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()

    from . import jobs as J
    from .models import Encoder, ProtoModel
    from .train import embed, meta_train, build_memory
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = J.make_cfg(a.override)
    S = J._load_all(J._cache_dir(a), cfg=cfg)
    cxr_arr, cxr_df = S["cxr"]; mem_arr, mem_labels = S["mem"]; chex_arr, chex_df = S["chex"]
    val = J._rows(cxr_df, split="val")
    src = J._pool(J._rows(cxr_df, split="train"), cfg.source_pool_per_class, 12345)
    pool = J._rows(chex_df, role="pool")          # the ONLY CheXpert rows used here
    rows = []
    for meth in a.methods:
        for seed in a.seeds:
            ck = os.path.join(a.ckpt_dir, f"full_seed{seed}.pt") if (a.ckpt_dir and meth == "full") else None
            if meth == "full" and not (ck and os.path.exists(ck)):
                ck = J.find_file(f"full_seed{seed}.pt")
            if ck and os.path.exists(ck):
                st = torch.load(ck, map_location="cpu")
                mcfg = J.make_cfg(json.dumps(st["cfg"]))
                model = ProtoModel(Encoder(mcfg.emb_dim, mcfg.hidden_dim, pretrained=False), "full", mcfg).to(dev)
                model.load_state_dict(st["state_dict"]); model.eval()
                model.memory = build_memory(model.encoder, mem_arr, mem_labels, dev, mcfg.eval_bs)
            else:
                enc_state = torch.load(a.encoder or J.find_file("stage1_encoder.pt"), map_location="cpu")
                model, _ = meta_train(J.METHOD_VARIANT[meth], enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed,
                                      dev, print, a.max_epochs)
            Zv = embed(model.encoder, cxr_arr, val.cache_row.values, dev)
            Zs = embed(model.encoder, cxr_arr, src.cache_row.values, dev)
            Zp = embed(model.encoder, chex_arr, pool.cache_row.values, dev)
            grid = GRID if model.uses_memory else {k: v for k, v in GRID.items() if v.tcr == 0}
            for r in evaluate_variants(model, Zv, val, Zs, src, Zp, pool, seed, cfg.n_support_sets, cfg.k_shot, grid):
                rows.append(dict(r, method=meth))
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "select_tta_results.csv"), index=False)
    df = pd.DataFrame(rows)
    keys = ["val_macro_f1", "pool_src_macro_f1", "pool_src_auroc", "pool_tgt_macro_f1", "pool_tgt_auroc",
            "prev25_macro_f1", "prev75_macro_f1", "pool_src_aurc_maxp", "pool_src_aurc_total", "pool_src_aurc_vac"]
    summ = df.groupby(["method", "variant"], sort=False)[keys].mean().round(4)
    summ.to_csv(os.path.join(a.out, "select_tta_summary.csv"))
    print(summ.to_string())


def selftest():
    """Random embeddings: only checks that every variant runs end to end and returns finite numbers."""
    from .config import Config
    from .models import MemoryBank, ProtoModel
    torch.manual_seed(0); rs = np.random.RandomState(0)

    class _Enc(torch.nn.Module):
        emb_dim = 32

    def dom(n, shift):
        y = rs.randint(0, 2, n)
        z = torch.randn(n, 32) * 0.5 + 2.0 + shift + torch.as_tensor(2 * y - 1, dtype=torch.float32).unsqueeze(-1) * 0.3
        return torch.nn.functional.normalize(z, dim=-1), y

    model = ProtoModel(_Enc(), "full", Config()).eval()
    mz, my = dom(600, 0.1); model.memory = MemoryBank(mz[my == 0], mz[my == 1])
    Zv, yv = dom(300, 0.0); Zs, ys = dom(400, 0.0); Zp, yp = dom(400, 0.4)
    val = pd.DataFrame({"label": yv}); src = pd.DataFrame({"label": ys})
    pool = pd.DataFrame({"label": yp, "group": [f"p{i // 2}" for i in range(400)]})
    pool["label"] = pool.groupby("group").label.transform("max")
    rows = evaluate_variants(model, Zv, val, Zs, src, Zp, pool, 42, 4, 5, log=lambda s: None)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    assert len(df) == len(GRID) and np.isfinite(num.values).all(), "non-finite metric"
    print(df[["variant", "val_macro_f1", "pool_src_macro_f1", "pool_src_auroc", "pool_tgt_auroc", "prev75_pi_hat",
              "pool_src_aurc_maxp", "pool_src_aurc_vac"]].round(3).to_string(index=False))
    print("select_tta selftest OK:", len(df), "variants x", num.shape[1], "numeric columns")


if __name__ == "__main__":
    main()
