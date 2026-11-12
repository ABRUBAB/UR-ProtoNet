"""Job entry points used on Kaggle (and locally):

  python -m urproto.jobs cache  --out OUT --assets ASSETS
  python -m urproto.jobs stage1 --out OUT
  python -m urproto.jobs seeds  --out OUT --seeds 42 123 --methods full proto ...
"""

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from .config import Config
from . import data as D
from .models import Encoder
from .train import (embed, meta_train, meta_train_decoupled, stage1_pretrain, stage1_pretrain_multilabel,
                    supervised_logits, train_supervised, build_memory)
from . import evaluate as EV
from . import tta as TTA

METHOD_VARIANT = {"full": "full", "proto": "proto", "proto_mem": "proto_mem", "proto_edl": "proto_edl",
                  "matching": "matching"}
METHOD_NAME = {"full": "UR-ProtoNet", "proto": "VanillaProtoNet", "proto_mem": "Proto+Memory",
               "proto_edl": "Proto+EDL", "matching": "MatchingNet", "simpleshot": "SimpleShot",
               "effnet": "EfficientNet-Binary"}
SUPPORT_OFFSET = {("cxr_val", "source"): 1, ("cxr_test", "source"): 2, ("cxr_test_all", "source"): 2, ("chexpert", "source"): 3,
                  ("chexpert", "target"): 4, ("siim", "source"): 5, ("siim", "target"): 6}


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_file(name, roots=(D.INPUT_ROOT, ".")):
    for r in roots:
        hit = D.find_file(name, r)
        if hit:
            return hit
    return None


def make_cfg(overrides: str) -> Config:
    cfg = Config()
    if overrides:
        for k, v in json.loads(overrides).items():
            setattr(cfg, k, v)
    return cfg


# ----------------------------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------------------------
def job_cache(a):
    log = Logger(os.path.join(a.out, "log.txt"))
    for d, dirs, files in D._walk_dirs(D.INPUT_ROOT, max_depth=4, max_children=50):
        log(f"  INPUT {d} | dirs={dirs[:8]} | files={files[:8]}")
    roots = D.discover_roots()
    log(f"roots: {json.dumps(roots, indent=1)}")
    missing = [k for k in ("nih", "cxr", "chexpert") if not roots.get(k)]
    if missing:
        raise RuntimeError(f"dataset roots not found: {missing}")
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    report = {"roots": roots}
    cfg = make_cfg(a.override)

    cxr = D.cxr_table(roots["cxr"], a.assets)
    log(f"CXR rows {len(cxr)} split counts {cxr['split'].value_counts().to_dict()}")
    cxr = D.build_cache(cxr, "cxr", cache)
    tr, te = cxr[cxr.split == "train"], cxr[cxr.split == "test"]
    report["cxr_counts"] = cxr.groupby(["split", "label"]).size().to_dict()
    report["cxr_test_dhash_in_train"] = int(te["dhash"].isin(set(tr["dhash"])).sum())
    report["cxr_test_patient_in_train"] = int(te["group"].isin(set(tr["group"])).sum())
    log(f"CXR test images whose pixels duplicate a train image: {report['cxr_test_dhash_in_train']} / {len(te)}; "
        f"whose parsed patient id appears in train: {report['cxr_test_patient_in_train']}")

    cx = D.chexpert_table(roots["chexpert"], a.assets, cfg.target_pool_per_class)
    q = cx[cx.role == "query"]
    report["chexpert_query_n"] = int(len(q))
    report["chexpert_label_mismatch"] = int((q["label"] != q["label_published"]).sum())
    report["chexpert_pool_query_patient_overlap"] = int(len(set(q.group) & set(cx[cx.role == "pool"].group)))
    log(f"CheXpert query {len(q)} pool {(cx.role == 'pool').sum()} label mismatches {report['chexpert_label_mismatch']} "
        f"pool/query patient overlap {report['chexpert_pool_query_patient_overlap']}")
    D.build_cache(cx, "chexpert", cache)

    if roots.get("siim_img") and roots.get("siim_lbl"):
        si = D.siim_table(roots["siim_img"], roots["siim_lbl"], cfg.siim_pool_studies_per_class)
        report["siim_counts"] = si.groupby(["role", "label"]).size().to_dict()
        report["siim_pool_query_study_overlap"] = int(len(set(si[si.role == "query"].group) & set(si[si.role == "pool"].group)))
        log(f"SIIM {report['siim_counts']} overlap {report['siim_pool_query_study_overlap']}")
        D.build_cache(si, "siim", cache)

    nih = D.nih_table(roots["nih"], cfg.mem_normal)
    report["nih_counts"] = nih.groupby(["label"]).size().to_dict()
    report["nih_memory"] = nih[nih.in_memory == 1].groupby("label").size().to_dict()
    log(f"NIH {report['nih_counts']} memory {report['nih_memory']}")
    if a.nih_memory_only:
        nih = nih[nih.in_memory == 1].reset_index(drop=True)
    D.build_cache(nih, "nih", cache)

    D.write_json({str(k): str(v) for k, v in report.items()}, os.path.join(cache, "urp_cache_index.json"))
    log("cache done")


def job_cache_nih14(a):
    """CPU job: preprocess the full ChestX-ray14 (all images, 14 finding labels) for multi-label Stage I."""
    log = Logger(os.path.join(a.out, "log.txt"))
    root = D.find_dir_with(["Data_Entry_2017.csv"], D.INPUT_ROOT)
    if not root:
        raise RuntimeError("NIH ChestX-ray14 root (Data_Entry_2017.csv) not found")
    df = D.nih14_table(root)
    cache = os.path.join(a.out, "cache_nih14")
    os.makedirs(cache, exist_ok=True)
    pos = {c[2:]: int(df[c].sum()) for c in df.columns if c.startswith("y_")}
    log(f"NIH14 root {root} rows {len(df)} split {df['split'].value_counts().to_dict()} positives {json.dumps(pos)}")
    out = D.build_cache(df, "nih14", cache)
    D.write_json({"n": int(len(out)), "ok": int(out["ok"].sum()), "labels": D.NIH14, "positives": pos},
                 os.path.join(cache, "urp_nih14_index.json"))
    log(f"nih14 cache done: {int(out['ok'].sum())}/{len(out)} readable")


def _find_encoder(hint: str, name: str = "stage1_encoder.pt") -> str:
    """Stage I checkpoint whose path contains `hint` (several Stage I kernels can be attached to one job)."""
    hits = []
    for r in (D.INPUT_ROOT, "."):
        if os.path.isdir(r):
            hits += [os.path.join(d, name) for d, _, files in D._walk_dirs(r, 7) if name in files]
    hits = sorted(h for h in hits if hint in h.replace("\\", "/"))
    if not hits:
        raise FileNotFoundError(f"no {name} whose path contains '{hint}'")
    return hits[0]


ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")


def _load_all(cache_dir, need_full_nih=False, cfg=None):
    cxr_arr, cxr_df = D.load_cache(cache_dir, "cxr")
    if cfg is not None and getattr(cfg, "cxr_exclude_nih", False):
        keys = D._read_list_gz(os.path.join(ASSETS, "cxr_nih_overlap.csv.gz"))["key"]
        cxr_df = D.cxr_clean_flags(cxr_df, nih_overlap_keys=keys, exclude_nih_from_train=True)
    else:
        cxr_df = D.cxr_clean_flags(cxr_df)
    cxr_df["split_orig"] = cxr_df["split"]
    cxr_df["split"] = cxr_df["eval_split"]          # strict val / test subsets; train unchanged
    nih_arr, nih_df = D.load_cache(cache_dir, "nih", in_ram=need_full_nih)
    mem = nih_df[nih_df.in_memory == 1]
    mem_arr = np.ascontiguousarray(nih_arr[mem["cache_row"].values])
    mem_labels = mem["label"].values
    chex_arr, chex_df = D.load_cache(cache_dir, "chexpert")
    siim = D.load_cache(cache_dir, "siim") if os.path.exists(os.path.join(cache_dir, "siim.npy")) else (None, None)
    return dict(cxr=(cxr_arr, cxr_df), nih=(nih_arr, nih_df), mem=(mem_arr, mem_labels), chex=(chex_arr, chex_df),
                siim=siim)


def _cache_dir(a):
    return a.cache or os.path.dirname(find_file("urp_cache_index.json"))


# ----------------------------------------------------------------------------------------------
# stage1
# ----------------------------------------------------------------------------------------------
def job_stage1(a):
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    cfg = make_cfg(a.override)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if getattr(cfg, "pre_task", "binary") == "multilabel":
        c14 = a.cache or os.path.dirname(find_file("urp_nih14_index.json"))
        nih_arr, nih_df = D.load_cache(c14, "nih14")
        n_excl = 0
        if getattr(cfg, "pre_exclude_cxr_eval", False):  # keep the CXR val / test images out of Stage I
            cx = pd.read_csv(find_file("cxr.csv"), low_memory=False)
            held = set(cx.loc[(cx["split"] != "train") & (cx["ok"] == 1), "dhash"].astype(str))
            m = nih_df["dhash"].astype(str).isin(held).values
            n_excl = int(m.sum())
            nih_df = nih_df[~m].reset_index(drop=True)
        log(f"NIH14 cache {nih_arr.shape} split {nih_df.split.value_counts().to_dict()} excluded as CXR val/test "
            f"duplicates {n_excl} device {dev} cfg {json.dumps(cfg.to_dict())}")
        enc, lg, last_state = stage1_pretrain_multilabel(nih_arr, nih_df, cfg, dev, log)
    else:
        cache = _cache_dir(a)
        nih_arr, nih_df = D.load_cache(cache, "nih")
        log(f"NIH cache {nih_arr.shape} split {nih_df.split.value_counts().to_dict()} device {dev}")
        enc, lg, last_state = stage1_pretrain(nih_arr, nih_df, cfg, dev, log)
    torch.save(enc.state_dict(), os.path.join(a.out, "stage1_encoder.pt"))
    torch.save(last_state, os.path.join(a.out, "stage1_encoder_last_epoch.pt"))
    lg.to_csv(os.path.join(a.out, "stage1_log.csv"), index=False)
    log("stage1 done")


# ----------------------------------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------------------------------
def _rows(df, **conds):
    m = np.ones(len(df), dtype=bool)
    for k, v in conds.items():
        m &= (df[k] == v).values
    return df[m].reset_index(drop=True)


def _pool(df, per_class, seed):
    parts = [df[df.label == c].sample(min(per_class, int((df.label == c).sum())), random_state=seed) for c in (0, 1)]
    return pd.concat(parts).reset_index(drop=True)


def job_seeds(a):
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    cfg = make_cfg(a.override)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    t_start = time.time()
    cache = _cache_dir(a)
    S = _load_all(cache, cfg=cfg)
    cxr_arr, cxr_df = S["cxr"]
    mem_arr, mem_labels = S["mem"]
    chex_arr, chex_df = S["chex"]
    siim_arr, siim_df = S["siim"]
    enc_path = a.encoder or find_file("stage1_encoder.pt")
    enc_state = torch.load(enc_path, map_location="cpu")
    log(f"device {dev} cache {cache} encoder {enc_path} cfg {json.dumps(cfg.to_dict())}")

    sets = {
        "cxr_val": (cxr_arr, _rows(cxr_df, split="val")),
        "cxr_test": (cxr_arr, _rows(cxr_df, split="test")),
        "cxr_test_all": (cxr_arr, _rows(cxr_df, split_orig="test")),
        "chexpert": (chex_arr, _rows(chex_df, role="query")),
    }
    pools = {"source": (cxr_arr, _pool(_rows(cxr_df, split="train"), cfg.source_pool_per_class, 12345)),
             "chexpert": (chex_arr, _rows(chex_df, role="pool"))}
    if siim_arr is not None:
        sets["siim"] = (siim_arr, _rows(siim_df, role="query"))
        pools["siim"] = (siim_arr, _rows(siim_df, role="pool"))
    protocols = {"cxr_val": ["source"], "cxr_test": ["source"], "cxr_test_all": ["source"],
                 "chexpert": ["source", "target"], "siim": ["source", "target"]}
    log("sets: " + ", ".join(f"{k}={len(v[1])}" for k, v in sets.items()) + " | pools: " +
        ", ".join(f"{k}={len(v[1])}" for k, v in pools.items()))

    for seed in a.seeds:
        pred_rows, met_rows, sweep_rows, retr_rows = [], [], [], []
        for method in a.methods:
            if a.budget_h and (time.time() - t_start) / 3600 > a.budget_h:
                log(f"time budget reached, skipping {method} seed {seed}")
                continue
            t0 = time.time()
            try:
                out = _run_method(method, seed, cfg, dev, enc_state, S, sets, pools, protocols, log, a)
                pred_rows += out["preds"]; met_rows += out["metrics"]; sweep_rows += out["sweeps"]; retr_rows += out["retrieval"]
                if out.get("log") is not None:
                    out["log"].to_csv(os.path.join(a.out, f"trainlog_{method}_seed{seed}.csv"), index=False)
                log(f"== {METHOD_NAME[method]} seed {seed} done in {(time.time() - t0) / 60:.1f} min")
            except Exception:
                log(f"!! {method} seed {seed} failed:\n{traceback.format_exc()}")
            # write incrementally
            if pred_rows:
                pd.concat(pred_rows).to_csv(os.path.join(a.out, f"preds_seed{seed}.csv.gz"), index=False)
            for name, rows in (("metrics", met_rows), ("sweeps", sweep_rows), ("retrieval", retr_rows)):
                if rows:
                    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{name}_seed{seed}.csv"), index=False)
    log(f"all done in {(time.time() - t_start) / 3600:.2f} h")


def _summ(seed, method, dataset, protocol, df, thr, T, extra=None):
    rows = []
    for level, d in (("image", df), ("group", EV.group_aggregate(df))):
        m = EV.compute_metrics(d["label"].values, d["p_cal"].values, d["u"].values, thr)
        m.update({"seed": seed, "method": METHOD_NAME.get(method, method), "dataset": dataset, "protocol": protocol,
                  "level": level, "threshold": thr, "T": T})
        if extra:
            m.update(extra)
        rows.append(m)
    return rows


def _run_method(method, seed, cfg, dev, enc_state, S, sets, pools, protocols, log, a):
    cxr_arr, cxr_df = S["cxr"]
    mem_arr, mem_labels = S["mem"]
    res = {"preds": [], "metrics": [], "sweeps": [], "retrieval": [], "log": None}
    name = METHOD_NAME[method]

    # ---------------- supervised baseline ----------------
    if method == "effnet":
        model, lg = train_supervised(enc_state, cxr_arr, cxr_df, cfg, seed, dev, log, a.max_epochs)
        res["log"] = lg
        logits = {k: supervised_logits(model, arr, df["cache_row"].values, dev, cfg.eval_bs) for k, (arr, df) in sets.items()}
        y_val = sets["cxr_val"][1]["label"].values
        T = EV.fit_T_binary(logits["cxr_val"], y_val)
        p_val = 1 / (1 + np.exp(-logits["cxr_val"] / T))
        thr = EV.best_threshold(y_val, p_val)
        for k, (arr, df) in sets.items():
            p_raw = 1 / (1 + np.exp(-logits[k])); p_cal = 1 / (1 + np.exp(-logits[k] / T))
            d = pd.DataFrame({"key": df["key"], "group": df["group"], "label": df["label"], "p_raw": p_raw,
                              "p_cal": p_cal, "u": EV.binary_entropy(p_cal)})
            for proto in protocols[k]:
                res["metrics"] += _summ(seed, method, k, proto, d, thr, T)
            d.insert(0, "protocol", "n/a"); d.insert(0, "dataset", k); d.insert(0, "method", name); d.insert(0, "seed", seed)
            res["preds"].append(d)
        return res

    # ---------------- embedding-based methods ----------------
    if method == "simpleshot":
        enc = Encoder(cfg.emb_dim, cfg.hidden_dim, pretrained=False).to(dev)
        enc.load_state_dict(enc_state)
        model = EV.SimpleShotModel()
        src_arr, src_df = pools["source"]
        mu = embed(enc, src_arr, src_df["cache_row"].values, dev, cfg.eval_bs, raw_features=True).mean(0, keepdim=True)

        def emb_fn(arr, rows):
            return torch.nn.functional.normalize(embed(enc, arr, rows, dev, cfg.eval_bs, raw_features=True) - mu, dim=-1)
    else:
        if method == "full" and getattr(cfg, "decoupled", False):
            model, lg = meta_train_decoupled(enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed, dev, log, a.max_epochs)
        else:
            model, lg = meta_train(METHOD_VARIANT[method], enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed, dev,
                                   log, a.max_epochs)
        res["log"] = lg

        def emb_fn(arr, rows):
            return embed(model.encoder, arr, rows, dev, cfg.eval_bs)

    Z = {k: emb_fn(arr, df["cache_row"].values) for k, (arr, df) in sets.items()}
    PZ = {k: (emb_fn(arr, df["cache_row"].values), df["label"].values) for k, (arr, df) in pools.items()}

    def pool_for(dataset, protocol):
        return PZ["source"] if protocol == "source" else PZ[dataset]

    ens = {}
    for k in sets:
        for proto in protocols[k]:
            pz, py = pool_for(k, proto)
            ens[(k, proto)] = EV.proto_ensemble(model, Z[k], pz, py, cfg.n_support_sets, cfg.k_shot,
                                                seed * 100 + SUPPORT_OFFSET[(k, proto)])
    y_val = sets["cxr_val"][1]["label"].values
    T = EV.fit_T_ensemble(ens[("cxr_val", "source")]["logits"], y_val)
    extra = {}
    if hasattr(model, "head") and hasattr(model.head, "bounds"):
        extra = {f"head_{k}": v for k, v in model.head.bounds().items()}
    for (k, proto), e in ens.items():
        df = sets[k][1]
        d = pd.DataFrame({"key": df["key"], "group": df["group"], "label": df["label"], "p_raw": e["p_raw"],
                          "p_cal": EV.calibrated_from_logits(e["logits"], T), "u": e["u"]})
        ex = dict(extra, beta_normal=float(e["beta"][0]), beta_pneumonia=float(e["beta"][1]),
                  memsim_normal=float(e["mem_sim"][0]), memsim_pneumonia=float(e["mem_sim"][1]))
        if getattr(model, "uses_edl", False):
            unc = TTA.evidential_uncertainties(e["logits"], T)
            d["u_diss"], d["u_total"] = unc["dissonance"], unc["total"]
        res["metrics"] += _summ(seed, method, k, proto, d, 0.5, T, ex)
        d.insert(0, "protocol", proto); d.insert(0, "dataset", k); d.insert(0, "method", name); d.insert(0, "seed", seed)
        res["preds"].append(d)

    # ---------------- transductive inference (unlabelled query batch; declared part of the method) ----------------
    if getattr(a, "tta", "") and method in ("full", "proto"):
        tcfg = TTA.TTAConfig(**json.loads(a.tta))
        if not getattr(model, "uses_memory", False):
            tcfg = dataclasses.replace(tcfg, tcr=0.0)
        tcfg = dataclasses.replace(tcfg, em_temp=T)
        ens_t = {}
        for k in sets:
            for proto in protocols[k]:
                pz, py = pool_for(k, proto)
                ens_t[(k, proto)] = TTA.proto_ensemble_tta(model, Z[k], pz, py, cfg.n_support_sets, cfg.k_shot,
                                                           seed * 100 + SUPPORT_OFFSET[(k, proto)], tta=tcfg)
        ev_ = ens_t[("cxr_val", "source")]

        def _nll(Tt):
            pp = np.clip(TTA.calibrated_from_logits_prior(ev_["logits"], Tt, ev_["log_prior"]), 1e-7, 1 - 1e-7)
            return float(-np.mean(y_val * np.log(pp) + (1 - y_val) * np.log(1 - pp)))

        Tt = EV.fit_temperature(_nll)
        tname = f"{name} (transductive)"
        for (k, proto), e in ens_t.items():
            df = sets[k][1]
            d = pd.DataFrame({"key": df["key"], "group": df["group"], "label": df["label"], "p_raw": e["p_raw"],
                              "p_cal": TTA.calibrated_from_logits_prior(e["logits"], Tt, e["log_prior"]), "u": e["u"]})
            if getattr(model, "uses_edl", False):
                unc = TTA.evidential_uncertainties(e["logits"], Tt)
                d["u_diss"], d["u_total"] = unc["dissonance"], unc["total"]
            res["metrics"] += _summ(seed, tname, k, proto, d, 0.5, Tt, {"pi_hat": e["pi_hat"], "shift_norm": e["shift_norm"],
                                                                         "tta": tcfg.tag()})
            d.insert(0, "protocol", proto); d.insert(0, "dataset", k); d.insert(0, "method", tname); d.insert(0, "seed", seed)
            res["preds"].append(d)

    eval_sets = [k for k in ("cxr_test", "chexpert", "siim") if k in sets]
    # ---------------- K-shot sweep (all embedding methods) ----------------
    for K in cfg.kshot_eval:
        for k in eval_sets:
            for proto in protocols[k]:
                pz, py = pool_for(k, proto)
                e = EV.proto_ensemble(model, Z[k], pz, py, cfg.n_support_sets, K, seed * 100 + SUPPORT_OFFSET[(k, proto)] + 7919 * K)
                df = sets[k][1]
                d = pd.DataFrame({"group": df["group"], "label": df["label"], "p_raw": e["p_raw"],
                                  "p_cal": EV.calibrated_from_logits(e["logits"], T), "u": e["u"]})
                for r in _summ(seed, method, k, proto, d, 0.5, T, {"sweep": "kshot", "setting": f"K={K}"}):
                    res["sweeps"].append(r)

    # ---------------- memory analyses (UR-ProtoNet only) ----------------
    if method == "full":
        for k in ["cxr_val"] + eval_sets:
            r = EV.retrieval_quality(model.memory, Z[k], sets[k][1]["label"].values, cfg.mem_topk)
            r.update({"seed": seed, "dataset": k})
            for proto in protocols[k]:
                e = ens[(k, proto)]
                r[f"beta_{proto}"] = float(np.mean(e["beta"])); r[f"memsim_{proto}"] = float(np.mean(e["mem_sim"]))
            res["retrieval"].append(r)
        for k in eval_sets:
            for proto in protocols[k]:
                tz, ty = (PZ["source"] if k == "cxr_test" else PZ[k])
                variants = EV.memory_variants(model.memory, tz, ty, seed)
                pz, py = pool_for(k, proto)
                for vname, mem in variants.items():
                    e = EV.proto_ensemble(model, Z[k], pz, py, cfg.n_support_sets, cfg.k_shot,
                                          seed * 100 + SUPPORT_OFFSET[(k, proto)], memory=mem,
                                          force_beta=1.0 if mem is None else None)
                    df = sets[k][1]
                    d = pd.DataFrame({"group": df["group"], "label": df["label"], "p_raw": e["p_raw"],
                                      "p_cal": EV.calibrated_from_logits(e["logits"], T), "u": e["u"]})
                    ex = {"sweep": "memory", "setting": vname, "beta_mean": float(np.mean(e["beta"])),
                          "memsim_mean": float(np.mean(e["mem_sim"])),
                          "mem_size": "0" if mem is None else "/".join(map(str, mem.sizes))}
                    res["sweeps"] += _summ(seed, method, k, proto, d, 0.5, T, ex)
        mdir = os.path.join(a.out, "models"); os.makedirs(mdir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "T": T, "cfg": cfg.to_dict()}, os.path.join(mdir, f"full_seed{seed}.pt"))
        if seed == a.seeds[0]:
            rng = np.random.RandomState(0)
            emb = {}
            for k in eval_sets:
                n = min(1500, Z[k].size(0)); idx = rng.choice(Z[k].size(0), n, replace=False)
                emb[k] = (Z[k][idx].cpu().numpy().astype(np.float16), sets[k][1]["label"].values[idx])
            mz = torch.cat(model.memory.mem); ml = np.r_[np.zeros(model.memory.sizes[0]), np.ones(model.memory.sizes[1])]
            idx = rng.choice(mz.size(0), min(1500, mz.size(0)), replace=False)
            emb["nih_memory"] = (mz[idx].cpu().numpy().astype(np.float16), ml[idx])
            np.savez_compressed(os.path.join(a.out, f"embeddings_seed{seed}.npz"),
                                **{f"{k}_z": v[0] for k, v in emb.items()}, **{f"{k}_y": v[1] for k, v in emb.items()})
    return res


def job_select(a):
    """Design selection WITHOUT test data: CXR-Val (source supports) and the CheXpert support pool
    (patient-disjoint from every query image) split by patient into a support half and a query half."""
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    cache = _cache_dir(a)
    variants = json.loads(a.variants)
    rows = []
    for v in variants:
        enc_path = _find_encoder(v["encoder_hint"]) if v.get("encoder_hint") else (a.encoder or find_file("stage1_encoder.pt"))
        enc_state = torch.load(enc_path, map_location="cpu")
        log(f"variant {v['name']}: encoder {enc_path} override {json.dumps(v.get('override', {}))}")
        cfg = make_cfg(json.dumps(v.get("override", {})))
        S = _load_all(cache, cfg=cfg)
        cxr_arr, cxr_df = S["cxr"]; mem_arr, mem_labels = S["mem"]; chex_arr, chex_df = S["chex"]
        val = _rows(cxr_df, split="val")
        src = _pool(_rows(cxr_df, split="train"), cfg.source_pool_per_class, 12345)
        pool = _rows(chex_df, role="pool")
        for seed in a.seeds:
            t0 = time.time()
            meth = v["method"]
            if meth == "full_decoupled":
                model, lg = meta_train_decoupled(enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed, dev, log, a.max_epochs)
            else:
                model, lg = meta_train(METHOD_VARIANT[meth], enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed, dev, log,
                                       a.max_epochs)
            Zv = embed(model.encoder, cxr_arr, val.cache_row.values, dev); Zs = embed(model.encoder, cxr_arr, src.cache_row.values, dev)
            Zp = embed(model.encoder, chex_arr, pool.cache_row.values, dev)
            ev = EV.proto_ensemble(model, Zv, Zs, src.label.values, cfg.n_support_sets, cfg.k_shot, seed * 100 + 1)
            T = EV.fit_T_ensemble(ev["logits"], val.label.values)
            res = {"variant": v["name"], "method": meth, "seed": seed, "epochs": len(lg),
                   "n_train": int((cxr_df.split == "train").sum()), "n_val": len(val)}
            m = EV.compute_metrics(val.label.values, EV.calibrated_from_logits(ev["logits"], T), ev["u"])
            res.update({f"val_{k}": m[k] for k in ("macro_f1", "auroc", "ece")})
            # CheXpert pool, source supports
            ep = EV.proto_ensemble(model, Zp, Zs, src.label.values, cfg.n_support_sets, cfg.k_shot, seed * 100 + 3)
            d = pool.assign(p_cal=EV.calibrated_from_logits(ep["logits"], T), p_raw=ep["p_raw"], u=ep["u"])
            g = EV.group_aggregate(d)
            m = EV.compute_metrics(g.label.values, g.p_cal.values, g.u.values)
            res.update({f"pool_src_{k}": m[k] for k in ("macro_f1", "auroc", "ece")})
            # CheXpert pool, target supports from the other half of the pool (split by patient)
            pats = np.array(sorted(pool.group.unique())); rng = np.random.RandomState(seed)
            half = set(rng.choice(pats, len(pats) // 2, replace=False))
            f1s, aucs = [], []
            for sup_mask in (pool.group.isin(half).values, ~pool.group.isin(half).values):
                qi, si = np.where(~sup_mask)[0], np.where(sup_mask)[0]
                e = EV.proto_ensemble(model, Zp[torch.as_tensor(qi, device=dev)], Zp[torch.as_tensor(si, device=dev)],
                                      pool.label.values[si], cfg.n_support_sets, cfg.k_shot, seed * 100 + 4)
                dq = pool.iloc[qi].assign(p_cal=EV.calibrated_from_logits(e["logits"], T), p_raw=e["p_raw"], u=e["u"])
                gq = EV.group_aggregate(dq)
                mm = EV.compute_metrics(gq.label.values, gq.p_cal.values, gq.u.values)
                f1s.append(mm["macro_f1"]); aucs.append(mm["auroc"])
            res.update({"pool_tgt_macro_f1": float(np.mean(f1s)), "pool_tgt_auroc": float(np.mean(aucs)),
                        "beta": float(np.mean(ep["beta"])), "minutes": (time.time() - t0) / 60})
            rows.append(res)
            log(f"SELECT {json.dumps(res)}")
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "select_results.csv"), index=False)


def job_xai(a):
    """Real Grad-CAM / Grad-CAM++ / LIME for the first-seed UR-ProtoNet model + UMAP coordinates."""
    from .models import ProtoModel
    from . import xai as X
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "log.txt"))
    cfg = make_cfg(a.override)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = _load_all(_cache_dir(a), cfg=cfg)
    cxr_arr, cxr_df = S["cxr"]
    chex_arr, chex_df = S["chex"]
    mem_arr, mem_labels = S["mem"]
    seed = a.seeds[0]
    ck = torch.load(a.model or find_file(f"full_seed{seed}.pt"), map_location="cpu")
    model = ProtoModel(Encoder(cfg.emb_dim, cfg.hidden_dim, pretrained=False), "full", cfg).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    model.memory = build_memory(model.encoder, mem_arr, mem_labels, dev, cfg.eval_bs)
    T = float(ck.get("T", 1.0))
    rng = np.random.RandomState(seed)
    tr = _rows(cxr_df, split="train")
    sup = pd.concat([tr[tr.label == c].sample(cfg.k_shot, random_state=seed) for c in (0, 1)])
    s_z = embed(model.encoder, cxr_arr, sup["cache_row"].values, dev)
    with torch.no_grad():
        protos, info = model.prototypes(s_z, torch.as_tensor(sup["label"].values, device=dev))
    out = {}
    for name, (arr, df) in {"cxr_test": (cxr_arr, _rows(cxr_df, split="test")),
                            "chexpert": (chex_arr, _rows(chex_df, role="query"))}.items():
        z = embed(model.encoder, arr, df["cache_row"].values, dev)
        with torch.no_grad():
            p = (model.head(z, protos)["logits"] / T).softmax(-1)[:, 1].cpu().numpy()
        df = df.assign(p=p, correct=((p >= 0.5).astype(int) == df["label"].values))
        pick = []
        for c in (1, 0):
            cand = df[(df.label == c) & df.correct]
            pick.append(cand.sample(3, random_state=seed))
        pick = pd.concat(pick)
        imgs = arr[pick["cache_row"].values]
        cam, campp, pred, prob = X.explain_cams(model, protos, imgs, dev)
        lime_res = X.explain_lime(model, protos, imgs, dev, T=T) if a.lime else []
        out[name] = dict(imgs=imgs, cam=cam, campp=campp, pred=pred, prob=prob, label=pick["label"].values,
                         keys=pick["key"].values.astype(str),
                         lime_pos=np.stack([r[1] for r in lime_res]) if lime_res else np.zeros(0),
                         lime_all=np.stack([r[2] for r in lime_res]) if lime_res else np.zeros(0))
        log(f"xai {name}: keys {list(pick['key'])} probs {np.round(prob, 3).tolist()}")
    np.savez_compressed(os.path.join(a.out, f"xai_seed{seed}.npz"),
                        **{f"{d}_{k}": v for d, dd in out.items() for k, v in dd.items()})
    # UMAP of encoder embeddings (dataset and class structure)
    try:
        import umap
        rows = []
        for name, (arr, df) in {"CXR": (cxr_arr, _rows(cxr_df, split="test")), "CheXpert": (chex_arr, _rows(chex_df, role="query")),
                                "SIIM": S["siim"] if S["siim"][0] is not None else (None, None)}.items():
            if arr is None:
                continue
            df = df.sample(min(1500, len(df)), random_state=seed)
            z = embed(model.encoder, arr, df["cache_row"].values, dev).cpu().numpy()
            rows.append((name, z, df["label"].values))
        mz = torch.cat(model.memory.mem).cpu().numpy()
        ml = np.r_[np.zeros(model.memory.sizes[0]), np.ones(model.memory.sizes[1])]
        idx = rng.choice(len(mz), 1500, replace=False)
        rows.append(("NIH memory", mz[idx], ml[idx]))
        Z = np.concatenate([r[1] for r in rows])
        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(Z)
        lab = np.concatenate([r[2] for r in rows]); dsn = np.concatenate([[r[0]] * len(r[1]) for r in rows])
        pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "dataset": dsn, "label": lab}).to_csv(
            os.path.join(a.out, f"umap_seed{seed}.csv"), index=False)
        log("umap done")
    except Exception:
        log("umap failed: " + traceback.format_exc())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("job", choices=["cache", "cache_nih14", "stage1", "seeds", "xai", "select", "selecttta"])
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--tta", default="")
    p.add_argument("--variants", default="[]")
    p.add_argument("--model", default=None)
    p.add_argument("--lime", action="store_true")
    p.add_argument("--out", default="/kaggle/working/out")
    p.add_argument("--assets", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--encoder", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[42])
    p.add_argument("--methods", nargs="*", default=["full", "proto", "proto_mem", "proto_edl", "matching", "simpleshot", "effnet"])
    p.add_argument("--override", default="")
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--budget_h", type=float, default=None)
    p.add_argument("--nih_memory_only", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.job == "selecttta":
        from .select_tta import main as _st
        argv2 = ["--out", a.out, "--seeds", *map(str, a.seeds), "--methods", *a.methods]
        if a.ckpt_dir: argv2 += ["--ckpt_dir", a.ckpt_dir]
        if a.cache: argv2 += ["--cache", a.cache]
        if a.encoder: argv2 += ["--encoder", a.encoder]
        return _st(argv2)
    {"cache": job_cache, "cache_nih14": job_cache_nih14, "stage1": job_stage1, "seeds": job_seeds, "xai": job_xai,
     "select": job_select}[a.job](a)


if __name__ == "__main__":
    main()
