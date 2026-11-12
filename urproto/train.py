"""Training loops: Stage I (NIH pre-training), Stage III (episodic meta-training) and the
supervised EfficientNet-Binary baseline. All loops read images from the uint8 RAM cache."""

import copy
import math
import random
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .data import to_input
from .models import Encoder, MemoryBank, ProtoModel, SupervisedClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast(
        device_type="cpu", enabled=False)


def _lr_factor(ep: int, total: int, warmup: int) -> float:
    if warmup > 0 and ep <= warmup:
        return ep / warmup
    t = (ep - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def embed(encoder: Encoder, arr: np.ndarray, rows: np.ndarray, device, bs: int = 256, raw_features: bool = False):
    was = encoder.training
    encoder.eval()
    outs = []
    for i in range(0, len(rows), bs):
        x = to_input(arr[rows[i:i + bs]], device)
        with _amp(device):
            z = encoder.features(x).float() if raw_features else encoder(x)
        outs.append(z.float())
    encoder.train(was)
    return torch.cat(outs, 0) if outs else torch.zeros(0, encoder.emb_dim, device=device)


def build_memory(encoder, mem_arr, mem_labels, device, bs=256) -> MemoryBank:
    z = embed(encoder, mem_arr, np.arange(len(mem_arr)), device, bs)
    lab = torch.as_tensor(mem_labels, device=device)
    return MemoryBank(z[lab == 0], z[lab == 1])


# ----------------------------------------------------------------------------------------------
# Stage I
# ----------------------------------------------------------------------------------------------
def stage1_pretrain(nih_arr, nih_df, cfg, device, log_cb=print):
    set_seed(cfg.seed)
    enc = Encoder(cfg.emb_dim, cfg.hidden_dim, pretrained=True).to(device)
    clf = nn.Linear(cfg.emb_dim, 1).to(device)
    tr = nih_df.index[nih_df["split"] == "train"].values
    va = nih_df.index[nih_df["split"] == "val"].values
    rows_tr, rows_va = nih_df.loc[tr, "cache_row"].values, nih_df.loc[va, "cache_row"].values
    y_tr, y_va = nih_df.loc[tr, "label"].values, nih_df.loc[va, "label"].values
    pos_w = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    params = list(enc.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=cfg.pre_lr, weight_decay=cfg.pre_wd)
    steps_per_ep = int(math.ceil(len(rows_tr) / cfg.pre_bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pre_epochs * steps_per_ep)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    rng = np.random.RandomState(cfg.seed)
    log = []
    best_auc, best_state, best_ep = -1.0, None, 0
    for ep in range(1, cfg.pre_epochs + 1):
        t0 = time.time()
        enc.train(); clf.train()
        perm = rng.permutation(len(rows_tr))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), cfg.pre_bs):
            b = perm[i:i + cfg.pre_bs]
            if len(b) < 2:
                continue
            x = to_input(nih_arr[rows_tr[b]], device, augment=True, aug_mode=getattr(cfg, "pre_aug_mode", "basic"))
            y = torch.as_tensor(y_tr[b], device=device).float()
            with _amp(device):
                z = enc(x)
            logit = clf(z.float()).squeeze(-1)
            loss = crit(logit, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(b); n += len(b)
            correct += ((logit > 0).long() == y.long()).sum().item()
        # monitoring split
        enc.eval(); clf.eval()
        with torch.no_grad():
            zs = embed(enc, nih_arr, rows_va, device, cfg.eval_bs)
            lv = clf(zs).squeeze(-1)
            yv = torch.as_tensor(y_va, device=device).float()
            vloss = crit(lv, yv).item()
            pv = torch.sigmoid(lv).cpu().numpy()
        pred = (pv >= 0.5).astype(int)
        row = {"epoch": ep, "train_loss": tot / max(1, n), "train_acc": correct / max(1, n), "val_loss": vloss,
               "val_acc": float((pred == y_va).mean()), "val_auroc": float(roc_auc_score(y_va, pv)),
               "val_bal_acc": float(0.5 * ((pred[y_va == 1] == 1).mean() + (pred[y_va == 0] == 0).mean())),
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0, "pos_weight": pos_w}
        if row["val_auroc"] > best_auc:
            best_auc, best_ep = row["val_auroc"], ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in enc.state_dict().items()})
        row["best_epoch_so_far"] = best_ep
        log.append(row)
        log_cb(f"[StageI] ep {ep:02d} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))
    last_state = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    enc.load_state_dict(best_state)  # checkpoint with the highest AUROC on held-out NIH patients
    log_cb(f"[StageI] selected epoch {best_ep} (held-out AUROC {best_auc:.4f})")
    return enc, pd.DataFrame(log), last_state


def stage1_pretrain_multilabel(nih_arr, nih_df, cfg, device, log_cb=print):
    """Stage I on the full ChestX-ray14 with the 14 finding labels (multi-label BCE, cfg.pre_task="multilabel").

    The encoder is the one used by the binary variant; only the linear classifier has 14 outputs. The
    checkpoint with the highest mean AUROC over the findings on held-out NIH patients is returned."""
    set_seed(cfg.seed)
    ycols = [c for c in nih_df.columns if c.startswith("y_")]
    enc = Encoder(cfg.emb_dim, cfg.hidden_dim, pretrained=True).to(device)
    clf = nn.Linear(cfg.emb_dim, len(ycols)).to(device)
    tr = nih_df.index[nih_df["split"] == "train"].values
    va = nih_df.index[nih_df["split"] == "val"].values
    rows_tr, rows_va = nih_df.loc[tr, "cache_row"].values, nih_df.loc[va, "cache_row"].values
    Y_tr = nih_df.loc[tr, ycols].values.astype(np.float32)
    Y_va = nih_df.loc[va, ycols].values.astype(np.float32)
    crit = nn.BCEWithLogitsLoss()
    scale = float(getattr(cfg, "pre_logit_scale", 1.0))  # logits of a linear layer on a unit vector are bounded by |w|
    aux = nn.Linear(enc.backbone.num_features, len(ycols)).to(device) if getattr(cfg, "pre_aux_head", False) else None
    params = list(enc.parameters()) + list(clf.parameters()) + (list(aux.parameters()) if aux is not None else [])
    opt = torch.optim.Adam(params, lr=cfg.pre_lr, weight_decay=cfg.pre_wd)
    steps_per_ep = int(math.ceil(len(rows_tr) / cfg.pre_bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pre_epochs * steps_per_ep)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    rng = np.random.RandomState(cfg.seed)
    aug_mode = getattr(cfg, "pre_aug_mode", "basic")
    log = []
    best_auc, best_state, best_ep = -1.0, None, 0
    for ep in range(1, cfg.pre_epochs + 1):
        t0 = time.time()
        enc.train(); clf.train()
        perm = rng.permutation(len(rows_tr))
        tot, n = 0.0, 0
        for i in range(0, len(perm), cfg.pre_bs):
            b = perm[i:i + cfg.pre_bs]
            if len(b) < 2:
                continue
            x = to_input(nih_arr[rows_tr[b]], device, augment=True, aug_mode=aug_mode)
            y = torch.as_tensor(Y_tr[b], device=device)
            with _amp(device):
                f = enc.features(x)
                z = F.normalize(enc.proj(f).float(), dim=-1)  # same as enc(x)
            loss = crit(clf(z.float() * scale), y)
            if aux is not None:  # auxiliary head on the pooled backbone features, discarded after Stage I
                loss = loss + crit(aux(f.float()), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(b); n += len(b)
        enc.eval(); clf.eval()
        with torch.no_grad():
            lv = clf(embed(enc, nih_arr, rows_va, device, cfg.eval_bs) * scale)
            vloss = crit(lv, torch.as_tensor(Y_va, device=device)).item()
            pv = torch.sigmoid(lv).cpu().numpy()
        aucs = {c[2:]: float(roc_auc_score(Y_va[:, j], pv[:, j])) for j, c in enumerate(ycols)
                if 0 < Y_va[:, j].sum() < len(Y_va)}
        mean_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        row = {"epoch": ep, "train_loss": tot / max(1, n), "val_loss": vloss, "val_auroc": mean_auc,
               "val_auroc_pneumonia": aucs.get("Pneumonia", float("nan")), "lr": opt.param_groups[0]["lr"],
               "seconds": time.time() - t0}
        row.update({f"auc_{k}": v for k, v in aucs.items()})
        if best_state is None or mean_auc > best_auc:
            best_auc, best_ep = mean_auc, ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in enc.state_dict().items()})
        row["best_epoch_so_far"] = best_ep
        log.append(row)
        log_cb(f"[StageI-ML] ep {ep:02d} train_loss={row['train_loss']:.4f} val_loss={vloss:.4f} "
               f"mean_auroc={mean_auc:.4f} pneumonia_auroc={row['val_auroc_pneumonia']:.4f} ({row['seconds']:.0f}s)")
    last_state = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    enc.load_state_dict(best_state)
    log_cb(f"[StageI-ML] selected epoch {best_ep} (held-out mean AUROC {best_auc:.4f})")
    return enc, pd.DataFrame(log), last_state


@torch.no_grad()
def _ema_update(ema, model, decay: float):
    """ema <- decay * ema + (1 - decay) * model for every floating-point parameter and buffer."""
    msd = model.state_dict()
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(msd[k].detach(), alpha=1.0 - decay)
        else:
            v.copy_(msd[k])


# ----------------------------------------------------------------------------------------------
# Stage III: episodic meta-training (UR-ProtoNet, ablations, VanillaProtoNet, MatchingNet)
# ----------------------------------------------------------------------------------------------
class EpisodeSampler:
    def __init__(self, rows: np.ndarray, labels: np.ndarray):
        self.by_class = [rows[labels == c] for c in (0, 1)]

    def sample(self, rng: np.random.RandomState, k: int, q: int):
        s, sy, qq, qy = [], [], [], []
        for c in (0, 1):
            pool = self.by_class[c]
            pick = rng.choice(pool, size=k + q, replace=len(pool) < k + q)
            s.append(pick[:k]); qq.append(pick[k:])
            sy += [c] * k; qy += [c] * q
        return np.concatenate(s), np.array(sy), np.concatenate(qq), np.array(qy)


def _episode_forward(model, arr, ep, device, augment, force_beta=None, aug_mode="basic", q_arr=None):
    s, sy, q, qy = ep
    if q_arr is None:
        x = to_input(arr[np.concatenate([s, q])], device, augment=augment, aug_mode=aug_mode)
    else:  # cross-domain episode: supports from `arr`, queries from `q_arr`
        x = torch.cat([to_input(arr[s], device, augment=augment, aug_mode=aug_mode),
                       to_input(q_arr[q], device, augment=augment, aug_mode=aug_mode)], 0)
    with _amp(device):
        z = model.encoder(x)
    z = z.float()
    s_z, q_z = z[:len(s)], z[len(s):]
    sy_t = torch.as_tensor(sy, device=device)
    qy_t = torch.as_tensor(qy, device=device)
    return model.predict(q_z, s_z, sy_t, force_beta=force_beta), qy_t


def meta_train(variant: str, enc_state: Dict, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed: int, device,
               log_cb=print, max_epochs: Optional[int] = None):
    set_seed(seed)
    enc = Encoder(cfg.emb_dim, cfg.hidden_dim, pretrained=False)
    enc.load_state_dict(enc_state)
    model = ProtoModel(enc, variant, cfg).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": cfg.meta_lr},
                             {"params": model.head_params(), "lr": cfg.head_lr}], weight_decay=cfg.meta_wd)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    tr = cxr_df[cxr_df["split"] == "train"]
    va = cxr_df[cxr_df["split"] == "val"]
    s_tr = EpisodeSampler(tr["cache_row"].values, tr["label"].values)
    s_va = EpisodeSampler(va["cache_row"].values, va["label"].values)
    rng_val = np.random.RandomState(seed + 1)
    val_eps = [s_va.sample(rng_val, cfg.k_shot, cfg.q_query) for _ in range(cfg.val_episodes)]
    rng = np.random.RandomState(seed)
    E = max_epochs or cfg.meta_epochs
    best, best_state, wait, log = -float("inf"), None, 0, []
    aug_mode = getattr(cfg, "aug_mode", "basic")
    ema_decay = float(getattr(cfg, "ema_decay", 0.0))
    xdom = float(getattr(cfg, "xdom_frac", 0.0))
    s_mem = EpisodeSampler(np.arange(len(mem_labels)), np.asarray(mem_labels)) if xdom > 0 else None
    ema, n_upd = None, 0
    if ema_decay > 0:  # weight averaging: the averaged model is validated, selected and returned
        ema = copy.deepcopy(model).eval()
        for prm in ema.parameters():
            prm.requires_grad_(False)
    for ep in range(1, E + 1):
        t0 = time.time()
        f = _lr_factor(ep, E, cfg.warmup_epochs)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        if model.uses_memory:
            model.memory = build_memory(model.encoder, mem_arr, mem_labels, device, cfg.eval_bs)
        kl_w = cfg.edl_lambda_max * min(1.0, ep / cfg.edl_anneal_epochs)
        model.train()
        losses, ces, kls, preds, tgts, uncs, betas = [], [], [], [], [], [], []
        for _ in range(cfg.train_episodes):
            k = cfg.k_shot if cfg.shot_sampling == "fixed" else int(rng.randint(1, cfg.k_shot + 1))
            fb = 1.0 if (model.uses_memory and getattr(cfg, "mem_dropout", 0.0) > 0 and rng.rand() < cfg.mem_dropout) else None
            ep_, q_arr = s_tr.sample(rng, k, cfg.q_query), None
            if s_mem is not None and rng.rand() < xdom:  # CXR supports, queries from the NIH memory images
                _, _, qm, qym = s_mem.sample(rng, 0, cfg.q_query)
                ep_, q_arr = (ep_[0], ep_[1], qm, qym), mem_arr
            out, qy = _episode_forward(model, cxr_arr, ep_, device, augment=True, force_beta=fb, aug_mode=aug_mode,
                                       q_arr=q_arr)
            loss, ce, kl = model.loss(out, qy, kl_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if ema is not None:
                n_upd += 1
                _ema_update(ema, model, min(ema_decay, (1.0 + n_upd) / (10.0 + n_upd)))
            losses.append(loss.item()); ces.append(ce.item()); kls.append(float(kl))
            preds.append(out["probs"].argmax(-1).cpu().numpy()); tgts.append(qy.cpu().numpy())
            uncs.append(out["uncertainty"].detach().mean().item()); betas.append(out["beta"].detach().mean().item())
        model.eval()
        ev_model = model
        if ema is not None:
            ev_model = ema
            if model.uses_memory:
                ema.memory = build_memory(ema.encoder, mem_arr, mem_labels, device, cfg.eval_bs)
        vp, vt, vnll, vu, vb = [], [], [], [], []
        with torch.no_grad():
            for e in val_eps:
                out, qy = _episode_forward(ev_model, cxr_arr, e, device, augment=False)
                vp.append(out["probs"].argmax(-1).cpu().numpy()); vt.append(qy.cpu().numpy())
                vnll.append(F.nll_loss(out["probs"].clamp_min(1e-8).log(), qy).item())
                vu.append(out["uncertainty"].mean().item()); vb.append(out["beta"].mean().item())
        vp, vt = np.concatenate(vp), np.concatenate(vt)
        val_f1 = f1_score(vt, vp, average="macro")
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "train_ce": float(np.mean(ces)),
               "train_kl": float(np.mean(kls)), "kl_weight": kl_w,
               "train_f1": f1_score(np.concatenate(tgts), np.concatenate(preds), average="macro"),
               "val_f1": val_f1, "val_nll": float(np.mean(vnll)), "train_unc": float(np.mean(uncs)),
               "val_unc": float(np.mean(vu)), "train_beta": float(np.mean(betas)), "val_beta": float(np.mean(vb)),
               "tau": float(getattr(ev_model.head, "tau", torch.tensor(float("nan"))).detach()),
               "gamma": float(getattr(ev_model.head, "gamma", torch.tensor(float("nan"))).detach()),
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        log.append(row)
        log_cb(f"[{variant} s{seed}] ep {ep:02d} loss={row['train_loss']:.4f} trF1={row['train_f1']:.4f} "
               f"vaF1={val_f1:.4f} vaNLL={row['val_nll']:.4f} u={row['val_unc']:.3f} beta={row['val_beta']:.3f} "
               f"tau={row['tau']:.3f} gamma={row['gamma']:.3f} ({row['seconds']:.0f}s)")
        score = -row["val_nll"] if getattr(cfg, "select_metric", "f1") == "nll" else val_f1 - 1e-3 * row["val_nll"]
        if score > best + 1e-6:
            best, wait = score, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in ev_model.state_dict().items()})
            row["best"] = 1
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    if model.uses_memory:
        model.memory = build_memory(model.encoder, mem_arr, mem_labels, device, cfg.eval_bs)
    return model, pd.DataFrame(log)


# ----------------------------------------------------------------------------------------------
# Supervised EfficientNet-Binary baseline
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def supervised_logits(model, arr, rows, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        x = to_input(arr[rows[i:i + bs]], device)
        with _amp(device):
            out.append(model(x).float())
    return torch.cat(out).cpu().numpy() if out else np.zeros(0)


def train_supervised(enc_state, cxr_arr, cxr_df, cfg, seed, device, log_cb=print, max_epochs=None):
    set_seed(seed)
    enc = Encoder(cfg.emb_dim, cfg.hidden_dim, pretrained=False)
    enc.load_state_dict(enc_state)
    model = SupervisedClassifier(enc).to(device)
    tr = cxr_df[cxr_df["split"] == "train"]
    va = cxr_df[cxr_df["split"] == "val"]
    r_tr, y_tr = tr["cache_row"].values, tr["label"].values
    r_va, y_va = va["cache_row"].values, va["label"].values
    E = max_epochs or cfg.sup_epochs
    opt = torch.optim.Adam(model.parameters(), lr=cfg.sup_lr, weight_decay=cfg.meta_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    best, best_state, wait, log = -1.0, None, 0, []
    for ep in range(1, E + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(r_tr))
        tot, n = 0.0, 0
        for i in range(0, len(perm), cfg.sup_bs):
            b = perm[i:i + cfg.sup_bs]
            if len(b) < 2:
                continue
            x = to_input(cxr_arr[r_tr[b]], device, augment=True, aug_mode=getattr(cfg, "aug_mode", "basic"))
            y = torch.as_tensor(y_tr[b], device=device).float()
            with _amp(device):
                logit = model(x)
            loss = crit(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(b); n += len(b)
        sched.step()
        lv = supervised_logits(model, cxr_arr, r_va, device, cfg.eval_bs)
        pv = 1 / (1 + np.exp(-lv))
        val_f1 = f1_score(y_va, (pv >= 0.5).astype(int), average="macro")
        vloss = float(F.binary_cross_entropy_with_logits(torch.tensor(lv), torch.tensor(y_va).float()))
        row = {"epoch": ep, "train_loss": tot / max(1, n), "val_f1": val_f1, "val_nll": vloss,
               "val_auroc": float(roc_auc_score(y_va, pv)), "lr": opt.param_groups[0]["lr"],
               "seconds": time.time() - t0}
        log.append(row)
        log_cb(f"[effnet s{seed}] ep {ep:02d} loss={row['train_loss']:.4f} vaF1={val_f1:.4f} "
               f"vaAUC={row['val_auroc']:.4f} ({row['seconds']:.0f}s)")
        score = val_f1 - 1e-3 * vloss
        if score > best + 1e-6:
            best, wait = score, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= cfg.sup_patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(log)


# ----------------------------------------------------------------------------------------------
# Decoupled training: (1) encoder + evidential head without memory, (2) FusionGate on frozen embeddings
# ----------------------------------------------------------------------------------------------
def meta_train_decoupled(enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed, device, log_cb=print,
                         max_epochs=None, gate_epochs: int = 20, gate_patience: int = 5):
    base, log1 = meta_train("proto_edl", enc_state, cxr_arr, cxr_df, mem_arr, mem_labels, cfg, seed, device, log_cb,
                            max_epochs)
    model = ProtoModel(base.encoder, "full", cfg).to(device)
    model.head.load_state_dict(base.head.state_dict())
    for prm in list(model.encoder.parameters()) + list(model.head.parameters()):
        prm.requires_grad_(False)
    model.eval()
    model.memory = build_memory(model.encoder, mem_arr, mem_labels, device, cfg.eval_bs)  # encoded once, frozen
    tr = cxr_df[cxr_df["split"] == "train"]
    va = cxr_df[cxr_df["split"] == "val"]
    E_tr = embed(model.encoder, cxr_arr, tr["cache_row"].values, device, cfg.eval_bs)
    E_va = embed(model.encoder, cxr_arr, va["cache_row"].values, device, cfg.eval_bs)
    s_tr = EpisodeSampler(np.arange(len(tr)), tr["label"].values)
    s_va = EpisodeSampler(np.arange(len(va)), va["label"].values)
    rng = np.random.RandomState(seed + 17)
    rng_val = np.random.RandomState(seed + 1)
    val_eps = [s_va.sample(rng_val, cfg.k_shot, cfg.q_query) for _ in range(cfg.val_episodes)]
    opt = torch.optim.AdamW(model.gate.parameters(), lr=1e-3, weight_decay=cfg.meta_wd)

    def run(E, ep):
        s, sy, q, qy = ep
        out = model.predict(E[torch.as_tensor(q, device=device)], E[torch.as_tensor(s, device=device)],
                            torch.as_tensor(sy, device=device))
        return out, torch.as_tensor(qy, device=device)

    best, best_state, wait, log2 = float("inf"), None, 0, []
    for ep in range(1, gate_epochs + 1):
        model.gate.train()
        losses, betas = [], []
        for _ in range(cfg.train_episodes):
            out, qy = run(E_tr, s_tr.sample(rng, cfg.k_shot, cfg.q_query))
            loss, _, _ = model.loss(out, qy, cfg.edl_lambda_max)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item()); betas.append(out["beta"].detach().mean().item())
        model.gate.eval()
        with torch.no_grad():
            vn, vf_p, vf_t = [], [], []
            for e in val_eps:
                out, qy = run(E_va, e)
                vn.append(F.nll_loss(out["probs"].clamp_min(1e-8).log(), qy).item())
                vf_p.append(out["probs"].argmax(-1).cpu().numpy()); vf_t.append(qy.cpu().numpy())
        vnll = float(np.mean(vn)); vf1 = f1_score(np.concatenate(vf_t), np.concatenate(vf_p), average="macro")
        log2.append({"epoch": len(log1) + ep, "phase": "gate", "train_loss": float(np.mean(losses)), "val_nll": vnll,
                     "val_f1": vf1, "train_beta": float(np.mean(betas))})
        log_cb(f"[decoupled gate s{seed}] ep {ep:02d} loss={np.mean(losses):.4f} vaNLL={vnll:.4f} vaF1={vf1:.4f} beta={np.mean(betas):.3f}")
        if vnll < best - 1e-5:
            best, wait = vnll, 0
            best_state = {k: v.detach().clone() for k, v in model.gate.state_dict().items()}
        else:
            wait += 1
            if wait >= gate_patience:
                break
    model.gate.load_state_dict(best_state)
    model.eval()
    log1 = log1.assign(phase="encoder")
    return model, pd.concat([log1, pd.DataFrame(log2)], ignore_index=True)
