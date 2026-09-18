"""Stage III: Episodic Meta-Training for UR-ProtoNet with Evidential Loss.

Implements N-way K-shot episodic learning with dynamic MemoryBank retrieval,
learned neural FusionGate prototype blending (beta ~ 0.58-0.60), and Evidential Deep Learning (EDL)
Dirichlet loss with linear annealing.
"""

import os
import sys
import time
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataclasses import asdict

from config import cfg
from model import FeatureEncoder, MemoryBank, FusionGate, EvidentialHead, URProtoNet
from data import (
    CXRDataset, EpisodeDataset, sample_episode,
    DEVICE, AMP_DEVICE, AMP_ENABLED,
    load_cxr_df, load_chexpert_df, patient_level_split, worker_init_fn
)


def make_grad_scaler(enabled: bool = True):
    try:
        return torch.amp.GradScaler(AMP_DEVICE, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_autocast(enabled: bool = True):
    try:
        return torch.amp.autocast(device_type=AMP_DEVICE, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def safe_l2_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable pairwise Euclidean distance avoiding numerical singularity."""
    cos_sim = torch.matmul(F.normalize(a, dim=-1), F.normalize(b, dim=-1).T)
    return (2.0 - 2.0 * cos_sim).clamp(min=0.0).sqrt()


def dirichlet_vacuity(alpha: torch.Tensor) -> torch.Tensor:
    """Epistemic vacuity u = K / S from Dirichlet concentration parameters."""
    K = alpha.size(-1)
    S = alpha.sum(dim=-1)
    return float(K) / S.clamp(min=1e-8)


def entropy_uncertainty(probs: torch.Tensor) -> torch.Tensor:
    """Predictive entropy H(p) = -sum(p * log(p))."""
    p = probs.clamp(min=1e-8)
    return -(p * torch.log(p)).sum(dim=-1)


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler divergence between Dir(alpha) and uniform Dir(1)."""
    K = alpha.size(-1)
    S = alpha.sum(dim=-1, keepdim=True)
    first_term = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    )
    second_term = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1, keepdim=True)
    return (first_term + second_term).squeeze(-1)


def evidential_loss(alpha: torch.Tensor, target_one_hot: torch.Tensor, epoch_frac: float = 1.0,
                    lambda_edl: float = 0.1, annealing_epochs: int = 10) -> torch.Tensor:
    """Evidential Loss penalizing misleading evidence with annealed KL divergence."""
    alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde)
    anneal = min(1.0, epoch_frac) * lambda_edl
    return (anneal * kl).mean()


def ur_protonet_step(encoder: FeatureEncoder, fusion_gate: FusionGate, memory_bank: Optional[MemoryBank],
                     s_x: torch.Tensor, s_y: torch.Tensor, q_x: torch.Tensor, q_y: torch.Tensor,
                     epoch: int = 1, edl_annealing_epochs: int = 10, lambda_edl: float = 0.1,
                     n_way: int = 2, topk_retrieval: int = 5) -> Dict:
    s_x, q_x = s_x.to(DEVICE, non_blocking=True), q_x.to(DEVICE, non_blocking=True)
    s_y, q_y = s_y.to(DEVICE, non_blocking=True), q_y.to(DEVICE, non_blocking=True)

    all_z = encoder(torch.cat([s_x, q_x], dim=0))
    s_z, q_z = all_z[:s_x.size(0)], all_z[s_x.size(0):]

    classes = torch.unique(s_y)
    if len(classes) != n_way:
        raise RuntimeError(f"Expected {n_way} classes in episode, got {len(classes)}")

    protos_local = torch.stack([s_z[s_y == c].mean(dim=0) for c in classes], dim=0)
    protos_local = torch.nan_to_num(F.normalize(protos_local, dim=-1), nan=0.0)

    if memory_bank is not None:
        sims, mem_vecs = memory_bank.retrieve(protos_local, classes, topk_retrieval)
        mem_vecs = torch.nan_to_num(mem_vecs, nan=0.0)
        weights = F.softmax(sims.clamp(-50, 50), dim=-1)
        protos_global = (weights.unsqueeze(-1) * mem_vecs).sum(dim=1)
        protos_global = torch.nan_to_num(F.normalize(protos_global, dim=-1), nan=0.0)
        protos_fused, beta = fusion_gate(protos_local, protos_global)
    else:
        protos_fused = protos_local
        beta = torch.full((len(classes), 1), 1.0, device=DEVICE)

    dists = safe_l2_dist(q_z, protos_fused).clamp(min=0.0)
    logits = -dists
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    probs = alpha / S
    vacuity = torch.nan_to_num(dirichlet_vacuity(alpha), nan=0.5)
    ent = torch.nan_to_num(entropy_uncertainty(probs), nan=0.5)

    label_to_idx = {c.item(): i for i, c in enumerate(classes)}
    q_idx = torch.tensor([label_to_idx[int(v)] for v in q_y.detach().cpu().tolist()],
                         dtype=torch.long, device=DEVICE)
    one_hot = F.one_hot(q_idx, num_classes=n_way).float()

    epoch_frac = min(epoch / max(edl_annealing_epochs, 1), 1.0)
    edl_l = evidential_loss(alpha, one_hot, epoch_frac=epoch_frac, lambda_edl=lambda_edl)
    ce_l = F.cross_entropy(logits, q_idx, label_smoothing=0.05)
    beta_reg = 0.1 * (beta - 0.5).pow(2).mean()
    loss = edl_l + 0.5 * ce_l + beta_reg

    preds_idx = logits.argmax(dim=-1)
    preds = torch.tensor([classes[i] for i in preds_idx], device=DEVICE)

    return {
        "loss": loss,
        "edl_loss": float(edl_l.item()),
        "ce_loss": float(ce_l.item()),
        "preds": preds,
        "targets": q_y,
        "vacuity": vacuity.detach(),
        "entropy": ent.detach(),
        "beta": beta.squeeze(-1).detach(),
        "alpha_mean": float(alpha.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="UR-ProtoNet Episodic Meta-Training")
    parser.add_argument("--epochs", type=int, default=cfg.meta_epochs)
    parser.add_argument("--k_shot", type=int, default=cfg.k_shot)
    parser.add_argument("--n_way", type=int, default=cfg.n_way)
    parser.add_argument("--q_queries", type=int, default=cfg.q_queries)
    parser.add_argument("--lr_enc", type=float, default=cfg.lr_meta)
    parser.add_argument("--lr_gate", type=float, default=cfg.meta_lr_gate)
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 70)
    print("Stage III: Episodic Meta-Training (UR-ProtoNet)")
    print(f"Device: {DEVICE} | AMP: {AMP_ENABLED} | Max Epochs: {args.epochs}")
    print("=" * 70)

    encoder = FeatureEncoder(backbone_name=cfg.backbone, pretrained=False,
                             proj_dim=cfg.emb_dim, hidden_dim=cfg.proj_hidden_dim).to(DEVICE)
    fusion_gate = FusionGate(emb_dim=cfg.emb_dim).to(DEVICE)

    # Load pretrained encoder if available
    pretrain_ckpt = os.path.join(args.output_dir, "pretrained_encoder.pt")
    if os.path.exists(pretrain_ckpt):
        encoder.load_state_dict(torch.load(pretrain_ckpt, map_location=DEVICE), strict=False)
        print(f"Loaded Stage I pretrained encoder from {pretrain_ckpt}")

    cxr_df = load_cxr_df()
    if cxr_df is None or len(cxr_df) == 0:
        print("\nNotice: CXR dataset files not found locally.")
        print(f"To run episodic meta-training, provide CXR images under '{cfg.cxr_dir}' or set custom path.")
        return

    train_df, val_df, _ = patient_level_split(cxr_df, group_col="patient_id", seed=cfg.seed)
    train_ds = EpisodeDataset(train_df, cfg.train_episodes, args.n_way, args.k_shot, args.q_queries, seed=cfg.seed)
    val_ds = EpisodeDataset(val_df, cfg.val_episodes, args.n_way, args.k_shot, args.q_queries, seed=cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_enc, "initial_lr": args.lr_enc},
        {"params": fusion_gate.parameters(), "lr": args.lr_gate, "initial_lr": args.lr_gate},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - cfg.warmup_epochs, 1), eta_min=1e-6
    )
    scaler = make_grad_scaler(enabled=AMP_ENABLED)

    best_val_f1 = -1.0
    wait = 0

    for ep in range(1, args.epochs + 1):
        encoder.train()
        fusion_gate.train()
        train_ds.set_epoch(ep)

        ep_losses, ep_targets, ep_preds = [], [], []
        ep_betas, ep_vacs = [], []

        for s_x, s_y, q_x, q_y in train_loader:
            s_x, s_y = s_x.squeeze(0), s_y.squeeze(0)
            q_x, q_y = q_x.squeeze(0), q_y.squeeze(0)

            optimizer.zero_grad(set_to_none=True)
            with make_autocast(enabled=AMP_ENABLED):
                out = ur_protonet_step(
                    encoder, fusion_gate, None, s_x, s_y, q_x, q_y,
                    epoch=ep, edl_annealing_epochs=cfg.edl_annealing_epochs,
                    lambda_edl=cfg.lambda_edl, n_way=args.n_way
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(fusion_gate.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            ep_losses.append(loss.item())
            ep_targets.append(out["targets"].cpu().numpy())
            ep_preds.append(out["preds"].cpu().numpy())
            ep_betas.append(out["beta"].cpu().numpy())
            ep_vacs.append(out["vacuity"].cpu().numpy())

        train_f1 = f1_score(np.concatenate(ep_targets), np.concatenate(ep_preds), average="macro")

        # Validate
        encoder.eval()
        fusion_gate.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for s_x, s_y, q_x, q_y in val_loader:
                s_x, s_y = s_x.squeeze(0), s_y.squeeze(0)
                q_x, q_y = q_x.squeeze(0), q_y.squeeze(0)
                out = ur_protonet_step(
                    encoder, fusion_gate, None, s_x, s_y, q_x, q_y,
                    epoch=ep, edl_annealing_epochs=cfg.edl_annealing_epochs,
                    lambda_edl=cfg.lambda_edl, n_way=args.n_way
                )
                val_targets.append(out["targets"].cpu().numpy())
                val_preds.append(out["preds"].cpu().numpy())

        val_f1 = f1_score(np.concatenate(val_targets), np.concatenate(val_preds), average="macro")
        if ep > cfg.warmup_epochs:
            scheduler.step()

        print(f"[Meta] Ep {ep:02d}/{args.epochs} | TrainLoss={np.mean(ep_losses):.4f} TrainF1={train_f1:.4f} ValF1={val_f1:.4f} MeanBeta={np.mean(ep_betas):.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wait = 0
            best_path = os.path.join(args.output_dir, "ur_protonet_best.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "fusion_gate": fusion_gate.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "epoch": ep,
            }, best_path)
            print(f"  -> Saved best model (ValF1={best_val_f1:.4f})")
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"Early stopping triggered at epoch {ep}")
                break


if __name__ == "__main__":
    main()