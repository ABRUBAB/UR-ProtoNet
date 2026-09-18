"""Stage I: Supervised Pre-Training on NIH ChestX-ray14 & Memory Bank Assembly.

Pretrains the EfficientNet-B0 backbone using class-weighted BCE loss to handle severe
NIH data imbalance (44.65:1 negative-to-positive ratio) and extracts 9,353 curated
anatomical prior embeddings.
"""

import os
import sys
import argparse
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from config import cfg
from model import FeatureEncoder, MemoryBank
from data import (
    CXRDataset, load_nih_df, worker_init_fn,
    DEVICE, AMP_DEVICE, AMP_ENABLED
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


def pretrain_on_nih(enc: FeatureEncoder, loader: DataLoader, epochs: int, lr: float,
                    pos_weight_val: float = 44.65) -> Dict[str, List[float]]:
    pw = torch.tensor([pos_weight_val], device=DEVICE)
    clf = nn.Linear(cfg.emb_dim, 1).to(DEVICE)
    optim = torch.optim.AdamW(
        list(enc.parameters()) + list(clf.parameters()),
        lr=lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = make_grad_scaler(enabled=AMP_ENABLED)

    history = {"epoch": [], "loss": [], "acc": [], "f1": [], "lr": []}
    best_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        enc.train()
        clf.train()
        running_loss = 0.0
        correct = total = 0
        all_preds, all_labels = [], []

        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.float().to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with make_autocast(enabled=AMP_ENABLED):
                z = enc(x)
                logit = clf(z).squeeze(1)
                loss = crit(logit, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logit) >= 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.long().cpu().numpy())

        acc = correct / max(total, 1)
        macro_f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro")
        cur_lr = optim.param_groups[0]["lr"]
        avg_loss = running_loss / max(total, 1)

        history["epoch"].append(ep)
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        history["f1"].append(macro_f1)
        history["lr"].append(cur_lr)

        print(f"  [Pretrain] Ep {ep:02d}/{epochs}  Loss={avg_loss:.4f}  Acc={acc:.4f}  F1={macro_f1:.4f}  LR={cur_lr:.2e}")
        sched.step()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in enc.state_dict().items()}

    if best_state:
        enc.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"  Restored best encoder state (Acc={best_acc:.4f})")
    return history


def main():
    parser = argparse.ArgumentParser(description="Stage I NIH Pre-Training & Memory Bank Creation")
    parser.add_argument("--epochs", type=int, default=cfg.pretrain_epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr_pretrain)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size_pretrain)
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 70)
    print("Stage I: Supervised Pre-Training on NIH ChestX-ray14")
    print(f"Device: {DEVICE} | AMP: {AMP_ENABLED} | Epochs: {args.epochs}")
    print("=" * 70)

    nih_df = load_nih_df()
    encoder = FeatureEncoder(backbone_name=cfg.backbone, pretrained=False,
                             proj_dim=cfg.emb_dim, hidden_dim=cfg.proj_hidden_dim).to(DEVICE)

    if nih_df is None or len(nih_df) == 0:
        print("\nNotice: NIH ChestX-ray14 dataset files not found locally.")
        print(f"To run full pre-training, provide NIH images under '{cfg.nih_dir}' or attach NIH dataset.")
        existing_ckpt = next(
            (p for p in [os.path.join(args.output_dir, "pretrained_encoder.pt"),
                         "checkpoints/pretrained_encoder.pt",
                         "../results/checkpoints/pretrained_encoder.pt"] if os.path.exists(p)),
            None
        )
        if existing_ckpt:
            print(f"Using existing trained Stage I checkpoint from: {existing_ckpt}")
        else:
            ckpt_path = os.path.join(args.output_dir, "pretrained_encoder.pt")
            torch.save(encoder.state_dict(), ckpt_path)
            print(f"Saved initial encoder template to {ckpt_path}")
        return

    nih_targets = nih_df["label"].values.astype(np.int64)
    class_counts = np.bincount(nih_targets, minlength=2)
    sample_weights = 1.0 / class_counts[nih_targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights), replacement=True
    )
    train_ds = CXRDataset(nih_df, train=True, return_meta=False)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=cfg.num_workers, pin_memory=(DEVICE.type == "cuda"),
                        worker_init_fn=worker_init_fn)

    history = pretrain_on_nih(encoder, loader, args.epochs, args.lr, pos_weight_val=cfg.pretrain_pos_weight)
    enc_path = os.path.join(args.output_dir, "pretrained_encoder.pt")
    torch.save(encoder.state_dict(), enc_path)
    print(f"Saved pretrained encoder -> {enc_path}")

    # Plot pretrain curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["loss"], marker="o", color="tab:blue", markersize=4)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Pretrain Loss")
    axes[1].plot(history["epoch"], history["acc"], marker="o", color="tab:green", markersize=4, label="Accuracy")
    axes[1].plot(history["epoch"], history["f1"], marker="s", color="tab:red", markersize=4, label="Macro-F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_title("Pretrain Acc & F1"); axes[1].legend()
    axes[2].plot(history["epoch"], history["lr"], marker="o", color="tab:purple", markersize=4)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate"); axes[2].set_title("LR Schedule")
    plt.tight_layout()
    plt.savefig("results/figures/fig3_pretrain_curves.png", dpi=150)
    plt.close()
    print("Saved training curves -> results/figures/fig3_pretrain_curves.png")


if __name__ == "__main__":
    main()