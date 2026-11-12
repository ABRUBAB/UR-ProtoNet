"""Hyperparameters for all three training stages and the evaluation protocol."""

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    seed: int = 42
    img_size: int = 224
    emb_dim: int = 512
    hidden_dim: int = 1024
    amp: bool = True
    num_workers: int = 4

    # Stage I: supervised pre-training on NIH ChestX-ray14
    pre_epochs: int = 10
    pre_lr: float = 1e-4
    pre_wd: float = 1e-4
    pre_bs: int = 64
    pre_val_frac: float = 0.1          # patient-level hold-out used only for monitoring curves

    # Stage II: labeled memory bank
    mem_normal: int = 8000             # Normal partition capped at 8,000 (uniform random)
    mem_topk: int = 5                  # neighbours retrieved per prototype
    mem_refresh: bool = True           # re-encode memory with the current encoder every epoch

    # Stage III: episodic meta-training
    n_way: int = 2
    k_shot: int = 5
    q_query: int = 15
    train_episodes: int = 150
    val_episodes: int = 80
    shot_sampling: str = "fixed"       # "fixed" (K every episode) or "uniform" (K ~ U{1..k_shot})
    meta_epochs: int = 30
    meta_lr: float = 1e-4
    head_lr: float = 2e-4
    meta_wd: float = 1e-4
    warmup_epochs: int = 3
    patience: int = 6
    edl_lambda_max: float = 0.5
    edl_anneal_epochs: int = 20
    tau_init: float = 0.25             # evidence temperature
    gamma_init: float = 1.0            # evidence radius (log-evidence = (gamma - d^2) / tau)
    gate_hidden: int = 256
    gate_features: str = "concat"      # "concat": [c, c_mem]; "interaction": + |c - c_mem|, c * c_mem, cos(c, c_mem)
    cxr_exclude_nih: bool = False      # drop CXR images that duplicate NIH ChestX-ray14 images (Stage I / memory)
    decoupled: bool = False            # train encoder+head without memory, then the gate on frozen embeddings
    mem_dropout: float = 0.0           # probability of skipping the memory (beta = 1) in a training episode

    # Round-2 training-recipe flags (every default reproduces the pipeline of the first revision runs)
    aug_mode: str = "basic"            # episodic / supervised augmentation: "basic" (flip, rotation, shift) or
                                       # "dg" (+ zoom / aspect, gamma, contrast, blur <-> sharpen, resolution, noise)
    pre_aug_mode: str = "basic"        # Stage I augmentation ("basic" or "dg")
    pre_task: str = "binary"           # Stage I objective: "binary" (Pneumonia vs No Finding) or "multilabel"
                                       # (14 findings on the full ChestX-ray14; needs the nih14 cache)
    pre_exclude_cxr_eval: bool = False # multilabel Stage I: drop NIH images whose dHash equals a CXR val/test image
    pre_logit_scale: float = 1.0       # multilabel Stage I: fixed scale on the unit-norm embedding before the classifier
    pre_aux_head: bool = False         # multilabel Stage I: auxiliary 14-label head on the pooled backbone features
    ema_decay: float = 0.0             # > 0: validate, select and return an exponential moving average of the weights
    select_metric: str = "f1"          # early-stopping criterion on the CXR-Val episodes: "f1" or "nll"
    xdom_frac: float = 0.0             # fraction of training episodes with CXR supports and NIH-memory queries

    # Supervised baseline (EfficientNet-Binary)
    sup_epochs: int = 30
    sup_patience: int = 8
    sup_bs: int = 64
    sup_lr: float = 1e-4

    # Evaluation
    n_support_sets: int = 20           # ensemble size M
    source_pool_per_class: int = 1000  # candidate support images drawn from CXR-train
    target_pool_per_class: int = 500   # patient-disjoint CheXpert support pool
    siim_pool_studies_per_class: int = 300
    chexpert_per_class: int = 5000
    kshot_eval: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    eval_bs: int = 256

    def to_dict(self):
        return asdict(self)


cfg = Config()
