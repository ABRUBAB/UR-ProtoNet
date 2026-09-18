"""Configuration and Hyperparameters for UR-ProtoNet.

Provides standardized configuration attributes, directory paths, and backward-compatible
aliases across pretraining, meta-training, evaluation, and triage pipelines.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class Config:
    # Experiment metadata
    exp_name: str = "ur_protonet_cxr"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    use_amp: bool = True

    # Image preprocessing
    image_size: int = 224
    in_channels: int = 3
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Architecture
    backbone: str = "efficientnet_b0"
    backbone_pretrained: bool = True
    proj_dim: int = 512
    proj_hidden_dim: int = 1024
    temperature: float = 0.5

    # Stage I: NIH Pre-training
    pretrain_epochs: int = 10
    pretrain_lr: float = 1e-4
    pretrain_batch_size: int = 64
    pretrain_pos_weight: float = 44.65  # Inversion of 45:1 imbalance

    # Stage II: Memory Bank
    memory_bank_size: int = 9353
    memory_normal_count: int = 8000
    memory_pneumonia_count: int = 1353
    memory_per_class: int = 4000

    # Stage III: Episodic Meta-Training
    meta_epochs: int = 30
    meta_episodes_per_epoch: int = 100
    n_way: int = 2
    k_shot: int = 5
    q_query: int = 15
    top_k_memory: int = 5
    meta_lr_encoder: float = 5e-5
    meta_lr_gate: float = 1e-3
    edl_annealing_epochs: int = 10
    warmup_epochs: int = 5
    patience: int = 7
    weight_decay: float = 1e-4
    lambda_edl: float = 0.1
    num_support_sets: int = 5

    # Evaluation & Data paths
    ext_batch_size: int = 64
    chexpert_eval_max_per_class: int = 5000
    val_episodes: int = 50

    # Paths (supporting both relative repo structure and custom mounts)
    nih_dir: str = "data/nih"
    cxr_dir: str = "data/cxr"
    chexpert_dir: str = "data/chexpert"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    figures_dir: str = "figures"

    # Compatibility properties / aliases
    @property
    def img_size(self) -> int:
        return self.image_size

    @img_size.setter
    def img_size(self, val: int):
        self.image_size = val

    @property
    def emb_dim(self) -> int:
        return self.proj_dim

    @emb_dim.setter
    def emb_dim(self, val: int):
        self.proj_dim = val

    @property
    def nih_root(self) -> str:
        return self.nih_dir

    @nih_root.setter
    def nih_root(self, val: str):
        self.nih_dir = val

    @property
    def cxr_root(self) -> str:
        return self.cxr_dir

    @cxr_root.setter
    def cxr_root(self, val: str):
        self.cxr_dir = val

    @property
    def chexpert_root(self) -> str:
        return self.chexpert_dir

    @chexpert_root.setter
    def chexpert_root(self, val: str):
        self.chexpert_dir = val

    @property
    def batch_size_pretrain(self) -> int:
        return self.pretrain_batch_size

    @batch_size_pretrain.setter
    def batch_size_pretrain(self, val: int):
        self.pretrain_batch_size = val

    @property
    def lr_pretrain(self) -> float:
        return self.pretrain_lr

    @lr_pretrain.setter
    def lr_pretrain(self, val: float):
        self.pretrain_lr = val

    @property
    def lr_meta(self) -> float:
        return self.meta_lr_encoder

    @lr_meta.setter
    def lr_meta(self, val: float):
        self.meta_lr_encoder = val

    @property
    def train_episodes(self) -> int:
        return self.meta_episodes_per_epoch

    @train_episodes.setter
    def train_episodes(self, val: int):
        self.meta_episodes_per_epoch = val

    @property
    def q_queries(self) -> int:
        return self.q_query

    @q_queries.setter
    def q_queries(self, val: int):
        self.q_query = val

    @property
    def topk_retrieval(self) -> int:
        return self.top_k_memory

    @topk_retrieval.setter
    def topk_retrieval(self, val: int):
        self.top_k_memory = val


cfg = Config()
