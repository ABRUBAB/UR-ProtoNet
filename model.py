"""UR-ProtoNet Neural Network Architecture Modules.

Implements the complete deep evidential retrieval-augmented prototypical network architecture:
- FeatureEncoder: EfficientNet-B0 backbone with projection head to unit hypersphere
- MemoryBank: Persistent anatomical memory bank (9,353 NIH CXR embeddings) with cosine retrieval
- FusionGate: Learned neural gating module interpolating episode support prototypes and memory priors
- EvidentialHead: Subjective Logic Evidential Deep Learning classifier parameterizing Dirichlet distributions
- URProtoNet: Integrated end-to-end episodic model with factory methods
"""

import os
from typing import Tuple, Optional, Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class FeatureEncoder(nn.Module):
    """EfficientNet-B0 backbone with hypersphere projection head."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        proj_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        if HAS_TIMM:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0
            )
            in_features = self.backbone.num_features  # 1280 for EfficientNet-B0
        else:
            # Fallback lightweight CNN feature extractor when timm is unavailable
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            in_features = 64

        # Primary projection layer matching the trained checkpoint
        self.proj = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
        self.proj_dim = proj_dim

    @property
    def projection(self) -> nn.Module:
        """Compatibility property for scripts referencing .projection."""
        return self.proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        proj = self.proj(feat)
        norm_feat = F.normalize(proj, p=2, dim=-1)
        return norm_feat

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Remap projection.* -> proj.* if needed for backward compatibility
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "projection."):
                new_k = prefix + "proj." + k[len(prefix + "projection."):]
                state_dict[new_k] = state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class MemoryBank(nn.Module):
    """Persistent labeled anatomical prior memory bank with cosine-similarity lookup."""

    def __init__(self, normal_embeddings: torch.Tensor, pneumonia_embeddings: torch.Tensor):
        super().__init__()
        self.register_buffer("mem_normal", F.normalize(normal_embeddings, p=2, dim=-1))
        self.register_buffer("mem_pneumonia", F.normalize(pneumonia_embeddings, p=2, dim=-1))

    @classmethod
    def from_file(cls, filepath: str, device: Union[str, torch.device] = "cpu") -> "MemoryBank":
        """Loads memory bank embeddings directly from .pt checkpoint."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Memory bank checkpoint not found: {filepath}")
        state = torch.load(filepath, map_location=device)
        if isinstance(state, dict):
            if 0 in state and 1 in state:
                norm_emb = state[0]
                pneu_emb = state[1]
            elif "mem_normal" in state and "mem_pneumonia" in state:
                norm_emb = state["mem_normal"]
                pneu_emb = state["mem_pneumonia"]
            else:
                keys = list(state.keys())
                norm_emb = state[keys[0]]
                pneu_emb = state[keys[1]]
        else:
            raise ValueError(f"Unrecognized memory bank format in {filepath}")
        return cls(norm_emb, pneu_emb).to(device)

    def retrieve_single(
        self, base_prototype: torch.Tensor, class_idx: int, top_k: int = 5
    ) -> torch.Tensor:
        """Retrieves top-k closest exemplars for a single class and returns their mean prototype."""
        mem = self.mem_normal if class_idx == 0 else self.mem_pneumonia
        p = F.normalize(base_prototype, p=2, dim=-1)
        if p.ndim == 1:
            sims = torch.matmul(mem, p)
        else:
            sims = torch.matmul(mem, p.squeeze(0))
        _, topk_indices = torch.topk(sims, k=min(top_k, mem.size(0)), largest=True)
        topk_vecs = mem[topk_indices]
        mem_proto = topk_vecs.mean(dim=0)
        return F.normalize(mem_proto, p=2, dim=-1)

    def retrieve_batch(
        self, protos_local: torch.Tensor, classes: Union[torch.Tensor, List[int]], top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch retrieval for episodic training, returning similarity scores and memory vectors."""
        all_sims = []
        all_vecs = []
        for i, c in enumerate(classes):
            c_val = int(c.item() if hasattr(c, "item") else c)
            mem = self.mem_normal if c_val == 0 else self.mem_pneumonia
            p = protos_local[i]
            sims = torch.matmul(mem, p)
            topk_s, topk_i = torch.topk(sims, k=min(top_k, mem.size(0)), largest=True)
            all_sims.append(topk_s)
            all_vecs.append(mem[topk_i])
        return torch.stack(all_sims, dim=0), torch.stack(all_vecs, dim=0)

    def retrieve(
        self, protos: torch.Tensor, class_idx_or_classes: Optional[Union[int, torch.Tensor, List[int]]] = None,
        top_k: int = 5
    ):
        """Flexible retrieve interface handling both single-prototype and batch-episode calls."""
        if class_idx_or_classes is None:
            classes = [0, 1] if protos.size(0) == 2 else list(range(protos.size(0)))
            return self.retrieve_batch(protos, classes, top_k)
        if isinstance(class_idx_or_classes, int):
            return self.retrieve_single(protos, class_idx_or_classes, top_k)
        if isinstance(class_idx_or_classes, (list, tuple, torch.Tensor)):
            return self.retrieve_batch(protos, class_idx_or_classes, top_k)
        return self.retrieve_single(protos, int(class_idx_or_classes), top_k)


class FusionGate(nn.Module):
    """Adaptive neural gate to interpolate episode support prototypes and memory priors.
    
    Uses interaction feature concatenation [c_sup, c_mem, c_sup * c_mem] to compute
    gating scalar beta in [0, 1], converging stably to ~0.58-0.60 as shown in the paper.
    """

    def __init__(self, emb_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def gate_net(self) -> nn.Module:
        """Alias for backward compatibility."""
        return self.net

    def get_beta(self, c_support: torch.Tensor, c_memory: torch.Tensor) -> torch.Tensor:
        """Computes beta scalar from support and memory prototype embeddings."""
        interaction = c_support * c_memory
        concat = torch.cat([c_support, c_memory, interaction], dim=-1)
        beta = torch.sigmoid(self.net(concat))
        return beta

    def forward(
        self, c_support: torch.Tensor, c_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.get_beta(c_support, c_memory)
        c_fused = beta * c_support + (1.0 - beta) * c_memory
        c_fused = F.normalize(c_fused, p=2, dim=-1)
        return c_fused, beta


class EvidentialHead(nn.Module):
    """Evidential Deep Learning head parameterizing Dirichlet distributions.
    
    Computes exact epistemic vacuity u = K/S (K=2 for binary classification)
    and class probabilities in a single deterministic pass.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, query_feats: torch.Tensor, prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # query_feats: [Q, D], prototypes: [N, D]
        dists_sq = torch.cdist(query_feats, prototypes, p=2) ** 2
        evidence = torch.exp(-dists_sq / max(self.temperature, 1e-4))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / S
        vacuity = 2.0 / S.squeeze(-1)
        return probs, vacuity, evidence


class URProtoNet(nn.Module):
    """Complete UR-ProtoNet system integrating Encoder, MemoryBank, FusionGate, and EDL Head."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        memory_bank: Optional[MemoryBank] = None,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.memory_bank = memory_bank
        self.fusion_gate = FusionGate(emb_dim=encoder.proj_dim)
        self.edl_head = EvidentialHead(temperature=temperature)

    def load_checkpoint(self, checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
        """Robustly loads model weights from either structured or monolithic checkpoints."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict):
            if "encoder" in state:
                self.encoder.load_state_dict(state["encoder"], strict=False)
            if "fusion_gate" in state:
                self.fusion_gate.load_state_dict(state["fusion_gate"], strict=False)
            if "model_state_dict" in state:
                self.load_state_dict(state["model_state_dict"], strict=False)
            elif "encoder" not in state and "fusion_gate" not in state:
                self.load_state_dict(state, strict=False)
        else:
            self.load_state_dict(state, strict=False)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        memory_bank_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        temperature: float = 0.5,
    ) -> "URProtoNet":
        """Factory method to instantiate a fully configured URProtoNet model ready for inference."""
        encoder = FeatureEncoder(backbone_name="efficientnet_b0", pretrained=False)
        mem_bank = None
        if memory_bank_path and os.path.exists(memory_bank_path):
            mem_bank = MemoryBank.from_file(memory_bank_path, device=device)
        model = cls(encoder=encoder, memory_bank=mem_bank, temperature=temperature).to(device)
        model.load_checkpoint(checkpoint_path, device=device)
        model.eval()
        return model

    def compute_prototypes(
        self, support_feats: torch.Tensor, support_labels: torch.Tensor,
        n_way: int = 2, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prototypes = []
        betas = []
        for c in range(n_way):
            mask = (support_labels == c)
            if mask.sum() > 0:
                c_sup = support_feats[mask].mean(dim=0)
            else:
                c_sup = torch.zeros(self.encoder.proj_dim, device=support_feats.device)
            c_sup = F.normalize(c_sup, p=2, dim=-1)

            if self.memory_bank is not None:
                c_mem = self.memory_bank.retrieve_single(c_sup, class_idx=c, top_k=top_k)
                c_fused, beta = self.fusion_gate(c_sup.unsqueeze(0), c_mem.unsqueeze(0))
                c_fused = c_fused.squeeze(0)
                beta = beta.squeeze(0)
            else:
                c_fused = c_sup
                beta = torch.tensor(1.0, device=support_feats.device)

            prototypes.append(c_fused)
            betas.append(beta)

        return torch.stack(prototypes, dim=0), torch.stack(betas, dim=0)

    def forward_episode(
        self, support_x: torch.Tensor, support_y: torch.Tensor,
        query_x: torch.Tensor, n_way: int = 2, top_k: int = 5
    ):
        sup_feats = self.encoder(support_x)
        qry_feats = self.encoder(query_x)
        prototypes, betas = self.compute_prototypes(sup_feats, support_y, n_way=n_way, top_k=top_k)
        probs, vacuity, evidence = self.edl_head(qry_feats, prototypes)
        return probs, vacuity, evidence, betas

