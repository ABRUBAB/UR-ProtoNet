"""Model components: encoder, labeled memory bank, FusionGate, evidential head and baselines."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# ----------------------------------------------------------------------------------------------
# Encoder (Eq. 1-2): EfficientNet-B0 -> LayerNorm(W2 ReLU(BN(W1 f))) -> L2 normalisation
# ----------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, emb_dim: int = 512, hidden_dim: int = 1024, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg")
        nf = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(nf, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.emb_dim = emb_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(x)).float(), dim=-1)


# ----------------------------------------------------------------------------------------------
# Labeled memory bank (Eq. 7-9) with similarity-weighted retrieval
# ----------------------------------------------------------------------------------------------
class MemoryBank:
    def __init__(self, mem0: torch.Tensor, mem1: torch.Tensor):
        self.mem = [F.normalize(mem0.float(), dim=-1), F.normalize(mem1.float(), dim=-1)]

    @property
    def sizes(self):
        return [m.size(0) for m in self.mem]

    def retrieve(self, protos: torch.Tensor, topk: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """protos: [2, D] (class order 0, 1). Returns memory prototypes [2, D] and top-k sims [2, k]."""
        out, sims = [], []
        for c in range(len(self.mem)):
            s = self.mem[c] @ protos[c]
            v, i = torch.topk(s, k=min(topk, s.numel()))
            w = v.clamp(min=1e-6)
            m = (w.unsqueeze(-1) * self.mem[c][i]).sum(0) / w.sum()
            out.append(F.normalize(m, dim=-1))
            sims.append(v)
        return torch.stack(out, 0), torch.stack(sims, 0)


# ----------------------------------------------------------------------------------------------
# FusionGate (Eq. 11-12): beta = sigmoid(g([c || c_mem])),  c~ = beta c + (1 - beta) c_mem
# ----------------------------------------------------------------------------------------------
class FusionGate(nn.Module):
    """features="concat": g([c || c_mem]);  features="interaction": g([c || c_mem || |c - c_mem| || c * c_mem || cos])
    so that the gate can measure the agreement between episode support and retrieved memory."""

    def __init__(self, emb_dim: int = 512, hidden: int = 256, features: str = "concat"):
        super().__init__()
        self.features = features
        d_in = 2 * emb_dim if features == "concat" else 4 * emb_dim + 1
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor, c_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.features == "concat":
            x = torch.cat([c, c_mem], dim=-1)
        else:
            cos = (c * c_mem).sum(-1, keepdim=True)
            x = torch.cat([c, c_mem, (c - c_mem).abs(), c * c_mem, cos], dim=-1)
        beta = torch.sigmoid(self.net(x))
        fused = F.normalize(beta * c + (1.0 - beta) * c_mem, dim=-1)
        return fused, beta.squeeze(-1)


# ----------------------------------------------------------------------------------------------
# Distance heads
# ----------------------------------------------------------------------------------------------
def sq_dist(z: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between unit vectors: ||z - c||^2 = 2 - 2 z.c, in [0, 4]."""
    return (2.0 - 2.0 * z @ protos.t()).clamp(min=0.0)


class EvidentialHead(nn.Module):
    """log e_n = (gamma - d_n^2) / tau,  alpha_n = e_n + 1,  u = K / S.

    gamma (evidence radius) and tau (temperature) are learned. Evidence exceeds the unit prior mass
    only for queries closer than sqrt(gamma) to a prototype, and lies in
    (exp((gamma - 4) / tau), exp(gamma / tau)], so S ranges over (K, K (1 + exp(gamma / tau))].
    """

    def __init__(self, tau_init: float = 0.25, gamma_init: float = 1.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.02, 2.0)

    def log_evidence(self, z: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        return (self.gamma.clamp(0.0, 4.0) - sq_dist(z.float(), protos.float())) / self.tau

    def forward(self, z: torch.Tensor, protos: torch.Tensor):
        le = self.log_evidence(z, protos)
        alpha = le.clamp(max=40.0).exp() + 1.0
        S = alpha.sum(-1, keepdim=True)
        probs = alpha / S
        vac = alpha.size(-1) / S.squeeze(-1)
        return {"log_evidence": le, "alpha": alpha, "probs": probs, "uncertainty": vac, "logits": le}

    def bounds(self):
        g, t = float(self.gamma.clamp(0.0, 4.0)), float(self.tau)
        e_max = math.exp(min(g / t, 40.0))
        return {"gamma": g, "tau": t, "evidence_max": e_max, "S_max": 2 * (1 + e_max), "u_min": 2 / (2 * (1 + e_max))}


class SoftmaxDistanceHead(nn.Module):
    """Prototypical (Snell et al.) head: p = softmax(-d^2 / tau) with a learned temperature."""

    def __init__(self, tau_init: float = 0.25):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.02, 2.0)

    def forward(self, z, protos):
        logits = -sq_dist(z.float(), protos.float()) / self.tau
        probs = logits.softmax(-1)
        return {"logits": logits, "probs": probs, "uncertainty": normalized_entropy(probs)}


class MatchingHead(nn.Module):
    """Matching Network (cosine attention over all support embeddings, without FCE)."""

    def __init__(self, tau_init: float = 0.1):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    def forward(self, z, s_z, s_y, n_way: int = 2):
        att = ((z.float() @ s_z.float().t()) / self.log_tau.exp().clamp(0.01, 2.0)).softmax(-1)
        onehot = F.one_hot(s_y, n_way).float()
        probs = (att @ onehot).clamp(1e-6, 1.0)
        probs = probs / probs.sum(-1, keepdim=True)
        return {"logits": probs.log(), "probs": probs, "uncertainty": normalized_entropy(probs)}


def normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-8).log()).sum(-1) / math.log(p.size(-1))


# ----------------------------------------------------------------------------------------------
# Losses (Eq. 17-19)
# ----------------------------------------------------------------------------------------------
def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    K = alpha.size(-1)
    S = alpha.sum(-1, keepdim=True)
    return (torch.lgamma(S).squeeze(-1) - math.lgamma(K) - torch.lgamma(alpha).sum(-1)
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(-1))


def edl_loss(alpha: torch.Tensor, y: torch.Tensor, kl_weight: float):
    """CE on the Dirichlet mean + annealed KL on misleading evidence (alpha~ = y + (1 - y) alpha)."""
    S = alpha.sum(-1)
    ce = -(alpha.gather(1, y.view(-1, 1)).squeeze(1) / S).log().mean()
    onehot = F.one_hot(y, alpha.size(-1)).float()
    alpha_t = onehot + (1.0 - onehot) * alpha
    kl = kl_dirichlet_uniform(alpha_t).mean()
    return ce + kl_weight * kl, ce.detach(), kl.detach()


# ----------------------------------------------------------------------------------------------
# Prototype construction shared by all prototype-based methods
# ----------------------------------------------------------------------------------------------
def class_prototypes(s_z: torch.Tensor, s_y: torch.Tensor, n_way: int = 2) -> torch.Tensor:
    return torch.stack([F.normalize(s_z[s_y == c].mean(0), dim=-1) for c in range(n_way)], 0)


class ProtoModel(nn.Module):
    """One class covers UR-ProtoNet and its ablations.

    variant: "full" (memory + gate + EDL), "proto_mem" (memory + gate + softmax),
             "proto_edl" (EDL, no memory), "proto" (VanillaProtoNet), "matching" (MatchingNet)
    """

    def __init__(self, encoder: Encoder, variant: str, cfg):
        super().__init__()
        self.encoder = encoder
        self.variant = variant
        self.uses_memory = variant in ("full", "proto_mem")
        self.uses_edl = variant in ("full", "proto_edl")
        self.gate = FusionGate(encoder.emb_dim, cfg.gate_hidden, getattr(cfg, "gate_features", "concat")) if self.uses_memory else None
        if variant == "matching":
            self.head = MatchingHead()
        elif self.uses_edl:
            self.head = EvidentialHead(cfg.tau_init, cfg.gamma_init)
        else:
            self.head = SoftmaxDistanceHead(cfg.tau_init)
        self.memory: Optional[MemoryBank] = None
        self.topk = cfg.mem_topk

    def head_params(self):
        ps = list(self.head.parameters())
        if self.gate is not None:
            ps += list(self.gate.parameters())
        return ps

    def prototypes(self, s_z, s_y, memory: Optional[MemoryBank] = None, force_beta: Optional[float] = None):
        c = class_prototypes(s_z, s_y)
        info = {"beta": torch.ones(2, device=c.device), "mem_sim": torch.zeros(2, device=c.device)}
        mem = memory if memory is not None else self.memory
        if self.uses_memory and mem is not None and force_beta != 1.0:
            c_mem, sims = mem.retrieve(c, self.topk)
            if force_beta is None:
                fused, beta = self.gate(c, c_mem)
            else:
                beta = torch.full((2,), float(force_beta), device=c.device)
                fused = F.normalize(beta.unsqueeze(-1) * c + (1 - beta.unsqueeze(-1)) * c_mem, dim=-1)
            info = {"beta": beta, "mem_sim": sims.mean(-1)}
            c = fused
        return c, info

    def predict(self, q_z, s_z, s_y, memory=None, force_beta=None):
        if self.variant == "matching":
            out = self.head(q_z, s_z, s_y)
            out["beta"] = torch.ones(2, device=q_z.device)
            out["mem_sim"] = torch.zeros(2, device=q_z.device)
            return out
        protos, info = self.prototypes(s_z, s_y, memory, force_beta)
        out = self.head(q_z, protos)
        out.update(info)
        return out

    def loss(self, out, q_y, kl_weight: float):
        if self.uses_edl:
            return edl_loss(out["alpha"], q_y, kl_weight)
        ce = F.nll_loss(out["probs"].clamp_min(1e-8).log(), q_y)
        return ce, ce.detach(), torch.zeros((), device=q_y.device)


class SupervisedClassifier(nn.Module):
    """EfficientNet-Binary baseline: backbone + linear logit, trained with BCE on CXR-train."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.backbone = encoder.backbone
        self.fc = nn.Linear(encoder.backbone.num_features, 1)

    def forward(self, x):
        return self.fc(self.backbone(x)).squeeze(-1)
