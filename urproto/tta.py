"""Transductive test-time adaptation (TTA) for UR-ProtoNet -- PROTOTYPE code, inference only.

Nothing here touches labels of the images being classified. The only inputs are
  * the trained model (encoder already applied: we work on embeddings),
  * the labelled K-shot support set (source or target protocol, exactly as before),
  * the labelled NIH memory bank,
  * the UNLABELLED batch of target embeddings `q_z` (the images to be classified, or any other
    unlabelled images of the target hospital).

Three components (see PROPOSALS.md):
  P1  class-balanced transductive alignment   -> TTAConfig.align / .center
  P2a target-conditioned memory retrieval      -> target_conditioned_prototypes(), TTAConfig.tcr
  P2b evidential EM prototype rectification    -> TTAConfig.rho / .topk_frac
  P4  prevalence (label-shift) estimation      -> TTAConfig.est_prior / .decision_prior
  P3  vacuity / dissonance decomposition       -> evidential_uncertainties()

Drop-in: `proto_ensemble_tta` has the signature and the return keys of `urproto.evaluate.proto_ensemble`
(same RNG stream -> identical support sets), plus a few diagnostics. With TTAConfig(align="none")
it reproduces `proto_ensemble` exactly (checked in the self-test).

Self-test on synthetic embeddings (CPU):  python tta.py
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TTAConfig:
    align: str = "shift"           # "none" | "shift": move the target batch so that its class-balanced centre
    #                                coincides with the midpoint of the two (fused) prototypes
    center: bool = False           # additionally remove the common centre and re-normalise (domain-wise CL2N)
    iters: int = 3                 # EM iterations (iteration 0 assumes balanced classes)
    rho: float = 0.0               # prototype rectification: c <- norm((1-rho) c~ + rho mu_target); 0 = off
    topk_frac: float = 1.0         # use only this fraction of most confident target images per class in mu_target
    tcr: float = 0.0               # weight of the target-conditioned memory prototype inside c_mem (0 = off)
    est_prior: bool = False        # estimate the target prevalence by EM (Saerens et al.)
    decision_prior: str = "balanced"   # "balanced": evidence-only decision; "estimated": add log prior ratio
    em_temp: float = 1.0           # temperature of the E-step responsibilities (1 = the head's own tau)

    def tag(self) -> str:
        return (f"align={self.align}|center={int(self.center)}|it={self.iters}|rho={self.rho}|top={self.topk_frac}"
                f"|tcr={self.tcr}|prior={int(self.est_prior)}:{self.decision_prior}")


def _class_prototypes(s_z: torch.Tensor, s_y: torch.Tensor, n_way: int = 2) -> torch.Tensor:
    return torch.stack([F.normalize(s_z[s_y == c].mean(0), dim=-1) for c in range(n_way)], 0)


# ----------------------------------------------------------------------------------------------
# P2a. Target-conditioned retrieval: which labelled memory entries does the TARGET domain look like?
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def target_conditioned_prototypes(memory, t_z: torch.Tensor, topk: int = 5, chunk: int = 2048):
    """Every unlabelled target embedding retrieves its top-k neighbours inside EACH class partition of the
    labelled memory (so the 8000:1353 imbalance plays no role). The similarity-weighted hit counts define one
    memory prototype per class: "class n of the memory, as seen from the target hospital".
    Returns (c_tmem [2, D], effective number of memory entries used per class)."""
    out, n_eff = [], []
    for m in memory.mem:
        w = torch.zeros(m.size(0), device=m.device)
        k = min(topk, m.size(0))
        for i in range(0, t_z.size(0), chunk):
            v, idx = (t_z[i:i + chunk].float() @ m.t()).topk(k, dim=-1)
            w.index_add_(0, idx.reshape(-1), v.clamp_min(1e-6).reshape(-1))
        out.append(F.normalize((w.unsqueeze(-1) * m).sum(0), dim=-1))
        n_eff.append(float(w.sum() ** 2 / (w ** 2).sum().clamp_min(1e-12)))
    return torch.stack(out, 0), n_eff


def fused_prototypes(model, s_z, s_y, memory=None, force_beta=None, c_tmem: Optional[torch.Tensor] = None,
                     tcr: float = 0.0):
    """ProtoModel.prototypes with an optional target-conditioned memory prototype mixed into c_mem.
    The trained FusionGate is reused unchanged."""
    if c_tmem is None or tcr <= 0 or not getattr(model, "uses_memory", False) or force_beta == 1.0:
        if hasattr(model, "prototypes"):
            return model.prototypes(s_z, s_y, memory, force_beta)
        c = _class_prototypes(s_z, s_y)                      # SimpleShot-like objects without .prototypes
        return c, {"beta": torch.ones(2, device=c.device), "mem_sim": torch.zeros(2, device=c.device)}
    c = _class_prototypes(s_z, s_y)
    mem = memory if memory is not None else model.memory
    c_mem, sims = mem.retrieve(c, model.topk)
    c_mem = F.normalize((1.0 - tcr) * c_mem + tcr * c_tmem.to(c_mem), dim=-1)
    if force_beta is None:
        fused, beta = model.gate(c, c_mem)
    else:
        beta = torch.full((2,), float(force_beta), device=c.device)
        fused = F.normalize(beta.unsqueeze(-1) * c + (1 - beta.unsqueeze(-1)) * c_mem, dim=-1)
    return fused, {"beta": beta, "mem_sim": sims.mean(-1)}


# ----------------------------------------------------------------------------------------------
# P1 + P2b + P4. Evidential EM on the unlabelled target batch
# ----------------------------------------------------------------------------------------------
def _head_logits(model, z, protos):
    """Log-evidence (EvidentialHead) or -d^2/tau (SoftmaxDistanceHead); SimpleShot: -d^2."""
    if hasattr(model, "head"):
        return model.head(z, protos)["logits"].float()
    return -(2.0 - 2.0 * z @ protos.t()).clamp(min=0.0)


@torch.no_grad()
def adapt_logits(model, q_z: torch.Tensor, protos: torch.Tensor, cfg: TTAConfig) -> Dict:
    """EM in the mixture whose two components are the evidential prototypes.

    The evidence e_n(z) = exp((gamma - ||z - c_n||^2) / tau) = const * exp((2 / tau) z.c_n) is an unnormalised
    von Mises-Fisher likelihood with concentration 2 / tau, so "responsibility" = Dirichlet-mean-free
    normalised evidence. E-step: responsibilities of the unlabelled target images. M-step:
      (a) shift xi that maps the class-balanced target centre onto the prototype midpoint   [align="shift"]
      (b) prototypes  c_n <- norm((1 - rho) c~_n + rho mu_n)                                   [rho > 0]
      (c) prevalence  pi_n <- mean responsibility                                              [est_prior]
    Iteration 0 uses uniform responsibilities, i.e. plain mean-centring under a balanced-class assumption.
    """
    z = q_z.float()
    c0 = protos.float()
    m_s = c0.mean(0)                                       # midpoint of the (fused) prototypes
    r = z.new_full((z.size(0), 2), 0.5)
    pi = z.new_full((2,), 0.5)
    xi = torch.zeros_like(m_s)
    c = c0
    le = None

    def geom(z_al, c_cur):
        if cfg.center:                                     # z + xi - m_s == z - (class-balanced target centre)
            return F.normalize(z + xi - m_s, dim=-1), F.normalize(c_cur - m_s, dim=-1)
        return z_al, c_cur

    for it in range(max(1, cfg.iters)):
        # ---- M-step from the current responsibilities
        rr = r
        if cfg.topk_frac < 1.0 and it > 0:
            k = max(1, int(round(cfg.topk_frac * z.size(0) * 0.5)))
            rr = torch.zeros_like(r)
            for n in range(2):
                idx = r[:, n].topk(min(k, z.size(0))).indices
                rr[idx, n] = r[idx, n]
        mu = (r.t() @ z) / r.sum(0).clamp_min(1e-6).unsqueeze(-1)          # class means, native coordinates
        if cfg.align == "shift":
            xi = m_s - mu.mean(0)
        z_al = F.normalize(z + xi, dim=-1)
        if cfg.rho > 0 and it > 0:
            mu_r = (rr.t() @ z) / rr.sum(0).clamp_min(1e-6).unsqueeze(-1)
            c = F.normalize((1.0 - cfg.rho) * c0 + cfg.rho * F.normalize(mu_r + xi, dim=-1), dim=-1)
        # ---- E-step
        zz, cc = geom(z_al, c)
        le = _head_logits(model, zz, cc)
        r = (le / cfg.em_temp + pi.log()).softmax(-1)
        if cfg.est_prior:
            pi = r.mean(0).clamp(0.02, 0.98)
            pi = pi / pi.sum()
    alpha = le.clamp(max=40.0).exp() + 1.0
    log_prior = (pi / 0.5).log() if cfg.decision_prior == "estimated" else torch.zeros_like(pi)
    return {"logits": le, "log_prior": log_prior, "pi": pi, "protos": c, "xi": xi, "shift_norm": xi.norm(),
            "uncertainty": alpha.size(-1) / alpha.sum(-1)}


@torch.no_grad()
def proto_ensemble_tta(model, q_z, pool_z, pool_y, M: int, K: int, seed: int, memory=None, force_beta=None,
                       tta: Optional[TTAConfig] = None, t_z: Optional[torch.Tensor] = None):
    """Drop-in replacement of evaluate.proto_ensemble. `t_z`: unlabelled target embeddings used for adaptation
    statistics (default: the query batch itself = transductive inference)."""
    tta = tta or TTAConfig()
    rng = np.random.RandomState(seed)
    dev = q_z.device
    c_tmem = None
    mem = memory if memory is not None else getattr(model, "memory", None)
    if tta.tcr > 0 and getattr(model, "uses_memory", False) and mem is not None:
        c_tmem, _ = target_conditioned_prototypes(mem, q_z if t_z is None else t_z, getattr(model, "topk", 5))
    L, P, U, B, MS, PI, LP, SH = [], [], [], [], [], [], [], []
    for _ in range(M):
        i0, i1 = np.where(pool_y == 0)[0], np.where(pool_y == 1)[0]          # identical to sample_support_indices
        s = np.concatenate([rng.choice(i0, K, replace=len(i0) < K), rng.choice(i1, K, replace=len(i1) < K)])
        s_z = pool_z[torch.as_tensor(s, device=dev)]
        s_y = torch.as_tensor(np.array([0] * K + [1] * K), device=dev)
        if getattr(model, "variant", "") == "matching" or tta.align == "none" and tta.rho == 0 and tta.tcr == 0 \
                and not tta.est_prior and not tta.center:
            out = model.predict(q_z, s_z, s_y, memory, force_beta)            # untouched baseline path
            le, u, info = out["logits"].float(), out["uncertainty"].float(), out
            lp, pi, sh = torch.zeros(2, device=dev), torch.full((2,), 0.5, device=dev), torch.zeros((), device=dev)
        else:
            protos, info = fused_prototypes(model, s_z, s_y, memory, force_beta, c_tmem, tta.tcr)
            if t_z is None:
                a = adapt_logits(model, q_z, protos, tta)
                le, u = a["logits"], a["uncertainty"]
            else:                       # statistics from a separate unlabelled set, then inductive on q_z
                a = adapt_logits(model, t_z, protos, tta)
                le, u = _apply_adapted(model, q_z, protos, a, tta)
            lp, pi, sh = a["log_prior"], a["pi"], a["shift_norm"]
        L.append(le); P.append((le + lp).softmax(-1)[:, 1] if lp.abs().sum() > 0 else _probs(model, le)[:, 1])
        U.append(u); B.append(info["beta"].float()); MS.append(info["mem_sim"].float())
        PI.append(pi); LP.append(lp); SH.append(sh)
    return {"logits": torch.stack(L).cpu(), "p_raw": torch.stack(P).mean(0).cpu().numpy(),
            "u": torch.stack(U).mean(0).cpu().numpy(), "beta": torch.stack(B).mean(0).cpu().numpy(),
            "mem_sim": torch.stack(MS).mean(0).cpu().numpy(),
            "log_prior": torch.stack(LP).cpu(), "pi_hat": float(torch.stack(PI)[:, 1].mean()),
            "shift_norm": float(torch.stack(SH).mean())}


def _probs(model, le):
    """Raw class probabilities consistent with the head: Dirichlet mean for EDL, softmax otherwise."""
    if getattr(model, "uses_edl", False):
        alpha = le.clamp(max=40.0).exp() + 1.0
        return alpha / alpha.sum(-1, keepdim=True)
    return le.softmax(-1)


def _apply_adapted(model, q_z, protos, a, cfg: TTAConfig):
    """Inductive use: the shift / rectified prototypes were estimated on a separate unlabelled set."""
    m_s, xi, z = protos.float().mean(0), a["xi"], q_z.float()
    if cfg.center:
        zz, cc = F.normalize(z + xi - m_s, dim=-1), F.normalize(a["protos"] - m_s, dim=-1)
    else:
        zz, cc = F.normalize(z + xi, dim=-1), a["protos"]
    le = _head_logits(model, zz, cc)
    alpha = le.clamp(max=40.0).exp() + 1.0
    return le, alpha.size(-1) / alpha.sum(-1)


def calibrated_from_logits_prior(logits: torch.Tensor, T: float, log_prior: Optional[torch.Tensor] = None) -> np.ndarray:
    """evaluate.calibrated_from_logits with an optional per-support-set log prior added AFTER tempering
    (the prior must not be divided by T)."""
    x = logits / T
    if log_prior is not None:
        x = x + log_prior.unsqueeze(1)
    return x.softmax(-1)[..., 1].mean(0).numpy()


# ----------------------------------------------------------------------------------------------
# P3. Vacuity / dissonance decomposition of the evidential output
# ----------------------------------------------------------------------------------------------
def evidential_uncertainties(logits: torch.Tensor, T: float = 1.0) -> Dict[str, np.ndarray]:
    """logits: [M, N, 2] log-evidence per support set (the "logits" entry of proto_ensemble).
    With alpha = exp(log e / T) + 1,  S = sum alpha,  b = e / S (belief masses):
        vacuity     u    = 2 / S                    (no evidence: far from both prototypes)
        dissonance  diss = 2 min(b_0, b_1)          (Josang's dissonance, closed form for K = 2)
        total            = u + diss = 2 (1 - max_k alpha_k / S)   <- exact identity for K = 2
    so the misclassification-relevant uncertainty 1 - max p splits into a "distance" and an "ambiguity" part.
    Also returns the disagreement between the M support sets (std of p)."""
    e = (logits.double() / T).clamp(max=40.0).exp()
    S = e.sum(-1) + 2.0
    u = 2.0 / S
    diss = 2.0 * e.min(-1).values / S
    p1 = (e[..., 1] + 1.0) / S
    return {"vacuity": u.mean(0).numpy(), "dissonance": diss.mean(0).numpy(), "total": (u + diss).mean(0).numpy(),
            "ens_std": p1.std(0).numpy() if logits.size(0) > 1 else np.zeros(logits.size(1)),
            "p_dirichlet": p1.mean(0).numpy()}


def risk_coverage(y, p, score, thr: float = 0.5):
    """Selective prediction: sort by increasing uncertainty `score`; returns coverage grid, selective risk, AURC."""
    y = np.asarray(y).astype(int); err = ((np.asarray(p) >= thr).astype(int) != y).astype(float)
    order = np.argsort(np.asarray(score), kind="stable")
    cum = np.cumsum(err[order]) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    return cov, cum, float(cum.mean())


# ----------------------------------------------------------------------------------------------
# Self-test on synthetic embeddings (checks the code path and the algebra, NOT real-data efficacy)
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
    from urproto.config import Config
    from urproto.models import MemoryBank, ProtoModel
    try:
        from sklearn.metrics import f1_score, roc_auc_score
    except ImportError:
        f1_score = roc_auc_score = None

    torch.manual_seed(0); rs = np.random.RandomState(0)
    D = 64

    class _Enc(torch.nn.Module):
        emb_dim = D

    def unit(v):
        return v / v.norm()

    def domain(offset, direction, n, prev, sep=0.55, noise=2.6):
        y = (rs.rand(n) < prev).astype(int)
        mu = offset + sep * torch.as_tensor(2 * y - 1, dtype=torch.float32).unsqueeze(-1) * direction
        return F.normalize(mu + noise * torch.randn(n, D) / math.sqrt(D), dim=-1), y

    common = 3.0 * unit(torch.randn(D))                       # all chest X-rays share a large common component
    d_src, d_perp = unit(torch.randn(D)), unit(torch.randn(D))
    d_tgt = unit(0.5 * d_src + 0.87 * d_perp)                 # class direction rotated by ~60 deg in the target
    o_src, o_nih = common, common + 0.8 * unit(torch.randn(D))
    o_tgt = common + 0.8 * unit(torch.randn(D)) + 0.45 * d_src   # target offset has a component ALONG the source
    #                                                             class direction -> "everything looks like pneumonia"
    model = ProtoModel(_Enc(), "full", Config()).eval()
    pool_z, pool_y = domain(o_src, d_src, 2000, 0.5)
    val_z, val_y = domain(o_src, d_src, 2000, 0.5)            # stands in for CXR-Val
    m_z, m_y = domain(o_nih, unit(0.3 * d_src + 0.95 * d_tgt), 6000, 0.15)
    model.memory = MemoryBank(m_z[m_y == 0], m_z[m_y == 1])

    def fit_T(logits, y):                                     # same criterion as evaluate.fit_T_ensemble (grid)
        best = (1e9, 1.0)
        for T in np.exp(np.linspace(-4, 3, 141)):
            p = np.clip((logits / T).softmax(-1)[..., 1].mean(0).numpy(), 1e-7, 1 - 1e-7)
            best = min(best, (float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), float(T)))
        return best[1]

    def report(name, y, e, T):
        p = calibrated_from_logits_prior(e["logits"], T, e["log_prior"] if "log_prior" in e else None)
        print(f"{name:30s} AUROC={roc_auc_score(y, p):.3f} macroF1={f1_score(y, (p >= .5).astype(int), average='macro'):.3f}"
              f" rec1={(p[y == 1] >= .5).mean():.2f} rec0={(p[y == 0] < .5).mean():.2f}"
              f" pi_hat={e.get('pi_hat', float('nan')):.2f} shift={e.get('shift_norm', 0):.2f} T={T:.3f}")

    T0 = fit_T(proto_ensemble_tta(model, val_z, pool_z, pool_y, 20, 5, 1, tta=TTAConfig(align="none"))["logits"], val_y)
    variants = [("P1 shift", TTAConfig()),
                ("P1 shift + centre", TTAConfig(center=True)),
                ("P1 shift, calibrated E-step", TTAConfig(em_temp=T0, iters=10)),
                ("P1 shift + prior EM (bal.)", TTAConfig(em_temp=T0, iters=10, est_prior=True)),
                ("P1 shift + prior EM (est.)", TTAConfig(em_temp=T0, iters=10, est_prior=True, decision_prior="estimated")),
                ("P1 + P2a TCR", TTAConfig(tcr=1.0)),
                ("P1 + P2b EM rho=0.5", TTAConfig(rho=0.5, em_temp=T0)),
                ("P1 + P2a + P2b", TTAConfig(tcr=1.0, rho=0.5, em_temp=T0)),
                ("P1 + P2 + prior(estimated)", TTAConfig(tcr=1.0, rho=0.5, em_temp=T0, iters=10, est_prior=True, decision_prior="estimated"))]
    for prev in (0.5, 0.75):
        q_z, q_y = domain(o_tgt, d_tgt, 4000, prev)
        print(f"--- synthetic target, prevalence {prev} (source supports) ---")
        base = proto_ensemble_tta(model, q_z, pool_z, pool_y, 20, 5, 7, tta=TTAConfig(align="none"))
        from urproto import evaluate as EV
        ref = EV.proto_ensemble(model, q_z, pool_z, pool_y, 20, 5, 7)
        assert np.allclose(ref["p_raw"], base["p_raw"], atol=1e-6), "align='none' must equal proto_ensemble"
        report("no TTA", q_y, base, T0)
        for name, cfg in variants:
            # the temperature is re-fitted on the validation stand-in with the SAME pipeline (no target labels)
            Tv = fit_T(proto_ensemble_tta(model, val_z, pool_z, pool_y, 20, 5, 1, tta=cfg)["logits"], val_y)
            report(name, q_y, proto_ensemble_tta(model, q_z, pool_z, pool_y, 20, 5, 7, tta=cfg), Tv)
    print("align='none' == evaluate.proto_ensemble : OK")

    # --- P3 identity: vacuity + dissonance == 2 (1 - max Dirichlet-mean probability), and bounds
    lg = base["logits"]
    unc = evidential_uncertainties(lg)
    e = lg.double().clamp(max=40).exp(); S = e.sum(-1) + 2
    lhs = (2 / S + 2 * e.min(-1).values / S); rhs = 2 * (1 - ((e + 1) / S.unsqueeze(-1)).max(-1).values)
    assert torch.allclose(lhs, rhs, atol=1e-9)
    assert (unc["dissonance"] <= 1 - unc["vacuity"] + 1e-9).all() and (unc["vacuity"] > 0).all()
    cov, risk, aurc = risk_coverage(q_y, base["p_raw"], unc["total"])
    _, _, aurc_v = risk_coverage(q_y, base["p_raw"], unc["vacuity"])
    print(f"P3 identity u + diss = 2(1 - max p): OK; diss <= 1 - u: OK; AURC(total)={aurc:.3f} AURC(vacuity)={aurc_v:.3f}")
