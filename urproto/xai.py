"""Gradient-based (Grad-CAM, Grad-CAM++) and perturbation-based (LIME) explanations of
UR-ProtoNet decisions. The explained score is the log-evidence of the predicted class with respect
to fixed fused prototypes, computed through the real encoder."""

import numpy as np
import torch
import torch.nn.functional as F

from .data import to_input


class CamExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.acts = None
        self.grads = None
        layer = encoder.backbone.bn2 if hasattr(encoder.backbone, "bn2") else encoder.backbone.conv_head
        self.h = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self.h.remove()


def explain_cams(model, protos, imgs_u8, device):
    """Returns (cam, campp, pred, prob) for each uint8 image; protos are fixed [2, D] prototypes."""
    enc = model.encoder
    enc.eval()
    ext = CamExtractor(enc)
    cams, campps, preds, probs = [], [], [], []
    for a in imgs_u8:
        x = to_input(a[None], device).requires_grad_(False)
        enc.zero_grad(set_to_none=True)
        z = enc(x)
        out = model.head(z, protos)
        c = int(out["probs"][0].argmax())
        score = out["logits"][0, c]
        score.backward()
        A, G = ext.acts.detach()[0], ext.grads.detach()[0]            # [C, h, w]
        w = G.mean(dim=(1, 2))
        cam = F.relu((w[:, None, None] * A).sum(0))
        g2, g3 = G ** 2, G ** 3
        alpha = g2 / (2 * g2 + A.sum(dim=(1, 2), keepdim=True) * g3 + 1e-8)
        wpp = (alpha * F.relu(G)).sum(dim=(1, 2))
        campp = F.relu((wpp[:, None, None] * A).sum(0))
        up = lambda m: F.interpolate(m[None, None], size=a.shape, mode="bilinear", align_corners=False)[0, 0]
        norm = lambda m: (m - m.min()) / (m.max() - m.min() + 1e-8)
        cams.append(norm(up(cam)).cpu().numpy()); campps.append(norm(up(campp)).cpu().numpy())
        preds.append(c); probs.append(float(out["probs"][0, 1].detach()))
    ext.remove()
    return np.stack(cams), np.stack(campps), np.array(preds), np.array(probs)


def explain_lime(model, protos, imgs_u8, device, T=1.0, num_samples=1000, seed=0):
    from lime import lime_image
    enc = model.encoder

    @torch.no_grad()
    def clf(batch):
        g = (np.asarray(batch)[..., 0] * 255.0).clip(0, 255).astype(np.uint8)
        outs = []
        for i in range(0, len(g), 128):
            z = enc(to_input(g[i:i + 128], device))
            le = model.head(z, protos)["logits"]
            outs.append((le / T).softmax(-1).cpu().numpy())
        return np.concatenate(outs)

    exp = lime_image.LimeImageExplainer(random_state=seed)
    res = []
    for a in imgs_u8:
        rgb = np.repeat(a[..., None], 3, axis=-1).astype(np.float64) / 255.0
        e = exp.explain_instance(rgb, clf, top_labels=2, hide_color=0, num_samples=num_samples, random_seed=seed)
        lab = e.top_labels[0]
        _, m_pos = e.get_image_and_mask(lab, positive_only=True, num_features=5, hide_rest=False)
        _, m_all = e.get_image_and_mask(lab, positive_only=False, num_features=10, hide_rest=False)
        res.append((lab, m_pos, m_all))
    return res
