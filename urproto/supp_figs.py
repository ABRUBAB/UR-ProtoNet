"""Supplementary Fig. S2: examples of the Stage III augmentation, on the XAI example radiographs.

    python -m urproto.supp_figs --xai runs/ac_1/urp-xai2/out/xai_seed42.npz --out revision/figs/augmentation.png
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import gpu_augment_dg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--xai", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_aug", type=int, default=5)
    a = p.parse_args(argv)
    d = np.load(a.xai, allow_pickle=True)
    imgs = [d["cxr_test_imgs"][0], d["cxr_test_imgs"][3], d["chexpert_imgs"][2], d["chexpert_imgs"][4]]
    names = ["CXR-Test, Pneumonia", "CXR-Test, Normal", "CheXpert, Pneumonia", "CheXpert, Normal"]
    torch.manual_seed(0)
    fig, axes = plt.subplots(len(imgs), a.n_aug + 1, figsize=(1.8 * (a.n_aug + 1), 1.9 * len(imgs)))
    for i, (im, nm) in enumerate(zip(imgs, names)):
        x = torch.as_tensor(np.asarray(im, np.float32) / 255.0)[None, None].repeat(a.n_aug, 1, 1, 1)
        aug = gpu_augment_dg(x).squeeze(1).numpy()
        axes[i, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_ylabel(nm, fontsize=8)
        for j in range(a.n_aug):
            axes[i, j + 1].imshow(aug[j], cmap="gray", vmin=0, vmax=1)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("Preprocessed", fontsize=9)
    for j in range(a.n_aug):
        axes[0, j + 1].set_title(f"Augmented {j + 1}", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
