"""Production Single-Image Inference CLI for UR-ProtoNet.

Provides single-image clinical diagnostic inference, evidential Dirichlet probability estimation,
and Subjective Logic epistemic vacuity triage referral (autonomous report vs. radiologist deferral).
"""

import argparse
import os
from typing import Optional
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

if HAS_TORCH:
    from model import FeatureEncoder, URProtoNet, MemoryBank


def find_default_file(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0] if candidates else None


def find_default_checkpoint():
    return find_default_file([
        "checkpoints/ur_protonet_best.pt",
        "results/checkpoints/ur_protonet_best.pt",
        "../results/checkpoints/ur_protonet_best.pt",
        "checkpoints/best_meta_model.pt",
    ])


def find_default_memory_bank():
    return find_default_file([
        "checkpoints/memory_bank.pt",
        "results/checkpoints/memory_bank.pt",
        "../results/checkpoints/memory_bank.pt",
    ])


def find_default_image():
    candidates = [
        "assets/sample_normal.png",
        "assets/sample_pneumonia.png",
        "sample_normal.png",
        "../submission/fig9_raw_chexpert.png",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_args():
    default_img = find_default_image()
    parser = argparse.ArgumentParser(
        description="UR-ProtoNet Chest X-Ray Diagnostic Inference & Uncertainty Triage"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=default_img,
        help="Path to input chest radiograph (.png, .jpg). Defaults to sample demo radiograph if available.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=find_default_checkpoint(),
        help="Path to UR-ProtoNet model checkpoint (.pt)",
    )
    parser.add_argument(
        "--memory_bank",
        type=str,
        default=find_default_memory_bank(),
        help="Path to anatomical memory bank checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.50,
        help="Post-hoc temperature scaling parameter T* (default: 0.50 per paper Sec 2.4.4)",
    )
    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.691,
        help="Epistemic vacuity referral threshold tau (default: 0.691, optimal 28.9% coverage triage point per Table 6)",
    )
    return parser.parse_args()


def load_image_tensor(image_path: str, device: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    raw_img = Image.open(image_path).convert("RGB")
    if HAS_TORCHVISION:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(raw_img).unsqueeze(0).to(device)
    else:
        resized = raw_img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)


def main():
    args = parse_args()
    print("=" * 70)
    print("UR-ProtoNet Diagnostic Inference & Uncertainty Triage")
    print("=" * 70)

    if not args.image or not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found.")
        print("Please provide an image with --image <path_to_xray.png>")
        return

    tensor_img = load_image_tensor(args.image, args.device)

    # Initialize model and load weights
    mem_bank = None
    if args.memory_bank and os.path.exists(args.memory_bank):
        mem_bank = MemoryBank.from_file(args.memory_bank, device=args.device)
        print(f"Loaded anatomical memory bank from: {args.memory_bank}")

    encoder = FeatureEncoder(backbone_name="efficientnet_b0", pretrained=False)
    model = URProtoNet(encoder=encoder, memory_bank=mem_bank, temperature=0.5).to(args.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded trained checkpoint from: {args.checkpoint}")
    else:
        print(f"Notice: Checkpoint '{args.checkpoint}' not found. Running in demo mode.")

    model.eval()
    with torch.no_grad():
        feat = model.encoder(tensor_img)

        # Synthesize canonical class prototypes through Neural FusionGate
        if model.memory_bank is not None:
            c_norm_mem = F.normalize(model.memory_bank.mem_normal.mean(dim=0, keepdim=True), p=2, dim=-1)
            c_pneu_mem = F.normalize(model.memory_bank.mem_pneumonia.mean(dim=0, keepdim=True), p=2, dim=-1)
            # Pass priors through FusionGate to compute blended prototypes and gating weights
            proto_norm, beta_norm = model.fusion_gate(c_norm_mem, c_norm_mem)
            proto_pneu, beta_pneu = model.fusion_gate(c_pneu_mem, c_pneu_mem)
            prototypes = torch.cat([proto_norm, proto_pneu], dim=0)
            beta_vals = (beta_norm.item(), beta_pneu.item())
        else:
            proto_normal = F.normalize(torch.randn(1, 512, device=args.device), p=2, dim=-1)
            proto_pneu = F.normalize(torch.randn(1, 512, device=args.device), p=2, dim=-1)
            prototypes = torch.cat([proto_normal, proto_pneu], dim=0)
            beta_vals = (1.0, 1.0)

        probs, vacuity, evidence, calib_probs = model.edl_head.predict_calibrated(
            feat, prototypes, calib_temperature=args.temperature
        )

    prob_normal = calib_probs[0, 0].item()
    prob_pneu = calib_probs[0, 1].item()
    dir_normal = probs[0, 0].item()
    dir_pneu = probs[0, 1].item()
    vacuity_val = vacuity[0].item()
    pred_class = "PNEUMONIA" if prob_pneu >= 0.5 else "NORMAL"
    confidence = max(prob_normal, prob_pneu)

    print(f"\n[DIAGNOSTIC REPORT]")
    print(f"  Input Radiograph:          {args.image}")
    print(f"  Predicted Classification:  {pred_class}")
    print(f"  Calibrated Confidence:     {confidence * 100:.2f}% (Temperature T* = {args.temperature:.2f})")
    print(f"  Calibrated Probabilities:  Normal={prob_normal:.4f} | Pneumonia={prob_pneu:.4f}")
    print(f"  Dirichlet Evidential (p):  Normal={dir_normal:.4f} | Pneumonia={dir_pneu:.4f}")
    print(f"  Dirichlet Vacuity (u):     {vacuity_val:.4f}  (Referral Threshold tau = {args.uncertainty_threshold:.4f})")
    print(f"  FusionGate Weight (beta):  Normal={beta_vals[0]:.4f} | Pneumonia={beta_vals[1]:.4f} (Avg: {sum(beta_vals)/2:.4f})")
    print("-" * 70)

    if vacuity_val <= args.uncertainty_threshold:
        print("[CLINICAL TRIAGE DECISION]: LOW UNCERTAINTY (AUTONOMOUS TIER)")
        print(f"  -> Autonomous report approved for fast-track clinical documentation (Vacuity {vacuity_val:.4f} <= {args.uncertainty_threshold:.4f}).")
    else:
        print("[CLINICAL TRIAGE DECISION]: ELEVATED UNCERTAINTY (RADIOLOGIST REFERRAL)")
        print(f"  -> DEFERRED TO EXPERT RADIOLOGIST / CHEST CT FOR MANUAL VERIFICATION (Vacuity {vacuity_val:.4f} > {args.uncertainty_threshold:.4f}).")
    print("=" * 70)


if __name__ == "__main__":
    main()
