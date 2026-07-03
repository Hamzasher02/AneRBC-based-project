# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
explain.py
----------
CLI entry-point for generating XAI saliency maps using Captum.
Supports Integrated Gradients and GradCAM.

Usage:
    python scripts/explain.py --method gradcam --model resnet50 \
        --checkpoint checkpoints/resnet50_best.pth \
        --image data/raw/AneRBC/class_1/sample.png

    python scripts/explain.py --method integrated_gradients --model custom_cnn \
        --checkpoint checkpoints/custom_cnn_best.pth \
        --image data/raw/AneRBC/class_2/sample.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_model import build_transfer_model
from src.xai.explainer import integrated_gradients, gradcam, visualise_attribution


def parse_args():
    p = argparse.ArgumentParser(description="XAI explanations for AneRBC classifier")
    p.add_argument("--method",      type=str, required=True,
                   choices=["integrated_gradients", "gradcam"])
    p.add_argument("--model",       type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--image",       type=str, required=True, help="Path to input image")
    p.add_argument("--target",      type=int, default=None,  help="Target class (auto if omitted)")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--save_dir",    type=str, default="outputs/figures/xai")
    return p.parse_args()


def load_image(image_path, img_size):
    """Load and preprocess a single image for inference."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf   = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img_pil = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img_np  = np.array(img_pil)                   # for overlay display
    tensor  = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    return tensor, img_np


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    if args.model == "custom_cnn":
        model = build_custom_cnn(num_classes=args.num_classes)
    else:
        model = build_transfer_model(backbone=args.model, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device).eval()

    # ── Image ────────────────────────────────────────────────────────────────
    tensor, img_np = load_image(args.image, args.img_size)
    tensor = tensor.to(device)

    # Auto-detect target class from model prediction
    with torch.no_grad():
        logits = model(tensor)
        pred_class = logits.argmax(dim=1).item()
    target_class = args.target if args.target is not None else pred_class
    print(f"Predicted class: {pred_class}  |  Target class for XAI: {target_class}")

    # ── Explain ───────────────────────────────────────────────────────────────
    stem = Path(args.image).stem
    if args.method == "integrated_gradients":
        attr = integrated_gradients(model, tensor, target_class)
        title = f"Integrated Gradients — class {target_class}"
    else:  # gradcam
        # Default: last conv layer (model-specific)
        if args.model == "custom_cnn":
            target_layer = model.features[-1].block[0]
        elif args.model in ("resnet18", "resnet50"):
            target_layer = model.layer4[-1]
        elif args.model == "efficientnet_b0":
            target_layer = model.features[-1]
        else:
            target_layer = model.features[-1]

        attr  = gradcam(model, tensor, target_class, target_layer)
        title = f"GradCAM — class {target_class}"

    save_path = save_dir / f"{args.method}_{args.model}_{stem}.png"
    visualise_attribution(img_np, attr, title=title,
                          save_path=str(save_path), show=False)
    print(f"Saliency map saved → {save_path}")


if __name__ == "__main__":
    main()
