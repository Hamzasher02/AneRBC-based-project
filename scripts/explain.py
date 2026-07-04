# ============================================================
# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 4 - XAI explanations for best custom and pretrained models
# ============================================================
"""
explain.py
----------
CLI entry-point for generating XAI saliency maps using Captum.
Supports Integrated Gradients and GradCAM.
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.custom_cnn import build_custom_model
from src.models.transfer_model import build_transfer_model
from src.xai.explainer import integrated_gradients, gradcam, visualise_attribution


def parse_args():
    """Parse command line arguments for the explanation script."""
    p = argparse.ArgumentParser(description="XAI explanations for AneRBC classifier")
    p.add_argument("--method",      type=str, required=True,
                   choices=["integrated_gradients", "gradcam"])
    p.add_argument("--model",       type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--image",       type=str, required=True, help="Path to input image")
    p.add_argument("--target",      type=int, default=None,  help="Target class (auto if omitted)")
    p.add_argument("--num_classes", type=int, default=2,
                   help="Number of output classes (default: 2 for AneRBC)")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--save_dir",    type=str, default="outputs/figures/xai")
    return p.parse_args()


def load_image(image_path, img_size):
    """
    Load and preprocess a single image for inference.

    Args:
        image_path (str): Path to input image.
        img_size (int): Size to resize the image to.

    Returns:
        tuple: (preprocessed_tensor, original_numpy_array)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf   = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img_pil = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img_np  = np.array(img_pil)
    tensor  = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    return tensor, img_np


def update_xai_summary(model_name, method, image_path, target_class, save_path, attr):
    """
    Save/update a summary of the XAI run in outputs/reports/xai_summary.json.

    Args:
        model_name (str): Name of the model.
        method (str): Explanation method used.
        image_path (str): Path to the input image.
        target_class (int): Target class index.
        save_path (str): Path to saved saliency map.
        attr (np.ndarray): Saliency/attribution values.
    """
    summary_path = Path("outputs/reports/xai_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    new_entry = {
        "model_name": model_name,
        "method": method,
        "image_path": str(image_path),
        "target_class": int(target_class),
        "save_path": str(save_path),
        "attribution_mean": float(attr.mean()),
        "attribution_std": float(attr.std())
    }

    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
    else:
        data = []

    # Check if entry already exists (by model, method, and image)
    updated = False
    for i, entry in enumerate(data):
        if (entry.get("model_name") == model_name and
            entry.get("method") == method and
            entry.get("image_path") == str(image_path)):
            data[i] = new_entry
            updated = True
            break

    if not updated:
        data.append(new_entry)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Updated XAI summary -> {summary_path}")


def main():
    """Main function to run explanations."""
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # -- Model ----------------------------------------------------------------
    if args.model in ("custom_cnn_3", "custom_cnn_4", "custom_cnn_5", "custom_cnn"):
        model = build_custom_model(args.model, num_classes=args.num_classes)
    else:
        model = build_transfer_model(model_name=args.model, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device).eval()

    # -- Image ----------------------------------------------------------------
    tensor, img_np = load_image(args.image, args.img_size)
    tensor = tensor.to(device)

    # Auto-detect target class from model prediction
    with torch.no_grad():
        logits = model(tensor)
        pred_class = logits.argmax(dim=1).item()
    target_class = args.target if args.target is not None else pred_class
    print(f"Predicted class: {pred_class}  |  Target class for XAI: {target_class}")

    # -- Explain ---------------------------------------------------------------
    stem = Path(args.image).stem
    if args.method == "integrated_gradients":
        attr = integrated_gradients(model, tensor, target_class)
        title = f"Integrated Gradients - class {target_class}"
    else:  # gradcam
        # Select target convolution layer precisely
        if args.model in ("custom_cnn_3", "custom_cnn_4", "custom_cnn_5", "custom_cnn"):
            target_layer = model.features[-1].block[0]
        elif args.model == "resnet18" or args.model == "resnet50":
            target_layer = model.layer4[-1].conv2
        elif args.model == "mobilenet_v2":
            target_layer = model.features[-1][0]
        elif args.model == "squeezenet1_0":
            last_conv = None
            for m in model.features.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = m
            target_layer = last_conv
        else:
            # Fallback helper to find last convolution layer
            last_conv = None
            target_module = getattr(model, "features", model)
            for m in target_module.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = m
            target_layer = last_conv

        attr  = gradcam(model, tensor, target_class, target_layer)
        title = f"GradCAM - class {target_class}"

    save_path = save_dir / f"{args.method}_{args.model}_{stem}.png"
    visualise_attribution(img_np, attr, title=title,
                          save_path=str(save_path), show=False)
    print(f"Saliency map saved -> {save_path}")

    # Save XAI summary JSON
    update_xai_summary(args.model, args.method, args.image, target_class, save_path, attr)


if __name__ == "__main__":
    main()
