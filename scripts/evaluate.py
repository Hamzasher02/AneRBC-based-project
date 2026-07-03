# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
evaluate.py
-----------
CLI entry-point for evaluating a trained model on the test split.
Saves classification report, confusion matrix, and ROC-AUC to outputs/.

Usage:
    python scripts/evaluate.py --model resnet50 --checkpoint checkpoints/resnet50_best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_dataloader
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_model import build_transfer_model
from src.evaluation.metrics import get_predictions, compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AneRBC classifier")
    p.add_argument("--model",       type=str, required=True,
                   help="Model name: custom_cnn | resnet18 | resnet50 | efficientnet_b0 | vgg16")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to saved .pth checkpoint")
    p.add_argument("--num_classes", type=int, default=2,
                   help="Number of classes (default: 2 for AneRBC healthy/anaemic)")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--splits_dir",  type=str, default="data/splits")
    p.add_argument("--data_root",   type=str, default="data/raw")
    p.add_argument("--device",      type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    splits     = Path(args.splits_dir)
    out_figs   = Path("outputs/figures"); out_figs.mkdir(parents=True, exist_ok=True)
    out_reports = Path("outputs/reports"); out_reports.mkdir(parents=True, exist_ok=True)

    # -- Load class names from test CSV (no separate class_map.csv needed) ---
    import pandas as pd
    test_df = pd.read_csv(splits / "test.csv")
    if "class_name" in test_df.columns:
        class_names = (
            test_df[["label", "class_name"]]
            .drop_duplicates()
            .sort_values("label")["class_name"]
            .tolist()
        )
    else:
        class_names = [str(i) for i in range(args.num_classes)]

    # -- DataLoader -----------------------------------------------------------
    test_loader = get_dataloader(splits / "test.csv", args.data_root, mode="test",
                                 img_size=args.img_size, batch_size=args.batch_size)

    # -- Model ----------------------------------------------------------------
    if args.model == "custom_cnn":
        model = build_custom_cnn(num_classes=args.num_classes)
    else:
        model = build_transfer_model(backbone=args.model, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    # -- Evaluate -------------------------------------------------------------
    labels, preds, probs = get_predictions(model, test_loader, device)
    results = compute_metrics(labels, preds, probs, class_names)

    print(f"\nAccuracy : {results['accuracy']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(results["classification_report"])

    # -- Save confusion matrix -------------------------------------------------
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {args.model}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    cm_path = out_figs / f"confusion_matrix_{args.model}.png"
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Confusion matrix saved -> {cm_path}")

    # -- Save report JSON -----------------------------------------------------
    report_path = out_reports / f"metrics_{args.model}.json"
    with open(report_path, "w") as f:
        json.dump({
            "model":    args.model,
            "accuracy": results["accuracy"],
            "roc_auc":  results["roc_auc"],
            "classification_report": results["classification_report"],
        }, f, indent=2)
    print(f"Metrics report saved -> {report_path}")


if __name__ == "__main__":
    main()
