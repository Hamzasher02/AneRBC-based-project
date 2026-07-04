# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
train.py
--------
CLI entry-point for training Custom CNN or Transfer Learning models
on the AneRBC dataset.

Usage examples:
    python scripts/train.py --model custom_cnn --epochs 50 --batch_size 32 --lr 1e-3
    python scripts/train.py --model resnet50 --pretrained --epochs 30 --lr 1e-4
    python scripts/train.py --model resnet50 --resume checkpoints/resnet50_best.pth
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.optim as optim

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_dataloader
from src.models.custom_cnn import build_custom_model
from src.models.transfer_model import build_transfer_model, get_trainable_parameter_summary
from src.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train AneRBC classifier")
    p.add_argument("--model",       type=str,   default="custom_cnn",
                   help="Model: 'custom_cnn_3' | 'custom_cnn_4' | 'custom_cnn_5' | 'custom_cnn' | 'resnet18' | 'resnet50' | 'efficientnet_b0' | 'vgg16' | 'mobilenet_v2' | 'squeezenet1_0'")
    p.add_argument("--pretrained",  action="store_true", help="Use ImageNet weights (transfer only)")
    p.add_argument("--freeze",      action="store_true", help="Freeze backbone features")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--num_classes", type=int,   default=2)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--splits_dir",  type=str,   default="data/splits")
    p.add_argument("--data_root",   type=str,   default="data/processed")
    p.add_argument("--resume",      type=str,   default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device",      type=str,   default=None, help="'cpu' or 'cuda' (auto-detect if omitted)")
    return p.parse_args()


def save_history_and_plots(history, model_name):
    """
    Save the training history to JSON, plot the learning curves, and update the
    validation summary CSV file.
    """
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    
    reports_dir = Path("outputs/reports")
    figures_dir = Path("outputs/figures")
    reports_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Save history JSON
    history_path = reports_dir / f"history_{model_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"Saved history -> {history_path}")
    
    # 2. Save learning curve PNG
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title(f"Loss Curves - {model_name}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_title(f"Accuracy Curves - {model_name}")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    curve_path = figures_dir / f"learning_curve_{model_name}.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved learning curve -> {curve_path}")
    
    # 3. Update validation summary CSV
    best_idx = 0
    best_val_acc = 0.0
    for i, acc in enumerate(history["val_acc"]):
        if acc > best_val_acc:
            best_val_acc = acc
            best_idx = i
    best_val_loss = history["val_loss"][best_idx]
    
    if any(x in model_name for x in ["resnet", "mobilenet", "squeezenet", "vgg", "efficientnet"]):
        csv_path = reports_dir / "transfer_validation_summary.csv"
    else:
        csv_path = reports_dir / "custom_cnn_validation_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["model_name", "best_val_loss", "best_val_acc"])
        
    if model_name in df["model_name"].values:
        df.loc[df["model_name"] == model_name, "best_val_loss"] = best_val_loss
        df.loc[df["model_name"] == model_name, "best_val_acc"] = best_val_acc
    else:
        new_row = pd.DataFrame([{"model_name": model_name, "best_val_loss": best_val_loss, "best_val_acc": best_val_acc}])
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_csv(csv_path, index=False)
    print(f"Updated validation summary -> {csv_path}")


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- DataLoaders ----------------------------------------------------------
    splits = Path(args.splits_dir)
    train_loader = get_dataloader(splits / "train.csv", args.data_root, mode="train",
                                  img_size=args.img_size, batch_size=args.batch_size,
                                  num_workers=args.workers)
    val_loader   = get_dataloader(splits / "val.csv",   args.data_root, mode="val",
                                  img_size=args.img_size, batch_size=args.batch_size,
                                  num_workers=args.workers)

    # -- Model ----------------------------------------------------------------
    if args.model in ("custom_cnn_3", "custom_cnn_4", "custom_cnn_5", "custom_cnn"):
        model = build_custom_model(args.model, num_classes=args.num_classes)
    else:
        model = build_transfer_model(model_name=args.model, num_classes=args.num_classes,
                                     pretrained=args.pretrained, freeze_backbone=args.freeze)
        if args.pretrained:
            summary = get_trainable_parameter_summary(model)
            print(f"\n--- {args.model} Parameter Summary ---")
            print(f"Total parameters      : {summary['total_params']:,}")
            print(f"Trainable parameters  : {summary['trainable_params']:,}")
            print(f"Frozen parameters     : {summary['frozen_params']:,}")
            print(f"Trainable layers count: {len(summary['trainable_layer_names'])}")
            print(f"Trainable layers      : {summary['trainable_layer_names']}\n")
    model = model.to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    # -- Optimiser & Scheduler ------------------------------------------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # -- Train ----------------------------------------------------------------
    save_path = f"checkpoints/{args.model}_best.pth"
    Path("checkpoints").mkdir(exist_ok=True)

    history = train(model, train_loader, val_loader, optimizer, scheduler,
                    epochs=args.epochs, device=device,
                    save_path=save_path, patience=args.patience)

    print(f"\nTraining complete. Best model saved -> {save_path}")

    # Save training evidence (Task 2.2)
    save_history_and_plots(history, args.model)


if __name__ == "__main__":
    main()
