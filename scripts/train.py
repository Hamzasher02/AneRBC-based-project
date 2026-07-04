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
from src.models.transfer_model import build_transfer_model
from src.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train AneRBC classifier")
    p.add_argument("--model",       type=str,   default="custom_cnn",
                   help="Model: 'custom_cnn_3' | 'custom_cnn_4' | 'custom_cnn_5' | 'custom_cnn' | 'resnet18' | 'resnet50' | 'efficientnet_b0' | 'vgg16'")
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
        model = build_transfer_model(backbone=args.model, num_classes=args.num_classes,
                                     pretrained=args.pretrained, freeze_features=args.freeze)
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


if __name__ == "__main__":
    main()
