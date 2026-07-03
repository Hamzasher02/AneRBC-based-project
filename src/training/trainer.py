# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
trainer.py
----------
Training loop with validation, early stopping, and checkpoint saving.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    """Run one validation/test epoch and return average loss and accuracy."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return running_loss / total, correct / total


def train(model, train_loader, val_loader, optimizer, scheduler=None,
          epochs=50, device="cpu", save_path="checkpoints/best.pth",
          patience=10):
    """
    Full training loop with early stopping and best-model checkpointing.

    Args:
        model        : nn.Module to train.
        train_loader : DataLoader for training split.
        val_loader   : DataLoader for validation split.
        optimizer    : PyTorch optimiser.
        scheduler    : LR scheduler (optional).
        epochs       : Maximum number of epochs.
        device       : 'cpu' or 'cuda'.
        save_path    : Where to save the best model checkpoint.
        patience     : Early-stopping patience (epochs without improvement).
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    no_improve   = 0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch [{epoch:03d}/{epochs}] "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")

        # Checkpoint best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    return history
