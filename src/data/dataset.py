# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
dataset.py
----------
PyTorch Dataset class for the AneRBC image classification dataset.
Handles image loading, label encoding, and optional transforms.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class AneRBCDataset(Dataset):
    """
    AneRBC Image Classification Dataset.

    Expects a CSV split file with columns:
        filepath  (relative to data_root)
        label     (integer class index)

    Args:
        split_csv (str | Path): Path to the CSV split file.
        data_root (str | Path): Root directory that contains the images.
        transform (callable, optional): torchvision transforms to apply.
    """

    def __init__(self, split_csv, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.df = pd.read_csv(split_csv)

        # Validate required columns
        required = {"filepath", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["filepath"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode: str = "train", img_size: int = 224):
    """
    Returns standard image transforms for train / val / test splits.

    Args:
        mode    : One of 'train', 'val', 'test'.
        img_size: Target image size (square).
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloader(split_csv, data_root, mode="train",
                   img_size=224, batch_size=32,
                   num_workers=4, shuffle=None):
    """
    Factory function that returns a DataLoader for a given split.

    Args:
        split_csv  : Path to the CSV split file.
        data_root  : Root image directory.
        mode       : 'train', 'val', or 'test'.
        img_size   : Target image size.
        batch_size : Mini-batch size.
        num_workers: DataLoader worker processes.
        shuffle    : Override shuffle; defaults to True for train.
    """
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = AneRBCDataset(
        split_csv=split_csv,
        data_root=data_root,
        transform=get_transforms(mode, img_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
