# Author: Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task   : 1.4 - Dataset and Dataloaders Integration
# ============================================================
"""
dataset.py
----------
PyTorch Dataset class for the AneRBC image classification dataset.
Handles image loading, label encoding, online transforms, and
extension-agnostic image path resolution.
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Allowed lowercase image extensions for fallback resolution
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def resolve_image_path(data_root: Path, relative_path: Path) -> Path:
    """
    Resolve an image path by first checking the exact relative path, and falling
    back to alternative extensions if not found (e.g. .png in CSV vs .jpg on disk).

    Args:
        data_root: Root image directory.
        relative_path: Relative image path (from the CSV splits file).

    Returns:
        Path: The resolved absolute or relative Path to the existing image file.

    Raises:
        FileNotFoundError: If the image cannot be found with any allowed extension.
    """
    data_root = Path(data_root)
    relative_path = Path(relative_path)
    
    # 1. Check exact match
    exact_path = data_root / relative_path
    if exact_path.exists():
        return exact_path

    # 2. Try alternative extensions
    attempted = [str(exact_path)]
    for ext in ALLOWED_EXTENSIONS:
        candidate_rel = relative_path.with_suffix(ext)
        candidate_abs = data_root / candidate_rel
        attempted.append(str(candidate_abs))
        if candidate_abs.exists():
            return candidate_abs

    # 3. Raise error if still not found
    raise FileNotFoundError(
        f"Could not resolve image path for '{relative_path}' under '{data_root}'.\n"
        f"Attempted paths:\n" + "\n".join(f"  - {p}" for p in attempted)
    )


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
        rel_path = Path(row["filepath"])
        
        # Resolve path dynamically to handle extension mismatches (.png in CSV, .jpg on disk)
        img_path = resolve_image_path(self.data_root, rel_path)
        
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


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    data_root: str = "data/processed",
    splits_dir: str = "data/splits",
    img_size: int = 224
) -> tuple:
    """
    Convenience wrapper that returns loaders for all phases (train, val, test) and
    the list of class names.

    Args:
        batch_size : Mini-batch size for each loader.
        num_workers: Worker processes for loading.
        data_root  : Root processed or raw directory containing class folders.
        splits_dir : Directory containing split CSVs (train.csv, val.csv, test.csv).
        img_size   : Target size of images (square).

    Returns:
        tuple: (loaders, class_names)
               - loaders (dict): Dict containing train, val, and test DataLoader objects.
               - class_names (list): List of class names sorted by their integer label.
    """
    splits_path = Path(splits_dir)
    
    # Infer class names mapping from splits
    train_csv = splits_path / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        class_names = (
            df[["label", "class_name"]]
            .drop_duplicates()
            .sort_values("label")["class_name"]
            .tolist()
        )
    else:
        class_names = ["healthy", "anaemic"]

    loaders = {
        "train": get_dataloader(splits_path / "train.csv", data_root, "train",
                                img_size, batch_size, num_workers),
        "val":   get_dataloader(splits_path / "val.csv",   data_root, "val",
                                img_size, batch_size, num_workers),
        "test":  get_dataloader(splits_path / "test.csv",  data_root, "test",
                                img_size, batch_size, num_workers)
    }

    return loaders, class_names
