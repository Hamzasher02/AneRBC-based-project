# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
prepare_data.py
---------------
Scans data/raw/AneRBC/, encodes class labels, and writes
train / val / test CSV split files to data/splits/.

Usage:
    python scripts/prepare_data.py [--val_ratio 0.15] [--test_ratio 0.15] [--seed 42]
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Prepare AneRBC data splits")
    p.add_argument("--raw_dir",    type=str, default="data/raw/AneRBC")
    p.add_argument("--splits_dir", type=str, default="data/splits")
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    raw_dir    = Path(args.raw_dir)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Please download AneRBC and place it under data/raw/AneRBC/"
        )

    # Discover classes (sub-folders)
    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class sub-directories found in {raw_dir}")

    print(f"Found {len(classes)} classes: {classes}")
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    records = []
    for cls in classes:
        for img_path in (raw_dir / cls).iterdir():
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}:
                rel_path = img_path.relative_to(raw_dir.parent)
                records.append({"filepath": str(rel_path), "label": class_to_idx[cls]})

    random.shuffle(records)
    n     = len(records)
    n_val  = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": records[:n_train],
        "val":   records[n_train:n_train + n_val],
        "test":  records[n_train + n_val:],
    }

    for split_name, rows in splits.items():
        out_path = splits_dir / f"{split_name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  {split_name:5s}: {len(rows):5d} samples → {out_path}")

    # Save class mapping
    pd.DataFrame(
        [{"class_name": k, "label": v} for k, v in class_to_idx.items()]
    ).to_csv(splits_dir / "class_map.csv", index=False)
    print("Class mapping saved → data/splits/class_map.csv")


if __name__ == "__main__":
    main()
