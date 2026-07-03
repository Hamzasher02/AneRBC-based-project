# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task   : 1.2 - Dataset Preprocessing and Split Generation
# ============================================================
"""
prepare_data.py
---------------
Task 1.2: Preprocess the AneRBC dataset.

What this script does:
  1. Searches data/raw_downloads/ for the AneRBC folder structure.
  2. Identifies the two main classes:
       - Healthy individuals  -> label 0
       - Anemic individuals   -> label 1
  3. Copies Original_images only into data/raw/<class>/.
  4. Generates stratified train / val / test split CSVs:
       data/splits/train.csv
       data/splits/val.csv
       data/splits/test.csv
     Each CSV has columns: filepath (relative to data/raw/), label, class_name.

AneRBC expected raw_downloads structure (after extracting archive.zip):
  data/raw_downloads/
    AneRBC-I/
      Healthy_individuals/
        Original_images/   <-- used
        Binary_segmented/  (not used)
        RGB_segmented/     (not used)
      Anemic_individuals/
        Original_images/   <-- used

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --raw_dl data/raw_downloads --dry_run
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Folder-name keyword -> (canonical label name, integer label)
CLASS_MAP = [
    (["healthy", "normal", "non_anemic", "nonanemic"], "healthy", 0),
    (["anemic",  "anaemic", "anemia", "anaemia"],      "anaemic", 1),
]

# Preferred image sub-folder names (checked in order)
ORIGINAL_HINTS = ["Original_images", "original_images", "Images", "images"]

# ANSI colours (safe on modern Windows terminals)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg):   return f"{GREEN}[OK]    {RESET}{msg}"
def _warn(msg): return f"{YELLOW}[WARN]  {RESET}{msg}"
def _err(msg):  return f"{RED}[ERROR] {RESET}{msg}"
def _info(msg): return f"{CYAN}[INFO]  {RESET}{msg}"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def is_image(path: Path) -> bool:
    """Return True if path is a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_class(folder_name: str):
    """
    Map a folder name to (canonical_class_name, int_label).

    Returns (None, None) if no keyword matches.
    """
    name = folder_name.lower().replace(" ", "_").replace("-", "_")
    for keywords, canonical, label in CLASS_MAP:
        if any(kw in name for kw in keywords):
            return canonical, label
    return None, None


def find_class_image_dirs(raw_dl: Path) -> list:
    """
    Walk raw_dl recursively, return list of
    (image_dir: Path, canonical: str, label: int)
    for every folder that matches a known class and contains images.
    """
    found = []
    visited_canonical: dict[str, set] = {}

    for folder in sorted(raw_dl.rglob("*")):
        if not folder.is_dir():
            continue

        canonical, label = resolve_class(folder.name)
        if canonical is None:
            continue

        # Prefer an "Original_images" subfolder
        img_dir = folder
        for hint in ORIGINAL_HINTS:
            candidate = folder / hint
            if candidate.is_dir():
                img_dir = candidate
                break

        images = [f for f in img_dir.iterdir() if is_image(f)]
        if not images:
            continue

        # Avoid duplicating the same canonical class from nested paths
        key = str(img_dir.resolve())
        if canonical not in visited_canonical:
            visited_canonical[canonical] = set()
        if key in visited_canonical[canonical]:
            continue
        visited_canonical[canonical].add(key)

        found.append((img_dir, canonical, label))

    return found


def collect_all_images(class_dirs: list) -> dict:
    """
    Build {canonical: [(src_path, label), ...]} from discovered dirs.
    """
    class_images: dict[str, list] = {}
    for img_dir, canonical, label in class_dirs:
        paths = sorted([f for f in img_dir.iterdir() if is_image(f)])
        class_images.setdefault(canonical, []).extend(
            (p, label) for p in paths
        )
    return class_images


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_splits(items: list, val_ratio: float, test_ratio: float, seed: int):
    """
    Random stratified split into train / val / test.

    Returns (train_list, val_list, test_list).
    """
    rng = random.Random(seed)
    pool = items[:]
    rng.shuffle(pool)
    n = len(pool)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    return pool[n_test + n_val:], pool[n_test:n_test + n_val], pool[:n_test]


def safe_copy(src: Path, dst_dir: Path, dry_run: bool) -> Path:
    """
    Copy src to dst_dir.  Rename on collision: stem_1.ext, stem_2.ext ...
    Returns the final destination path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        count = 1
        while dst.exists():
            dst = dst_dir / f"{src.stem}_{count}{src.suffix}"
            count += 1
    if not dry_run:
        shutil.copy2(src, dst)
    return dst


def save_csv(rows: list, csv_path: Path, dry_run: bool):
    """Write (filepath, label, class_name) rows to CSV."""
    if dry_run:
        print(_info(f"  [DRY-RUN] Would write {len(rows)} rows -> {csv_path}"))
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "class_name"])
        writer.writerows(rows)
    print(_ok(f"{len(rows):>6} rows  ->  {csv_path}"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="AneRBC dataset preprocessor and split generator (Task 1.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw_dl",     default="data/raw_downloads",
                   help="Extracted downloads folder (default: data/raw_downloads)")
    p.add_argument("--raw_out",    default="data/raw",
                   help="Organised class-image output (default: data/raw)")
    p.add_argument("--splits_dir", default="data/splits",
                   help="CSV output directory (default: data/splits)")
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--dry_run",    action="store_true",
                   help="Simulate without writing any files")
    return p.parse_args()


def main() -> None:
    """
    Main entry-point for Task 1.2 dataset preprocessing.

    Steps:
      1. Validate raw_downloads directory.
      2. Discover class image directories.
      3. Copy images into data/raw/<class>/.
      4. Generate and save train/val/test CSV splits.
    """
    args = parse_args()
    raw_dl     = Path(args.raw_dl)
    raw_out    = Path(args.raw_out)
    splits_dir = Path(args.splits_dir)

    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  AneRBC Dataset Preprocessor - Task 1.2{RESET}")
    print(f"  Author : Hamza Sher  |  SRN : 3012260007")
    print(f"{BOLD}{'='*62}{RESET}\n")

    if args.dry_run:
        print(_warn("DRY-RUN mode: no files will be written.\n"))

    # Step 1 - validate
    if not raw_dl.exists():
        print(_err(f"Directory not found: {raw_dl.resolve()}"))
        print(_info("Extract archive.zip first, then re-run this script."))
        raise SystemExit(1)

    # Step 2 - discover
    print(_info(f"Scanning:  {raw_dl.resolve()}"))
    class_dirs = find_class_image_dirs(raw_dl)

    if not class_dirs:
        print(_err("No matching class folders found inside raw_downloads."))
        print(_info("Expected folders with keywords: healthy / normal  OR  anemic / anaemic"))
        raise SystemExit(1)

    print()
    print(f"{BOLD}[>>] DISCOVERED CLASS DIRECTORIES{RESET}")
    print("-" * 62)
    for img_dir, canonical, label in class_dirs:
        n = sum(1 for f in img_dir.iterdir() if is_image(f))
        rel = img_dir.relative_to(raw_dl)
        print(f"  Label {label}  '{canonical:<10}'  <-  {rel}  ({n} images)")
    print()

    # Step 3 - collect, split, copy
    class_images = collect_all_images(class_dirs)
    split_rows: dict[str, list] = {"train": [], "val": [], "test": []}

    print(f"{BOLD}[>>] COPYING IMAGES  ->  data/raw/{RESET}")
    print("-" * 62)

    for canonical in sorted(class_images):
        items = class_images[canonical]
        label = items[0][1]
        train, val, test = make_splits(items, args.val_ratio, args.test_ratio, args.seed)
        print(f"  '{canonical}' (label={label}):  total={len(items):>5}  "
              f"train={len(train)}  val={len(val)}  test={len(test)}")

        dst_dir = raw_out / canonical
        for split_name, bucket in [("train", train), ("val", val), ("test", test)]:
            for src, lbl in bucket:
                dst = safe_copy(src, dst_dir, dry_run=args.dry_run)
                rel = dst.relative_to(raw_out)
                split_rows[split_name].append(
                    (str(rel).replace("\\", "/"), lbl, canonical)
                )

    print()

    # Step 4 - write CSVs
    print(f"{BOLD}[>>] WRITING SPLIT CSVs  ->  data/splits/{RESET}")
    print("-" * 62)
    for split_name in ("train", "val", "test"):
        save_csv(split_rows[split_name], splits_dir / f"{split_name}.csv",
                 dry_run=args.dry_run)

    total = sum(len(v) for v in split_rows.values())
    print()
    print(f"{BOLD}{'='*62}{RESET}")
    print(f"{GREEN}{BOLD}  [OK]  Task 1.2 complete.{RESET}")
    print(f"       Total images  : {total}")
    print(f"       Train / Val / Test : "
          f"{len(split_rows['train'])} / "
          f"{len(split_rows['val'])} / "
          f"{len(split_rows['test'])}")
    if not args.dry_run:
        print(f"       Splits folder: {splits_dir.resolve()}")
        print(f"\n       Next step: python scripts/train.py --model custom_cnn")
    print(f"{BOLD}{'='*62}{RESET}\n")


if __name__ == "__main__":
    main()
