# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task   : 1.1 - Dataset Preparation (placement check)
# ============================================================
"""
download_data.py
----------------
AneRBC dataset placement checker and instructions script.

WHY NO AUTOMATIC DOWNLOAD?
    The AneRBC (Anaemia Red Blood Cell Classification) dataset must be
    downloaded manually.  The recommended source is Figshare (open access).
    An alternative copy is also hosted on Kaggle under the slug:
        jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images

    Kaggle downloads require an account and API credentials.  Some medical
    image datasets are also distributed under restricted licences that
    prohibit automated redistribution.

    This script therefore:
      1. Prints step-by-step instructions for obtaining the dataset from
         both Figshare (recommended) and Kaggle (alternative).
      2. Validates that the dataset has been placed in the correct
         location before any preprocessing or training is attempted.
      3. Never deletes, moves, or modifies any existing files.

EXPECTED DIRECTORY STRUCTURE after manual placement:

    data/raw/
      healthy/          (one folder per RBC morphology class)
        image001.jpg
        image002.jpg
        ...
      anaemic/
        image001.jpg
        ...

    Note: folder names must match the actual extracted class names from
    the dataset.  Adjust as needed (e.g. "normal", "sickle", "thalassemia").

Usage:
    python scripts/download_data.py [--raw_dir data/raw]
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Supported image extensions (case-insensitive)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# ANSI colour helpers (work on modern Windows terminals / Linux / macOS)
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"{GREEN}[OK]    {RESET}{msg}"
def _warn(msg: str) -> str: return f"{YELLOW}[WARN]  {RESET}{msg}"
def _err(msg: str)  -> str: return f"{RED}[ERROR] {RESET}{msg}"
def _info(msg: str) -> str: return f"{CYAN}[INFO]  {RESET}{msg}"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Print a formatted banner identifying the script and project."""
    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  AneRBC Dataset Placement Checker - Task 1.1{RESET}")
    print(f"  Author : Hamza Sher  |  SRN : 3012260007")
    print(f"  Project: Medical Image Classification (Deep Learning CW)")
    print(f"{BOLD}{'='*62}{RESET}\n")


def print_manual_instructions() -> None:
    """
    Print step-by-step manual download and placement instructions
    for the AneRBC dataset.

    Two sources are documented:
      - Figshare (recommended): open-access, no account required.
      - Kaggle (alternative): requires account + API token.
        Slug: jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images

    This function prints instructions only; it does not download anything.
    """
    print(f"{BOLD}[>>] MANUAL DATASET PLACEMENT INSTRUCTIONS{RESET}")
    print("-" * 62)
    print("""
  DATASET: AneRBC - Anaemia Red Blood Cell Classification

  The AneRBC dataset must be downloaded manually.
  Two sources are available:

  -----------------------------------------------------------
  RECOMMENDED SOURCE - Figshare (open access, no login needed)
  -----------------------------------------------------------
    * Visit the official AneRBC Figshare page:
      https://figshare.com/
      (search: "AneRBC" OR "Anaemia Red Blood Cell")
    * Download the dataset archive directly from Figshare.
    * No account or API token required.

  -----------------------------------------------------------
  ALTERNATIVE SOURCE - Kaggle
  -----------------------------------------------------------
    Option A (Browser):
      * Visit: https://www.kaggle.com/datasets/jocelyndumlao/
                  anerbc-anemia-diagnosis-using-rbc-images
      * Sign in to your Kaggle account.
      * Click 'Download' and save the ZIP archive.

    Option B (Kaggle CLI - requires kaggle.json API token):
      pip install kaggle
      kaggle datasets download \\
          -d jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images \\
          -p data/raw_downloads --unzip
      (then manually move class folders into data/raw/)

  -----------------------------------------------------------
  STEP 2 - Arrange class folders under data/raw/
  -----------------------------------------------------------
    After downloading and extracting, ensure each RBC class is
    its own direct sub-folder inside data/raw/:

      data/raw/
        healthy/
          image001.jpg
          image002.jpg
        anaemic/
          image001.jpg
          ...

    Folder names must match the actual class names in the archive.
    Common names: healthy, anaemic, normal, sickle, thalassemia.
    Adjust folder names to match your extracted archive.

  -----------------------------------------------------------
  STEP 3 - Verify placement
  -----------------------------------------------------------
      python scripts/download_data.py

  [WARN]  data/raw/ contents are listed in .gitignore.
          NEVER commit raw image files to the repository.
""")
    print("-" * 62)


def check_raw_dir_exists(raw_dir: Path) -> bool:
    """
    Check that the data/raw directory exists on disk.

    Args:
        raw_dir: Absolute or relative Path to the raw data directory.

    Returns:
        True if the directory exists, False otherwise.
    """
    if raw_dir.exists() and raw_dir.is_dir():
        print(_ok(f"data/raw directory found  ->  {raw_dir.resolve()}"))
        return True
    else:
        print(_err(f"data/raw directory NOT found at: {raw_dir.resolve()}"))
        print(_info("Create it with:  mkdir -p data/raw"))
        return False


def discover_class_folders(raw_dir: Path) -> list[Path]:
    """
    Discover class sub-folders inside data/raw.

    A valid class folder is any direct child directory of raw_dir.
    Files at the top level of raw_dir (e.g. README, .gitkeep) are ignored.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        Sorted list of Path objects - one per class folder found.
    """
    class_dirs = sorted([
        p for p in raw_dir.iterdir()
        if p.is_dir()
    ])
    return class_dirs


def count_images_in_folder(folder: Path) -> int:
    """
    Recursively count all image files inside a given folder.

    Args:
        folder: Path to the class directory to scan.

    Returns:
        Integer count of image files with a recognised extension.
    """
    return sum(
        1 for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(raw_dir: Path) -> bool:
    """
    Perform a full validation of the dataset directory.

    Checks performed:
      1. data/raw exists.
      2. At least one class sub-folder is present.
      3. Each class sub-folder contains at least one image.

    Prints a summary table of classes and image counts.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        True if the dataset appears valid, False otherwise.
    """
    # --- Check 1: directory exists ---
    if not check_raw_dir_exists(raw_dir):
        return False

    print()

    # --- Check 2: discover class folders ---
    class_dirs = discover_class_folders(raw_dir)

    if not class_dirs:
        print(_warn("data/raw exists but contains NO class sub-folders."))
        print(_info("Please follow the manual placement instructions above."))
        return False

    print(_ok(f"Total class folders found : {len(class_dirs)}"))
    print()

    # --- Check 3: image counts per class ---
    print(f"  {'Class Folder':<30}  {'Image Count':>12}")
    print(f"  {'-'*30}  {'-'*12}")

    total_images = 0
    empty_classes = []

    for cls_dir in class_dirs:
        n = count_images_in_folder(cls_dir)
        total_images += n
        status = "" if n > 0 else "  [WARN] empty"
        print(f"  {cls_dir.name:<30}  {n:>12}{status}")
        if n == 0:
            empty_classes.append(cls_dir.name)

    print(f"  {'-'*30}  {'-'*12}")
    print(f"  {'TOTAL':<30}  {total_images:>12}")
    print()

    # --- Summary ---
    if total_images == 0:
        print(_warn("data/raw is present but contains ZERO image files."))
        print(_warn("Please place AneRBC images inside the class sub-folders."))
        return False

    if empty_classes:
        print(_warn(f"The following class folders are empty: {empty_classes}"))
        print(_warn("Verify that all images were extracted correctly."))

    print(_ok(f"Dataset validated - {total_images} images across {len(class_dirs)} classes."))
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AneRBC dataset placement checker (Task 1.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Path to the raw data directory (default: data/raw)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry-point.

    Workflow:
      1. Print banner and manual placement instructions.
      2. Validate the raw data directory.
      3. Exit 0 on success, exit 1 if dataset is missing or empty.
    """
    args = parse_args()
    raw_dir = Path(args.raw_dir)

    print_banner()
    print_manual_instructions()

    print(f"{BOLD}[>>] VALIDATING DATASET PLACEMENT{RESET}")
    print("-" * 62)

    success = validate_dataset(raw_dir)

    print()
    print(f"{BOLD}{'='*62}{RESET}")
    if success:
        print(f"{GREEN}{BOLD}  [OK]  Dataset is in place. Proceed to Task 1.2 preprocessing.{RESET}")
        print(f"       Next step:  python scripts/prepare_data.py")
        sys.exit(0)
    else:
        print(f"{YELLOW}{BOLD}  [WARN]  Dataset NOT found or empty in: {raw_dir.resolve()}{RESET}")
        print(f"       Please follow the instructions printed above,")
        print(f"       then re-run:  python scripts/download_data.py")
        sys.exit(1)
    print(f"{BOLD}{'='*62}{RESET}\n")


if __name__ == "__main__":
    main()
