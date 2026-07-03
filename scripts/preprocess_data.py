# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 1.3 - Preprocessing and Cleaning CLI Script
# ============================================================
"""
preprocess_data.py
------------------
CLI script for executing the image preprocessing and cleaning pipeline.
Prepares the raw data for model training, saving output images in data/processed.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import (
    preprocess_dataset,
    save_preprocessing_report,
    write_preprocessing_summary
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset preprocessing script.

    Returns:
        argparse.Namespace: Object containing argument values.

    Assumptions:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Resize, clean, and convert raw RBC images to processed formats."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to the raw data root directory (default: data/raw)"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to save processed images (default: data/processed)"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="outputs/reports",
        help="Path to save report logs (default: outputs/reports)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target size for image resize (default: 224)"
    )
    parser.add_argument(
        "--use-clahe",
        action="store_true",
        help="Apply optional CLAHE contrast normalization"
    )
    parser.add_argument(
        "--use-denoise",
        action="store_true",
        help="Apply optional Gaussian denoising"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files"
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution logic for dataset preprocessing and cleaning.

    Flow:
      1. Parse CLI arguments.
      2. Verify directories exist.
      3. Call preprocessor from src/data/preprocess.
      4. Write output logs (CSV) and summary stats (JSON).
      5. Print a clean, ASCII-only execution report to the console.

    Returns:
        None.

    Assumptions:
        None.
    """
    args = parse_arguments()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    reports_dir = Path(args.reports_dir)

    print("==============================================================")
    print("  AneRBC Dataset Preprocessing Pipeline - Task 1.3")
    print("  Author : Hamza Sher  |  SRN : 3012260007")
    print("==============================================================")

    # Validate raw directory
    if not raw_dir.exists():
        print(f"[ERROR] Raw data root directory not found: {raw_dir.resolve()}")
        print("[INFO] Please verify raw image files are in place under data/raw/.")
        sys.exit(1)

    print(f"[INFO] Source Raw Directory     : {raw_dir.resolve()}")
    print(f"[INFO] Destination Processed    : {processed_dir.resolve()}")
    print(f"[INFO] Configuration Options    :")
    print(f"  * Target Size                 : {args.image_size}x{args.image_size}")
    print(f"  * CLAHE Contrast Enhancement  : {'ENABLED' if args.use_clahe else 'DISABLED'}")
    print(f"  * Gaussian Denoising          : {'ENABLED' if args.use_denoise else 'DISABLED'}")
    print(f"  * Overwrite Existing Files    : {'ENABLED' if args.overwrite else 'DISABLED'}")

    print("\n[INFO] Running preprocessing pipeline, please wait...")
    
    # Execute batch preprocessing
    records, summary = preprocess_dataset(
        raw_dir,
        processed_dir,
        args.image_size,
        args.use_clahe,
        args.use_denoise,
        args.overwrite
    )

    # Establish report file paths
    csv_path = reports_dir / "preprocessing_report.csv"
    json_path = reports_dir / "preprocessing_summary.json"

    # Save outputs
    save_preprocessing_report(records, csv_path)
    write_preprocessing_summary(summary, json_path)

    # Print summary report
    print("\n--------------------------------------------------------------")
    print("[>>] PREPROCESSING EXECUTION SUMMARY")
    print("--------------------------------------------------------------")
    print(f"  Total Images Discovered   : {summary['total_images_found']}")
    print(f"  Total Images Processed    : {summary['total_processed']}")
    print(f"  Total Images Skipped      : {summary['total_skipped']}")
    print(f"  Total Images Failed       : {summary['total_failed']}")
    print("\n  Class Distribution (Processed/Skipped):")
    for name, count in sorted(summary["class_distribution"].items()):
        print(f"    * {name:<15} : {count} files")
    print("--------------------------------------------------------------")
    
    if summary["total_failed"] == 0:
        print("[OK]    PREPROCESSING COMPLETED SUCCESSFULLY")
    else:
        print(f"[WARN]  PREPROCESSING COMPLETED WITH {summary['total_failed']} FAILURES")
        print("        Check outputs/reports/preprocessing_report.csv for details.")
    print("--------------------------------------------------------------")
    
    print("\n[INFO] Generated Artifacts:")
    print(f"  * Processed Images Dir  : {processed_dir.resolve()}")
    print(f"  * Detailed CSV Report   : Saved -> {csv_path}")
    print(f"  * Summary JSON Stats    : Saved -> {json_path}")
    print("==============================================================")


if __name__ == "__main__":
    main()
