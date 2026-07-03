# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 1.2 - Dataset Validation Script
# ============================================================
"""
validate_data.py
----------------
Command-line script for executing dataset validation checks.
Loads images, checks for corruption, reports stats, and saves reports.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate_data import (
    validate_dataset,
    save_validation_report,
    save_class_distribution_plot,
    write_summary_json
)

# Allowed lowercase image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset validation script.

    Returns:
        argparse.Namespace: Object containing argument values.

    Assumptions:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Verify raw dataset files for corruption and extensions."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to the raw data root directory (default: data/raw)"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="outputs/reports",
        help="Path to save output report files (default: outputs/reports)"
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="outputs/figures",
        help="Path to save output distribution figures (default: outputs/figures)"
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry-point for executing dataset validation checks.

    Flow:
      1. Parse CLI arguments.
      2. Verify directory existence.
      3. Call validation logic from src/data/validate_data.
      4. Write outputs: CSV report, JSON summary, PNG chart.
      5. Print a clean, ASCII-only text summary to the console.

    Returns:
        None.

    Assumptions:
        None.
    """
    args = parse_arguments()
    raw_dir = Path(args.raw_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)

    print("==============================================================")
    print("  AneRBC Dataset Validation Checker - Task 1.2")
    print("  Author : Hamza Sher  |  SRN : 3012260007")
    print("==============================================================")

    # Validate directory existence
    if not raw_dir.exists():
        print(f"[ERROR] Raw dataset folder not found: {raw_dir.resolve()}")
        print("[INFO] Please run scripts/prepare_data.py first to copy the images.")
        sys.exit(1)

    print(f"[INFO] Scanning raw directory: {raw_dir.resolve()}")
    
    # Run validation
    results, summary = validate_dataset(raw_dir, ALLOWED_EXTENSIONS)

    # Establish output file paths
    csv_path = reports_dir / "data_validation_report.csv"
    json_path = reports_dir / "data_validation_summary.json"
    plot_path = figures_dir / "class_distribution.png"

    # Save reports
    save_validation_report(results, csv_path)
    write_summary_json(summary, json_path)
    if summary["valid_images"] > 0:
        save_class_distribution_plot(summary["class_distribution"], plot_path)
        plot_status = f"Saved -> {plot_path}"
    else:
        plot_status = "Not generated (no valid images)"

    # Print summary to console (ASCII-only)
    print("\n--------------------------------------------------------------")
    print("[>>] DATA VALIDATION REPORT SUMMARY")
    print("--------------------------------------------------------------")
    print(f"  Dataset Path        : {summary['dataset_path']}")
    print(f"  Total Classes Found : {summary['total_classes']}")
    print("  Class Mapping       :")
    for name, idx in sorted(summary["class_mapping"].items(), key=lambda x: x[1]):
        print(f"    * {name:<15} -> Label {idx}")
    print("\n  Class Distribution (Valid Images):")
    for name, count in sorted(summary["class_distribution"].items()):
        print(f"    * {name:<15} : {count} files")
    
    print("\n  Scanned File Statistics:")
    print(f"    * Total Files Scanned    : {summary['total_files_scanned']}")
    print(f"    * Valid Image Files      : {summary['valid_images']}")
    print(f"    * Corrupted Image Files  : {summary['corrupted_images']}")
    print(f"    * Invalid Extension Files: {summary['invalid_extension_files']}")
    
    print("--------------------------------------------------------------")
    if summary["passes_validation"]:
        print("[OK]    DATASET PASSES VALIDATION CHECKS")
        print("        No corrupted files or invalid extensions found.")
    else:
        print("[WARN]  DATASET CONTAINS ERRORS")
        if summary["corrupted_images"] > 0:
            print(f"        * Found {summary['corrupted_images']} corrupted files.")
        if summary["invalid_extension_files"] > 0:
            print(f"        * Found {summary['invalid_extension_files']} invalid files.")
        print("        Please review the detailed CSV log for file paths.")
    print("--------------------------------------------------------------")
    
    print("\n[INFO] Generated Artifacts:")
    print(f"  * Detailed CSV Report : Saved -> {csv_path}")
    print(f"  * Summary JSON Stats  : Saved -> {json_path}")
    print(f"  * Distribution Plot   : {plot_status}")
    print("==============================================================")


if __name__ == "__main__":
    main()
