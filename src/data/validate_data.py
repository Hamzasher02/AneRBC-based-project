# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 1.2 - Dataset Validation Logic
# ============================================================
"""
validate_data.py
----------------
Module containing functions for data validation checks on the AneRBC dataset.
Detects corrupted/unreadable images, verifies label mapping and class
distribution, reports invalid extensions, and generates report artifacts.
"""

import csv
import json
from pathlib import Path
from PIL import Image
import matplotlib
# Use Agg backend for non-GUI environments (avoids window creation issues)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_class_folders(raw_dir: Path) -> list:
    """
    Discover all class directories inside the raw data directory.

    Args:
        raw_dir: Path to the raw data root directory.

    Returns:
        list of Path: Sorted list of class directories.

    Assumptions:
        raw_dir exists and is a directory. Non-directory or hidden folders
        starting with '.' are ignored.
    """
    return sorted([
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])


def build_class_mapping(class_folders: list) -> dict:
    """
    Build a deterministic class-to-integer label mapping.

    Args:
        class_folders: List of Path objects representing class directories.

    Returns:
        dict: Mapping from class names (str) to integer labels (int).

    Assumptions:
        Class folders are sorted alphabetically to ensure deterministic mapping.
    """
    class_names = sorted([p.name for p in class_folders])
    return {name: idx for idx, name in enumerate(class_names)}


def list_image_files(raw_dir: Path) -> list:
    """
    Recursively list all files within the class directories of raw_dir.

    Args:
        raw_dir: Root directory of raw data containing class folders.

    Returns:
        list of Path: Path objects for all files within class folders.

    Assumptions:
        raw_dir contains class directories. All files (regardless of extension)
        are returned to allow reporting invalid extensions.
    """
    class_folders = discover_class_folders(raw_dir)
    all_files = []
    for folder in class_folders:
        for p in folder.rglob("*"):
            if p.is_file():
                all_files.append(p)
    return sorted(all_files)


def validate_image_file(path: Path, allowed_extensions: set) -> dict:
    """
    Validate a single file: verify extension and check for image corruption.

    Args:
        path: Path to the file to validate.
        allowed_extensions: Set of allowed lowercase file extensions.

    Returns:
        dict: A record containing:
              - filepath (str, relative or absolute path)
              - class_name (str)
              - extension (str)
              - status (str: 'valid' | 'corrupt' | 'invalid_extension')
              - width (int or None)
              - height (int or None)
              - mode (str or None)
              - error_message (str)

    Assumptions:
        The path represents a valid existing file.
    """
    class_name = path.parent.name
    ext = path.suffix.lower()

    record = {
        "filepath": str(path),
        "class_name": class_name,
        "extension": ext,
        "status": "valid",
        "width": None,
        "height": None,
        "mode": None,
        "error_message": ""
    }

    # 1. Check extension
    if ext not in allowed_extensions:
        record["status"] = "invalid_extension"
        record["error_message"] = f"Extension '{ext}' is not in allowed set."
        return record

    # 2. Check corruption by attempting to open and verify the image
    try:
        with Image.open(path) as img:
            img.verify()  # Basic structural integrity verification
        
        # Verify closes the file; re-open to load pixels and read metadata
        with Image.open(path) as img:
            img.load()  # Decompresses pixel data to detect deeper corruptions
            record["width"], record["height"] = img.size
            record["mode"] = img.mode
    except Exception as e:
        record["status"] = "corrupt"
        record["error_message"] = str(e)

    return record


def validate_dataset(raw_dir: Path, allowed_extensions: set) -> tuple:
    """
    Validate the entire raw dataset directory.

    Args:
        raw_dir: Path to the raw dataset root directory.
        allowed_extensions: Set of allowed lowercase image extensions.

    Returns:
        tuple: (results, summary)
               - results (list of dict): Individual file records.
               - summary (dict): Consolidated stats and class mapping.

    Assumptions:
        raw_dir exists.
    """
    class_folders = discover_class_folders(raw_dir)
    class_mapping = build_class_mapping(class_folders)
    all_files = list_image_files(raw_dir)

    results = []
    class_counts = {name: 0 for name in class_mapping}
    corrupt_count = 0
    invalid_ext_count = 0
    valid_count = 0

    for path in all_files:
        record = validate_image_file(path, allowed_extensions)
        class_name = record["class_name"]
        
        # Add integer label if class is recognized
        record["label"] = class_mapping.get(class_name, -1)
        results.append(record)

        if record["status"] == "valid":
            valid_count += 1
            if class_name in class_counts:
                class_counts[class_name] += 1
        elif record["status"] == "corrupt":
            corrupt_count += 1
        elif record["status"] == "invalid_extension":
            invalid_ext_count += 1

    passes_validation = (valid_count > 0) and (corrupt_count == 0) and (invalid_ext_count == 0)

    summary = {
        "dataset_path": str(raw_dir.resolve()),
        "total_classes": len(class_folders),
        "class_mapping": class_mapping,
        "class_distribution": class_counts,
        "total_files_scanned": len(all_files),
        "valid_images": valid_count,
        "corrupted_images": corrupt_count,
        "invalid_extension_files": invalid_ext_count,
        "passes_validation": passes_validation
    }

    return results, summary


def save_validation_report(results: list, output_csv: Path) -> None:
    """
    Save detailed file validation results to a CSV file.

    Args:
        results: List of validation records (dictionaries).
        output_csv: Target path to write the CSV report.

    Assumptions:
        results list is not empty and keys match CSV columns.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["filepath", "class_name", "label", "extension", "status", "width", "height", "mode", "error_message"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            # Format filepath to use forward slashes for cross-platform reports
            row_copy = row.copy()
            row_copy["filepath"] = row_copy["filepath"].replace("\\", "/")
            writer.writerow(row_copy)


def save_class_distribution_plot(class_counts: dict, output_png: Path) -> None:
    """
    Create and save a bar chart of the class distribution.

    Args:
        class_counts: Dict mapping class name (str) to image count (int).
        output_png: Target path to save the generated PNG plot.

    Assumptions:
        class_counts is not empty. Matplotlib is configured.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color="skyblue", edgecolor="navy")
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center", va="bottom"
        )

    plt.title("AneRBC Dataset Class Distribution (Valid Images)", fontsize=14, fontweight="bold")
    plt.xlabel("Class Name", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output_png, dpi=150)
    plt.close()


def write_summary_json(summary: dict, output_json: Path) -> None:
    """
    Write dataset validation summary statistics to a JSON file.

    Args:
        summary: Summary statistics dictionary.
        output_json: Target path to write the JSON file.

    Assumptions:
        summary dictionary contains JSON serializable values.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
