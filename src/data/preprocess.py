# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 1.3 - Preprocessing and Cleaning Pipeline
# ============================================================
"""
preprocess.py
-------------
Module containing functions for data preprocessing and cleaning on the AneRBC dataset.
Supports resizing, converting to RGB, CLAHE contrast enhancement, Gaussian
denoising, and saving cleaned outputs to data/processed.
"""

import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def get_image_files(raw_dir: Path, allowed_extensions: set) -> list:
    """
    Recursively find all valid image files under raw_dir.

    Args:
        raw_dir: Path to the raw data root directory.
        allowed_extensions: Set of allowed lowercase image extensions.

    Returns:
        list of Path: Path objects for discovered image files.

    Assumptions:
        raw_dir exists. Files starting with '.' are ignored.
    """
    image_paths = []
    # Recurse through directories
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_extensions and not p.name.startswith("."):
            image_paths.append(p)
    return sorted(image_paths)


def ensure_output_folder(output_path: Path) -> None:
    """
    Ensure the parent directory of output_path exists.

    Args:
        output_path: Target file path whose directory must exist.

    Assumptions:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)


def load_image_rgb(image_path: Path) -> Image.Image:
    """
    Load an image from disk and guarantee it is in RGB mode.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL.Image.Image: The image loaded in RGB mode.

    Assumptions:
        The file is a valid image and can be opened by PIL.
    """
    return Image.open(image_path).convert("RGB")


def resize_image(image: Image.Image, size: int) -> Image.Image:
    """
    Resize a PIL Image to a square dimension using bilinear interpolation.

    Args:
        image: PIL Image object.
        size: Target size (integer for square dimensions).

    Returns:
        PIL.Image.Image: Resized PIL Image object.

    Assumptions:
        image is a valid PIL Image.
    """
    return image.resize((size, size), Image.Resampling.BILINEAR)


def apply_clahe_rgb(image: Image.Image) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an RGB image.
    Converts to LAB space, applies CLAHE to the L (luminance) channel, and converts back.

    Args:
        image: PIL Image object.

    Returns:
        PIL.Image.Image: Contrast-enhanced PIL Image object.

    Assumptions:
        image is in RGB mode.
    """
    # Convert PIL Image to OpenCV NumPy array (RGB to BGR/RGB)
    img_np = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Create CLAHE operator and apply to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced_np)


def apply_gaussian_denoise(image: Image.Image) -> Image.Image:
    """
    Apply Gaussian denoising (blurring) to an RGB image using a 3x3 kernel.

    Args:
        image: PIL Image object.

    Returns:
        PIL.Image.Image: Denoised PIL Image object.

    Assumptions:
        image is in RGB mode.
    """
    img_np = np.array(image)
    # Apply Gaussian Blur with kernel size (3,3)
    denoised_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    return Image.fromarray(denoised_np)


def preprocess_single_image(
    input_path: Path,
    output_path: Path,
    image_size: int,
    use_clahe: bool,
    use_denoise: bool,
    overwrite: bool
) -> dict:
    """
    Preprocess a single image: load, resize, apply optional cleaning, and save.

    Args:
        input_path: Path to the raw source image.
        output_path: Target path to write the preprocessed image.
        image_size: Target square dimensions (e.g., 224).
        use_clahe: If True, apply CLAHE contrast enhancement.
        use_denoise: If True, apply Gaussian denoising.
        overwrite: If True, overwrite output file if it exists.

    Returns:
        dict: A record containing filepath, status (processed/skipped/failed),
              and optional error message.

    Assumptions:
        input_path exists.
    """
    record = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "processed",
        "error_message": ""
    }

    if output_path.exists() and not overwrite:
        record["status"] = "skipped"
        return record

    try:
        ensure_output_folder(output_path)
        
        # Load in RGB mode
        img = load_image_rgb(input_path)
        
        # Resize to square
        img = resize_image(img, image_size)
        
        # Apply optional CLAHE
        if use_clahe:
            img = apply_clahe_rgb(img)
            
        # Apply optional Gaussian Denoise
        if use_denoise:
            img = apply_gaussian_denoise(img)
            
        # Save preprocessed image to disk (non-normalized RGB representation)
        img.save(output_path, format="JPEG", quality=95)
        
    except Exception as e:
        record["status"] = "failed"
        record["error_message"] = str(e)

    return record


def preprocess_dataset(
    raw_dir: Path,
    processed_dir: Path,
    image_size: int,
    use_clahe: bool,
    use_denoise: bool,
    overwrite: bool
) -> tuple:
    """
    Preprocess the entire raw dataset and preserve class structure.

    Args:
        raw_dir: Path to raw dataset root.
        processed_dir: Path to processed output root.
        image_size: Target square dimensions.
        use_clahe: If True, apply CLAHE.
        use_denoise: If True, apply Gaussian denoising.
        overwrite: If True, overwrite outputs.

    Returns:
        tuple: (records, summary)
               - records (list of dict): Individual processing results.
               - summary (dict): Summary statistics of the run.

    Assumptions:
        raw_dir exists.
    """
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = get_image_files(raw_dir, allowed_extensions)

    records = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    class_counts = {}

    for input_path in image_paths:
        class_name = input_path.parent.name
        output_path = processed_dir / class_name / f"{input_path.stem}.jpg"
        
        record = preprocess_single_image(
            input_path, output_path, image_size, use_clahe, use_denoise, overwrite
        )
        records.append(record)

        if record["status"] == "processed":
            processed_count += 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        elif record["status"] == "skipped":
            skipped_count += 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        elif record["status"] == "failed":
            failed_count += 1

    summary = {
        "raw_dir": str(raw_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "image_size": image_size,
        "use_clahe": use_clahe,
        "use_denoise": use_denoise,
        "total_images_found": len(image_paths),
        "total_processed": processed_count,
        "total_skipped": skipped_count,
        "total_failed": failed_count,
        "class_distribution": class_counts
    }

    return records, summary


def save_preprocessing_report(records: list, output_csv: Path) -> None:
    """
    Save detailed preprocessing file logs to a CSV report.

    Args:
        records: List of preprocessing records (dictionaries).
        output_csv: Target path to write the CSV report.

    Assumptions:
        records list is not empty and keys match CSV columns.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["input_path", "output_path", "status", "error_message"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in records:
            # Format paths to use forward slashes for cross-platform reports
            row_copy = row.copy()
            row_copy["input_path"] = row_copy["input_path"].replace("\\", "/")
            row_copy["output_path"] = row_copy["output_path"].replace("\\", "/")
            writer.writerow(row_copy)


def write_preprocessing_summary(summary: dict, output_json: Path) -> None:
    """
    Save preprocessing summary statistics to a JSON report.

    Args:
        summary: Summary statistics dictionary.
        output_json: Target path to write the JSON report.

    Assumptions:
        summary contains JSON serializable values.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
