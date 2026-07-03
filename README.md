# Medical Image Classification with Custom CNNs, Transfer Learning, and XAI

> **Author:** Hamza Sher | **SRN:** 3012260007  
> **Course:** Deep Learning Coursework  
> **Dataset:** AneRBC Image Classification Dataset  
> **Framework:** PyTorch

---

## Project Overview

This project implements a full pipeline for medical image classification on the **AneRBC** (Anaemia Red Blood Cell) dataset, covering:

- Custom CNN architectures built from scratch in PyTorch
- Transfer learning with pretrained backbone models
- Explainable AI (XAI) using Captum (GradCAM, Integrated Gradients, SHAP)
- Quantitative evaluation with metrics, confusion matrices, and classification reports

---

## Project Structure

```
uniproject/
├── data/
│   ├── raw/            # Original downloaded dataset (not committed to git)
│   ├── processed/      # Preprocessed & normalised images (not committed)
│   └── splits/         # Train / val / test CSV split files
├── notebooks/          # Jupyter notebooks for exploration & reporting
├── scripts/            # CLI helper scripts (download, preprocess, etc.)
├── src/
│   ├── data/           # Dataset classes, transforms, data loaders
│   ├── models/         # Custom CNN + transfer learning model definitions
│   ├── training/       # Training loops, loss functions, optimisers
│   ├── evaluation/     # Evaluation metrics and report generation
│   └── xai/            # Explainability (GradCAM, Integrated Gradients)
├── outputs/
│   ├── figures/        # Saved plots and visualisations
│   └── reports/        # JSON / CSV evaluation reports
├── checkpoints/        # Saved model weights (not committed to git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd uniproject
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Task 1.1 -- Dataset Preparation

### Dataset: AneRBC (Anaemia Red Blood Cell Classification)

The AneRBC dataset consists of microscopic blood-smear images categorised by
RBC morphology. It must be **downloaded manually**.

> Full dataset documentation: [data/README_DATA.md](data/README_DATA.md)

### Sources

| Source | URL | Account needed? |
|---|---|---|
| **Figshare** (recommended) | https://figshare.com/ -- search "AneRBC" | No |
| Kaggle (alternative) | https://www.kaggle.com/datasets/jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images | Yes |

### Step 1 -- Download

**Recommended -- Figshare (no account required):**

Visit https://figshare.com/ and search for **AneRBC** or **Anaemia Red Blood Cell**.
Download the archive directly.

**Alternative -- Kaggle CLI** (requires `kaggle.json` token at `~/.kaggle/`):

```bash
kaggle datasets download \
    -d jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images \
    -p data/raw_downloads --unzip
```

### Step 2 -- Arrange Class Folders

After extracting, place each RBC class in its own direct sub-folder inside `data/raw/`:

```
data/raw/
  healthy/
    image001.jpg
    image002.jpg
  anaemic/
    image001.jpg
    ...
```

> Folder names must match the actual class names in your downloaded archive.
> Common AneRBC names: `healthy`, `anaemic`.
> Adjust if the archive uses different names (e.g. `normal`, `sickle`).

### Step 3 -- Verify Placement

```bash
python scripts/download_data.py
```

This script will print:
- Total class folders found
- Class folder names and image counts per class
- A warning if `data/raw/` is empty or missing
- Next-step instructions on success

### Git Tracking Note

`data/raw/` contents are **excluded from git** via `.gitignore`:

```gitignore
data/raw/*
!data/raw/.gitkeep    (only the anchor file is tracked)
```

**Never commit raw image files.** Only `.gitkeep`, scripts, and documentation
are tracked in version control.

---

## Task 1.2 -- Dataset Validation

### Run Dataset Validation checks

Run the validation script to scan all files inside `data/raw/` for image corruption, incorrect class label mappings, invalid file extensions, and class imbalance:

```bash
venv\Scripts\python scripts/validate_data.py
```

### Generated Artifacts

The script performs validation checks (verifying images with `PIL.Image.open().verify()` and loading the data) and saves the following marking evidence:
* **Detailed CSV Log**: `outputs/reports/data_validation_report.csv` (contains width, height, format mode, status, and error messages for every scanned file)
* **Summary JSON Stats**: `outputs/reports/data_validation_summary.json` (contains class mapping, class counts, corrupted counts, and overall pass status)
* **Class Distribution Plot**: `outputs/figures/class_distribution.png` (bar chart representing valid image counts per class)

---

## Training Commands

### Train a Custom CNN from Scratch

```bash
python scripts/train.py --model custom_cnn --epochs 50 --batch_size 32 --lr 1e-3
```

### Train with Transfer Learning (e.g., ResNet-50)

```bash
python scripts/train.py --model resnet50 --pretrained --epochs 30 --batch_size 32 --lr 1e-4
```

### Resume from Checkpoint

```bash
python scripts/train.py --model resnet50 --resume checkpoints/resnet50_best.pth
```

---

## Evaluation Commands

```bash
python scripts/evaluate.py --model resnet50 --checkpoint checkpoints/resnet50_best.pth
```

Outputs saved to `outputs/reports/` (JSON metrics) and `outputs/figures/` (confusion matrix, ROC curves).

---

## XAI Commands

### GradCAM Visualisation

```bash
python scripts/explain.py --method gradcam --model resnet50 --checkpoint checkpoints/resnet50_best.pth --image data/raw/sample.png
```

### Integrated Gradients

```bash
python scripts/explain.py --method integrated_gradients --model resnet50 --checkpoint checkpoints/resnet50_best.pth --image data/raw/sample.png
```

Saliency maps saved to `outputs/figures/xai/`.

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis of AneRBC dataset |
| `notebooks/02_custom_cnn.ipynb` | Custom CNN training & evaluation |
| `notebooks/03_transfer_learning.ipynb` | Transfer learning experiments |
| `notebooks/04_xai.ipynb` | XAI visualisations & interpretation |

---

## License

This project is submitted as part of university coursework and is not for redistribution.
