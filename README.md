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

## Dataset Instructions

1. Download the **AneRBC** dataset from Kaggle or the official source:
   - Kaggle: [AneRBC Dataset](https://www.kaggle.com/datasets/...)
2. Place the raw files inside `data/raw/`:
   ```
   data/raw/
   └── AneRBC/
       ├── class_1/
       ├── class_2/
       └── ...
   ```
3. Run preprocessing to generate splits:

```bash
python scripts/prepare_data.py
```

> **Note:** `data/raw/` and `data/processed/` are excluded from git via `.gitignore`.

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
