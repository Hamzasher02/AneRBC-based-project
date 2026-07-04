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

## Task 1.3 -- Dataset Preprocessing

### Run Preprocessing & Cleaning Pipeline

Run the preprocessing script to load, convert to RGB, and resize all images from `data/raw/` to a uniform 224x224 dimension, saving results under `data/processed/`:

```bash
# Standard resizing and RGB conversion
venv\Scripts\python scripts/preprocess_data.py --overwrite

# Enhanced preprocessing (with optional CLAHE contrast enhancement and Gaussian denoising)
venv\Scripts\python scripts/preprocess_data.py --use-clahe --use-denoise --overwrite
```

### Generated Local Artifacts

* **Processed Dataset**: `data/processed/healthy/` and `data/processed/anaemic/` (JPEG representations of the resized and cleaned images)
* **Detailed CSV Log**: `outputs/reports/preprocessing_report.csv` (contains input/output file paths and status for every processed file)
* **Summary JSON Stats**: `outputs/reports/preprocessing_summary.json` (contains runtime config, total count, processed/skipped counts, and class distributions)

### Normalization & Dataloader Resolution Notes

* **No Float Images**: PyTorch float tensor normalizations (e.g. ImageNet standard subtraction and division) are **not** saved as image files.
* **Online Transforms**: Subtraction of ImageNet mean (`[0.485, 0.456, 0.406]`) and division by standard deviation (`[0.229, 0.224, 0.225]`) are performed **online** in PyTorch dataset transforms during training/inference.
* **Relative Paths & Extension Resolution**: Split CSVs store relative paths (`class_name/filename.png`) using the original filenames. The dataloader helper `resolve_image_path` dynamically resolves these to matching extensions (such as `.jpg`) on disk, allowing training on either `data/raw` or preprocessed `data/processed` folders seamlessly.
* **Default Folders**: By default, training and evaluation scripts use the preprocessed `data/processed/` images.
* **Git Excluded**: The `data/processed/` outputs are excluded from version control in `.gitignore`.

---

## Task 2.1 -- Custom CNN Architectures

Custom deep convolutional neural network architectures with 3, 4, and 5 layers. Features include Convolution blocks (Conv, BatchNorm, ReLU), MaxPooling, Dropout, and fully-connected classifiers.

### Architecture Specifications

| Model Name | Conv Layers | Filters | Activation | Dropout | Classifier Neurons |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `custom_cnn_3` | 3 | 32, 64, 128 | ReLU | 0.25 | 128 -> 256 -> `num_classes` |
| `custom_cnn_4` | 4 | 32, 64, 128, 256 | ReLU | 0.30 | 256 -> 512 -> `num_classes` |
| `custom_cnn_5` | 5 | 32, 64, 128, 256, 512 | ReLU | 0.40 | 512 -> 512 -> 256 -> `num_classes` |

### Training Example Commands (CPU/GPU-friendly)

To train the custom models:
```bash
# Train Custom 3-Layer CNN
venv\Scripts\python scripts\train.py --model custom_cnn_3 --epochs 30

# Train Custom 4-Layer CNN (equivalent to default custom_cnn)
venv\Scripts\python scripts\train.py --model custom_cnn_4 --epochs 30

# Train Custom 5-Layer CNN
venv\Scripts\python scripts\train.py --model custom_cnn_5 --epochs 30
```

---

## Task 2.2 -- Custom CNN Model Training & Validation

Each custom CNN model was trained and validated on the preprocessed dataset `data/processed/` using early stopping (patience = 8 epochs).

### Validation Results Summary

| Model Name | Best Validation Loss | Best Validation Accuracy | Epochs Trained |
| :--- | :---: | :---: | :---: |
| `custom_cnn_3` | 0.5860 | 75.33% | 11 (Early stopped) |
| `custom_cnn_4` | 0.4998 | 78.00% | 20 (Early stopped) |
| `custom_cnn_5` | 0.5466 | 74.00% | 15 (Early stopped) |

### Local Artifacts Generated (Git-Ignored)

* **Checkpoints**:
  * `checkpoints/custom_cnn_3_best.pth`
  * `checkpoints/custom_cnn_4_best.pth`
  * `checkpoints/custom_cnn_5_best.pth`
* **History Logs**:
  * `outputs/reports/history_custom_cnn_3.json`
  * `outputs/reports/history_custom_cnn_4.json`
  * `outputs/reports/history_custom_cnn_5.json`
* **Learning Curves**:
  * `outputs/figures/learning_curve_custom_cnn_3.png`
  * `outputs/figures/learning_curve_custom_cnn_4.png`
  * `outputs/figures/learning_curve_custom_cnn_5.png`
* **Validation summary**:
  * `outputs/reports/custom_cnn_validation_summary.csv`

---

## Task 2.3 -- Custom CNN Model Evaluation

Each custom CNN model was evaluated on the independent test split using the saved best checkpoints.

### Test Results Summary

| Model Name | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `custom_cnn_3` | 59.33% | 0.5984 | 0.5933 | 0.5880 | 0.6315 | 150 |
| `custom_cnn_4` | **66.67%** | **0.6769** | **0.6667** | **0.6618** | **0.6967** | 150 |
| `custom_cnn_5` | 62.67% | 0.6267 | 0.6267 | 0.6267 | 0.6873 | 150 |

**Best Custom CNN Model**: `custom_cnn_4` (4-layer CNN). It achieves the highest macro F1-score (0.6618) and highest overall test accuracy (66.67%) on the test set.

### Evaluation Commands (CPU/GPU-friendly)

To evaluate the custom models on the preprocessed test set:
```bash
# Evaluate Custom 3-Layer CNN
venv\Scripts\python scripts\evaluate.py --model custom_cnn_3 --checkpoint checkpoints/custom_cnn_3_best.pth --workers 0

# Evaluate Custom 4-Layer CNN
venv\Scripts\python scripts\evaluate.py --model custom_cnn_4 --checkpoint checkpoints/custom_cnn_4_best.pth --workers 0

# Evaluate Custom 5-Layer CNN
venv\Scripts\python scripts\evaluate.py --model custom_cnn_5 --checkpoint checkpoints/custom_cnn_5_best.pth --workers 0
```

### Local Artifacts Generated (Git-Ignored)

* **Classification Reports**:
  * `outputs/reports/classification_report_custom_cnn_3.txt`
  * `outputs/reports/classification_report_custom_cnn_4.txt`
  * `outputs/reports/classification_report_custom_cnn_5.txt`
* **Confusion Matrices**:
  * `outputs/figures/confusion_matrix_custom_cnn_3.png`
  * `outputs/figures/confusion_matrix_custom_cnn_4.png`
  * `outputs/figures/confusion_matrix_custom_cnn_5.png`
* **Test summary**:
  * `outputs/reports/custom_cnn_test_summary.csv`

---

## Task 3.1 -- Transfer Learning Architecture Support

Support for three pretrained CNN models as feature extractors with frozen convolutional backbones and customized classifier heads.

### Pretrained Models Specifications

| Pretrained Model | PyTorch backbone name | Replaced Classifier Head | Freeze Backbone Explanation | Trainable Top Layers Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | `mobilenet_v2` | `classifier[1]` (Linear) | Set all convolutional/inverted residual block parameters `requires_grad = False`. | The final `nn.Linear` classifier head is trained for 2 classes. |
| **SqueezeNet 1.0** | `squeezenet1_0` | `classifier[1]` (Conv2d) + `num_classes` | Set all fire block and squeeze parameters `requires_grad = False`. | The classifier Conv2d layer is trained for 2 classes. |
| **ResNet18** | `resnet18` | `fc` (Linear) | Set all residual block parameters `requires_grad = False`. | The final fully-connected Linear layer is trained for 2 classes. |

### Training Example Commands (CPU/GPU-friendly)

To train these models with a frozen backbone:
```bash
# Train MobileNetV2 Transfer Learning Model
venv\Scripts\python scripts\train.py --model mobilenet_v2 --pretrained --freeze --epochs 20 --batch_size 16 --workers 0

# Train SqueezeNet 1.0 Transfer Learning Model
venv\Scripts\python scripts\train.py --model squeezenet1_0 --pretrained --freeze --epochs 20 --batch_size 16 --workers 0

# Train ResNet18 Transfer Learning Model
venv\Scripts\python scripts\train.py --model resnet18 --pretrained --freeze --epochs 20 --batch_size 16 --workers 0
```

---

## Task 3.2 -- Train/Validate Pretrained Models

The three transfer learning models were fine-tuned with a frozen feature extraction backbone and trainable classifier heads on the preprocessed dataset.

### Training Commands Run
```bash
# MobileNetV2
venv\Scripts\python scripts\train.py --model mobilenet_v2 --pretrained --freeze --epochs 20 --batch_size 16 --data_root data/processed --workers 0 --patience 6

# SqueezeNet 1.0
venv\Scripts\python scripts\train.py --model squeezenet1_0 --pretrained --freeze --epochs 20 --batch_size 16 --data_root data/processed --workers 0 --patience 6

# ResNet18
venv\Scripts\python scripts\train.py --model resnet18 --pretrained --freeze --epochs 20 --batch_size 16 --data_root data/processed --workers 0 --patience 6
```

### Pretrained Validation Results Summary

This is saved locally under `outputs/reports/transfer_validation_summary.csv`:

| Model Name | Best Validation Loss | Best Validation Accuracy | Epochs Run | Checkpoint Path | Learning Curve Plot |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `mobilenet_v2` | 0.4907 | 78.00% | 8 epochs (early stopped) | `checkpoints/mobilenet_v2_best.pth` | `outputs/figures/learning_curve_mobilenet_v2.png` |
| `squeezenet1_0` | **0.4085** | **80.67%** | 17 epochs (early stopped) | `checkpoints/squeezenet1_0_best.pth` | `outputs/figures/learning_curve_squeezenet1_0.png` |
| `resnet18` | 0.5022 | 74.67% | 11 epochs (early stopped) | `checkpoints/resnet18_best.pth` | `outputs/figures/learning_curve_resnet18.png` |

> [!NOTE]
> For all transfer learning runs, the backbone features were completely frozen (`requires_grad = False`). Only the custom replaced classifier heads were optimized, enabling fast convergence and preventing overfitting on the medical image dataset.

---

## Task 3.3 -- Test/Evaluate Pretrained Models

The three trained pretrained models were evaluated on the independent test set (150 images total, stratified split).

### Evaluation Commands Run
```bash
# MobileNetV2
venv\Scripts\python scripts\evaluate.py --model mobilenet_v2 --checkpoint checkpoints/mobilenet_v2_best.pth --data_root data/processed --batch_size 16 --workers 0

# SqueezeNet 1.0
venv\Scripts\python scripts\evaluate.py --model squeezenet1_0 --checkpoint checkpoints/squeezenet1_0_best.pth --data_root data/processed --batch_size 16 --workers 0

# ResNet18
venv\Scripts\python scripts\evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_best.pth --data_root data/processed --batch_size 16 --workers 0
```

### Pretrained Test Results Summary

This is saved locally under `outputs/reports/transfer_test_summary.csv`:

| Model Name | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Support | Confusion Matrix | Classification Report |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `mobilenet_v2` | **75.33%** | **0.7707** | **0.7533** | **0.7493** | 0.8228 | 150 | `outputs/figures/confusion_matrix_mobilenet_v2.png` | `outputs/reports/classification_report_mobilenet_v2.txt` |
| `squeezenet1_0` | 74.00% | 0.7421 | 0.7400 | 0.7394 | **0.8455** | 150 | `outputs/figures/confusion_matrix_squeezenet1_0.png` | `outputs/reports/classification_report_squeezenet1_0.txt` |
| `resnet18` | 70.00% | 0.7083 | 0.7000 | 0.6970 | 0.8084 | 150 | `outputs/figures/confusion_matrix_resnet18.png` | `outputs/reports/classification_report_resnet18.txt` |

* **Best Pretrained Model**: `mobilenet_v2`
* **Reason**: Achieved the highest macro F1-score (**0.7493**) and highest overall test accuracy (**75.33%**) on the test set.

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
