# Data README — AneRBC Dataset

> **Author:** Hamza Sher | **SRN:** 3012260007  
> **Project:** Medical Image Classification with Custom CNNs, Transfer Learning, and XAI  
> **Task:** 1.1 — Dataset Preparation

---

## Dataset Selected

**AneRBC** -- Anaemia Red Blood Cell Image Classification Dataset

The AneRBC dataset consists of microscopic blood-smear images labelled by
Red Blood Cell (RBC) morphology categories relevant to anaemia diagnosis.
It is the primary dataset for all experiments in this coursework project.

| Property | Detail |
|---|---|
| Task | Multi-class image classification |
| Domain | Medical imaging (haematology) |
| Image type | Microscopic blood-smear photographs |
| Recommended source | Figshare (open access) |
| Alternative source | Kaggle |
| Kaggle slug | `jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images` |

---

## Expected Raw Dataset Folder Structure

After manual placement, `data/raw/` must look exactly like this:

```
data/raw/
├── class_1/                   ← one folder per RBC morphology class
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── class_2/
│   ├── image001.jpg
│   └── ...
├── class_3/
│   └── ...
└── class_N/
    └── ...
```

> **Each direct sub-folder of `data/raw/` is treated as one class.**  
> The folder name becomes the class label used throughout the pipeline.

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

---

## How to Obtain and Place the Dataset

### Recommended Source -- Figshare (open access, no account needed)

1. Visit the official AneRBC Figshare repository:
   - URL: **https://figshare.com/** (search `AneRBC` or `Anaemia Red Blood Cell`)
2. Download the dataset archive directly -- no login or API token required.
3. Extract the archive and proceed to **Step 2** below.

### Alternative Source -- Kaggle

**Kaggle dataset page:**
https://www.kaggle.com/datasets/jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images

**Option A -- Browser download:**

1. Sign in to your Kaggle account.
2. Open the dataset page above and click **Download**.
3. Save the `.zip` archive to your local machine.

**Option B -- Kaggle CLI** (requires `kaggle.json` API token at `~/.kaggle/`):

```bash
# Install Kaggle CLI inside your virtual environment
pip install kaggle

# Download and unzip into a staging folder
kaggle datasets download \
    -d jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images \
    -p data/raw_downloads --unzip

# Then move the class folders into data/raw/ manually
```

### Step 2 -- Arrange Class Folders Under data/raw/

After extracting, ensure each RBC class occupies its own direct sub-folder
inside `data/raw/`. The expected structure using the real AneRBC class names is:

```
data/raw/
  healthy/
    image001.jpg
    image002.jpg
    ...
  anaemic/
    image001.jpg
    ...
```

> **Note:** Folder names must match the actual class names in the downloaded
> archive. Common names in AneRBC: `healthy`, `anaemic`.
> Adjust folder names if the archive uses different naming conventions
> (e.g. `normal`, `sickle`, `thalassemia`).
>
> Each direct sub-folder of `data/raw/` is treated as one class label
> throughout the full pipeline.

### Step 3 -- Verify the Structure

Ensure that each class of images is in its **own direct sub-folder** inside
`data/raw/`. If the archive extracted into a single nested folder, move the
class folders up one level so they are immediate children of `data/raw/`.

### Step 4 -- Run the Placement Checker

```bash
python scripts/download_data.py
```

A successful run will print:

```
[OK]    data/raw directory found  ->  ...\data\raw
[OK]    Total class folders found : 2
  Class Folder                     Image Count
  healthy                              1200
  anaemic                               800
  TOTAL                                2000
[OK]    Dataset validated - 2000 images across 2 classes.
  [OK]  Dataset is in place. Proceed to Task 1.2 preprocessing.
```

---

## Git Tracking Rules

> ⚠️ **Raw dataset files must NEVER be committed to git.**

| Path | Git Status | Reason |
|---|---|---|
| `data/raw/*.jpg` / `*.png` | **Ignored** | Large binary files; excluded via `.gitignore` |
| `data/raw/<class_folder>/` | **Ignored** | Same rule applies to sub-directories |
| `data/raw/.gitkeep` | **Tracked** | Empty-dir anchor file only |
| `scripts/download_data.py` | **Tracked** | Instructions and checker script |
| `data/README_DATA.md` | **Tracked** | This documentation file |
| `data/splits/*.csv` | **Ignored** | Generated split files (re-creatable) |

The relevant `.gitignore` rules are:

```gitignore
data/raw/*
!data/raw/.gitkeep
```

This means:
- Everything inside `data/raw/` is silently ignored by git.
- The `.gitkeep` anchor file is explicitly un-ignored so git retains the
  directory in the repository even when it is otherwise empty.

---

## Next Steps After Placing Data

| Step | Command |
|---|---|
| **Verify placement** | `python scripts/download_data.py` |
| **Generate train/val/test splits** | `python scripts/prepare_data.py` *(Task 1.2)* |
| **Train a model** | `python scripts/train.py --model custom_cnn ...` *(Task 2)* |

---

*Last updated: Task 1.1 (updated with specific AneRBC sources)*
