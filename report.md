# Medical Image Classification with Custom CNNs, Transfer Learning, and XAI

**Author:** Hamza Sher  
**SRN:** 3012260007  
**Coursework Submission Report**  

---

## 1. Title Page & Metadata
- **Project Title:** Medical Image Classification with Custom CNNs, Transfer Learning, and Explainable AI (XAI)
- **Author:** Hamza Sher
- **SRN:** 3012260007
- **Target Task:** Classification of healthy vs. anaemic red blood cells (AneRBC dataset)

---

## 2. Dataset Description
The model development and evaluation are based on the **AneRBC** medical image dataset.
- **Source:** AneRBC database of red blood cells.
- **Classes:** Binary classification task featuring:
  - `healthy`: Healthy red blood cells (RBCs).
  - `anaemic`: Anaemic/abnormal red blood cells (RBCs).
- **Counts:** 1,000 images total, split equally:
  - 500 healthy cell images.
  - 500 anaemic cell images.
- **Splits:** A stratified split (70% train, 15% validation, 15% test) was used to ensure identical class distributions across subsets:
  - **Train:** 700 images (350 healthy, 350 anaemic)
  - **Validation:** 150 images (75 healthy, 75 anaemic)
  - **Test:** 150 images (75 healthy, 75 anaemic)
- **Preprocessing & Normalisation:**
  - Raw images are converted to RGB format.
  - Resized to $224 \times 224$ pixels.
  - Standard ImageNet normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`) is applied online inside PyTorch Dataloaders for transfer compatibility.

---

## 3. Task 1 Summary: Data pipeline
- **Data Placement:** Raw images were safely downloaded, verified, and placed under `data/raw/`.
- **Validation Checks:** Asserted that directory structures contain only the two target classes, that image extensions are valid (JPEG/PNG), and that no corruption or empty files exist.
- **Preprocessing Pipeline:** Resized all raw images to $224 \times 224$ JPEG format in `data/processed/`.
- **Stratified Split:** Generated randomized, class-stratified splits using `scikit-learn` and stored indexes in `train.csv`, `val.csv`, and `test.csv` to ensure perfect class representation during training and testing.

---

## 4. Task 2: Custom CNN Evaluation
Three custom deep CNN architectures with 3, 4, and 5 convolutional layers were built from scratch in PyTorch, trained using early stopping (patience = 8, batch size = 16) on CPU, and evaluated on the test set.

### Custom CNN Performance on Test Set
| Model Name | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `custom_cnn_3` | 59.33% | 0.5984 | 0.5933 | 0.5880 | 0.6315 | 150 |
| `custom_cnn_4` | **66.67%** | **0.6769** | **0.6667** | **0.6618** | **0.6967** | 150 |
| `custom_cnn_5` | 62.67% | 0.6267 | 0.6267 | 0.6267 | 0.6873 | 150 |

### Critical Evaluation of Custom CNNs
- **Best Model:** `custom_cnn_4` (4-layer CNN) achieved the highest macro F1-score (**0.6618**) and accuracy (**66.67%**).
- **Overfitting & Capacity Critique:** Going from 3 convolutional layers (`custom_cnn_3`) to 4 convolutional layers (`custom_cnn_4`) significantly improved the representation capacity. However, further increasing the capacity to 5 layers (`custom_cnn_5`) led to a drop in performance. This suggests that the 5-layer model suffered from overfitting, given the relatively small training split (700 images).

---

## 5. Task 3: Transfer Learning Evaluation
Three pretrained models (`mobilenet_v2`, `squeezenet1_0`, and `resnet18`) were fine-tuned as feature extractors. The convolutional backbones were completely frozen, and the final classifier heads were replaced to classify 2 classes.

### Pretrained CNN Performance on Test Set
| Model Name | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mobilenet_v2` | **75.33%** | **0.7707** | **0.7533** | **0.7493** | 0.8228 | 150 |
| `squeezenet1_0` | 74.00% | 0.7421 | 0.7400 | 0.7394 | **0.8455** | 150 |
| `resnet18` | 70.00% | 0.7083 | 0.7000 | 0.6970 | 0.8084 | 150 |

### Critical Evaluation of Pretrained CNNs
- **Best Pretrained Model:** `mobilenet_v2` achieved the highest macro F1-score (**0.7493**) and accuracy (**75.33%**).
- **Feature Extraction Efficacy:** SqueezeNet 1.0 also achieved a very high ROC-AUC (**0.8455**) and F1 (**0.7394**). ResNet18 performed slightly worse (**0.6970** F1), possibly because the frozen ResNet features are less optimized for cellular structures without adjusting intermediate representations.

---

## 6. Custom vs. Pretrained Critical Comparison
- The best pretrained model (**MobileNetV2: 75.33% accuracy, 0.7493 F1**) significantly outperformed the best custom CNN model (**custom_cnn_4: 66.67% accuracy, 0.6618 F1**).
- **Representation Learning Theory:** Transfer learning models benefit from high-quality features learned from millions of general images (ImageNet). These features generalize exceptionally well as edge detectors, shape descriptors, and textures, which the custom models had to learn from scratch from only 700 training samples. Hence, feature extraction is far superior for small-scale medical datasets.

---

## 7. Task 4: XAI Critique
We applied **Grad-CAM** (coarse regional attribution) and **Integrated Gradients** (pixel-level attribution) to `custom_cnn_4` and `mobilenet_v2` using Captum.

### Saliency Map Findings
- **Attribution Regions:** For anaemic cells, both Grad-CAM and Integrated Gradients highlighted the central pallor region of the cell, its outer boundaries, and shape irregular regions. This matches clinical logic, as anaemic cells differ in shape and light transmission/absorption due to hemoglobin concentration variations.
- **Comparative Resolution:** Integrated Gradients showed finer pixel-level boundaries, while Grad-CAM visualised coarse regional activation maps.
- **XAI Limitations:** 
  - Visual attribution maps are qualitative and can be subjective.
  - Heatmaps show correlation rather than causal medical indicators.
  - Saliency maps must be interpreted with clinical expert validation to avoid confirmation bias.

---

## 8. General Project Limitations
1. **Dataset Size:** 1,000 images is extremely small for training high-parameter neural networks from scratch.
2. **Clinical Validation:** The classification is binary (healthy vs. anaemic) and does not segment or classify specific subclasses of anaemia (e.g., sickle-cell, microcytic) which are clinically necessary.
3. **No External Testing:** Testing was done on the split of the same original dataset; performance on external clinical samples is untested.

---

## 9. Conclusion
We successfully designed and executed an end-to-end medical image classification pipeline. Transfer learning models (especially `mobilenet_v2` and `squeezenet1_0`) showed a clear performance advantage over custom networks trained from scratch, illustrating the power of pre-trained feature extractors. XAI techniques provided qualitative transparency by showing that models look at clinically relevant RBC cell regions for classifications.

---

## 10. Reproducibility & Local Commands
All tasks are fully reproducible. Checkpoints, logs, and plots are stored locally and git-ignored.

```bash
# 1. Preprocess raw data
python scripts/download_data.py --data_root data/raw

# 2. Train Custom CNN Models
python scripts/train.py --model custom_cnn_3 --epochs 50 --batch_size 16 --workers 0 --patience 8
python scripts/train.py --model custom_cnn_4 --epochs 50 --batch_size 16 --workers 0 --patience 8
python scripts/train.py --model custom_cnn_5 --epochs 50 --batch_size 16 --workers 0 --patience 8

# 3. Train Pretrained Models
python scripts/train.py --model mobilenet_v2 --pretrained --freeze --epochs 20 --batch_size 16 --workers 0 --patience 6
python scripts/train.py --model squeezenet1_0 --pretrained --freeze --epochs 20 --batch_size 16 --workers 0 --patience 6
python scripts/train.py --model resnet18 --pretrained --freeze --epochs 20 --batch_size 16 --workers 0 --patience 6

# 4. Evaluate Models on Test Set
python scripts/evaluate.py --model custom_cnn_4 --checkpoint checkpoints/custom_cnn_4_best.pth --workers 0
python scripts/evaluate.py --model mobilenet_v2 --checkpoint checkpoints/mobilenet_v2_best.pth --workers 0

# 5. Generate XAI Maps
python scripts/explain.py --method gradcam --model custom_cnn_4 --checkpoint checkpoints/custom_cnn_4_best.pth --image data/processed/anaemic/165_a.jpg
python scripts/explain.py --method integrated_gradients --model mobilenet_v2 --checkpoint checkpoints/mobilenet_v2_best.pth --image data/processed/anaemic/165_a.jpg
```
