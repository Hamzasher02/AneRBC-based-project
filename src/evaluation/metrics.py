# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
metrics.py
----------
Evaluation metrics: accuracy, precision, recall, F1, confusion matrix,
classification report, and ROC-AUC for the AneRBC dataset.
"""

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Collect all ground-truth labels and model predictions from a DataLoader.

    Returns:
        all_labels (np.ndarray): Ground-truth class indices.
        all_preds  (np.ndarray): Predicted class indices.
        all_probs  (np.ndarray): Softmax probabilities (N × C).
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(labels, preds, probs, class_names=None):
    """
    Compute a full evaluation report.

    Returns a dict containing:
        accuracy, classification_report, confusion_matrix, roc_auc (OvR macro).
    """
    acc  = accuracy_score(labels, preds)
    cm   = confusion_matrix(labels, preds)
    cr   = classification_report(labels, preds, target_names=class_names, digits=4)

    try:
        if len(probs.shape) == 2 and probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "accuracy":              acc,
        "roc_auc":               auc,
        "confusion_matrix":      cm,
        "classification_report": cr,
    }
