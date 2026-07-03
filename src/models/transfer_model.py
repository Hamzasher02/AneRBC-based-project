# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
transfer_model.py
-----------------
Transfer-learning wrappers around torchvision pretrained backbones
for AneRBC medical image classification.

Supported backbones: resnet18, resnet50, efficientnet_b0, vgg16.
"""

import torch
import torch.nn as nn
from torchvision import models


_SUPPORTED = ("resnet18", "resnet50", "efficientnet_b0", "vgg16")


def build_transfer_model(
    backbone: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_features: bool = False,
) -> nn.Module:
    """
    Load a pretrained torchvision backbone and replace the classifier head.

    Args:
        backbone       : One of 'resnet18', 'resnet50', 'efficientnet_b0', 'vgg16'.
        num_classes    : Number of output classes.
        pretrained     : Load ImageNet weights when True.
        freeze_features: Freeze convolutional layers (feature extraction mode).

    Returns:
        nn.Module ready for training.
    """
    if backbone not in _SUPPORTED:
        raise ValueError(f"backbone must be one of {_SUPPORTED}, got '{backbone}'")

    weights_arg = "DEFAULT" if pretrained else None

    # ── ResNet family ───────────────────────────────────────────────────────
    if backbone in ("resnet18", "resnet50"):
        model = getattr(models, backbone)(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # ── EfficientNet ────────────────────────────────────────────────────────
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ── VGG-16 ──────────────────────────────────────────────────────────────
    elif backbone == "vgg16":
        model = models.vgg16(weights=weights_arg)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    # Optionally freeze all layers except the new head
    if freeze_features:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model
