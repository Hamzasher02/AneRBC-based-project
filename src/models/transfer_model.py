# ============================================================
# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 3.1 - Pretrained CNN model support for transfer learning
# ============================================================
"""
transfer_model.py
-----------------
Transfer-learning wrappers around torchvision pretrained backbones
for AneRBC medical image classification.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_transfer_model(
    model_name: str = None,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    **kwargs
) -> nn.Module:
    """
    Load a pretrained torchvision backbone and replace the classifier head.

    Args:
        model_name      : One of 'mobilenet_v2', 'squeezenet1_0', 'resnet18'.
        num_classes     : Number of output classes (default: 2).
        pretrained      : Load ImageNet weights when True.
        freeze_backbone : Freeze convolutional layers (feature extraction mode).

    Returns:
        nn.Module: PyTorch model object.
    """
    # Map backbone / freeze_features to keep compatibility
    if model_name is None:
        model_name = kwargs.get("backbone", "resnet18")
    freeze_features = kwargs.get("freeze_features", freeze_backbone)

    weights_arg = "DEFAULT" if pretrained else None

    # ── ResNet family ───────────────────────────────────────────────────────
    if model_name == "resnet18" or model_name == "resnet50":
        model = getattr(models, model_name)(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # ── MobileNetV2 ─────────────────────────────────────────────────────────
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ── SqueezeNet 1.0 ──────────────────────────────────────────────────────
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(weights=weights_arg)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes

    # ── EfficientNet ────────────────────────────────────────────────────────
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ── VGG-16 ──────────────────────────────────────────────────────────────
    elif model_name == "vgg16":
        model = models.vgg16(weights=weights_arg)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model name: '{model_name}'")

    # Optionally freeze features (feature extraction mode)
    if freeze_features:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model


def get_trainable_parameter_summary(model: nn.Module) -> dict:
    """
    Returns a summary of total, trainable, and frozen parameters in the model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        dict: Parameter count stats and trainable layer names.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    trainable_layer_names = [name for name, p in model.named_parameters() if p.requires_grad]

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_layer_names": trainable_layer_names,
    }


def describe_transfer_models() -> list:
    """
    Returns descriptions of the supported transfer learning architectures.

    Returns:
        list: List of dictionaries describing each pretrained model.
    """
    return [
        {
            "model_name": "mobilenet_v2",
            "description": "MobileNetV2: lightweight network with inverted residuals and linear bottlenecks",
            "classifier_head": "classifier[1] Linear",
        },
        {
            "model_name": "squeezenet1_0",
            "description": "SqueezeNet 1.0: extremely small CNN architecture achieving AlexNet-level accuracy",
            "classifier_head": "classifier[1] Conv2d",
        },
        {
            "model_name": "resnet18",
            "description": "ResNet18: standard 18-layer residual network",
            "classifier_head": "fc Linear",
        }
    ]
