# ============================================================
# Author: Hamza Sher
# SRN: 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# Task: 2.1 - Custom Deep CNN architectures with 3, 4, and 5 conv layers
# ============================================================
"""
custom_cnn.py
-------------
Custom CNN architectures built from scratch in PyTorch for
AneRBC medical image classification with 3, 4, and 5 convolutional layers.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Reusable Conv -> BN -> ReLU block with optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the conv block."""
        return self.block(x)


class CustomCNN3(nn.Module):
    """
    Custom CNN with 3 convolutional layers for AneRBC classification.

    Architecture:
        3 x ConvBlock (3->32->64->128)  +  Global Avg Pool  +  FC head

    Args:
        num_classes (int): Number of output classes.
        dropout     (float): Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 224->112
            ConvBlock(32,  64,  pool=True),   # 112->56
            ConvBlock(64,  128, pool=True),   # 56->28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CustomCNN3."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class CustomCNN4(nn.Module):
    """
    Custom CNN with 4 convolutional layers for AneRBC classification.

    Architecture:
        4 x ConvBlock (3->32->64->128->256)  +  Global Avg Pool  +  FC head

    Args:
        num_classes (int): Number of output classes.
        dropout     (float): Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 224->112
            ConvBlock(32,  64,  pool=True),   # 112->56
            ConvBlock(64,  128, pool=True),   # 56->28
            ConvBlock(128, 256, pool=True),   # 28->14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CustomCNN4."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class CustomCNN5(nn.Module):
    """
    Custom CNN with 5 convolutional layers for AneRBC classification.

    Architecture:
        5 x ConvBlock (3->32->64->128->256->512)  +  Global Avg Pool  +  FC head

    Args:
        num_classes (int): Number of output classes.
        dropout     (float): Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.40):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 224->112
            ConvBlock(32,  64,  pool=True),   # 112->56
            ConvBlock(64,  128, pool=True),   # 56->28
            ConvBlock(128, 256, pool=True),   # 28->14
            ConvBlock(256, 512, pool=True),   # 14->7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CustomCNN5."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Backward Compatibility ───────────────────────────────────────────────

class CustomCNN(CustomCNN4):
    """Original CustomCNN representation mapped to CustomCNN4 for compatibility."""
    pass


class ResidualBlock(nn.Module):
    """Basic residual block: two 3x3 convs with a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of residual block."""
        return self.relu(self.block(x) + x)


class DeepCNN(nn.Module):
    """Deeper Custom CNN with residual skip connections for AneRBC."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.stage1 = nn.Sequential(ConvBlock(3,   32,  pool=True),  ResidualBlock(32))
        self.stage2 = nn.Sequential(ConvBlock(32,  64,  pool=True),  ResidualBlock(64))
        self.stage3 = nn.Sequential(ConvBlock(64,  128, pool=True),  ResidualBlock(128))
        self.stage4 = nn.Sequential(ConvBlock(128, 256, pool=True),  ResidualBlock(256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DeepCNN."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


def build_custom_cnn(num_classes: int = 2, dropout: float = 0.5,
                     deep: bool = False) -> nn.Module:
    """Convenience factory for compatibility."""
    if deep:
        return DeepCNN(num_classes=num_classes, dropout=dropout)
    return CustomCNN(num_classes=num_classes, dropout=dropout)


# ── New Task 2.1 Factory and Description Functions ───────────────────────

def build_custom_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """
    Factory function to build one of the custom CNN architectures.

    Args:
        model_name  : Name of the architecture ('custom_cnn_3', 'custom_cnn_4', 'custom_cnn_5', 'custom_cnn').
        num_classes : Number of output classes.

    Returns:
        nn.Module   : PyTorch model object.
    """
    if model_name == "custom_cnn_3":
        return CustomCNN3(num_classes=num_classes)
    elif model_name == "custom_cnn_4" or model_name == "custom_cnn":
        return CustomCNN4(num_classes=num_classes)
    elif model_name == "custom_cnn_5":
        return CustomCNN5(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown custom model name: '{model_name}'")


def describe_custom_architectures() -> list:
    """
    Describe custom CNN architectures details.

    Returns:
        list of dict: Architecture descriptions containing layers, filters, activation, dropout, classifier.
    """
    return [
        {
            "model_name": "custom_cnn_3",
            "conv_layers": 3,
            "filters": [32, 64, 128],
            "activation": "ReLU",
            "dropout": 0.25,
            "classifier_neurons": [128, 256, "num_classes"]
        },
        {
            "model_name": "custom_cnn_4",
            "conv_layers": 4,
            "filters": [32, 64, 128, 256],
            "activation": "ReLU",
            "dropout": 0.30,
            "classifier_neurons": [256, 512, "num_classes"]
        },
        {
            "model_name": "custom_cnn_5",
            "conv_layers": 5,
            "filters": [32, 64, 128, 256, 512],
            "activation": "ReLU",
            "dropout": 0.40,
            "classifier_neurons": [512, 512, 256, "num_classes"]
        }
    ]
