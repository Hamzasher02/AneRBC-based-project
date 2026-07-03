# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
custom_cnn.py
-------------
Custom CNN architectures built from scratch in PyTorch for
AneRBC medical image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Reusable Conv -> BN -> ReLU block with optional MaxPool."""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Lightweight Custom CNN for AneRBC classification.

    Architecture:
        4 x ConvBlock (3->32->64->128->256)  +  Global Avg Pool  +  FC head

    Args:
        num_classes (int): Number of output classes.
        dropout     (float): Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 224->112
            ConvBlock(32,  64,  pool=True),   # 112->56
            ConvBlock(64,  128, pool=True),   # 56->28
            ConvBlock(128, 256, pool=True),   # 28->14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # 14->1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


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

    def forward(self, x):
        return self.relu(self.block(x) + x)


class DeepCNN(nn.Module):
    """
    Deeper Custom CNN with residual skip connections for AneRBC.

    Architecture:
        4 x (ConvBlock + ResidualBlock) + Global Avg Pool + FC head

    Args:
        num_classes (int): Number of output classes.
        dropout     (float): Dropout rate in the classifier head.
    """

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

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


def build_custom_cnn(num_classes: int = 2, dropout: float = 0.5,
                     deep: bool = False) -> nn.Module:
    """
    Convenience factory for custom CNN architectures.

    Args:
        num_classes: Number of output classes (default 2 for AneRBC).
        dropout    : Dropout rate.
        deep       : If True, returns DeepCNN; else returns CustomCNN.
    """
    if deep:
        return DeepCNN(num_classes=num_classes, dropout=dropout)
    return CustomCNN(num_classes=num_classes, dropout=dropout)
