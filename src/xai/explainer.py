# ============================================================
# Author : Hamza Sher
# SRN    : 3012260007
# Project: Medical Image Classification with Custom CNNs,
#          Transfer Learning, and XAI
# ============================================================
"""
explainer.py
------------
XAI utilities using Captum:
  - Integrated Gradients
  - GradCAM  (via captum.attr.LayerGradCam)
  - Saliency map visualisation helpers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerGradCam, NoiseTunnel
from captum.attr import visualization as viz


def integrated_gradients(model, image_tensor, target_class, baseline=None):
    """
    Compute Integrated Gradients attribution for a single image.

    Args:
        model        : Trained nn.Module (eval mode).
        image_tensor : (1, C, H, W) FloatTensor on correct device.
        target_class : Integer class index.
        baseline     : Baseline tensor; defaults to black image.

    Returns:
        attributions (np.ndarray): Shape (C, H, W).
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(image_tensor)

    ig   = IntegratedGradients(model)
    attr = ig.attribute(image_tensor, baselines=baseline,
                        target=target_class, n_steps=50)
    return attr.squeeze().cpu().detach().numpy()


def gradcam(model, image_tensor, target_class, target_layer):
    """
    Compute GradCAM heatmap for a single image.

    Args:
        model        : Trained nn.Module (eval mode).
        image_tensor : (1, C, H, W) FloatTensor.
        target_class : Integer class index.
        target_layer : The layer to hook (e.g., model.features[-1]).

    Returns:
        heatmap (np.ndarray): Upsampled GradCAM map (H, W).
    """
    model.eval()
    gc   = LayerGradCam(model, target_layer)
    attr = gc.attribute(image_tensor, target=target_class)
    # Upsample to input size
    upsampled = torch.nn.functional.interpolate(
        attr, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )
    return upsampled.squeeze().cpu().detach().numpy()


def visualise_attribution(image_np, attribution_np, title="Attribution Map",
                           save_path=None, show=True):
    """
    Overlay attribution map on the original image and optionally save.

    Args:
        image_np      : (H, W, C) uint8 numpy array.
        attribution_np: (H, W) or (C, H, W) attribution map.
        title         : Figure title.
        save_path     : If given, saves the figure to this path.
        show          : If True, calls plt.show().
    """
    if attribution_np.ndim == 3:
        attribution_np = attribution_np.mean(0)  # average across channels

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(attribution_np, cmap="hot")
    axes[1].set_title("Attribution Map")
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(attribution_np, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
