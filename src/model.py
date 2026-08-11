"""
Model module for the rust vs non_rust binary image classifier.

Provides a pretrained EfficientNet-B0 configured for transfer learning
and automatic MPS / CPU device selection for Apple Silicon Macs.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_device():
    """
    Select the best available device.

    Returns ``torch.device("mps")`` on Apple Silicon Macs with MPS support,
    otherwise falls back to ``torch.device("cpu")``.

    Returns
    -------
    device : torch.device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def build_model():
    """
    Build a ResNet-34 model for binary classification via transfer learning.

    * Loads ``resnet34`` with ImageNet V1 pretrained weights.
    * All backbone layers are unfrozen for full fine-tuning.
    * Replaces the final ``fc`` layer with Dropout(0.3) → Linear(512, 2).

    Returns
    -------
    model : torchvision.models.ResNet
    """
    model = models.resnet34(weights="IMAGENET1K_V1")

    # Unfreeze all layers for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Replace the final fc with dropout + linear
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )

    return model


def get_trainable_params(model):
    """
    Return an iterator over only the parameters that require gradients.

    Pass the result directly to an optimizer, e.g.::

        optimizer = torch.optim.Adam(get_trainable_params(model), lr=1e-3)

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    params : filter object
        Iterator of ``nn.Parameter`` with ``requires_grad=True``.
    """
    return filter(lambda p: p.requires_grad, model.parameters())


# ── sanity-check entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    model = build_model()
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in get_trainable_params(model))

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {total - trainable:,}")

