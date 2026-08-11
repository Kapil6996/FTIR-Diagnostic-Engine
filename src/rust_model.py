"""
Stage 1: Binary Rust vs. Non-Rust Classifier Module (src.rust_model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reusable PyTorch vision classifier designed to filter incoming FTIR records in Stage 1.
Evaluates normalized attachments (still images, video frames, PDF illustrations) and
determines whether an FTIR record exhibits corrosion defect symptoms and should proceed
to Stage 2 SBPR defect diagnosis.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from .utils import get_bundle_path, get_device

logger = logging.getLogger(__name__)

# ImageNet normalization statistics matching training distribution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class label mappings (0: non_rust, 1: rust)
CLASS_LABELS = {0: "non_rust", 1: "rust"}

# Standard test-time inference transformations
_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_rust_model(arch: str = "resnet34", freeze_backbone: bool = True) -> nn.Module:
    """
    Build ResNet classifier for binary rust classification (rust vs non_rust).
    
    Supports both ResNet18 and ResNet34 architectures to accommodate experimental
    checkpoints as well as our finalized 85.37% accuracy trained checkpoint.
    
    Parameters
    ----------
    arch : str
        Target architecture ("resnet34" or "resnet18").
    freeze_backbone : bool
        If True, freezes feature extractor backbone parameters for inference or transfer tuning.

    Returns
    -------
    nn.Module
        Configured binary PyTorch model.
    """
    arch_lower = arch.lower()
    if arch_lower == "resnet18":
        model = models.resnet18(weights=None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        # Default ResNet-34 structure matching our finalized checkpoint
        model = models.resnet34(weights=None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2),
        )
    return model

def get_image_embedding(image_path: str, model: nn.Module, device: torch.device) -> List[float]:
    """
    Extract a 512-dim visual feature embedding for an image using the loaded ResNet backbone.
    This vector represents the visual fingerprint of the image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = _inference_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            x = model.conv1(tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            
            # Normalize embedding for stable cosine similarity
            x = F.normalize(x, p=2, dim=1)
            return x.cpu().numpy().tolist()[0]
    except Exception as e:
        logger.warning(f"Could not extract embedding for {image_path}: {e}")
        return []


def load_trained_rust_model(weights_path: str = None, device: Optional[torch.device] = None) -> nn.Module:
    """
    Loads build_rust_model() and applies the saved state dict, sets eval mode,
    and moves to the selected compute device.

    Parameters
    ----------
    weights_path : str
        Path to saved PyTorch model checkpoint (.pth).
    device : torch.device, optional
        Target hardware compute device. If None, resolved via get_device().

    Returns
    -------
    nn.Module
        Trained evaluation model in eval mode on target device.
    """
    if device is None:
        device = get_device()

    bundle = get_bundle_path("models")

    if weights_path is None:
        weights_path = os.path.join(bundle, "models", "rust_demo.pth")

    if not os.path.exists(weights_path):
        # Check fallback locations
        fallbacks = [
            os.path.join(bundle, "models", "rust_demo.pth"),
            os.path.join(bundle, "models", "rust_model.pth"),
            "models/rust_demo.pth",
            "models/rust_model.pth",
        ]
        found = False
        for alt in fallbacks:
            if os.path.exists(alt):
                weights_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Trained model checkpoint not found at: {weights_path}. Searched: {fallbacks}")

    state_dict = torch.load(weights_path, map_location=device)
    
    # Auto-detect ResNet-18 vs ResNet-34 from state_dict layer composition
    # ResNet-34 has layer1.2.* whereas ResNet-18 only reaches layer1.1.*
    has_layer_34 = any("layer1.2." in k for k in state_dict.keys())
    arch_detected = "resnet34" if has_layer_34 else "resnet18"
    
    model = build_rust_model(arch=arch_detected, freeze_backbone=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded trained Stage 1 ({arch_detected}) weights from '{weights_path}' onto device [{device}]")
    return model


def predict_rust(image_path: str, model: nn.Module, threshold: float = 0.75, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Run single-image binary inference using test-time transforms.

    Parameters
    ----------
    image_path : str
        Path to local media file to evaluate.
    model : nn.Module
        Loaded PyTorch classifier model.
    threshold : float
        Confidence threshold below which predictions are marked "uncertain".
    device : torch.device, optional
        Inference device (inferred from model parameters if None).

    Returns
    -------
    Dict[str, Any]
        {
            "label": "rust" | "non_rust" | "uncertain",
            "confidence": float,
            "image_path": str,
            "raw_prediction": "rust" | "non_rust"
        }
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device()

    if not os.path.isfile(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return {"label": "uncertain", "confidence": 0.0, "image_path": image_path, "raw_prediction": "non_rust"}

    try:
        image = Image.open(image_path).convert("RGB")
        tensor = _inference_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        confidence_val = conf.item()
        pred_label_idx = pred_idx.item()
        raw_pred_str = CLASS_LABELS[pred_label_idx]

        if confidence_val < threshold:
            label_str = "uncertain"
        else:
            label_str = raw_pred_str

        return {
            "label": label_str,
            "confidence": confidence_val,
            "image_path": image_path,
            "raw_prediction": raw_pred_str,
        }
    except Exception as e:
        logger.error(f"Failed inference during predict_rust on '{image_path}': {e}")
        return {"label": "uncertain", "confidence": 0.0, "image_path": image_path, "raw_prediction": "non_rust"}


def predict_rust_for_ftir(image_paths: List[str], model: nn.Module, threshold: float = 0.75, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Run predict_rust over a list of images (e.g. from media_normalize.py) and return
    the single highest-confidence "rust" result if any exists, otherwise the highest-confidence
    result overall. This is what decides whether an FTIR proceeds to Stage 2 SBPR classification.

    Parameters
    ----------
    image_paths : List[str]
        Normalized still image file paths belonging to an FTIR record.
    model : nn.Module
        Loaded Stage 1 PyTorch classifier.
    threshold : float
        Minimum probability threshold required to classify a photo as "rust".
    device : torch.device, optional
        Inference backend compute device.

    Returns
    -------
    Dict[str, Any]
        The winning prediction dictionary representing the entire FTIR record's Stage 1 verdict.
    """
    if not image_paths:
        logger.warning("predict_rust_for_ftir called with an empty list of images.")
        return {"label": "uncertain", "confidence": 0.0, "image_path": None, "raw_prediction": "non_rust", "total_images_evaluated": 0}

    all_results: List[Dict[str, Any]] = []
    rust_results: List[Dict[str, Any]] = []

    for path in image_paths:
        res = predict_rust(path, model, threshold=threshold, device=device)
        all_results.append(res)
        if res["label"] == "rust":
            rust_results.append(res)

    # 1. Return the highest-confidence confirmed "rust" result if any exists
    if rust_results:
        best_result = max(rust_results, key=lambda x: x["confidence"])
    else:
        # 2. Otherwise return the highest-confidence result overall (could be "non_rust" or "uncertain")
        best_result = max(all_results, key=lambda x: x["confidence"])

    # Attach diagnostic metadata about batch evaluation
    best_result["total_images_evaluated"] = len(all_results)
    best_result["all_predictions"] = all_results
    return best_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 65)
    print("  Stage 1: Rust vs. Non-Rust Pipeline Module Verification")
    print("=" * 65)

    dev = get_device()
    print(f"  Selected Compute Backend : [{dev}]")

    bundle = get_bundle_path("models")
    model_path = os.path.join(bundle, "models", "rust_demo.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(bundle, "models", "rust_model.pth")

    try:
        classifier = load_trained_rust_model(weights_path=model_path, device=dev)
        print("  Model Checkpoint Status  : Loaded successfully in eval mode")
    except Exception as e:
        print(f"  Model Checkpoint Error   : {e}")
        exit(1)

    print("-" * 65)
    # Test against sample folder from our test dataset
    test_folder = "data/SBIN201210B00011/test"
    if os.path.isdir(test_folder):
        sample_images = [
            os.path.join(test_folder, fname)
            for fname in sorted(os.listdir(test_folder))
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not fname.startswith('.')
        ][:8]  # Take up to 8 sample images

        print(f"Running batch predict_rust_for_ftir across 8 images from '{test_folder}':\n")
        ftir_verdict = predict_rust_for_ftir(sample_images, classifier, threshold=0.75)

        print(f"=== FTIR STAGE 1 VERDICT ===")
        print(f"  Overall Label     : [{ftir_verdict['label'].upper()}]")
        print(f"  Peak Confidence   : [{ftir_verdict['confidence']:.2%}]")
        print(f"  Winning Image     : {os.path.basename(str(ftir_verdict['image_path']))}")
        print(f"  Images Evaluated  : {ftir_verdict['total_images_evaluated']}\n")

        print("--- Breakdown of Evaluated Attachments ---")
        for idx, item in enumerate(ftir_verdict["all_predictions"], start=1):
            lbl = item["label"].ljust(9)
            conf = f"{item['confidence']:.2%}".rjust(7)
            fname = os.path.basename(str(item["image_path"]))
            print(f"  [{idx}] {fname.ljust(15)} -> Label: {lbl} | Confidence: {conf}")
    else:
        print(f"Test directory '{test_folder}' not found. Setup verification complete.")
    print("=" * 65)
