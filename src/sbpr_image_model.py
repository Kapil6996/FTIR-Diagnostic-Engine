"""
Stage 2 Secondary: 3-Class SBPR Image Classifier Module (src.sbpr_image_model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch ResNet-18 fine-tuned classifier that visually categorises confirmed
corrosion FTIR attachment images into one of 3 SBPR defect categories:

    sbpr_1 → SBIN201210B00011  (Seat frame / interior bracket corrosion)
    sbpr_2 → SBIN202310B06811  (Exterior trim / chrome stain / paint peeling)
    sbpr_3 → SBIN202507B07143  (Door panel / hinge / hard-water corrosion)

This model acts as the *secondary* verification channel in Stage 2's fusion
engine — its image-based prediction is compared against the primary metadata
Decision Tree's text-based prediction.  Agreement → high-confidence output;
disagreement → flagged for manual review.
"""

import os
import sys
import argparse
import logging
import shutil
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from PIL import Image

from .utils import get_bundle_path, get_device

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Canonical ordered mapping between ImageFolder subfolder names and real SBPR codes.
# ImageFolder sorts subfolders alphabetically, so sbpr_1 < sbpr_2 < sbpr_3.
SBPR_CLASS_NAMES = ["sbpr_1", "sbpr_2", "sbpr_3"]
SBPR_CODE_MAP = {
    "sbpr_1": "SBIN201210B00011",
    "sbpr_2": "SBIN202310B06811",
    "sbpr_3": "SBIN202507B07143",
}
# Reverse lookup
SBPR_CODE_TO_CLASS = {v: k for k, v in SBPR_CODE_MAP.items()}

# Paths where the original per-SBPR image folders live (from project setup)
_SOURCE_DIRS = {
    "sbpr_1": "data/SBIN201210B00011",
    "sbpr_2": "data/SBIN202310B06811",
    "sbpr_3": "data/SBIN202507B07143",
}

DEFAULT_WEIGHTS_PATH = os.path.join(get_bundle_path("models"), "models", "sbpr_image.pth")

# ── Transforms ─────────────────────────────────────────────────────────────────

_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Device Selection ───────────────────────────────────────────────────────────




# ── Model Architecture ────────────────────────────────────────────────────────

def build_sbpr_image_model(num_classes: int = 3, freeze_backbone: bool = True, use_pretrained_weights: bool = True) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet with frozen convolutional backbone and
    a freshly-initialised fully-connected head for 3-class SBPR classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (3 for SBPR).
    freeze_backbone : bool
        If True, freezes all feature-extraction layers so only the final fc
        layer is updated during training (transfer learning).
    use_pretrained_weights : bool
        If True (during training from scratch), uses ImageNet weights.
        If False (during offline checkpoint loading/inference), uses weights=None to avoid network requests.

    Returns
    -------
    nn.Module
        Configured PyTorch model ready for training or weight loading.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained_weights else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── ImageFolder Directory Preparation ──────────────────────────────────────────

def _prepare_imagefolder_layout(data_dir: str = "data/sbpr") -> bool:
    """
    Construct an ImageFolder-compatible directory tree by symlinking images
    from the project's existing flat per-SBPR directories into:

        data/sbpr/train/sbpr_1/  ← symlinks to data/SBIN201210B00011/train/*
        data/sbpr/train/sbpr_2/  ← symlinks to data/SBIN202310B06811/train/*
        data/sbpr/train/sbpr_3/  ← symlinks to data/SBIN202507B07143/train/*
        data/sbpr/test/sbpr_1/   ← ...
        data/sbpr/test/sbpr_2/
        data/sbpr/test/sbpr_3/

    Symlinks are used to avoid duplicating image data on disk.  The function
    is idempotent — it skips creation if the layout already exists with images.

    Returns True if layout is ready, False on failure.
    """
    ready = True
    for split in ("train", "test"):
        for class_name, source_base in _SOURCE_DIRS.items():
            source_split = os.path.join(source_base, split)
            target_dir = os.path.join(data_dir, split, class_name)

            if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 0:
                continue  # Already populated

            if not os.path.isdir(source_split):
                logger.warning(f"Source directory missing: {source_split}")
                ready = False
                continue

            os.makedirs(target_dir, exist_ok=True)
            for fname in os.listdir(source_split):
                src = os.path.abspath(os.path.join(source_split, fname))
                dst = os.path.join(target_dir, fname)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        # Windows fallback: copy instead of symlink
                        shutil.copy2(src, dst)
    return ready


# ── Training ───────────────────────────────────────────────────────────────────

def train_sbpr_image_model(
    data_dir: str = "data/sbpr",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: str = DEFAULT_WEIGHTS_PATH,
) -> nn.Module:
    """
    Fine-tune the ResNet-18 fc head on the 3-class SBPR image dataset using
    ImageFolder on data_dir/train and data_dir/test (subfolders sbpr_1,
    sbpr_2, sbpr_3).

    Uses the same transform philosophy as the rust model:
      • Train: random crop + flip + rotation + colour jitter (augmentation)
      • Test : resize 256 → centre crop 224 (deterministic)

    Only the new fc layer is trained (backbone frozen).  Tracks best
    validation accuracy across epochs and saves that checkpoint to disk.

    Parameters
    ----------
    data_dir : str
        Root of the ImageFolder tree (must contain train/ and test/ subdirs).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    lr : float
        Learning rate for the Adam optimiser.
    save_path : str
        File path for saving the best checkpoint.

    Returns
    -------
    nn.Module
        Trained model (best validation checkpoint), in eval mode on device.
    """
    device = get_device()
    logger.info(f"Training Stage 2 Secondary CNN on device [{device}]")

    # Prepare ImageFolder symlink layout from existing per-SBPR directories
    _prepare_imagefolder_layout(data_dir)

    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Expected train/ and test/ subdirs under '{data_dir}'")

    train_dataset = datasets.ImageFolder(train_dir, transform=_train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=_test_transform)

    logger.info(f"Training set: {len(train_dataset)} images across classes {train_dataset.classes}")
    logger.info(f"Test set:     {len(test_dataset)} images across classes {test_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_sbpr_image_model(num_classes=len(train_dataset.classes), freeze_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimise parameters with requires_grad=True (the fc layer)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Evaluate ──
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ★ New best validation accuracy: {best_val_acc:.2%} — checkpoint saved")

        scheduler.step()

    logger.info(f"Training complete.  Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"Saved best checkpoint to '{save_path}'")

    # Reload best checkpoint before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


# ── Weight Loading ─────────────────────────────────────────────────────────────

def load_trained_sbpr_image_model(
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a previously trained 3-class SBPR image model checkpoint (strictly offline, zero network requests)."""
    if device is None:
        device = get_device()

    if not os.path.exists(weights_path):
        # Search fallback locations
        bundle = _get_bundle_path()
        fallbacks = [
            os.path.join(bundle, "models", "sbpr_image.pth"),
            "models/sbpr_image.pth",
        ]
        found = False
        for alt in fallbacks:
            if os.path.exists(alt):
                weights_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"SBPR image model weights not found at: {weights_path}. Searched: {fallbacks}")

    model = build_sbpr_image_model(num_classes=3, freeze_backbone=True, use_pretrained_weights=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded Stage 2 Secondary CNN weights from '{weights_path}' onto [{device}]")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_sbpr_image(
    image_path: str,
    model: nn.Module,
    threshold: float = 0.6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run single-image 3-class SBPR inference.

    Parameters
    ----------
    image_path : str
        Path to a still image file.
    model : nn.Module
        Loaded 3-class SBPR PyTorch model.
    threshold : float
        Confidence floor below which the prediction is marked "uncertain".
        Default is **0.6** (vs 0.75 for the binary rust model) because with
        3 output classes the softmax probability mass is distributed across
        more bins — a perfectly balanced random baseline would yield ~0.33,
        so 0.6 already represents strong class preference while 0.75 would
        discard too many correct but naturally lower-confidence predictions.
    device : torch.device, optional
        Inference device (inferred from model parameters if None).

    Returns
    -------
    Dict[str, Any]
        {
            "sbpr_no": str — full SBPR code (e.g. "SBIN201210B00011") or "uncertain",
            "sbpr_class": str — short class name (e.g. "sbpr_1") or "uncertain",
            "confidence": float,
            "image_path": str,
        }
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device()

    if not os.path.isfile(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return {
            "sbpr_no": "uncertain", "sbpr_class": "uncertain",
            "confidence": 0.0, "image_path": image_path,
        }

    try:
        image = Image.open(image_path).convert("RGB")
        tensor = _test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        confidence_val = conf.item()
        pred_class_idx = pred_idx.item()

        if pred_class_idx < len(SBPR_CLASS_NAMES):
            class_name = SBPR_CLASS_NAMES[pred_class_idx]
            sbpr_code = SBPR_CODE_MAP.get(class_name, "unknown")
        else:
            class_name = f"class_{pred_class_idx}"
            sbpr_code = "unknown"

        if confidence_val < threshold:
            return {
                "sbpr_no": "uncertain", "sbpr_class": "uncertain",
                "confidence": confidence_val, "image_path": image_path,
                "raw_sbpr_no": sbpr_code, "raw_class": class_name,
            }

        return {
            "sbpr_no": sbpr_code,
            "sbpr_class": class_name,
            "confidence": confidence_val,
            "image_path": image_path,
        }

    except Exception as e:
        logger.error(f"Failed inference on '{image_path}': {e}")
        return {
            "sbpr_no": "uncertain", "sbpr_class": "uncertain",
            "confidence": 0.0, "image_path": image_path,
        }


def predict_sbpr_image_for_ftir(
    image_paths: List[str],
    model: nn.Module,
    threshold: float = 0.6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Aggregate single-image predictions across all normalised images belonging
    to one FTIR record.  Returns the highest-confidence non-uncertain result,
    or the highest-confidence result overall if all are uncertain.
    """
    if not image_paths:
        return {
            "sbpr_no": "uncertain", "sbpr_class": "uncertain",
            "confidence": 0.0, "image_path": None,
            "total_images_evaluated": 0,
        }

    all_results = [predict_sbpr_image(p, model, threshold, device) for p in image_paths]
    confident = [r for r in all_results if r["sbpr_no"] != "uncertain"]

    best = max(confident, key=lambda r: r["confidence"]) if confident else max(all_results, key=lambda r: r["confidence"])
    best["total_images_evaluated"] = len(all_results)
    best["all_predictions"] = all_results
    return best


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage 2 Secondary: 3-Class SBPR Image Classifier (ResNet-18)"
    )
    parser.add_argument("--train", action="store_true",
                        help="Train the model from scratch on data/sbpr/{train,test}")
    parser.add_argument("--data-dir", type=str, default="data/sbpr",
                        help="Root directory for ImageFolder train/test splits")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training mini-batch size (default: 16)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image for inference (requires trained weights)")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH,
                        help=f"Path to model weights (default: {DEFAULT_WEIGHTS_PATH})")
    args = parser.parse_args()

    print("=" * 68)
    print("  Stage 2 Secondary: 3-Class SBPR Image Classifier (ResNet-18)")
    print("=" * 68)
    print(f"  Compute Device: [{get_device()}]")

    if args.train:
        print(f"\n--- Training Mode ---")
        print(f"  Data Dir  : {args.data_dir}")
        print(f"  Epochs    : {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Save Path : {args.weights}")
        print()
        trained_model = train_sbpr_image_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.weights,
        )
        print(f"\n✓ Training complete.  Weights saved to '{args.weights}'.")

        # Quick sanity prediction after training
        test_dir = os.path.join(args.data_dir, "test")
        if os.path.isdir(test_dir):
            for class_dir in sorted(os.listdir(test_dir)):
                class_path = os.path.join(test_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                imgs = [f for f in os.listdir(class_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
                if imgs:
                    sample = os.path.join(class_path, imgs[0])
                    res = predict_sbpr_image(sample, trained_model)
                    print(f"  Sample [{class_dir}] {imgs[0]} → {res['sbpr_no']} "
                          f"(conf: {res['confidence']:.2%})")

    elif args.image:
        print(f"\n--- Inference Mode ---")
        print(f"  Image   : {args.image}")
        print(f"  Weights : {args.weights}")

        model = load_trained_sbpr_image_model(args.weights)
        result = predict_sbpr_image(args.image, model)

        print(f"\n  Prediction  : {result['sbpr_no']}")
        print(f"  Class       : {result.get('sbpr_class', 'N/A')}")
        print(f"  Confidence  : {result['confidence']:.2%}")

    else:
        print("\nUsage:")
        print("  Train:   python -m src.sbpr_image_model --train [--epochs 30] [--batch-size 16]")
        print("  Predict: python -m src.sbpr_image_model --image path/to/photo.jpg")
        print("\nRun with --help for all options.")

    print("=" * 68)
