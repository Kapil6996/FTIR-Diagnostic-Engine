"""
Visual demo grid for the rust vs non_rust binary classifier.

Picks 12 random test images, runs inference, and saves a 3×4 grid to
``outputs/demo_grid.png`` with green/red borders indicating correct/wrong
predictions.

Usage::

    python -m src.demo_grid
"""

import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model, get_device

# ── constants ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "models/rust_demo.pth"
OUTPUT_PATH = os.path.join("outputs", "demo_grid.png")
NUM_IMAGES = 12
ROWS, COLS = 3, 4
THRESHOLD = 0.75

CLASS_LABELS = {0: "No rust", 1: "Rust"}
TRUE_LABEL_MAP = {"non_rust": 0, "rust": 1}

_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _collect_test_images(test_dir="data/test"):
    """Return a list of (image_path, true_class_index) for every test image."""
    samples = []
    for class_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        true_idx = TRUE_LABEL_MAP.get(class_name)
        if true_idx is None:
            continue
        for fname in os.listdir(class_dir):
            if fname.startswith('.') or fname == 'Thumbs.db':
                continue
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath):
                samples.append((fpath, true_idx))
    return samples


def _predict_single(model, image_path, device):
    """Run inference on one image and return (pred_label, confidence, pred_idx)."""
    image = Image.open(image_path).convert("RGB")
    tensor = _inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred_idx = pred_idx.item()

    if confidence < THRESHOLD:
        label = "Can't classify"
    else:
        label = CLASS_LABELS[pred_idx]

    return label, confidence, pred_idx


def build_demo_grid():
    """Create and save the 3×4 demo grid."""

    device = get_device()

    # ── model --------------------------------------------------------------
    model = build_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # ── sample images ------------------------------------------------------
    samples = _collect_test_images()
    if len(samples) < NUM_IMAGES:
        print(
            f"Warning: only {len(samples)} test images found, "
            f"expected at least {NUM_IMAGES}."
        )
    picked = random.sample(samples, min(NUM_IMAGES, len(samples)))

    # ── build figure -------------------------------------------------------
    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, 12))
    fig.suptitle("Rust Detection — Demo Grid", fontsize=18, fontweight="bold", y=0.98)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(picked):
            ax.axis("off")
            continue

        img_path, true_idx = picked[idx]
        label, confidence, pred_idx = _predict_single(model, img_path, device)

        # Display the original image (un-normalised)
        display_img = Image.open(img_path).convert("RGB")
        ax.imshow(display_img)

        # Determine correctness
        is_correct = pred_idx == true_idx
        border_color = "green" if is_correct else "red"

        # Colour the subplot border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)

        ax.set_xticks([])
        ax.set_yticks([])

        true_name = CLASS_LABELS[true_idx]
        ax.set_title(
            f"Pred: {label}  ({confidence:.2%})\nTrue: {true_name}",
            fontsize=10,
            color=border_color,
            fontweight="bold",
            pad=6,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Demo grid saved to {OUTPUT_PATH}")


# ── entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_demo_grid()
