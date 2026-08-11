"""
Evaluation script for the rust vs non_rust binary image classifier.

Loads the best checkpoint from models/rust_demo.pth, runs inference on the
test set WITH Test-Time Augmentation (TTA), and reports:
  • Overall accuracy (standard + TTA)
  • Confusion matrix
  • Per-class precision / recall / F1
  • Softmax confidence distributions for correct vs misclassified predictions
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model, get_device

CHECKPOINT_PATH = "models/rust_demo.pth"
TTA_TRANSFORMS_COUNT = 7  # number of augmented views per image


def _get_tta_transforms():
    """Return a list of test-time augmentation transforms."""
    return [
        # Original (standard test transform)
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slight rotation left
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slight rotation right
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slightly larger crop
        transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slightly smaller crop
        transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]


def evaluate():
    """Load the saved checkpoint and evaluate on the test set with TTA."""

    device = get_device()

    # ── model --------------------------------------------------------------
    model = build_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # ── data ---------------------------------------------------------------
    _, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset
    class_names = test_dataset.classes  # e.g. ['non_rust', 'rust']

    # ── Standard inference (no TTA) ----------------------------------------
    all_labels = []
    all_preds_std = []
    all_confs_std = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds_std.extend(preds.cpu().tolist())
            all_confs_std.extend(max_probs.cpu().tolist())

    # ── TTA inference ------------------------------------------------------
    tta_transforms = _get_tta_transforms()
    # For TTA we need raw images, so use the dataset's samples list
    from PIL import Image

    all_preds_tta = []
    all_confs_tta = []

    print(f"\nRunning TTA with {len(tta_transforms)} augmented views...")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, true_label = test_dataset.samples[idx]
            raw_img = Image.open(img_path).convert("RGB")

            # Average softmax across all TTA transforms
            avg_probs = torch.zeros(2, device=device)
            for t in tta_transforms:
                img_tensor = t(raw_img).unsqueeze(0).to(device)
                output = model(img_tensor)
                avg_probs += F.softmax(output, dim=1).squeeze(0)
            avg_probs /= len(tta_transforms)

            max_prob, pred = torch.max(avg_probs, dim=0)
            all_preds_tta.append(pred.item())
            all_confs_tta.append(max_prob.item())

    # ── Standard metrics ---------------------------------------------------
    acc_std = accuracy_score(all_labels, all_preds_std)

    print()
    print("=" * 64)
    print(f"  Standard Accuracy (no TTA): {acc_std * 100:.2f}%")
    print("=" * 64)

    # ── TTA metrics --------------------------------------------------------
    acc_tta = accuracy_score(all_labels, all_preds_tta)
    cm = confusion_matrix(all_labels, all_preds_tta)
    report = classification_report(
        all_labels, all_preds_tta, target_names=class_names, digits=4
    )

    print()
    print("=" * 64)
    print(f"  TTA Accuracy ({len(tta_transforms)} views): {acc_tta * 100:.2f}%")
    print("=" * 64)

    print("\nConfusion Matrix (TTA):")
    header = "            " + "  ".join(f"{name:>10}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>10}" for val in row)
        print(f"  {class_names[i]:>10}  {row_str}")

    print(f"\nClassification Report (TTA):\n{report}")

    # ── confidence distributions (TTA) ------------------------------------
    correct_confs = []
    wrong_confs = []
    for label, pred, conf in zip(all_labels, all_preds_tta, all_confs_tta):
        if pred == label:
            correct_confs.append(round(conf, 4))
        else:
            wrong_confs.append(round(conf, 4))

    print("-" * 60)
    print(f"Correct predictions  ({len(correct_confs):>4}): confidences ↓")
    print(sorted(correct_confs, reverse=True))
    print()
    print(f"Wrong predictions    ({len(wrong_confs):>4}): confidences ↓")
    print(sorted(wrong_confs, reverse=True))
    print("-" * 60)

    if correct_confs and wrong_confs:
        min_correct = min(correct_confs)
        max_wrong = max(wrong_confs)
        print(f"\nLowest correct-prediction confidence : {min_correct:.4f}")
        print(f"Highest wrong-prediction confidence  : {max_wrong:.4f}")
        if min_correct > max_wrong:
            suggested = round((min_correct + max_wrong) / 2, 4)
            print(f"→ Suggested 'can't classify' threshold: < {suggested}")
        else:
            print(
                "→ Confidence ranges overlap — inspect the lists above "
                "to pick a threshold manually."
            )
    print()


# ── entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()
