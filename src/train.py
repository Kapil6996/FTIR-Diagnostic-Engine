"""
Training script for the rust vs non_rust binary image classifier.

Uses a 3-phase training strategy with NO validation split (uses all training
data since we have a separate test set). Evaluates on the test set to track
progress:
  Phase 1: Freeze backbone, train only the classifier head (10 epochs).
  Phase 2: Unfreeze everything with differential LRs + cosine annealing (50 epochs).
  Phase 3: Hard-example mining — find misclassified training images and retrain
           with a weighted sampler that oversamples hard examples (30 epochs).

Saves the best checkpoint (by test accuracy) to models/rust_demo.pth.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data import get_dataloaders
from src.model import build_model, get_device

# ── hyperparameters ───────────────────────────────────────────────────────
PHASE1_EPOCHS = 10          # frozen backbone, train head only
PHASE2_EPOCHS = 50          # full fine-tuning
PHASE3_EPOCHS = 30          # hard-example refinement
PHASE1_LR = 1e-3            # head-only LR
PHASE2_BACKBONE_LR = 5e-5   # low LR for pretrained backbone
PHASE2_HEAD_LR = 5e-4       # higher LR for classifier head
PHASE3_LR = 1e-5            # very low LR for refinement
HARD_EXAMPLE_WEIGHT = 5.0   # how much more to sample hard examples
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
CHECKPOINT_PATH = os.path.join("models", "rust_demo.pth")
BATCH_SIZE = 16


# ── helpers ───────────────────────────────────────────────────────────────
def _run_epoch(model, loader, criterion, device, optimizer=None):
    """Run a single training or validation epoch."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def _freeze_backbone(model):
    """Freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
        else:
            param.requires_grad = True


def _unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def _count_params(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _find_hard_examples(model, dataset, device):
    """
    Run inference on the training set and return per-sample weights.
    Misclassified samples get weight = HARD_EXAMPLE_WEIGHT, correct = 1.0.
    Low-confidence correct predictions (< 0.8) get weight = 2.0.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    weights = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = nnF.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            for pred, label, conf in zip(preds, labels, max_probs):
                if pred.item() != label.item():
                    weights.append(HARD_EXAMPLE_WEIGHT)
                elif conf.item() < 0.8:
                    weights.append(2.0)  # borderline correct
                else:
                    weights.append(1.0)

    n_hard = sum(1 for w in weights if w == HARD_EXAMPLE_WEIGHT)
    n_borderline = sum(1 for w in weights if w == 2.0)
    print(f"  Hard (wrong): {n_hard}, Borderline (low conf): {n_borderline}, "
          f"Easy: {len(weights) - n_hard - n_borderline}")
    return weights


def train():
    """3-phase training using ALL training data (no val split)."""

    device = get_device()

    # ── data ──────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    train_dataset = train_loader.dataset

    print(f"Train size : {len(train_dataset)}")
    print(f"Test size  : {len(test_loader.dataset)}")
    print()

    # ── model setup ───────────────────────────────────────────────────────
    model = build_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_test_acc = 0.0
    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Frozen backbone — train classifier head only
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 64)
    print("  PHASE 1: Training classifier head only (backbone frozen)")
    print("=" * 64)

    _freeze_backbone(model)
    trainable, total_p = _count_params(model)
    print(f"Trainable params: {trainable:,} / {total_p:,}\n")

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=PHASE1_EPOCHS, eta_min=1e-5)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer_p1
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, criterion, device, optimizer=None
        )

        current_lr = scheduler_p1.get_last_lr()[0]
        scheduler_p1.step()

        print(
            f"Epoch {epoch:>2}/{total_epochs}  │  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}%  │  "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:6.2f}%  │  "
            f"LR: {current_lr:.2e}",
            end="",
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  ✓ saved", end="")
        print()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: Full fine-tuning with differential learning rates
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 64)
    print("  PHASE 2: Full fine-tuning (all layers unfrozen)")
    print("=" * 64)

    _unfreeze_all(model)
    trainable, total_p = _count_params(model)
    print(f"Trainable params: {trainable:,} / {total_p:,}\n")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": PHASE2_BACKBONE_LR},
            {"params": head_params, "lr": PHASE2_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        global_epoch = PHASE1_EPOCHS + epoch

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer_p2
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, criterion, device, optimizer=None
        )

        current_lr = scheduler_p2.get_last_lr()[0]
        scheduler_p2.step()

        print(
            f"Epoch {global_epoch:>2}/{total_epochs}  │  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}%  │  "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:6.2f}%  │  "
            f"LR: {current_lr:.2e}",
            end="",
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  ✓ saved", end="")
        print()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: Hard-example mining + refinement
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 64)
    print("  PHASE 3: Hard-example refinement (learning from mistakes)")
    print("=" * 64)

    # Load the best checkpoint before mining
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Find hard examples in the training set
    sample_weights = _find_hard_examples(model, train_dataset, device)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights) * 2,  # oversample to see more hard examples
        replacement=True,
    )
    hard_train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )

    _unfreeze_all(model)
    optimizer_p3 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": PHASE3_LR},
            {"params": head_params, "lr": PHASE3_LR * 5},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_p3 = CosineAnnealingLR(optimizer_p3, T_max=PHASE3_EPOCHS, eta_min=1e-7)

    for epoch in range(1, PHASE3_EPOCHS + 1):
        global_epoch = PHASE1_EPOCHS + PHASE2_EPOCHS + epoch

        train_loss, train_acc = _run_epoch(
            model, hard_train_loader, criterion, device, optimizer=optimizer_p3
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, criterion, device, optimizer=None
        )

        current_lr = scheduler_p3.get_last_lr()[0]
        scheduler_p3.step()

        print(
            f"Epoch {global_epoch:>2}/{total_epochs}  │  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}%  │  "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:6.2f}%  │  "
            f"LR: {current_lr:.2e}",
            end="",
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  ✓ saved", end="")
        print()

    print()
    print(f"Training complete. Best test accuracy: {best_test_acc:.2f}%")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


# ── entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
