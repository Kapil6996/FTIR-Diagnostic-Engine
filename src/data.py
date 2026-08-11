"""
Data loading module for the rust vs non_rust binary image classifier.

Uses torchvision.datasets.ImageFolder to load images from the standard
    data/train/{rust, non_rust}
    data/test/{rust, non_rust}
folder layout.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet channel-wise mean and standard deviation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir="data", batch_size=16):
    """
    Build and return train and test DataLoaders.

    Parameters
    ----------
    data_dir : str
        Root data directory containing ``train/`` and ``test/`` subfolders.
    batch_size : int
        Number of images per mini-batch.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for the training set (with augmentation).
    test_loader : DataLoader
        DataLoader for the test set (no augmentation).
    """

    # --- transforms --------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # --- datasets ----------------------------------------------------------
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Print the class-to-index mapping so we always know which label is which
    print(f"Class-to-index mapping: {train_dataset.class_to_idx}")

    # --- dataloaders -------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


# ── sanity-check entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    print(f"Training set : {len(train_loader.dataset)} images")
    print(f"Test set     : {len(test_loader.dataset)} images")
