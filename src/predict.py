"""
Single-image inference for the rust vs non_rust binary classifier.

Usage::

    python -m src.predict --image path/to/photo.jpg
    python -m src.predict --image path/to/photo.jpg --threshold 0.85
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model, get_device

# Class index → human-readable label (matches ImageFolder alphabetical order)
CLASS_LABELS = {0: "No rust", 1: "Rust"}

# Same transform used for the test set in src/data.py
_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict_image(image_path, model_path="models/rust_demo.pth", threshold=0.75):
    """
    Run inference on a single image and return a label with confidence.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    model_path : str
        Path to the saved model checkpoint (state_dict).
    threshold : float
        Minimum softmax confidence required to commit to a prediction.
        Below this value the function returns ``"Can't classify"``.

    Returns
    -------
    label : str
        One of ``"Rust"``, ``"No rust"``, or ``"Can't classify"``.
    confidence : float
        Softmax probability of the predicted class (0-1).
    """
    device = get_device()

    # ── model --------------------------------------------------------------
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ── image preprocessing ------------------------------------------------
    image = Image.open(image_path).convert("RGB")
    tensor = _inference_transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # ── inference ----------------------------------------------------------
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred_idx = pred_idx.item()

    if confidence < threshold:
        label = "Can't classify"
    else:
        label = CLASS_LABELS[pred_idx]

    return label, confidence


# ── entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict whether a single image shows rust or not."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/rust_demo.pth",
        help="Path to the model checkpoint (default: models/rust_demo.pth).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for classification (default: 0.75).",
    )

    args = parser.parse_args()

    label, confidence = predict_image(
        args.image, model_path=args.model, threshold=args.threshold
    )

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.4f}")
