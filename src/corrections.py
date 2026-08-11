import os
import json
import shutil
import logging
import math
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

from .utils import get_persistent_dir

# Base directory for the learning layer persistent storage
_BASE_DIR = get_persistent_dir()
_CORRECTIONS_DIR = os.path.join(_BASE_DIR, "corrections")
_DB_FILE = os.path.join(_CORRECTIONS_DIR, "corrections_db.json")

def _load_db() -> Dict[str, Any]:
    if not os.path.exists(_DB_FILE):
        return {"overrides": {}}
    try:
        with open(_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load corrections DB: {e}")
        return {"overrides": {}}

def _save_db(db: Dict[str, Any]):
    os.makedirs(_CORRECTIONS_DIR, exist_ok=True)
    with open(_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)

def save_correction(
    ftir_no: str,
    correction_type: str,
    original_prediction: str,
    correct_label: str,
    user_reason: str,
    image_paths: List[str],
    metadata: Dict[str, str],
):
    """Save a human-reported correction for a specific FTIR."""
    db = _load_db()
    
    # Ensure images are copied to a permanent learning folder and compute embeddings
    saved_images = []
    embeddings = []
    
    if image_paths:
        from src.rust_model import load_trained_rust_model, get_image_embedding
        from src.utils import get_device
        device = get_device()
        try:
            model = load_trained_rust_model(device=device)
        except Exception as e:
            logger.warning(f"Could not load rust model for embeddings: {e}")
            model = None

        ftir_img_dir = os.path.join(_CORRECTIONS_DIR, "images", ftir_no)
        os.makedirs(ftir_img_dir, exist_ok=True)
        for path in image_paths:
            if os.path.exists(path):
                dest = os.path.join(ftir_img_dir, os.path.basename(path))
                try:
                    shutil.copy2(path, dest)
                    saved_images.append(dest)
                    if model:
                        emb = get_image_embedding(dest, model, device)
                        if emb:
                            embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to process image {path} for learning layer: {e}")
    
    # Add or update the record
    db["overrides"][ftir_no] = {
        "correction_type": correction_type,
        "original_prediction": original_prediction,
        "correct_label": correct_label,
        "user_reason": user_reason,
        "metadata": metadata,
        "saved_images": saved_images,
        "embeddings": embeddings,
        "timestamp": datetime.now().isoformat(),
    }
    
    _save_db(db)
    logger.info(f"Saved learning correction for FTIR {ftir_no} -> {correct_label} with {len(embeddings)} visual embeddings.")

def get_override_for_ftir(ftir_no: str) -> Optional[Dict[str, Any]]:
    """Check if there is an exact human override for this FTIR number."""
    db = _load_db()
    return db["overrides"].get(ftir_no)

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    # Assuming vec1 and vec2 are L2-normalized, cosine similarity is just dot product
    return sum(a * b for a, b in zip(vec1, vec2))

def find_similar_override(metadata: Dict[str, str], image_paths: List[str] = None, text_threshold: float = 0.6, image_threshold: float = 0.90) -> Optional[Dict[str, Any]]:
    """
    Check if a similar FTIR has been manually corrected before.
    If image_paths are provided, heavily prioritizes visual similarity via embeddings.
    """
    db = _load_db()
    overrides = db.get("overrides", {})
    if not overrides:
        return None
        
    current_embeddings = []
    if image_paths:
        from src.rust_model import load_trained_rust_model, get_image_embedding
        from src.utils import get_device
        device = get_device()
        try:
            model = load_trained_rust_model(device=device)
            for path in image_paths:
                emb = get_image_embedding(path, model, device)
                if emb:
                    current_embeddings.append(emb)
        except Exception:
            pass

    current_text = " ".join([
        metadata.get("Subject (English)", ""),
        metadata.get("Customer Complaint", ""),
        metadata.get("Subject", "")
    ]).lower()
    
    current_words = set(current_text.split()) if current_text.strip() else set()
        
    best_match = None
    best_score = 0.0
    match_reason = ""
    
    for ftir_no, record in overrides.items():
        # Visual similarity takes precedence
        visual_score = 0.0
        rec_embs = record.get("embeddings", [])
        if current_embeddings and rec_embs:
            # Find the best matching pair of images between the current FTIR and historical FTIR
            for c_emb in current_embeddings:
                for r_emb in rec_embs:
                    sim = _cosine_similarity(c_emb, r_emb)
                    if sim > visual_score:
                        visual_score = sim
        
        if visual_score >= image_threshold and visual_score > best_score:
            best_score = visual_score
            best_match = record
            match_reason = f"Visual match ({visual_score:.2f} >= {image_threshold})"
            continue

        # Text similarity fallback
        rec_meta = record.get("metadata", {})
        rec_text = " ".join([
            rec_meta.get("Subject (English)", ""),
            rec_meta.get("Customer Complaint", ""),
            rec_meta.get("Subject", "")
        ]).lower()
        
        rec_words = set(rec_text.split())
        if rec_words and current_words:
            intersection = current_words.intersection(rec_words)
            union = current_words.union(rec_words)
            text_score = len(intersection) / len(union) if union else 0.0
            
            if text_score >= text_threshold and text_score > best_score and not best_match:
                best_score = text_score
                best_match = record
                match_reason = f"Text match ({text_score:.2f} >= {text_threshold})"
            
    if best_match:
        logger.info(f"Learning Layer: Found similar past correction -> {match_reason}")
        
    return best_match
