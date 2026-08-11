"""
Stage 2 Fusion Engine (src.fusion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combines the two independent SBPR classification channels into one final,
auditable decision per FTIR record:

    Primary channel   → metadata Decision Tree  (sbpr_tree.py,   ~93% accuracy)
    Secondary channel → image CNN classifier     (sbpr_image_model.py, ~72% accuracy)

Fusion Decision Matrix:
┌───────────────────────┬────────────────────────┬───────────────────────────────┐
│ Metadata prediction   │ Image prediction       │ Fusion outcome                │
├───────────────────────┼────────────────────────┼───────────────────────────────┤
│ SBPR-X                │ SBPR-X  (agree)        │ ✓ High confidence SBPR-X      │
│ SBPR-X                │ SBPR-Y  (disagree)     │ ⚠ Flag for manual review      │
│ SBPR-X                │ uncertain              │ ✓ Trust metadata, note caveat │
│ uncertain             │ SBPR-X                 │ ✓ Trust image, note caveat    │
│ uncertain             │ uncertain              │ ⚠ Flag for manual review      │
└───────────────────────┴────────────────────────┴───────────────────────────────┘
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def fuse_sbpr_predictions(
    metadata_result: Dict[str, Any],
    image_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fuse metadata-based and image-based SBPR predictions into a single
    auditable diagnostic decision.

    Parameters
    ----------
    metadata_result : dict
        Output from ``sbpr_tree.predict_sbpr_metadata()``.
        Expected keys: ``sbpr_no``, ``confidence``, ``reason``.
    image_result : dict
        Output from ``sbpr_image_model.predict_sbpr_image()`` or
        ``predict_sbpr_image_for_ftir()``.
        Expected keys: ``sbpr_no``, ``confidence``.

    Returns
    -------
    dict
        {
            "final_sbpr": str       — agreed SBPR code or "manual_review",
            "reason": str           — plain-English audit trail,
            "flag_for_review": bool — True when human inspection is needed,
            "metadata_confidence": float,
            "image_confidence": float,
            "metadata_sbpr": str    — raw metadata prediction for traceability,
            "image_sbpr": str       — raw image prediction for traceability,
        }
    """
    meta_sbpr  = str(metadata_result.get("sbpr_no", "uncertain"))
    meta_conf  = float(metadata_result.get("confidence", 0.0))
    meta_reason = str(metadata_result.get("reason", ""))

    img_sbpr   = str(image_result.get("sbpr_no", "uncertain"))
    img_conf   = float(image_result.get("confidence", 0.0))

    meta_uncertain = meta_sbpr.lower() in ("uncertain", "unknown", "manual_review")
    img_uncertain  = img_sbpr.lower()  in ("uncertain", "unknown", "manual_review")

    # ── Branch 1: Both channels agree on the same SBPR ─────────────────
    if not meta_uncertain and not img_uncertain and meta_sbpr == img_sbpr:
        reason = f"{meta_reason} (image verification agrees, conf {img_conf:.0%})"
        return {
            "final_sbpr":          meta_sbpr,
            "reason":              reason,
            "flag_for_review":     False,
            "metadata_confidence": meta_conf,
            "image_confidence":    img_conf,
            "metadata_sbpr":       meta_sbpr,
            "image_sbpr":          img_sbpr,
        }

    # ── Branch 2: Image prediction is uncertain / low confidence ───────
    #    Trust the metadata prediction alone — it is the stronger, more
    #    interpretable channel (~93% CV accuracy vs ~72% for the CNN).
    if not meta_uncertain and img_uncertain:
        reason = (
            f"{meta_reason} "
            f"(image verification was inconclusive, conf {img_conf:.0%} "
            f"— relying on metadata classification alone)"
        )
        return {
            "final_sbpr":          meta_sbpr,
            "reason":              reason,
            "flag_for_review":     False,
            "metadata_confidence": meta_conf,
            "image_confidence":    img_conf,
            "metadata_sbpr":       meta_sbpr,
            "image_sbpr":          img_sbpr,
        }

    # ── Branch 3: Metadata is uncertain but image has a prediction ─────
    #    Fall back to the image channel — better than flagging everything.
    if meta_uncertain and not img_uncertain:
        reason = (
            f"Metadata classification was inconclusive (conf {meta_conf:.0%}). "
            f"Image classifier predicts {img_sbpr} with {img_conf:.0%} confidence."
        )
        return {
            "final_sbpr":          img_sbpr,
            "reason":              reason,
            "flag_for_review":     False,
            "metadata_confidence": meta_conf,
            "image_confidence":    img_conf,
            "metadata_sbpr":       meta_sbpr,
            "image_sbpr":          img_sbpr,
        }

    # ── Branch 4: Both channels disagree on SBPR number ────────────────
    if not meta_uncertain and not img_uncertain and meta_sbpr != img_sbpr:
        reason = (
            f"Metadata suggests {meta_sbpr} ({meta_reason}, conf {meta_conf:.0%}), "
            f"image suggests {img_sbpr} (conf {img_conf:.0%}) "
            f"— flagged for manual review."
        )
        return {
            "final_sbpr":          "manual_review",
            "reason":              reason,
            "flag_for_review":     True,
            "metadata_confidence": meta_conf,
            "image_confidence":    img_conf,
            "metadata_sbpr":       meta_sbpr,
            "image_sbpr":          img_sbpr,
        }

    # ── Branch 5: Both channels uncertain ──────────────────────────────
    reason = (
        f"Both metadata (conf {meta_conf:.0%}) and image (conf {img_conf:.0%}) "
        f"predictions are inconclusive — flagged for manual review."
    )
    return {
        "final_sbpr":          "manual_review",
        "reason":              reason,
        "flag_for_review":     True,
        "metadata_confidence": meta_conf,
        "image_confidence":    img_conf,
        "metadata_sbpr":       meta_sbpr,
        "image_sbpr":          img_sbpr,
    }


# ── CLI Demonstration ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 72)
    print("  Stage 2 Fusion Engine — Decision Branch Demonstration")
    print("=" * 72)

    # ── Case 1: Agreement ──────────────────────────────────────────────
    meta_1 = {
        "sbpr_no": "SBIN201210B00011",
        "confidence": 1.0,
        "reason": "subject mentions 'seat' AND vehicle model = Model 7",
    }
    img_1 = {
        "sbpr_no": "SBIN201210B00011",
        "confidence": 0.70,
    }

    result_1 = fuse_sbpr_predictions(meta_1, img_1)

    print("\n┌─ CASE 1: Both channels AGREE ─────────────────────────────────┐")
    print(f"  Metadata prediction : {meta_1['sbpr_no']} (conf {meta_1['confidence']:.0%})")
    print(f"  Image prediction    : {img_1['sbpr_no']} (conf {img_1['confidence']:.0%})")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Final SBPR          : {result_1['final_sbpr']}")
    print(f"  Flag for Review     : {result_1['flag_for_review']}")
    print(f"  Reason              : {result_1['reason']}")
    print("└───────────────────────────────────────────────────────────────┘")

    # ── Case 2: Disagreement ───────────────────────────────────────────
    meta_2 = {
        "sbpr_no": "SBIN202310B06811",
        "confidence": 0.64,
        "reason": "subject lacks keyword 'seat' AND subject lacks keyword 'hw'",
    }
    img_2 = {
        "sbpr_no": "SBIN201210B00011",
        "confidence": 0.68,
    }

    result_2 = fuse_sbpr_predictions(meta_2, img_2)

    print("\n┌─ CASE 2: Channels DISAGREE ───────────────────────────────────┐")
    print(f"  Metadata prediction : {meta_2['sbpr_no']} (conf {meta_2['confidence']:.0%})")
    print(f"  Image prediction    : {img_2['sbpr_no']} (conf {img_2['confidence']:.0%})")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Final SBPR          : {result_2['final_sbpr']}")
    print(f"  Flag for Review     : {result_2['flag_for_review']}")
    print(f"  Reason              : {result_2['reason']}")
    print("└───────────────────────────────────────────────────────────────┘")

    # ── Case 3: Image uncertain ────────────────────────────────────────
    meta_3 = {
        "sbpr_no": "SBIN202507B07143",
        "confidence": 1.0,
        "reason": "subject mentions 'hw'",
    }
    img_3 = {
        "sbpr_no": "uncertain",
        "confidence": 0.42,
    }

    result_3 = fuse_sbpr_predictions(meta_3, img_3)

    print("\n┌─ CASE 3: Image UNCERTAIN, trust metadata ─────────────────────┐")
    print(f"  Metadata prediction : {meta_3['sbpr_no']} (conf {meta_3['confidence']:.0%})")
    print(f"  Image prediction    : {img_3['sbpr_no']} (conf {img_3['confidence']:.0%})")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Final SBPR          : {result_3['final_sbpr']}")
    print(f"  Flag for Review     : {result_3['flag_for_review']}")
    print(f"  Reason              : {result_3['reason']}")
    print("└───────────────────────────────────────────────────────────────┘")

    # ── Case 4: Metadata uncertain, image has prediction ───────────────
    meta_4 = {
        "sbpr_no": "uncertain",
        "confidence": 0.38,
        "reason": "No strong metadata signal",
    }
    img_4 = {
        "sbpr_no": "SBIN202310B06811",
        "confidence": 0.81,
    }

    result_4 = fuse_sbpr_predictions(meta_4, img_4)

    print("\n┌─ CASE 4: Metadata UNCERTAIN, trust image ─────────────────────┐")
    print(f"  Metadata prediction : {meta_4['sbpr_no']} (conf {meta_4['confidence']:.0%})")
    print(f"  Image prediction    : {img_4['sbpr_no']} (conf {img_4['confidence']:.0%})")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Final SBPR          : {result_4['final_sbpr']}")
    print(f"  Flag for Review     : {result_4['flag_for_review']}")
    print(f"  Reason              : {result_4['reason']}")
    print("└───────────────────────────────────────────────────────────────┘")

    # ── Case 5: Both uncertain ─────────────────────────────────────────
    meta_5 = {
        "sbpr_no": "uncertain",
        "confidence": 0.35,
        "reason": "No strong metadata signal",
    }
    img_5 = {
        "sbpr_no": "uncertain",
        "confidence": 0.31,
    }

    result_5 = fuse_sbpr_predictions(meta_5, img_5)

    print("\n┌─ CASE 5: BOTH uncertain ──────────────────────────────────────┐")
    print(f"  Metadata prediction : {meta_5['sbpr_no']} (conf {meta_5['confidence']:.0%})")
    print(f"  Image prediction    : {img_5['sbpr_no']} (conf {img_5['confidence']:.0%})")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Final SBPR          : {result_5['final_sbpr']}")
    print(f"  Flag for Review     : {result_5['flag_for_review']}")
    print(f"  Reason              : {result_5['reason']}")
    print("└───────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 72)
