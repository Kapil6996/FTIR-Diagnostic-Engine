"""
End-to-End FTIR → SBPR Classification Pipeline (src.pipeline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Orchestrates every module in the ftir_sbpr_tool project into a single
sequential pipeline that processes an input Excel workbook of FTIR records
and produces an augmented output Excel with per-row defect classification,
SBPR assignment, human-readable reasoning, and review flags.

Pipeline Stages:
    1. Excel ingestion            (excel_io)
    2. Browser attachment download (browser_extract)  — with resumability
    3. Media normalisation         (media_normalize)
    4. Stage 1: Rust vs Non-Rust   (rust_model)       — gate filter
    5. Stage 2 Primary: Metadata   (sbpr_features + sbpr_tree)
    6. Stage 2 Secondary: Image    (sbpr_image_model)
    7. Stage 2 Fusion              (fusion)
    8. Output Excel generation     (excel_io)
"""

import os
import sys
import time
import logging
import argparse
import datetime
from typing import Optional, Callable, Any, Dict

import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# ── Module imports ─────────────────────────────────────────────────────────────

from src.excel_io import read_ftir_sheet, write_output_sheet
from src.media_normalize import get_all_images_for_ftir
from src.rust_model import load_trained_rust_model, predict_rust_for_ftir
from src.sbpr_features import build_features
from src.sbpr_tree import predict_sbpr_metadata
from src.sbpr_image_model import load_trained_sbpr_image_model, predict_sbpr_image_for_ftir
from src.fusion import fuse_sbpr_predictions

# ── Path Resolution for Desktop App ──────────────────────────────────────────────
def _get_bundle_path():
    """Returns the base path where bundled read-only assets (models/, config/) live.

    On macOS .app bundles, sys._MEIPASS points to Contents/Frameworks/ which is
    where --add-data assets are extracted. We check multiple candidates and return
    whichever actually contains the 'models' directory.
    """
    if not getattr(sys, 'frozen', False):
        # Running from source
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    candidates = []
    # 1. _MEIPASS  (where --add-data extracts to inside .app)
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(meipass)
    # 2. Directory containing the actual executable binary
    exe_dir = os.path.dirname(sys.executable)
    candidates.append(exe_dir)
    # 3. The dist/AppName/ folder that sits *beside* the .app on macOS
    #    e.g. .../dist/Maruti_FTIR_Diagnostic_Engine.app/Contents/MacOS/exe
    #    -> go up 3 levels to get .../dist/Maruti_FTIR_Diagnostic_Engine/
    app_sibling = os.path.normpath(os.path.join(exe_dir, '..', '..', '..'))
    # Strip .app suffix to find the companion folder
    app_name = os.path.basename(app_sibling)
    if app_name.endswith('.app'):
        companion = os.path.join(os.path.dirname(app_sibling), app_name[:-4])
        candidates.append(companion)
    candidates.append(app_sibling)

    for path in candidates:
        if os.path.isdir(os.path.join(path, 'models')):
            return path

    # Ultimate fallback — prefer _MEIPASS if available
    return meipass or exe_dir

def _get_workspace_path():
    """Returns a writable user-facing folder path for data and outputs.
    
    When running from source (terminal): use the project directory (original behavior).
    When running as frozen .app: use ~/Desktop/Maruti_FTIR_Workspace.
    """
    if getattr(sys, 'frozen', False):
        return os.path.expanduser("~/Desktop/Maruti_FTIR_Workspace")
    else:
        # Running from source — use the project root (parent of src/)
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

BUNDLE_DIR = _get_bundle_path()
WORKSPACE_DIR = _get_workspace_path()

# Ensure writable workspace structure exists
os.makedirs(os.path.join(WORKSPACE_DIR, "outputs"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "ftir_records"), exist_ok=True)

# ── Default paths ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_PATH      = os.path.join(WORKSPACE_DIR, "outputs", "ftir_results.xlsx")
DEFAULT_LOG_PATH         = os.path.join(WORKSPACE_DIR, "outputs", "pipeline_log.txt")
DEFAULT_RUST_WEIGHTS     = os.path.join(BUNDLE_DIR, "models", "rust_demo.pth")
DEFAULT_SBPR_TREE_BUNDLE = os.path.join(BUNDLE_DIR, "models", "sbpr_tree.joblib")
DEFAULT_SBPR_IMG_WEIGHTS = os.path.join(BUNDLE_DIR, "models", "sbpr_image.pth")
FTIR_RECORDS_DIR         = os.path.join(WORKSPACE_DIR, "ftir_records")


# ── Logging to file + console ─────────────────────────────────────────────────

def _setup_logging(log_path: str) -> None:
    """Configure dual logging: INFO to console, DEBUG to file."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(ch)

    # File handler — append mode so crash-resume preserves prior entries
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    root.addHandler(fh)


# ── FTIR number extraction helper ─────────────────────────────────────────────

def _find_ftir_column(columns) -> Optional[str]:
    """Identify the column holding the FTIR record identifier."""
    for col in columns:
        cl = str(col).lower()
        if ("ftir" in cl and "date" not in cl and "url" not in cl) or "sno" in cl or "record" in cl:
            return col
    # Fallback: check for standard masked identifier columns like Masked VIN
    for col in columns:
        if "vin" in str(col).lower():
            return col
    return columns[0] if len(columns) > 0 else None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    rust_weights: str = DEFAULT_RUST_WEIGHTS,
    sbpr_tree_bundle: str = DEFAULT_SBPR_TREE_BUNDLE,
    sbpr_img_weights: str = DEFAULT_SBPR_IMG_WEIGHTS,
    ftir_records_dir: str = FTIR_RECORDS_DIR,
    skip_browser: bool = False,
    profile_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], Any]] = None,
) -> pd.DataFrame:
    """
    Execute the full FTIR → SBPR classification pipeline.

    Parameters
    ----------
    input_path : str
        Path to the source FTIR Excel workbook.
    output_path : str
        Destination path for the augmented results Excel.
    log_path : str
        Path for the running plaintext pipeline log.
    rust_weights : str
        Path to the Stage 1 rust classifier checkpoint.
    sbpr_tree_bundle : str
        Path to the Stage 2 metadata tree joblib bundle.
    sbpr_img_weights : str
        Path to the Stage 2 image CNN checkpoint.
    ftir_records_dir : str
        Root directory where per-FTIR attachment folders are stored.
    skip_browser : bool
        If True, skip browser extraction entirely (use only already-downloaded
        attachments in ftir_records_dir or images in the source data folders).
    profile_dir : str, optional
        Chrome/Edge persistent profile directory for browser automation.

    Returns
    -------
    pd.DataFrame
        The augmented DataFrame with classification result columns appended.
    """
    _setup_logging(log_path)
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  FTIR → SBPR Classification Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Input Excel    : {input_path}")
    logger.info(f"  Output Excel   : {output_path}")
    logger.info(f"  Pipeline Log   : {log_path}")
    logger.info(f"  Skip Browser   : {skip_browser}")
    logger.info(f"  Started at     : {datetime.datetime.now().isoformat()}")
    logger.info("=" * 70)

    # ── 1. Read input Excel ────────────────────────────────────────────
    logger.info("Stage 0: Reading input Excel workbook...")
    df = read_ftir_sheet(input_path)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Identify FTIR number column
    ftir_col = _find_ftir_column(df.columns)
    logger.info(f"  FTIR identifier column: '{ftir_col}'")

    # ── 2. Load all models once ────────────────────────────────────────
    logger.info("Stage 0: Loading model checkpoints...")

    rust_model = load_trained_rust_model(weights_path=rust_weights)
    logger.info("  ✓ Stage 1 Rust classifier loaded")

    tree_bundle = joblib.load(sbpr_tree_bundle)
    tree_model = tree_bundle["model"]
    tree_feature_names = tree_bundle["feature_names"]
    logger.info(f"  ✓ Stage 2 Metadata tree loaded ({len(tree_feature_names)} features)")

    sbpr_img_model = load_trained_sbpr_image_model(weights_path=sbpr_img_weights)
    logger.info("  ✓ Stage 2 Image CNN loaded")

    # Optionally initialise browser driver
    browser_driver = None
    if not skip_browser:
        try:
            from src.browser_extract import get_driver, process_ftir
            # Set headless=False so the user can visually confirm the browser is opening the FTIR
            browser_driver = get_driver(profile_dir=profile_dir, headless=False)
            logger.info("  ✓ Browser driver initialised (Visible Mode)")
        except Exception as e:
            logger.warning(f"  ✗ Browser driver unavailable ({e}). "
                           f"Will use only pre-downloaded attachments.")
            skip_browser = True

    # ── 3. Prepare result columns ──────────────────────────────────────
    result_cols = {
        "Defect_Type":          [],
        "SBPR_Number":          [],
        "Reason":               [],
        "Metadata_Confidence":  [],
        "Image_Confidence":     [],
        "Flag_For_Review":      [],
        "Pipeline_Status":      [],
        "Failure_Stage":        [],
        "Failure_Diagnostics":  [],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(ftir_records_dir, exist_ok=True)

    total_rows = len(df)
    rust_count = 0
    non_rust_count = 0
    review_count = 0
    error_count = 0
    warning_count = 0

    logger.info(f"\nProcessing {total_rows} FTIR records...\n")
    logger.info("-" * 70)
    if progress_callback:
        progress_callback(0, total_rows, f"Starting inference across {total_rows} records...", {})

    # ── 4. Per-row processing loop ─────────────────────────────────────
    for idx in range(total_rows):
        row = df.iloc[idx]
        ftir_no = str(row.get(ftir_col, f"row_{idx}")).strip()
        ftir_url = str(row.get("ftir_url", "")).strip()
        row_label = f"[{idx + 1}/{total_rows}] FTIR {ftir_no}"

        logger.info(f"{row_label}: Processing...")

        pipeline_status = "SUCCESS"
        failure_stage = "None"
        failure_diagnostics = "All pipeline stages executed cleanly."
        extraction_source = "None"

        # ── 4a. Browser extraction (with resumability) ─────────────────
        ftir_folder = os.path.join(ftir_records_dir, ftir_no)
        if not skip_browser and ftir_url and ftir_url.startswith("http"):
            try:
                from src.browser_extract import process_ftir
                extract_result = process_ftir(
                    driver=browser_driver,
                    ftir_no=ftir_no,
                    url=ftir_url,
                    base_save_dir=ftir_records_dir,
                )
                extraction_source = extract_result.get("extraction_source", "Unknown")
                if extract_result.get("skipped"):
                    logger.debug(f"{row_label}: Attachments already downloaded (resumability skip)")
                    extraction_source = "Local Cache"
                else:
                    n_files = len(extract_result.get("downloaded_files", []))
                    logger.info(f"{row_label}: Downloaded {n_files} attachment(s)")
            except Exception as e:
                err_str = str(e)
                logger.warning(f"{row_label}: Browser extraction failed — {err_str}")
                error_count += 1
                if any(w in err_str.lower() for w in ("login", "auth", "cookie", "session", "unauthorized", "timeout", "sign in")):
                    pipeline_status = "FAILED_PORTAL_LOGIN"
                    failure_stage = "Stage 1a (Portal Login & Authentication)"
                    failure_diagnostics = f"Portal authentication/login failed or session expired: {err_str}"
                else:
                    pipeline_status = "FAILED_HYPERLINK_DOWNLOAD"
                    failure_stage = "Stage 1a (Hyperlink Attachment Download)"
                    failure_diagnostics = f"Hyperlink attachment extraction error: {err_str}"

        # ── 4b. Media normalisation & picture extraction ───────────────
        image_paths = []
        if os.path.isdir(ftir_folder):
            image_paths = get_all_images_for_ftir(ftir_folder)
            logger.debug(f"{row_label}: Normalised {len(image_paths)} image(s)")
            if len(image_paths) == 0 and len(os.listdir(ftir_folder)) > 0:
                logger.warning(f"{row_label}: Files found in attachment folder, but zero valid pictures could be extracted!")
                error_count += 1
                pipeline_status = "FAILED_PICTURE_EXTRACTION"
                failure_stage = "Stage 1b (Media Normalization & Picture Extraction)"
                failure_diagnostics = "Attachment files present, but picture extraction failed (corrupted video/PDF/image format or unsupported codec)."
        else:
            logger.debug(f"{row_label}: No attachment folder found at '{ftir_folder}'")
            if pipeline_status == "SUCCESS":
                if not ftir_url or not ftir_url.startswith("http"):
                    warning_count += 1
                    pipeline_status = "WARNING_MISSING_HYPERLINK"
                    failure_stage = "Stage 1a (Excel Hyperlink Check)"
                    failure_diagnostics = "No valid portal hyperlink present in Excel row and no cached pictures found. Evaluated via tabular metadata alone."
                else:
                    warning_count += 1
                    pipeline_status = "WARNING_NO_CACHED_ATTACHMENTS"
                    failure_stage = "Stage 1a / 1b (Offline Attachment Lookups)"
                    failure_diagnostics = "No cached attachments folder found in ftir_records. Evaluated via tabular metadata alone."

        # ── Learning Layer: Check for human-reported corrections ────────
        # If this FTIR was previously reported as misclassified, use the
        # human-corrected label directly and skip model inference.
        # Also checks for SIMILAR FTIRs (same keywords + mileage range).
        try:
            from src.corrections import get_override_for_ftir
            override = get_override_for_ftir(ftir_no)
        except Exception:
            override = None

        if override:
            ctype = override["correction_type"]
            clabel = override["correct_label"]
            creason = override.get("user_reason", "")
            match_type = override.get("match_type", "exact")
            if match_type == "similar":
                sim_score = override.get("similarity_score", 0)
                matched = override.get("matched_ftir", "?")
                logger.info(f"{row_label}: → SIMILAR LEARNING OVERRIDE applied "
                            f"({ctype} → {clabel}, {sim_score:.0%} similar to {matched})")
                reason_prefix = (f"[LEARNING: SIMILAR to {matched} ({sim_score:.0%} match)] "
                                 f"Corrected by operator: {creason}")
            else:
                logger.info(f"{row_label}: → EXACT LEARNING OVERRIDE applied ({ctype} → {clabel})")
                reason_prefix = f"[LEARNING OVERRIDE] Previously corrected by operator: {creason}"

            if ctype in ("model1_wrongly_rust", "model1_wrongly_nonrust"):
                # Model 1 was wrong: use corrected rust/non_rust label
                if clabel == "non_rust":
                    non_rust_count += 1
                    result_cols["Defect_Type"].append("non_rust")
                    result_cols["SBPR_Number"].append("N/A")
                else:
                    rust_count += 1
                    result_cols["Defect_Type"].append("rust")
                    result_cols["SBPR_Number"].append("pending_sbpr")
                result_cols["Reason"].append(reason_prefix)
            else:
                # Model 2 was wrong: rust was correct, but SBPR was wrong
                rust_count += 1
                result_cols["Defect_Type"].append("rust")
                result_cols["SBPR_Number"].append(clabel)
                result_cols["Reason"].append(reason_prefix)

            result_cols["Metadata_Confidence"].append(1.0)
            result_cols["Image_Confidence"].append(1.0)
            result_cols["Flag_For_Review"].append(False)
            result_cols["Pipeline_Status"].append("LEARNING_OVERRIDE")
            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(
                f"Human correction applied. Original error type: {ctype}"
            )
            if progress_callback:
                progress_callback(idx + 1, total_rows, row_label, {
                    "defect_type": result_cols["Defect_Type"][-1],
                    "sbpr": result_cols["SBPR_Number"][-1],
                    "rust_count": rust_count, "non_rust_count": non_rust_count,
                    "review_count": review_count, "reason": result_cols["Reason"][-1],
                    "error_count": error_count, "warning_count": warning_count,
                    "pipeline_status": "LEARNING_OVERRIDE", "failure_stage": "None",
                    "failure_diagnostics": "", "ftir_no": ftir_no
                })
            continue

        # ── 4c. Late Learning Layer: Visually Similar Override ─────────
        # Check if we have seen these images before and they were manually corrected
        if not override:
            try:
                from src.corrections import find_similar_override
                row_meta = {}
                for col in ["Subject (English)", "Customer Complaint", "Mileage - Using Time", "Subject"]:
                    if col in row.index:
                        row_meta[col] = str(row[col])
                override = find_similar_override(row_meta, image_paths=image_paths, text_threshold=0.6, image_threshold=0.90)
            except Exception as e:
                logger.warning(f"Failed during similar override check: {e}")
                override = None
        
        if override:
            ctype = override.get("correction_type", "")
            clabel = override.get("correct_label", "")
            creason = override.get("user_reason", "")
            logger.info(f"{row_label}: → VISUAL/SIMILAR LEARNING OVERRIDE applied ({ctype} → {clabel})")
            reason_prefix = f"[LEARNING OVERRIDE] Visually similar to corrected past record: {creason}"

            if ctype in ("model1_wrongly_rust", "model1_wrongly_nonrust"):
                if clabel == "non_rust":
                    non_rust_count += 1
                    result_cols["Defect_Type"].append("non_rust")
                    result_cols["SBPR_Number"].append("N/A")
                else:
                    rust_count += 1
                    result_cols["Defect_Type"].append("rust")
                    result_cols["SBPR_Number"].append("pending_sbpr")
            else:
                rust_count += 1
                result_cols["Defect_Type"].append("rust")
                result_cols["SBPR_Number"].append(clabel)

            result_cols["Reason"].append(reason_prefix)
            result_cols["Metadata_Confidence"].append(1.0)
            result_cols["Image_Confidence"].append(1.0)
            result_cols["Flag_For_Review"].append(False)
            result_cols["Pipeline_Status"].append("LEARNING_OVERRIDE")
            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(f"Human correction applied via similarity. Error type: {ctype}")
            
            if progress_callback:
                progress_callback(idx + 1, total_rows, row_label, {
                    "defect_type": result_cols["Defect_Type"][-1], "sbpr": result_cols["SBPR_Number"][-1],
                    "rust_count": rust_count, "non_rust_count": non_rust_count, "review_count": review_count, 
                    "reason": result_cols["Reason"][-1], "error_count": error_count, "warning_count": warning_count,
                    "pipeline_status": "LEARNING_OVERRIDE", "failure_stage": "None", "failure_diagnostics": "", "ftir_no": ftir_no
                })
            continue

        # ── 4d. Stage 1: Rust classification ───────────────────────────
        if not image_paths:
            logger.info(f"{row_label}: No images available, proceeding with metadata-only classification")
            rust_label = "uncertain"
            rust_confidence = 0.0
        else:
            try:
                rust_result = predict_rust_for_ftir(image_paths, rust_model, threshold=0.85)
                rust_label = rust_result["label"]
                rust_confidence = rust_result["confidence"]
                logger.debug(f"{row_label}: Rust verdict = {rust_label} ({rust_confidence:.1%})")
            except Exception as e:
                logger.error(f"{row_label}: Stage 1 rust classification error: {e}")
                error_count += 1
                pipeline_status = "FAILED_RUST_INFERENCE"
                failure_stage = "Stage 1c (Rust vs Non-Rust Vision Model)"
                failure_diagnostics = f"Stage 1 computer vision model inference threw error: {e}"
                rust_label = "uncertain"
                rust_confidence = 0.0

        if rust_label == "non_rust":
            # Not a corrosion defect — skip SBPR classification entirely
            non_rust_count += 1
            result_cols["Defect_Type"].append("non_rust")
            result_cols["SBPR_Number"].append("N/A")
            result_cols["Reason"].append(
                f"Stage 1 classified as non-rust (confidence {rust_confidence:.0%}). "
                f"SBPR classification skipped."
            )
            result_cols["Metadata_Confidence"].append(0.0)
            result_cols["Image_Confidence"].append(0.0)
            result_cols["Flag_For_Review"].append(False)
            result_cols["Pipeline_Status"].append(pipeline_status)
            result_cols["Failure_Stage"].append(failure_stage)
            result_cols["Failure_Diagnostics"].append(failure_diagnostics)
            logger.info(f"{row_label}: → NON-RUST ({rust_confidence:.0%}) — skipping SBPR")
            if progress_callback:
                progress_callback(idx + 1, total_rows, row_label, {
                    "defect_type": "non_rust", "sbpr": "N/A",
                    "rust_count": rust_count, "non_rust_count": non_rust_count,
                    "review_count": review_count, "reason": result_cols["Reason"][-1],
                    "error_count": error_count, "warning_count": warning_count,
                    "pipeline_status": pipeline_status, "failure_stage": failure_stage,
                    "failure_diagnostics": failure_diagnostics, "ftir_no": ftir_no
                })
            continue

        # Record is rust (or uncertain about rust) — proceed to Stage 2
        rust_count += 1
        defect_type = "rust" if rust_label == "rust" else "uncertain_rust"

        # ── 4d. Stage 2 Primary: Metadata prediction ──────────────────
        try:
            single_row_df = pd.DataFrame([row])
            X_row, _, _ = build_features(single_row_df)
            X_aligned = X_row.reindex(columns=tree_feature_names, fill_value=0)

            metadata_result = predict_sbpr_metadata(
                feature_row=X_aligned,
                model=tree_bundle,
                feature_names=tree_feature_names,
            )
        except Exception as e:
            logger.warning(f"{row_label}: Metadata prediction failed — {e}")
            metadata_result = {
                "sbpr_no": "uncertain",
                "confidence": 0.0,
                "reason": f"Metadata prediction error: {e}",
            }
            if pipeline_status == "SUCCESS":
                error_count += 1
                pipeline_status = "FAILED_METADATA_INFERENCE"
                failure_stage = "Stage 2 Primary (Tabular Metadata Tree)"
                failure_diagnostics = f"Metadata feature engineering or tree evaluation failed: {e}"

        # ── 4e. Stage 2 Secondary: Image prediction ───────────────────
        if image_paths:
            try:
                image_result = predict_sbpr_image_for_ftir(
                    image_paths, sbpr_img_model, threshold=0.6,
                )
            except Exception as e:
                logger.warning(f"{row_label}: Image prediction failed — {e}")
                image_result = {"sbpr_no": "uncertain", "confidence": 0.0}
                if pipeline_status == "SUCCESS":
                    error_count += 1
                    pipeline_status = "FAILED_IMAGE_INFERENCE"
                    failure_stage = "Stage 2 Secondary (3-Class Vision CNN)"
                    failure_diagnostics = f"Stage 2 CNN inference failed: {e}"
        else:
            image_result = {"sbpr_no": "uncertain", "confidence": 0.0}

        # ── 4f. Stage 2 Fusion ─────────────────────────────────────────
        fused = fuse_sbpr_predictions(metadata_result, image_result)

        if fused["flag_for_review"]:
            review_count += 1

        result_cols["Defect_Type"].append(defect_type)
        result_cols["SBPR_Number"].append(fused["final_sbpr"])
        result_cols["Reason"].append(fused["reason"])
        result_cols["Metadata_Confidence"].append(fused["metadata_confidence"])
        result_cols["Image_Confidence"].append(fused["image_confidence"])
        result_cols["Flag_For_Review"].append(fused["flag_for_review"])
        result_cols["Pipeline_Status"].append(pipeline_status)
        result_cols["Failure_Stage"].append(failure_stage)
        result_cols["Failure_Diagnostics"].append(failure_diagnostics)

        flag_str = " ⚠ REVIEW" if fused["flag_for_review"] else ""
        err_str_log = f" [! {pipeline_status}]" if pipeline_status != "SUCCESS" else ""
        logger.info(
            f"{row_label}: → {defect_type.upper()} | "
            f"SBPR: {fused['final_sbpr']} | "
            f"Meta: {fused['metadata_confidence']:.0%} | "
            f"Img: {fused['image_confidence']:.0%}"
            f"{flag_str}{err_str_log}"
        )
        if progress_callback:
            progress_callback(idx + 1, total_rows, row_label, {
                "defect_type": defect_type, "sbpr": fused["final_sbpr"],
                "rust_count": rust_count, "non_rust_count": non_rust_count,
                "review_count": review_count, "reason": fused["reason"],
                "flag_for_review": fused["flag_for_review"],
                "error_count": error_count, "warning_count": warning_count,
                "pipeline_status": pipeline_status, "failure_stage": failure_stage,
                "failure_diagnostics": failure_diagnostics, "ftir_no": ftir_no,
                "extraction_source": extraction_source
            })

    # ── 5. Append result columns to DataFrame ──────────────────────────
    for col_name, values in result_cols.items():
        df[col_name] = values

    # ── 6. Write augmented output Excel ────────────────────────────────
    logger.info("-" * 70)
    logger.info(f"Writing augmented results to '{output_path}'...")
    write_output_sheet(df, output_path)
    logger.info(f"  ✓ Output Excel saved successfully")

    # ── 7. Summary statistics ──────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Pipeline Summary")
    logger.info("=" * 70)
    logger.info(f"  Total FTIR Records   : {total_rows}")
    logger.info(f"  Rust (→ Stage 2)     : {rust_count}")
    logger.info(f"  Non-Rust (skipped)   : {non_rust_count}")
    logger.info(f"  Flagged for Review   : {review_count}")
    logger.info(f"  Errors / Faults      : {error_count}")
    logger.info(f"  Missing Imgs/Warnings: {warning_count}")
    logger.info(f"  Elapsed Time         : {elapsed:.1f}s")
    logger.info(f"  Output File          : {output_path}")
    logger.info(f"  Pipeline Log         : {log_path}")
    logger.info("=" * 70)

    # Clean up browser
    if browser_driver is not None:
        try:
            browser_driver.quit()
        except Exception:
            pass

    if progress_callback:
        progress_callback(total_rows, total_rows, f"Completed in {elapsed:.1f}s — Saved to {output_path}", {
            "completed": True,
            "elapsed": elapsed,
            "total": total_rows,
            "rust": rust_count,
            "non_rust": non_rust_count,
            "review": review_count,
            "errors": error_count,
            "warnings": warning_count,
            "output_path": output_path
        })

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FTIR → SBPR Classification Pipeline: End-to-End Orchestrator"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the source FTIR Excel workbook",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT_PATH,
        help=f"Destination path for augmented results Excel (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH,
        help=f"Path for the running pipeline log file (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--rust-weights", default=DEFAULT_RUST_WEIGHTS,
        help=f"Path to Stage 1 rust classifier weights (default: {DEFAULT_RUST_WEIGHTS})",
    )
    parser.add_argument(
        "--tree-bundle", default=DEFAULT_SBPR_TREE_BUNDLE,
        help=f"Path to Stage 2 metadata tree bundle (default: {DEFAULT_SBPR_TREE_BUNDLE})",
    )
    parser.add_argument(
        "--img-weights", default=DEFAULT_SBPR_IMG_WEIGHTS,
        help=f"Path to Stage 2 image CNN weights (default: {DEFAULT_SBPR_IMG_WEIGHTS})",
    )
    parser.add_argument(
        "--skip-browser", action="store_true",
        help="Skip browser extraction; use only pre-downloaded attachments",
    )
    parser.add_argument(
        "--profile-dir", default=None,
        help="Chrome/Edge persistent user-data-dir for browser automation",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        rust_weights=args.rust_weights,
        sbpr_tree_bundle=args.tree_bundle,
        sbpr_img_weights=args.img_weights,
        skip_browser=args.skip_browser,
        profile_dir=args.profile_dir,
    )
