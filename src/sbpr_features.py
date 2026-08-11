"""
SBPR Tabular Feature Engineering Module (src.sbpr_features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transforms raw FTIR spreadsheet tabular metadata (mileage intervals and textual
description fields) into a clean, numerical feature matrix ready for
scikit-learn Stage 2 Primary Decision Tree / Random Forest models.

Note on Vehicle Models: 'Masked Model' and generic 'Model' columns in source Excel
sheets are uninformative/obfuscated placeholder codes and do not represent actual
vehicle models. They are explicitly excluded from feature extraction.

Key Transformations:
1. Mileage: Continuous numerical values parsed from text formatted strings
   (e.g., "15,250 [km]") and quantized into logical service intervals
   (0-10k, 10k-30k, 30k-60k, 60k+, and unknown), then one-hot encoded.
2. Subject Keywords: Binary indicators matching diagnostic domain text vocabulary
   loaded directly from offline configuration (`config/sbpr_keywords.yaml`).
"""

import os
import sys
import re
import logging

import pandas as pd
import yaml

from .utils import get_bundle_path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(get_bundle_path("config"), "config", "sbpr_keywords.yaml")
MILEAGE_BUCKETS = ["0-10k", "10k-30k", "30k-60k", "60k+", "unknown"]


def parse_mileage_km(val: any) -> float:
    """
    Extract continuous float mileage value from formatted string cells
    (e.g., "15,250 [km]", "13 [km]", "27,493 km" -> 15250.0).
    """
    if pd.isna(val):
        return float("nan")
    val_str = str(val).strip().replace(",", "")
    # Find first integer or decimal number sequence
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return float("nan")
    return float("nan")


def bucket_mileage(km: float) -> str:
    """Quantize raw kilometer readings into automotive warranty/service interval ranges."""
    if km != km:  # fast NaN check
        return "unknown"
    if km < 10000:
        return "0-10k"
    elif km < 30000:
        return "10k-30k"
    elif km < 60000:
        return "30k-60k"
    else:
        return "60k+"


def load_keywords_from_config(yaml_path: str = DEFAULT_CONFIG_PATH) -> list[str]:
    """
    Load distinct keyword vocabulary from YAML configuration file.
    Supports both flat YAML lists or hierarchical dicts keyed by SBPR defect class.
    """
    if not os.path.exists(yaml_path):
        logger.warning(f"Keyword configuration file not found at '{yaml_path}'. Returning defaults.")
        return ["rust", "corrosion", "seat", "door", "paint", "hard water", "bird"]

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        keywords = set()
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict) and "keywords" in val and isinstance(val["keywords"], list):
                    keywords.update([k.lower().strip() for k in val["keywords"] if isinstance(k, str)])
                elif isinstance(val, list):
                    keywords.update([k.lower().strip() for k in val if isinstance(k, str)])
        elif isinstance(data, list):
            keywords.update([k.lower().strip() for k in data if isinstance(k, str)])
            
        return sorted(list(keywords))
    except Exception as e:
        logger.error(f"Failed parsing keyword config '{yaml_path}': {e}")
        return []


def _find_column_by_keyword(columns: list[str], target_substrings: list[str]) -> str | None:
    """Helper to flexibly identify relevant column names in heterogeneous spreadsheets."""
    for col in columns:
        col_lower = str(col).lower()
        if any(sub in col_lower for sub in target_substrings):
            return col
    return None


def build_features(
    df: pd.DataFrame, 
    keywords_yaml_path: str = DEFAULT_CONFIG_PATH,
    target_col: str | None = None
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """
    Transform raw FTIR metadata DataFrame into numerical Scikit-Learn feature matrix X,
    label Series y, and readable feature column names list.

    Parameters
    ----------
    df : pd.DataFrame
        Input table imported from FTIR Excel spreadsheets via `excel_io.read_ftir_sheet()`.
    keywords_yaml_path : str
        Path to YAML vocabulary file containing defect symptoms.
    target_col : str, optional
        Column name containing historical SBPR ground truth labels (default auto-searches
        for 'sbpr_no', 'sbpr', 'defect_type', or 'label').

    Returns
    -------
    X : pd.DataFrame
        Clean numerical feature matrix consisting of 0/1 one-hot variables.
    y : pd.Series or None
        Target label vector if target column exists; otherwise None.
    feature_names : list of str
        Ordered list of column titles in X, essential for interpreting decision tree branches.
    """
    features_df = pd.DataFrame(index=df.index)

    # ── 1. Vehicle Model Exclusion Notice ──────────────────────────────
    # 'Masked Model' and generic 'Model' columns in incoming Excel spreadsheets
    # are obfuscated or placeholder codes that do not represent actual vehicle models.
    # They are explicitly EXCLUDED from tabular feature engineering to avoid learning
    # erroneous or spurious rules in the Decision Tree / Random Forest models.
    logger.debug("Skipping 'Masked Model' / 'Model' columns (uninformative placeholder feature).")

    # ── 2. Mileage Interval Bucketing & Encoding ───────────────────────
    mileage_col = _find_column_by_keyword(df.columns, ["mileage", "odometer", "using time"])
    if mileage_col:
        parsed_km = df[mileage_col].apply(parse_mileage_km)
        buckets = parsed_km.apply(bucket_mileage)
        
        # Explicitly enforce all 5 bucket intervals exist even if unrepresented in current sample
        mileage_dummies = pd.get_dummies(buckets, prefix="mileage", dtype=int)
        for expected_b in MILEAGE_BUCKETS:
            col_name = f"mileage_{expected_b}"
            if col_name not in mileage_dummies.columns:
                mileage_dummies[col_name] = 0
        
        # Reorder to standard schema order
        ordered_mileage_cols = [f"mileage_{b}" for b in MILEAGE_BUCKETS]
        features_df = pd.concat([features_df, mileage_dummies[ordered_mileage_cols]], axis=1)
    else:
        logger.info("No Mileage column detected in input DataFrame.")

    # ── 3. Subject & Complaint Keyword Features ────────────────────────
    keywords = load_keywords_from_config(keywords_yaml_path)
    
    # Locate all relevant unstructured text column headers
    text_cols = [col for col in df.columns if any(w in str(col).lower() for w in ["subject", "complaint", "description", "reason", "detail"])]
    if not text_cols and "ftir_url" not in df.columns:
        # Fallback: scan all object/string type columns except known identifiers
        text_cols = [col for col in df.select_dtypes(include=["object"]).columns if not any(k in str(col).lower() for k in ["url", "link", "vin", "date"])]

    if text_cols:
        # Combine all relevant text fields into one lowercase searchable corpus per record
        unified_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        
        for kw in keywords:
            clean_col_name = f"contains_{kw.replace(' ', '_')}"
            # Word or partial token substring matching
            features_df[clean_col_name] = unified_text.str.contains(kw, regex=False).astype(int)
    else:
        logger.warning("No textual subject/complaint columns identified for keyword feature extraction.")

    # ── 4. Target Label Extraction (y) ─────────────────────────────────
    y = None
    if target_col is None:
        # Try finding standard historical label column
        label_col = _find_column_by_keyword(df.columns, ["sbpr_no", "sbpr", "label", "target_sbpr", "defect_type"])
    else:
        label_col = target_col if target_col in df.columns else None

    if label_col and label_col in df.columns:
        y = df[label_col].copy()
        logger.info(f"Target label column identified: '{label_col}' ({y.nunique()} distinct classes)")
    else:
        logger.debug("No ground truth label column found (inference mode or unlabelled test data).")

    # Final cleanup & validation
    features_df.fillna(0, inplace=True)
    feature_names = list(features_df.columns)

    return features_df, y, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.excel_io import read_ftir_sheet

    print("=" * 65)
    print("  Stage 2 Primary: SBPR Feature Engineering Verification")
    print("=" * 65)

    if len(sys.argv) < 2:
        # Auto-test: load all three sample historical spreadsheets from data/excel/ and merge
        excel_dir = os.path.join("data", "excel")
        if os.path.isdir(excel_dir):
            files = sorted([f for f in os.listdir(excel_dir) if f.endswith(".xlsx") and not f.startswith(".")])
            print(f"No custom sheet passed via sys.argv. Auto-aggregating {len(files)} Historical Excel Workbooks:")
            
            dfs = []
            for fname in files:
                fpath = os.path.join(excel_dir, fname)
                try:
                    df_sub = read_ftir_sheet(fpath)
                    # Extract historical SBPR class code directly from filename for training demonstration!
                    sbpr_code = "Unknown_SBPR"
                    match = re.search(r"(SBIN\d+[A-Z]\d+)", fname)
                    if match:
                        sbpr_code = match.group(1)
                    df_sub["sbpr_no"] = sbpr_code
                    dfs.append(df_sub)
                    print(f"  -> Loaded '{fname}' with ground truth label [{sbpr_code}] ({len(df_sub)} rows)")
                except Exception as e:
                    print(f"  -> Skipping '{fname}' due to error: {e}")
            
            df_test = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            print(f"\nUnified Input Table Shape: {df_test.shape}")
        else:
            print("Usage: python -m src.sbpr_features <path_to_ftir.xlsx>")
            sys.exit(1)
    else:
        sheet_path = sys.argv[1]
        print(f"Loading custom sheet: {sheet_path}")
        df_test = read_ftir_sheet(sheet_path)

    print("-" * 65)
    print("Building numerical feature matrix X from tabular metadata...")
    X, y, fname_list = build_features(df_test)

    print(f"\n=== FEATURE ENGINEERING RESULTS ===")
    print(f"  Feature Matrix X Shape : {X.shape}")
    print(f"  Total Feature Columns  : {len(fname_list)}")
    print(f"  Target Vector y Shape  : {y.shape if y is not None else 'None'}\n")

    if y is not None:
        print("--- Ground Truth Target Class Distribution (y) ---")
        for cls_name, cnt in y.value_counts().items():
            print(f"  -> {cls_name.ljust(20)} : {cnt} rows ({cnt/len(y):.1%})")
        print()

    print("--- Sample Feature Columns Generated ---")
    model_features = [c for c in fname_list if c.startswith("model_")]
    mileage_features = [c for c in fname_list if c.startswith("mileage_")]
    keyword_features = [c for c in fname_list if c.startswith("contains_")]

    print(f"  * Model One-Hot Columns ({len(model_features)} total) : {model_features[:6]} ...")
    print(f"  * Mileage Bucket Columns ({len(mileage_features)} total): {mileage_features}")
    print(f"  * Keyword Boolean Columns ({len(keyword_features)} total): {keyword_features[:8]} ...\n")

    print("--- First 5 Rows of Generated Feature Matrix X (Subset) ---")
    # Show a readable preview of key feature column types
    preview_cols = (model_features[:2] + mileage_features[:3] + keyword_features[:4])[:10]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 150)
    print(X[preview_cols].head(5).to_string())
    print("=" * 65)
