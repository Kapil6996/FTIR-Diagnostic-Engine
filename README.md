# FTIR → SBPR Automated Defect Diagnosis & Fusion Engine

An offline, on-device Python pipeline with a full **Desktop GUI** for automated defect triage and SBPR classification of Automotive Field Technical Investigation Reports (FTIRs). All processing runs locally without any external API calls, ensuring high privacy, reliability, and portability across Mac (Apple Silicon MPS backend) and Windows office workstations (CPU fallback).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Desktop GUI
python app.py
```

### SIFT Maruti Portal (Online Mode Only)

If using **Online mode** to automatically download FTIR attachment photos from Excel hyperlinks, you must log into the **SIFT Maruti platform once** in your default browser before running the app. After that first login, the tool reuses your saved session — no repeated logins needed.

If using **Offline / Cache-Only mode** (the default), no internet or login is required. The pipeline analyzes Excel tabular features and local photos already saved in `ftir_records/`.

---

## Features

| Feature | Description |
| :--- | :--- |
| **Desktop GUI** | Professional Tkinter/ttk desktop window with high-contrast theme |
| **Excel Input** | Reads FTIR spreadsheets with auto-header detection |
| **Stage 1: Rust Filter** | Binary CNN (Rust vs Non-Rust) to gate corrosion defects |
| **Stage 2: SBPR Fusion** | Dual-model decision fusion (metadata tree + image CNN) |
| **KPI Dashboard** | Live counters for total rows, rust, non-rust, manual review, faults |
| **Failure Tracker** | Stage-level error diagnostics (portal, hyperlink, extraction, inference) |
| **FTIR Photo Browser** | Post-pipeline viewer to browse all FTIRs and their extracted photos |
| **Manual Review** | Classify flagged uncertain FTIRs with photo gallery and SBPR dropdown |
| **One-Click Report** | Open generated Excel output directly from the app |
| **Standalone Executable** | Package into a double-click app via `python build_executable.py` |

---

## Folder Layout

```text
ftir_sbpr_tool/
├── app.py                   # Desktop GUI entry point
├── build_executable.py      # PyInstaller standalone packaging utility
├── requirements.txt         # Offline package requirements
├── config/
│   └── sbpr_keywords.yaml   # SBPR subject keyword vocabulary & feature hints
├── data/                    # Labeled image & tabular datasets for training
├── ftir_records/            # Downloaded/cached attachment files per FTIR record
├── models/                  # Saved model checkpoints (.pth & .joblib)
├── outputs/                 # Generated Excel reports
└── src/                     # Core pipeline modules
    ├── __init__.py
    ├── excel_io.py          # Spreadsheet input loading and output formatting
    ├── browser_extract.py   # Selenium web extraction (one-time SIFT login)
    ├── media_normalize.py   # PDF/video/image normalization
    ├── rust_model.py        # Stage 1: Binary vision AI (Rust vs Non-Rust)
    ├── sbpr_features.py     # Stage 2: Tabular & NLP text feature engineering
    ├── sbpr_tree.py         # Stage 2: Decision tree / random forest model
    ├── sbpr_image_model.py  # Stage 2: 3-class vision CNN
    ├── fusion.py            # Stage 2: Decision fusion & review flagging
    ├── pipeline.py          # Master pipeline orchestrator
    └── review_viewer.py     # Post-pipeline FTIR browser & manual review windows
```

---

## Pipeline Stages

1. **Excel Ingestion** (`src.excel_io`) — Reads FTIR worksheet with record IDs, vehicle metadata, defect keywords, and hyperlinks.

2. **Browser Extraction** (`src.browser_extract`) — Uses Selenium with persistent session cookies from SIFT Maruti. User logs in once; all subsequent FTIR hyperlink downloads are automatic.

3. **Media Normalization** (`src.media_normalize`) — Converts PDFs, videos, and images into normalized RGB pictures for model evaluation.

4. **Stage 1: Rust vs Non-Rust** (`src.rust_model`) — Binary CNN filter. Non-rust FTIRs are logged and skip Stage 2.

5. **Stage 2: SBPR Decision Fusion** (`src.fusion`) — Dual-model verification:
   - **Metadata Tree** (`src.sbpr_features` + `src.sbpr_tree`) — Evaluates mileage, subject keywords, and complaint text.
   - **Image CNN** (`src.sbpr_image_model`) — 3-class vision classifier on extracted photos.
   - **Fusion** — If both models agree → confirmed. If they disagree → flagged for manual review.

6. **Output Generation** — Augmented Excel with: `Defect_Type`, `SBPR_Number`, `Reason`, `Confidence`, `Flag_For_Review`, `Pipeline_Status`, `Failure_Stage`, `Failure_Diagnostics`.

---

## Standalone Deployment

Package the entire tool into a **double-click executable** for teammates who don't have Python installed:

```bash
pip install pyinstaller
python build_executable.py
```

The generated `dist/Maruti_FTIR_Diagnostic_Engine/` folder contains everything needed. Copy it to a USB stick or share via network — recipients just double-click the executable.

---

## Offline & Cross-Platform

All PyTorch models use automatic hardware detection:
- **Apple Silicon Mac** → MPS GPU acceleration
- **NVIDIA GPU** → CUDA acceleration
- **Windows/Linux CPU** → Standard CPU fallback

No internet APIs are called during inference. Everything runs 100% on-device.
