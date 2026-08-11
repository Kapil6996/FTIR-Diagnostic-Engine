# Maruti FTIR → SBPR Automated Defect Diagnosis & Fusion Engine

An offline, on-device Python AI pipeline with a full **Desktop GUI** for automated defect triage and SBPR classification of Automotive Field Technical Investigation Reports (FTIRs). All processing runs locally without any external API calls, ensuring high privacy, reliability, and portability across Mac (Apple Silicon MPS backend) and Windows office workstations (CPU fallback).

---

## 🚀 Quick Start

### For Developers (Running from Source)
```bash
# 1. Clone the repository (Requires Git LFS for models)
git clone https://github.com/Kapil6996/FTIR-Diagnostic-Engine.git
cd FTIR-Diagnostic-Engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Desktop GUI
python app.py
```

### For End-Users (Running the App)
Navigate to the `dist/` directory and simply double-click the **`Maruti_FTIR_Diagnostic_Engine.app`** (or `.exe` on Windows). No terminal or Python installation is required!

---

## ✨ Features & Architecture

| Feature | Description |
| :--- | :--- |
| **Dynamic Scrollable GUI** | Professional Tkinter desktop window that adapts to any screen size. |
| **Excel Input** | Reads FTIR spreadsheets with auto-header detection. |
| **Stage 1: Rust Filter** | Binary CNN (Rust vs Non-Rust) to gate corrosion defects. |
| **Stage 2: SBPR Fusion** | Dual-model decision fusion (metadata tree + image CNN). |
| **KPI Dashboard** | Live counters for total rows, rust, non-rust, manual review, faults. |
| **Robust Extraction** | Automatically logs into SIFT Maruti to extract photos via URLs. Features a **Search Fallback Engine** if URLs are broken. |
| **Multi-Media Support** | Extracts frames and images from PDFs, Videos, and handles multiple photos per FTIR. |
| **Active Learning Loop** | Operators can report wrong classifications. Corrections are saved to `corrections_db.json` to automatically override future pipeline runs and provide a dataset for retraining. |

---

## 📂 Folder Layout

```text
ftir_sbpr_tool/
├── app.py                   # Desktop GUI entry point
├── build_executable.py      # PyInstaller standalone packaging utility
├── requirements.txt         # Offline package requirements
├── config/
│   └── sbpr_keywords.yaml   # SBPR subject keyword vocabulary & feature hints
├── data/                    # Labeled image & tabular datasets for training (Ignored in Git)
├── ftir_records/            # Downloaded/cached attachment files per FTIR record (Ignored in Git)
├── models/                  # Saved model checkpoints (.pth & .joblib) [Tracked via Git LFS]
├── outputs/                 # Generated Excel reports
└── src/                     # Core pipeline modules
    ├── __init__.py
    ├── excel_io.py          # Spreadsheet input loading and output formatting
    ├── browser_extract.py   # Selenium web extraction (URL direct + Fallback Search)
    ├── media_normalize.py   # PDF/video/image normalization & multi-image extraction
    ├── rust_model.py        # Stage 1: Binary vision AI (Rust vs Non-Rust)
    ├── sbpr_features.py     # Stage 2: Tabular & NLP text feature engineering
    ├── sbpr_tree.py         # Stage 2: Decision tree / random forest model
    ├── sbpr_image_model.py  # Stage 2: 3-class vision CNN
    ├── fusion.py            # Stage 2: Decision fusion & review flagging
    ├── pipeline.py          # Master pipeline orchestrator
    ├── corrections.py       # Active learning loop & JSON database manager
    └── review_viewer.py     # Post-pipeline FTIR browser & manual review windows
```

---

## ⚙️ Pipeline Stages

1. **Excel Ingestion** (`src.excel_io`) — Reads FTIR worksheet with record IDs, vehicle metadata, defect keywords, and hyperlinks.
2. **Browser Extraction** (`src.browser_extract`) — Uses Selenium with persistent session cookies from SIFT Maruti. Attempts direct URL download first. If it fails, falls back to searching the FTIR number in the portal text box.
3. **Media Normalization** (`src.media_normalize`) — Converts PDFs, videos, and multi-image batches into normalized RGB pictures for model evaluation.
4. **Stage 1: Rust vs Non-Rust** (`src.rust_model`) — Binary CNN filter. Non-rust FTIRs are logged and skip Stage 2.
5. **Stage 2: SBPR Decision Fusion** (`src.fusion`) — Dual-model verification:
   - **Metadata Tree** (`src.sbpr_tree`) — Evaluates mileage, subject keywords, and complaint text.
   - **Image CNN** (`src.sbpr_image_model`) — 3-class vision classifier on extracted photos.
   - **Fusion** — If models agree → confirmed. If they disagree → flagged for manual review.
6. **Output Generation** — Augmented Excel with: `Defect_Type`, `SBPR_Number`, `Reason`, `Confidence`, `Flag_For_Review`, and `Failure_Diagnostics`.

---

## 🧑‍💻 How to Upgrade & Build

If you are a colleague taking over the project, you can easily upgrade the models or tweak the logic:
1. Make your code changes in `src/` or drop new `.pth` weights into `models/`.
2. Open your terminal in the project directory and run:
   ```bash
   python build_executable.py
   ```
3. The script will automatically package your new AI weights and Python scripts into a fresh `.app` (Mac) or `.exe` (Windows) in the `dist/` folder!

### Git LFS Note
Because the AI models are large, this repository uses **Git LFS (Large File Storage)**. 
- If you are pushing new models to GitHub, ensure you have Git LFS installed (`brew install git-lfs` on Mac) and run `git lfs install` before pushing.
- Standard source code files (`.py`) are pushed normally.
