"""
FTIR → SBPR Automated Diagnostic Engine — Desktop GUI Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A standalone, zero-dependency desktop graphical user interface built using
Python's built-in Tkinter/ttk toolkit. Designed specifically for cross-platform
offline engineering workstations (Mac Apple Silicon & Windows CPU machines).

Features:
    - Intuitive Excel sheet selection and configuration.
    - One-time SIFT Maruti portal login for authenticated FTIR hyperlink access.
    - Real-time progress bar and KPI diagnostic dashboard (Rust vs Non-Rust counters).
    - Live streaming execution console with color-coded warning/review indicators.
    - Comprehensive failure & fault diagnostic tracker with stage-level error reporting.
    - Post-pipeline FTIR photo browser for viewing all extracted attachment images.
    - Manual review window for human classification of flagged uncertain records.
    - One-click report launching in system Excel.

Usage:
    python app.py
"""

import os
import sys
import time
import queue
import logging
import threading
import subprocess

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import pandas as pd
import yaml

# Ensure project root is in path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline, DEFAULT_OUTPUT_PATH, DEFAULT_LOG_PATH
from src.review_viewer import FtirBrowserWindow, ManualReviewWindow

# ── Custom Queue Handler for Real-Time Logging to UI ─────────────────────────

class QueueLoggingHandler(logging.Handler):
    """Intercepts Python logging events and routes them to a thread-safe UI Queue."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.log_queue.put(("LOG", record.levelname, msg))
        except Exception:
            pass


# ── Main Application GUI ─────────────────────────────────────────────────────

class FtirSbprDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("FTIR → SBPR Automated Defect Diagnosis & Fusion Engine")
        self.geometry("980x740")
        self.minsize(780, 560)
        self.configure(bg="#F4F6F8")

        # UI Thread Synchronization Queue
        self.msg_queue = queue.Queue()
        self.pipeline_thread = None
        self.result_df = None       # Stores pipeline output DataFrame after completion
        self.last_output_path = None
        self.is_running = False

        # Apply modern Tk styling
        self._setup_styles()
        
        # ── Build scrollable master container ──────────────────────────────
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#EAEEF3")
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.main_container = ttk.Frame(self.canvas, style="TFrame")
        
        self.main_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        # Expand canvas window width on resize
        def _configure_canvas(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        self.canvas.bind("<Configure>", _configure_canvas)
        
        # Mousewheel support
        def _on_mousewheel(event):
            if sys.platform == 'darwin':
                self.canvas.yview_scroll(int(-1*(event.delta)), "units")
            else:
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")

        # Build interactive interface sections (now inside self.main_container)
        self._build_header()
        self._build_input_section()
        self._build_kpi_dashboard()
        self._build_action_deck()
        self._build_console()
        self._build_footer()

        # Connect log handler
        self._setup_log_interception()

        # Start reactive queue processing loop
        self.after(100, self._process_queue)

    def _setup_styles(self):
        """Configure clean, high-contrast executive theme and crystal-clear typography."""
        self.style = ttk.Style(self)
        # Using 'clam' forces consistent high-contrast rendering across macOS, Windows, and Linux
        self.style.theme_use("clam")

        self.configure(bg="#EAEEF3")
        self.style.configure("TFrame", background="#EAEEF3")
        self.style.configure("Header.TFrame", background="#0F172A")
        self.style.configure("Card.TFrame", background="#FFFFFF", relief="solid", borderwidth=1, bordercolor="#CBD5E1")
        self.style.configure("KPI.TFrame", background="#F8FAFC", relief="solid", borderwidth=1, bordercolor="#CBD5E1")
        
        # Typography hierarchy
        self.style.configure("Title.TLabel", background="#0F172A", foreground="#FFFFFF", font=("Segoe UI", 17, "bold"))
        self.style.configure("Subtitle.TLabel", background="#0F172A", foreground="#38BDF8", font=("Segoe UI", 11, "bold"))
        self.style.configure("Section.TLabel", font=("Segoe UI", 13, "bold"), foreground="#0F172A", background="#FFFFFF")
        self.style.configure("KPIValue.TLabel", font=("Segoe UI", 24, "bold"), foreground="#0F172A", background="#F8FAFC", anchor="center")
        self.style.configure("KPILabel.TLabel", font=("Segoe UI", 10, "bold"), foreground="#334155", background="#F8FAFC", anchor="center")
        
        self.style.configure("TLabel", background="#FFFFFF", foreground="#0F172A", font=("Segoe UI", 11))
        self.style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
        self.style.configure("Action.TButton", font=("Segoe UI", 11, "bold"), padding=8)

    def _build_header(self):
        """Top executive banner strip."""
        header = ttk.Frame(self.main_container, style="Header.TFrame", padding=(25, 18))
        header.pack(fill=tk.X, side=tk.TOP)
        title = ttk.Label(header, text="FTIR → SBPR Automated Defect Diagnosis & Fusion Engine", style="Title.TLabel")
        title.pack(anchor=tk.W)
        subtitle = ttk.Label(
            header,
            text="On-Device AI Pipeline | Stage 1 Corrosion Filter + Stage 2 Multimodality Decision Fusion",
            style="Subtitle.TLabel"
        )
        subtitle.pack(anchor=tk.W, pady=(5, 0))

    def _build_input_section(self):
        """Spacious, readable Excel spreadsheet parameter configuration and operation mode selection."""
        frame = ttk.Frame(self.main_container, style="Card.TFrame", padding=20)
        frame.pack(fill=tk.X, padx=15, pady=12)

        title = ttk.Label(frame, text="1. Configuration & Operation Mode", style="Section.TLabel")
        title.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))

        # 1. Input Excel Spreadsheet
        ttk.Label(frame, text="Input FTIR Excel Sheet:*", font=("Segoe UI", 11, "bold")).grid(row=1, column=0, sticky=tk.W, pady=6)
        self.var_input_path = tk.StringVar()
        entry_in = ttk.Entry(frame, textvariable=self.var_input_path, width=55, font=("Segoe UI", 10))
        entry_in.grid(row=1, column=1, sticky=tk.EW, padx=(12, 8), pady=6)
        ttk.Button(frame, text="📂 Browse Excel...", command=self._browse_input).grid(row=1, column=2, sticky=tk.EW, pady=6)

        # 2. Output Report Path
        ttk.Label(frame, text="Output Report Save Path:").grid(row=2, column=0, sticky=tk.W, pady=6)
        self.var_output_path = tk.StringVar(value=os.path.abspath(DEFAULT_OUTPUT_PATH))
        entry_out = ttk.Entry(frame, textvariable=self.var_output_path, width=55, font=("Segoe UI", 10))
        entry_out.grid(row=2, column=1, sticky=tk.EW, padx=(12, 8), pady=6)
        ttk.Button(frame, text="💾 Change Save As...", command=self._browse_output).grid(row=2, column=2, sticky=tk.EW, pady=6)

        # 3. SIFT Maruti Portal Notice
        lbl_sift_notice = ttk.Label(
            frame,
            text="📌 Important: Please ensure you have logged into the SIFT Maruti platform once in your default browser before running this tool. After that first login, FTIR links will open automatically without needing to log in again.",
            font=("Segoe UI", 10, "bold"),
            foreground="#15803D",
            wraplength=850
        )
        lbl_sift_notice.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(15, 4))

        self.var_test_mode = tk.BooleanVar(value=True) # Default to True right now for their testing
        chk_test = ttk.Checkbutton(frame, text="🧪 Test Mode (Process first row only for quick verification)", variable=self.var_test_mode)
        chk_test.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(5, 4))

        frame.columnconfigure(1, weight=1)

    def _build_kpi_dashboard(self):
        """Live diagnostic counter widgets with bold typography."""
        dash = ttk.Frame(self.main_container, padding=(15, 0))
        dash.pack(fill=tk.X, pady=6)

        # Create 5 equal high-contrast summary boxes
        self.kpi_total = self._create_kpi_box(dash, "TOTAL FTIR ROWS", "0", 0)
        self.kpi_rust = self._create_kpi_box(dash, "STAGE 2 RUST", "0", 1, color="#DC2626")
        self.kpi_nonrust = self._create_kpi_box(dash, "NON-RUST (SKIPPED)", "0", 2, color="#16A34A")
        self.kpi_review = self._create_kpi_box(dash, "MANUAL REVIEW", "0", 3, color="#D97706")
        self.kpi_faults = self._create_kpi_box(dash, "FAULTS / WARNINGS", "0", 4, color="#E11D48")

        for c in range(5):
            dash.columnconfigure(c, weight=1, uniform="kpi")

    def _create_kpi_box(self, parent, label_text, default_val, col, color="#0F172A"):
        box = ttk.Frame(parent, style="KPI.TFrame", padding=14)
        box.grid(row=0, column=col, sticky=tk.NSEW, padx=5)
        lbl_val = ttk.Label(box, text=default_val, style="KPIValue.TLabel", foreground=color)
        lbl_val.pack(expand=True)
        lbl_title = ttk.Label(box, text=label_text, style="KPILabel.TLabel")
        lbl_title.pack(expand=True, pady=(4, 0))
        return lbl_val

    def _build_action_deck(self):
        """Execution controls and status progression."""
        action_frame = ttk.Frame(self.main_container, padding=(15, 8))
        action_frame.pack(fill=tk.X, pady=6)

        # Buttons container
        btn_box = ttk.Frame(action_frame)
        btn_box.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        self.btn_run = ttk.Button(
            btn_box,
            text="▶ START DIAGNOSTIC PIPELINE",
            style="Action.TButton",
            command=self._start_pipeline
        )
        self.btn_run.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_stop = ttk.Button(
            btn_box,
            text="■ Cancel Operation",
            command=self._cancel_pipeline,
            state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=6)

        self.btn_open = ttk.Button(
            btn_box,
            text="📊 Open Output Report in Excel",
            command=self._open_output_report,
            state=tk.DISABLED
        )
        self.btn_open.pack(side=tk.RIGHT)

        # ── Post-Pipeline Viewer Buttons (second row) ────────────────────
        btn_box2 = ttk.Frame(action_frame)
        btn_box2.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        self.btn_browse_ftirs = ttk.Button(
            btn_box2,
            text="📋 Browse All FTIRs & Photos",
            command=self._open_ftir_browser,
            state=tk.DISABLED
        )
        self.btn_browse_ftirs.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_manual_review = ttk.Button(
            btn_box2,
            text="⚠️ Manual Review — Classify Flagged FTIRs",
            command=self._open_manual_review,
            state=tk.DISABLED
        )
        self.btn_manual_review.pack(side=tk.LEFT, padx=6)

        # Progress bar & Status Label
        self.lbl_status = ttk.Label(action_frame, text="Status: Ready — Awaiting spreadsheet selection", font=("Segoe UI", 11, "bold", "italic"), foreground="#0F172A", background="#EAEEF3")
        self.lbl_status.pack(anchor=tk.W, pady=(0, 5))

        self.progress = ttk.Progressbar(action_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X)

    def _build_console(self):
        """Dual-tab execution monitoring display: real-time streaming text logs & structured stage failure table."""
        console_box = ttk.Frame(self.main_container, style="Card.TFrame", padding=(18, 10, 18, 18))
        console_box.pack(fill=tk.BOTH, expand=True, padx=15, pady=(8, 18))

        header_frame = ttk.Frame(console_box, style="Card.TFrame")
        header_frame.pack(fill=tk.X, pady=(5, 8))
        ttk.Label(header_frame, text="2. Live Execution Monitor & Diagnostic Stage Failure Tracker", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(header_frame, text="🧹 Clear Console", command=self._clear_console).pack(side=tk.RIGHT)

        # Tabbed interface for Logs vs Failures
        self.notebook = ttk.Notebook(console_box)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ── Tab 1: Live Stream Log ──────────────────────────────────────────
        tab_log = ttk.Frame(self.notebook)
        self.notebook.add(tab_log, text=" 🖥️ Live Reasoning Stream & Logs ")

        self.txt_console = scrolledtext.ScrolledText(
            tab_log,
            wrap=tk.WORD,
            font=("Consolas", 10),
            background="#0F172A",
            foreground="#F8FAFC",
            relief="flat",
            height=12,
            insertbackground="#FFFFFF"
        )
        self.txt_console.pack(fill=tk.BOTH, expand=True)

        # Configure vibrant high-contrast color tags
        self.txt_console.tag_configure("INFO", foreground="#CBD5E1")
        self.txt_console.tag_configure("WARNING", foreground="#FACC15")
        self.txt_console.tag_configure("ERROR", foreground="#F87171", font=("Consolas", 10, "bold"))
        self.txt_console.tag_configure("SUCCESS", foreground="#4ADE80", font=("Consolas", 10, "bold"))
        self.txt_console.tag_configure("REVIEW", foreground="#FB923C", font=("Consolas", 10, "bold"))
        self.txt_console.tag_configure("TITLE", foreground="#38BDF8", font=("Consolas", 11, "bold"))

        self._log_to_console("System initialized and ready. Please select an input FTIR Excel sheet above.\n", "TITLE")

        # ── Tab 2: Stage Failure & Warning Tracker Table ───────────────────
        self.fault_count = 0
        tab_faults = ttk.Frame(self.notebook)
        self.notebook.add(tab_faults, text=" ⚠️ Diagnostic Stage Failures & Faults (0) ")

        ttk.Label(
            tab_faults,
            text="Audit records encountering attachment extraction, portal login, hyperlink, or inference stage issues:",
            font=("Segoe UI", 10, "italic"),
            foreground="#334155"
        ).pack(anchor=tk.W, padx=6, pady=(8, 6))

        tree_frame = ttk.Frame(tab_faults)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=2)

        cols = ("ftir", "status", "stage", "diag")
        self.tree_faults = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        self.tree_faults.heading("ftir", text="FTIR Number")
        self.tree_faults.heading("status", text="Fault Status / Code")
        self.tree_faults.heading("stage", text="Failed Stage")
        self.tree_faults.heading("diag", text="Root Cause & Diagnostics")

        self.tree_faults.column("ftir", width=120, minwidth=90)
        self.tree_faults.column("status", width=170, minwidth=140)
        self.tree_faults.column("stage", width=220, minwidth=180)
        self.tree_faults.column("diag", width=360, minwidth=250)

        sb_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree_faults.yview)
        self.tree_faults.configure(yscrollcommand=sb_y.set)
        self.tree_faults.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Selected diagnosis display box at bottom
        self.var_diag_detail = tk.StringVar(value="Click any row above to view full diagnostic inspection details...")
        lbl_detail = ttk.Label(tab_faults, textvariable=self.var_diag_detail, font=("Consolas", 9, "bold"), foreground="#B91C1C", wraplength=900)
        lbl_detail.pack(anchor=tk.W, fill=tk.X, padx=5, pady=(5, 3))

        def on_tree_select(event):
            selected = self.tree_faults.selection()
            if selected:
                vals = self.tree_faults.item(selected[0], "values")
                if len(vals) == 4:
                    self.var_diag_detail.set(f"[{vals[0]} | {vals[1]} | {vals[2]}]: {vals[3]}")
        self.tree_faults.bind("<<TreeviewSelect>>", on_tree_select)

    def _build_footer(self):
        """Bottom workspace signature."""
        footer = ttk.Frame(self.main_container, padding=(15, 5))
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(footer, text="Offline Engineering Diagnostic Tool v1.0 | PyTorch & Scikit-Learn Hybrid Engine", foreground="#94A3B8", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Label(footer, text="Backend: MPS (Mac Apple Silicon) / CPU (Windows)", foreground="#94A3B8", font=("Segoe UI", 8)).pack(side=tk.RIGHT)

    def _setup_log_interception(self):
        """Hook root logging into our UI queue."""
        handler = QueueLoggingHandler(self.msg_queue)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # ── User Interaction Handlers ────────────────────────────────────────────

    def _browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input FTIR Excel Spreadsheet",
            filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")]
        )
        if filename:
            self.var_input_path.set(filename)
            self.lbl_status.config(text=f"Status: Selected workbook -> {os.path.basename(filename)}")
            self._log_to_console(f"Selected input source: {filename}\n", "INFO")

    def _browse_output(self):
        folder = filedialog.askdirectory(
            title="Select Output Folder"
        )
        if folder:
            filename = os.path.join(folder, "ftir_results.xlsx")
            self.var_output_path.set(filename)
            self._log_to_console(f"Output path changed to: {filename}\n", "INFO")

    def _clear_console(self):
        self.txt_console.delete("1.0", tk.END)

    def _open_output_report(self):
        out_path = self.var_output_path.get().strip()
        if not os.path.exists(out_path):
            messagebox.showerror("File Not Found", f"The result file does not exist yet:\n{out_path}")
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(out_path)
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", out_path], check=False)
            else:
                subprocess.run(["xdg-open", out_path], check=False)
        except Exception as e:
            messagebox.showerror("Error Opening File", f"Could not launch Excel report:\n{e}")

    # ── Pipeline Execution Orchestration ─────────────────────────────────────

    def _start_pipeline(self):
        input_file = self.var_input_path.get().strip()
        if not input_file or not os.path.exists(input_file):
            messagebox.showwarning("Missing Input File", "Please select a valid input FTIR Excel spreadsheet before starting the pipeline.")
            return

        out_file = self.var_output_path.get().strip()
        if not out_file:
            out_file = os.path.abspath(DEFAULT_OUTPUT_PATH)
            self.var_output_path.set(out_file)

        # Automatically use our persistent default profile directory so SIFT Maruti login cookies are always active
        profile = None
        skip_web = False  # Always extract photos from FTIR URLs

        # Reset UI dashboards
        self.kpi_total.config(text="0")
        self.kpi_rust.config(text="0")
        self.kpi_nonrust.config(text="0")
        self.kpi_review.config(text="0")
        self.kpi_faults.config(text="0")
        self.progress["value"] = 0
        self.progress["maximum"] = 100

        # Reset faults tracker table
        self.fault_count = 0
        for item in self.tree_faults.get_children():
            self.tree_faults.delete(item)
        self.notebook.tab(1, text=" ⚠️ Diagnostic Stage Failures & Faults (0) ")
        self.var_diag_detail.set("Click any row above to view full diagnostic inspection details...")

        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.DISABLED)

        self.is_running = True
        self.lbl_status.config(text="Status: Initializing deep learning and tabular model weights...")
        self._log_to_console("\n" + "="*70 + "\n", "TITLE")
        self._log_to_console("LAUNCHING DIAGNOSTIC PIPELINE WORKFLOW\n", "TITLE")
        self._log_to_console("="*70 + "\n", "TITLE")

        # Start execution thread
        self.pipeline_thread = threading.Thread(
            target=self._run_worker,
            args=(input_file, out_file, profile, skip_web),
            daemon=True
        )
        self.pipeline_thread.start()

    def _cancel_pipeline(self):
        if self.is_running:
            self.is_running = False
            self.lbl_status.config(text="Status: Cancelling operation...")
            self._log_to_console("\n[!] User cancellation requested. Terminating worker...\n", "WARNING")
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_run.config(state=tk.NORMAL)

    def _run_worker(self, input_file, out_file, profile, skip_web):
        """Worker thread executing the heavy diagnostic classification models."""
        try:
            def on_progress(current_idx, total_count, label, stats):
                if not self.is_running:
                    raise InterruptedError("Pipeline cancelled by operator.")
                self.msg_queue.put(("PROGRESS", current_idx, total_count, label, stats))

            result_df = run_pipeline(
                input_path=input_file,
                output_path=out_file,
                profile_dir=profile,
                skip_browser=skip_web,
                progress_callback=on_progress
            )
            self.msg_queue.put(("DONE", out_file, result_df))
        except InterruptedError as e:
            self.msg_queue.put(("CANCELLED", str(e)))
        except Exception as e:
            logging.exception("Fatal pipeline execution error")
            self.msg_queue.put(("ERROR", str(e)))

    # ── Reactive UI Queue Reader ─────────────────────────────────────────────

    def _process_queue(self):
        """Regularly consumes updates from background workers to update GUI widgets cleanly."""
        while not self.msg_queue.empty():
            msg_type, *args = self.msg_queue.get_nowait()

            if msg_type == "LOG":
                level, text = args
                tag = "INFO"
                if level in ("WARNING", "ERROR"):
                    tag = level
                elif "REVIEW" in text:
                    tag = "REVIEW"
                elif "✓" in text or "Completed" in text:
                    tag = "SUCCESS"
                self._log_to_console(text + "\n", tag)

            elif msg_type == "PROGRESS":
                current, total, label, stats = args
                if total > 0:
                    self.progress["maximum"] = total
                    self.progress["value"] = current
                    self.lbl_status.config(text=f"Status: Analyzing Record [{current}/{total}] — {label}")
                    self.kpi_total.config(text=str(total))

                # Update live KPI Counters
                if "rust_count" in stats:
                    self.kpi_rust.config(text=str(stats["rust_count"]))
                if "non_rust_count" in stats:
                    self.kpi_nonrust.config(text=str(stats["non_rust_count"]))
                if "review_count" in stats:
                    self.kpi_review.config(text=str(stats["review_count"]))

                err_val = stats.get("error_count", stats.get("errors", 0)) + stats.get("warning_count", stats.get("warnings", 0))
                if err_val > 0:
                    self.kpi_faults.config(text=str(err_val))

                # If record experienced an issue or failure, record it into the Failure Tracker Table!
                p_status = str(stats.get("pipeline_status", "SUCCESS"))
                if p_status and p_status != "SUCCESS" and not stats.get("completed"):
                    ftir_val = stats.get("ftir_no", label.split(" ")[-1])
                    f_stage = stats.get("failure_stage", "Stage 1/2 Verification")
                    f_diag = stats.get("failure_diagnostics", "Pipeline notice logged.")
                    self.tree_faults.insert("", "end", values=(ftir_val, p_status, f_stage, f_diag))
                    self.fault_count += 1
                    self.notebook.tab(1, text=f" ⚠️ Diagnostic Stage Failures & Faults ({self.fault_count}) ")

                if stats.get("completed"):
                    self.lbl_status.config(text=f"Status: ✓ Operation Complete! Saved diagnostic report ({stats['elapsed']:.1f}s)")

            elif msg_type == "DONE":
                out_path = args[0]
                result_df = args[1] if len(args) > 1 else None
                self.is_running = False
                self.last_output_path = out_path
                # Store result DataFrame for viewer windows
                if result_df is not None:
                    self.result_df = result_df
                else:
                    try:
                        self.result_df = pd.read_excel(out_path)
                    except Exception:
                        self.result_df = None
                self.btn_run.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
                self.btn_open.config(state=tk.NORMAL)
                self.btn_browse_ftirs.config(state=tk.NORMAL)
                self.btn_manual_review.config(state=tk.NORMAL)
                messagebox.showinfo(
                    "Pipeline Successfully Complete!",
                    f"All FTIR defect records have been classified and fused.\n\n"
                    f"Formatted automated report generated at:\n{out_path}\n\n"
                    f"You can now use 'Browse All FTIRs' to view extracted photos,\n"
                    f"or 'Manual Review' to classify flagged records."
                )

            elif msg_type == "CANCELLED":
                self.is_running = False
                self.btn_run.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
                self.lbl_status.config(text="Status: Operation cancelled.")

            elif msg_type == "ERROR":
                err = args[0]
                self.is_running = False
                self.btn_run.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
                self.lbl_status.config(text="Status: Fatal Error Occurred (See Console)")
                messagebox.showerror("Pipeline Execution Error", f"A fatal error stopped execution:\n\n{err}")

        # Loop checking queue every 100 milliseconds
        self.after(100, self._process_queue)

    def _log_to_console(self, text, tag="INFO"):
        """Append text to the scrollable console window."""
        self.txt_console.insert(tk.END, text, tag)
        self.txt_console.see(tk.END)

    # ── Post-Pipeline Viewer Windows ─────────────────────────────────────────

    def _get_sbpr_list(self):
        """Load known SBPR numbers from config/sbpr_keywords.yaml."""
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "sbpr_keywords.yaml")
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return list(data.keys()) if data else []
        except Exception:
            return []

    def _open_ftir_browser(self):
        """Open the FTIR Results Browser window to view all processed FTIRs and their photos."""
        if self.result_df is None:
            messagebox.showwarning("No Results", "Please run the pipeline first before browsing results.")
            return
        try:
            out_path = self.last_output_path or self.var_output_path.get().strip()
            sbpr_list = self._get_sbpr_list()
            FtirBrowserWindow(self, self.result_df, output_path=out_path, sbpr_list=sbpr_list)
        except Exception as e:
            messagebox.showerror("Viewer Error", f"Could not open FTIR browser:\n{e}")

    def _open_manual_review(self):
        """Open the Manual Review window to classify flagged FTIRs."""
        if self.result_df is None:
            messagebox.showwarning("No Results", "Please run the pipeline first before reviewing results.")
            return
        out_path = self.last_output_path or self.var_output_path.get().strip()
        sbpr_list = self._get_sbpr_list()
        try:
            ManualReviewWindow(self, self.result_df, out_path, sbpr_list)
        except Exception as e:
            messagebox.showerror("Viewer Error", f"Could not open manual review window:\n{e}")


if __name__ == "__main__":
    # Mandatory for PyInstaller standalone compiled executables on Windows (.exe)
    # when underlying libraries (PyTorch / joblib / scikit-learn) spawn parallel workers or threads.
    import multiprocessing
    multiprocessing.freeze_support()

    app = FtirSbprDesktopApp()
    app.mainloop()
