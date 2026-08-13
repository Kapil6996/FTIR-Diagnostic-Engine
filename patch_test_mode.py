import re

with open("src/pipeline.py", "r") as f:
    code = f.read()

target_args = """    skip_browser: bool = False,
    profile_dir: Optional[str] = None,"""
replacement_args = """    skip_browser: bool = False,
    test_mode: bool = False,
    profile_dir: Optional[str] = None,"""
code = code.replace(target_args, replacement_args)

target_read = """    # ── 1. Read input Excel ────────────────────────────────────────────
    logger.info("Stage 0: Reading input Excel workbook...")
    df = read_ftir_sheet(input_path)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")"""
replacement_read = """    # ── 1. Read input Excel ────────────────────────────────────────────
    logger.info("Stage 0: Reading input Excel workbook...")
    df = read_ftir_sheet(input_path)
    if test_mode:
        df = df.head(1)
        logger.info("  [TEST MODE] Truncated dataset to 1 row for rapid testing.")
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")"""
code = code.replace(target_read, replacement_read)

target_cli = """    parser.add_argument(
        "--profile-dir", default=None,"""
replacement_cli = """    parser.add_argument(
        "--test-mode", action="store_true",
        help="Run on only the first row of the excel sheet to test extraction",
    )
    parser.add_argument(
        "--profile-dir", default=None,"""
code = code.replace(target_cli, replacement_cli)

target_cli_call = """        skip_browser=args.skip_browser,
        profile_dir=args.profile_dir,"""
replacement_cli_call = """        skip_browser=args.skip_browser,
        test_mode=args.test_mode,
        profile_dir=args.profile_dir,"""
code = code.replace(target_cli_call, replacement_cli_call)

with open("src/pipeline.py", "w") as f:
    f.write(code)

# ----------------- APP.PY -----------------
with open("app.py", "r") as f:
    code = f.read()

target_ui = """        lbl_sift_notice.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(15, 4))

        frame.columnconfigure(1, weight=1)"""
replacement_ui = """        lbl_sift_notice.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(15, 4))

        self.var_test_mode = tk.BooleanVar(value=True) # Default to True right now for their testing
        chk_test = ttk.Checkbutton(frame, text="🧪 Test Mode (Process first row only for quick verification)", variable=self.var_test_mode)
        chk_test.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(5, 4))

        frame.columnconfigure(1, weight=1)"""
code = code.replace(target_ui, replacement_ui)

target_kwargs = """                "skip_browser": False,
                "profile_dir": get_chrome_profile_dir(),"""
replacement_kwargs = """                "skip_browser": False,
                "test_mode": self.var_test_mode.get(),
                "profile_dir": get_chrome_profile_dir(),"""
code = code.replace(target_kwargs, replacement_kwargs)

with open("app.py", "w") as f:
    f.write(code)

print("test mode added")
