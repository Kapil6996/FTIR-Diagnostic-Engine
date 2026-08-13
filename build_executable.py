"""
Automated PyInstaller Executable Builder — Standalone Packaging Utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script automatically freezes and converts the entire Maruti FTIR Diagnostic
Application (including Python runtime, PyTorch models, Scikit-Learn decision trees,
and Tkinter GUI) into a standalone double-click executable (.exe on Windows,
macOS application / binary on Apple Silicon/Intel Mac).

Usage:
    # Build standard standalone app directory (Instant double-click startup time!):
    python build_executable.py

    # Optional: Build single monolithic .exe file (Compressed, slower initial launch):
    python build_executable.py --onefile
"""

import os
import sys
import shutil
import argparse
import subprocess


def check_requirements():
    """Verify PyInstaller is available in the Python workspace."""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("\n[!] PyInstaller library is not currently installed in your environment.")
        print("    To install it offline or online, simply run:")
        print("        pip install pyinstaller")
        print("    Then re-run this script: python build_executable.py\n")
        return False


def build_app():
    parser = argparse.ArgumentParser(description="Build Standalone Executable Application Bundle")
    parser.add_argument(
        "--onefile", "-f",
        action="store_true",
        help="Package everything into a single monolithic file (Note: ~1.5GB PyTorch archives take ~30 seconds to unpack into temp folder on startup when using onefile mode)."
    )
    args = parser.parse_args()

    if not check_requirements():
        sys.exit(1)

    app_name = "Maruti_FTIR_Diagnostic_Engine"
    entry_point = "app.py"
    
    # OS path delimiter for PyInstaller --add-data (';' on Windows, ':' on Mac/Linux)
    sep = os.pathsep

    print("=" * 75)
    print(f"  MARUTI SUZUKI — STANDALONE EXECUTABLE BUILD UTILITY")
    print(f"  Target OS Platform : {sys.platform.upper()}")
    print(f"  Packaging Mode     : {'Monolithic Single File (--onefile)' if args.onefile else 'Standalone Folder / Bundle (--onedir) [Recommended]'}")
    print("=" * 75)
    print("\n[1/4] Preparing PyInstaller build flags and hidden module imports...")

    # Define robust hidden imports required by dynamic PyTorch, Scikit-Learn & Excel engines
    hidden_imports = [
        "torch",
        "torch.backends.mps",
        "torchvision",
        "torchvision.models.resnet",
        "sklearn.tree",
        "sklearn.ensemble",
        "sklearn.utils._typedefs",
        "sklearn.neighbors._typedefs",
        "openpyxl",
        "openpyxl.cell._writer",
        "yaml",
        "PIL",
        "PIL.Image",
        "pymupdf",
        "selenium",
        "selenium.webdriver",
        "selenium.webdriver.chrome.webdriver",
        "selenium.webdriver.edge.webdriver",
        "selenium.webdriver.safari.webdriver",
        "webdriver_manager",
        "webdriver_manager.microsoft",
        "webdriver_manager.chrome",
        "tkinter",
        "tkinter.ttk",
        "tkinter.scrolledtext",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "pandas",
        "pandas.io.excel",
        "PIL.ImageTk",
        "src.review_viewer",
        "src.pipeline",
        "src.excel_io",
        "src.media_normalize",
        "src.rust_model",
        "src.sbpr_features",
        "src.sbpr_tree",
        "src.sbpr_image_model",
        "src.fusion",
        "src.browser_extract",
        "src.utils",
    ]

    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", app_name,
        "--windowed",  # Hide console / shell box behind GUI window
        "--clean",
        "--noconfirm",
        "--exclude-module", "cv2",  # Bypasses OpenCV recursion import bug on macOS
        "--add-data", f"config{sep}config",
        "--add-data", f"src{sep}src",
        "--add-data", f"models{sep}models",
    ]

    if args.onefile:
        pyinstaller_cmd.append("--onefile")
    else:
        pyinstaller_cmd.append("--onedir")

    for mod in hidden_imports:
        pyinstaller_cmd.extend(["--hidden-import", mod])

    pyinstaller_cmd.append(entry_point)

    print("\n[2/4] Executing PyInstaller compilation (this may take 2-4 minutes)...")
    print("      Command: " + " ".join(pyinstaller_cmd))

    res = subprocess.run(pyinstaller_cmd, check=False)
    if res.returncode != 0:
        print("\n[❌] PyInstaller build failed. See terminal errors above.")
        sys.exit(res.returncode)

    print("\n[3/4] Copying offline ML model checkpoint files into distributable bundle...")
    dist_dir = os.path.join("dist", app_name) if not args.onefile else "dist"
    models_source = "models"
    models_dest = os.path.join(dist_dir, "models")

    if os.path.exists(models_source):
        if os.path.exists(models_dest):
            shutil.rmtree(models_dest)
        shutil.copytree(models_source, models_dest)
        print(f"      ✓ Successfully embedded 'models/' directory into -> {models_dest}")
    else:
        print(f"      [! Warning] Local 'models/' directory not found at {models_source}. Remember to place trained weights beside the executable.")

    # Ensure output records directory exists in distribution
    records_dest = os.path.join(dist_dir, "ftir_records")
    os.makedirs(records_dest, exist_ok=True)

    # Ensure outputs directory exists in distribution
    outputs_dest = os.path.join(dist_dir, "outputs")
    os.makedirs(outputs_dest, exist_ok=True)

    # Copy config directory if not already included via --add-data
    config_dest = os.path.join(dist_dir, "config")
    if not os.path.exists(config_dest) and os.path.exists("config"):
        shutil.copytree("config", config_dest)
        print(f"      ✓ Copied 'config/' directory into -> {config_dest}")

    print("\n[4/4] 🎉 COMPILATION & EMBEDDING SUCCESSFULLY COMPLETED!")
    print("=" * 75)
    if not args.onefile:
        exe_ext = ".exe" if sys.platform.startswith("win") else (".app" if sys.platform.startswith("darwin") and os.path.exists(f"dist/{app_name}.app") else "")
        target_bin = os.path.join(dist_dir, f"{app_name}{exe_ext}")
        print(f"  Your zero-setup standalone application folder is ready at:")
        print(f"    📁 {os.path.abspath(dist_dir)}")
        print(f"\n  To deploy to teammates or Windows workstations:")
        print(f"    1. Copy or zip up the entire '{app_name}/' folder from 'dist/'.")
        print(f"    2. Hand it to coworkers on a USB stick or network folder.")
        print(f"    3. They simply double-click '{app_name}{'.exe' if sys.platform.startswith('win') else ''}' to run offline!")
    else:
        exe_ext = ".exe" if sys.platform.startswith("win") else ""
        target_bin = os.path.join(dist_dir, f"{app_name}{exe_ext}")
        print(f"  Your monolithic single-file executable is ready at:")
        print(f"    📦 {os.path.abspath(target_bin)}")
        print(f"\n  Note: Remember to distribute the 'models/' folder alongside this executable file!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    build_app()
