"""
Shared utility functions for the Maruti FTIR Diagnostic Engine.
Centralizes environment detection and hardware acceleration logic.
"""

import os
import sys

def get_bundle_path(check_dir: str = "models") -> str:
    """
    Returns the base path where bundled read-only assets live.
    Automatically resolves paths whether running from source or from a
    frozen PyInstaller executable bundle (macOS .app or Windows .exe).
    
    Parameters
    ----------
    check_dir : str
        A specific subdirectory (e.g., 'models' or 'config') to verify
        the root path against.
    """
    if not getattr(sys, 'frozen', False):
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        
    candidates = []
    
    # 1. PyInstaller One-File extracted temp dir
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(meipass)
        
    # 2. Executable directory (One-Dir mode)
    exe_dir = os.path.dirname(sys.executable)
    candidates.append(exe_dir)
    
    # 3. macOS .app bundle
    app_bundle = os.path.normpath(os.path.join(exe_dir, '..', '..'))
    app_name = os.path.basename(app_bundle)
    if app_name.endswith('.app'):
        companion = os.path.join(os.path.dirname(app_bundle), app_name[:-4])
        candidates.append(companion)
        
    # 4. macOS .app root directory itself
    candidates.append(app_bundle)
    
    for path in candidates:
        if os.path.isdir(os.path.join(path, check_dir)):
            return path
            
    return meipass or exe_dir

def get_persistent_dir() -> str:
    """
    Returns a directory path that is guaranteed to persist across application restarts,
    even when running as a PyInstaller frozen application.
    """
    if not getattr(sys, 'frozen', False):
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        
    exe_dir = os.path.dirname(sys.executable)
    app_bundle = os.path.normpath(os.path.join(exe_dir, '..', '..'))
    app_name = os.path.basename(app_bundle)
    if app_name.endswith('.app'):
        # macOS: Store next to the .app bundle
        return os.path.dirname(app_bundle)
    else:
        # Windows/Linux: Store next to the .exe
        return exe_dir

def get_device() -> "torch.device":
    """
    Select best available compute device for PyTorch.
    Prefers Apple Silicon MPS backend if available, then CUDA, falling back to CPU.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        # If torch is not available (e.g., in non-ML scripts), just return None or raise
        pass
    return None
