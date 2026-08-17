"""
Media Normalization Module (src.media_normalize)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~────────────────

Converts heterogeneous FTIR attachments (still images, multi-page PDFs, and video files)
into a flat, normalized list of still image file paths ready for computer vision
model inference (Stage 1 Rust classifier and Stage 2 SBPR Image CNN).

Supported File Types:
- Images (.jpg, .jpeg, .png, .bmp, .webp, .heic): Retained as-is.
- Videos (.mp4, .avi, .mov, .mkv, .wmv, .3gp, .webm): Sampled at fixed time intervals
  using OpenCV, saved to temporary directory.
- PDFs (.pdf): Embedded images extracted using PyMuPDF (with fallback to whole-page
  rendering for scanned/flattened documents), saved to temporary directory.
"""

import os
import sys
import logging
import tempfile
import hashlib
from typing import List

logger = logging.getLogger(__name__)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Video frame extraction will be disabled.")

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False
        pymupdf = None
        logger.warning("PyMuPDF (pymupdf / fitz) not available. PDF image extraction will be disabled.")

# Cross-platform safe default directory for extracted frames
_DEFAULT_TMP_DIR = "/tmp/ftir_frames" if os.name == "posix" else os.path.join(tempfile.gettempdir(), "ftir_frames")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".3gp", ".webm"}
PDF_EXTENSIONS = {".pdf"}
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}


def _get_output_dir_for_file(filepath: str, base_tmp_dir: str = _DEFAULT_TMP_DIR) -> str:
    """
    Generate a unique, collision-safe temporary output folder path for a specific media file.
    Uses the file stem plus a short hash of the full path to prevent overwrites between
    identically named attachments across different FTIR records.
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    # Short hash of absolute path to guarantee uniqueness across identical filenames
    path_hash = hashlib.md5(os.path.abspath(filepath).encode("utf-8")).hexdigest()[:6]
    folder_name = f"{stem}_{path_hash}"
    out_dir = os.path.join(base_tmp_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def extract_images_from_attachment(path: str, frame_interval_sec: float = 2.0, driver=None) -> List[str]:
    """
    Convert a single attachment file into a flat list of still image paths.

    Dispatch Rules:
    - Images: returned immediately as `[path]`.
    - Videos: opens with OpenCV and samples one frame every `frame_interval_sec` seconds.
      Saved under `/tmp/ftir_frames/<stem>/`.
    - PDFs: opens with PyMuPDF and extracts all embedded images from every page.
      If a page contains no embedded images (e.g., flattened scan), renders the page as an image.
      Saved under `/tmp/ftir_frames/<stem>/`.
    - Unrecognised types: prints a warning and returns `[]`.

    Parameters
    ----------
    path : str
        Filesystem path to the target attachment file.
    frame_interval_sec : float, optional
        Time spacing (in seconds) between video frame captures (default: 2.0 seconds).

    Returns
    -------
    List[str]
        List of absolute file paths to normalized still images.
    """
    if not os.path.exists(path):
        logger.error(f"Attachment file not found: {path}")
        return []

    ext = os.path.splitext(path)[1].lower()
    abs_path = os.path.abspath(path)

    # 1. Direct Image Formats
    if ext in IMAGE_EXTENSIONS:
        logger.debug(f"Image attachment retained directly: {abs_path}")
        return [abs_path]

    # 2. Video Formats (OpenCV Frame Extraction)
    if ext in VIDEO_EXTENSIONS:
        if not OPENCV_AVAILABLE:
            logger.error(f"Cannot extract frames from {abs_path}: OpenCV is not available in this build.")
            return []

        out_dir = _get_output_dir_for_file(abs_path)
        existing_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
        if existing_files:
            logger.info(f"Using {len(existing_files)} cached frames for video '{os.path.basename(abs_path)}'")
            return sorted(existing_files)

        logger.info(f"Extracting video frames every {frame_interval_sec}s from '{os.path.basename(abs_path)}' to {out_dir}")
        
        extracted_frames = []
        cap = cv2.VideoCapture(abs_path)
        
        if not cap.isOpened():
            logger.error(f"OpenCV failed to open video file: {abs_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle broken metadata default to 30 FPS
        if fps <= 0 or np_isnan(fps):
            fps = 30.0
        
        frame_step = max(1, int(fps * frame_interval_sec))
        frame_count = 0
        saved_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video stream

                if frame_count % frame_step == 0:
                    out_fname = f"frame_{saved_count:04d}_sec_{frame_count/fps:.1f}.jpg"
                    out_fpath = os.path.join(out_dir, out_fname)
                    
                    # Save frame as JPEG with high quality
                    cv2.imwrite(out_fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    extracted_frames.append(out_fpath)
                    saved_count += 1

                frame_count += 1
        except Exception as e:
            logger.error(f"Error during video decoding of {abs_path}: {e}")
        finally:
            cap.release()

        logger.info(f"  -> Extracted {len(extracted_frames)} frames from video.")
        return extracted_frames

    # 3. PDF Documents (PyMuPDF Embedded Image Extraction)
    if ext in PDF_EXTENSIONS:
        if not PYMUPDF_AVAILABLE:
            logger.error(f"Cannot extract images from {abs_path}: PyMuPDF is not installed.")
            return []

        out_dir = _get_output_dir_for_file(abs_path)
        existing_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
        if existing_files:
            logger.info(f"Using {len(existing_files)} cached images for PDF '{os.path.basename(abs_path)}'")
            return sorted(existing_files)

        logger.info(f"Extracting images from PDF '{os.path.basename(abs_path)}' to {out_dir}")
        
        extracted_images = []
        try:
            doc = pymupdf.open(abs_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                if image_list:
                    # Extract embedded image objects
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        try:
                            base_img = doc.extract_image(xref)
                            image_bytes = base_img["image"]
                            image_ext = base_img["ext"] or "png"
                            
                            out_fname = f"page_{page_num + 1}_img_{img_idx + 1}.{image_ext}"
                            out_fpath = os.path.join(out_dir, out_fname)
                            
                            with open(out_fpath, "wb") as f:
                                f.write(image_bytes)
                            
                            extracted_images.append(out_fpath)
                        except Exception as e:
                            logger.warning(f"Failed to extract embedded image xref {xref} on page {page_num+1}: {e}")
                else:
                    # Fallback: If no embedded images exist (e.g. flattened scanned page), render full page
                    logger.debug(f"Page {page_num + 1} has no embedded images. Rendering full page to image as fallback.")
                    pix = page.get_pixmap(dpi=150)
                    out_fname = f"page_{page_num + 1}_fullrender.png"
                    out_fpath = os.path.join(out_dir, out_fname)
                    pix.save(out_fpath)
                    extracted_images.append(out_fpath)

            doc.close()
            logger.info(f"  -> Extracted {len(extracted_images)} image(s) from PDF.")
            return extracted_images
        except Exception as e:
            logger.error(f"Error processing PDF {abs_path}: {e}")
            return []

    # 4. Excel Documents (Embedded Images Extraction from Response Form)
    if ext in EXCEL_EXTENSIONS:
        out_dir = _get_output_dir_for_file(abs_path)
        existing_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
        if existing_files:
            logger.info(f"Using {len(existing_files)} cached images for Excel '{os.path.basename(abs_path)}'")
            return sorted(existing_files)

        logger.info(f"Extracting images from Excel '{os.path.basename(abs_path)}' to {out_dir}")
        try:
            from src.browser_extract import _extract_images_from_xlsx
            extracted_images = _extract_images_from_xlsx(abs_path, out_dir, driver=driver)
            logger.info(f"  -> Extracted {len(extracted_images)} image(s) from Excel.")
            return extracted_images
        except Exception as e:
            logger.error(f"Error extracting images from Excel {abs_path}: {e}")
            return []

    # 5. Unrecognised / Non-Visual Attachment
    logger.warning(f"Skipping non-visual attachment format '{ext}' for file: {os.path.basename(abs_path)}")
    return []


def get_all_images_for_ftir(ftir_folder: str, driver=None) -> List[str]:
    """
    Scan an FTIR record folder and normalize all attachments into a flat list of still image paths.
    
    Walks through all files in `ftir_records/<ftir_no>/`, dispatching each to
    `extract_images_from_attachment()` and combining the resulting image file paths into a single list
    ready for computer vision AI evaluation.

    Parameters
    ----------
    ftir_folder : str
        Directory path containing downloaded FTIR attachments (e.g., `ftir_records/FTIR001`).

    Returns
    -------
    List[str]
        Unified flat list of image file paths suitable for Model Stage 1 & Stage 2 inference.
    """
    if not os.path.isdir(ftir_folder):
        logger.error(f"Target FTIR directory does not exist: {ftir_folder}")
        return []

    combined_images: List[str] = []
    
    # Sort files for deterministic processing sequence
    filenames = sorted(os.listdir(ftir_folder))
    
    for fname in filenames:
        if fname.startswith("."):
            continue  # Skip OS hidden files like .DS_Store
            
        fpath = os.path.join(ftir_folder, fname)
        if os.path.isfile(fpath):
            img_paths = extract_images_from_attachment(fpath, driver=driver)
            combined_images.extend(img_paths)
            
    logger.info(f"FTIR folder '{os.path.basename(ftir_folder)}' yielded {len(combined_images)} total normalized images.")
    return combined_images


def np_isnan(val: float) -> bool:
    """Helper to check NaN without mandatory numpy import overhead in inner loop."""
    return val != val


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 65)
    print("  FTIR Media Normalization Utility (src.media_normalize)")
    print("=" * 65)

    if len(sys.argv) < 2:
        # Default test: use sample images folder if available
        default_dir = os.path.join("data", "SBIN201210B00011", "test")
        if os.path.isdir(default_dir):
            target_path = default_dir
            print(f"No path provided via sys.argv — auto-detecting test directory:\n  -> {target_path}\n")
        else:
            print("Usage: python -m src.media_normalize <path_to_ftir_folder_or_file>")
            sys.exit(1)
    else:
        target_path = sys.argv[1]

    if os.path.isfile(target_path):
        print(f"Testing single attachment extraction: {target_path}")
        results = extract_images_from_attachment(target_path)
    elif os.path.isdir(target_path):
        print(f"Testing FTIR batch folder normalization: {target_path}")
        results = get_all_images_for_ftir(target_path)
    else:
        print(f"Error: path not found: {target_path}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"  Extraction Summary: Found {len(results)} normalized image(s)")
    print("=" * 65)
    for i, p in enumerate(results[:10], start=1):
        print(f"  [{i:2d}] {p}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more.")
    print("=" * 65)
