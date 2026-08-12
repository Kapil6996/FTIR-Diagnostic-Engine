"""
Browser Extraction & Attachment Download Module (src.browser_extract)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selenium-based module for extracting attachments from FTIR detail pages
behind the SIFT Maruti internal quality portal.

Strategy
--------
*  Uses a **persistent Chrome user-data-dir profile** so the operator
   only needs to log into SIFT Maruti **once** in their default browser;
   subsequent automated runs reuse the saved session cookies & tokens.
*  The script **never** handles login credentials itself.
*  Downloads are performed via **requests** using the browser session's
   cookies, which is faster and more reliable than Selenium's built-in
   download manager for binary files.

Usage::

    # Test against a single FTIR page:
    python -m src.browser_extract "https://sift.marutisuzuki.com/ftir/12345"

    # With a custom profile directory:
    python -m src.browser_extract "https://sift.marutisuzuki.com/ftir/12345" \\
        --profile /path/to/chrome-profile
"""

import os
import sys
import time
import logging
import argparse
import mimetypes
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urljoin, unquote

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)

logger = logging.getLogger(__name__)

# Default persistent profile location (platform-aware)
_DEFAULT_PROFILE_DIR = os.path.join(
    os.path.expanduser("~"), ".ftir_sbpr_tool", "browser_profile"
)

# ---------------------------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------------------------
# If the standard URL fails, the robot will navigate to this portal URL and
# try to search for the FTIR by typing its number into the first text box.
# REPLACE THIS WITH THE ACTUAL EMPLOYEE PORTAL SEARCH URL:
PORTAL_SEARCH_URL = "INSERT_URL_HERE"

# Delay (seconds) between consecutive network requests to avoid
# hammering the internal server.
_REQUEST_DELAY = 1.5

# Page-load timeout for Selenium waits (seconds).
_PAGE_TIMEOUT = 30

# Common attachment file extensions we care about.
_ATTACHMENT_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic",
    ".pdf",
    ".mp4", ".avi", ".mov", ".wmv", ".mkv", ".3gp", ".webm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z",
}


# ---------------------------------------------------------------------------
#  Driver setup
# ---------------------------------------------------------------------------

def get_driver(
    profile_dir: Optional[str] = None,
    headless: bool = False,
) -> webdriver.Chrome:
    """
    Return a configured Selenium Chrome WebDriver using a persistent
    user-data-dir profile.

    The profile directory stores cookies, local-storage, and session
    tokens so that a one-time manual login by the user is preserved
    across all future automated runs.

    Parameters
    ----------
    profile_dir : str, optional
        Filesystem path to the Chrome user-data directory.
        Defaults to ``~/.ftir_sbpr_tool/browser_profile``.
    headless : bool
        If True, launch Chrome in headless mode (no visible window).
        Defaults to False so the user can see the browser on first run
        and manually log in.

    Returns
    -------
    webdriver.Chrome
    """
    if profile_dir is None:
        profile_dir = _DEFAULT_PROFILE_DIR

    os.makedirs(profile_dir, exist_ok=True)

    options = ChromeOptions()

    # Persistent profile — preserves login sessions across runs
    options.add_argument(f"--user-data-dir={os.path.abspath(profile_dir)}")

    if headless:
        # True headless breaks enterprise portals. Move window off-screen instead to hide it.
        options.add_argument("--window-position=-32000,-32000")

    # Suppress noisy DevTools logging
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Disable automation banners that some portals detect
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Reasonable window size for element visibility
    options.add_argument("--window-size=1920,1080")

    # Disable pop-up blocker so download dialogs don't interfere
    options.add_argument("--disable-popup-blocking")

    try:
        driver = webdriver.Chrome(options=options)
        logger.info(f"Chrome driver started with profile: {profile_dir}")
    except WebDriverException:
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            logger.info(f"Chrome driver started with profile: {profile_dir}")
        except Exception:
            logger.info("Chrome not found or failed to start. Falling back to Microsoft Edge...")
            edge_options = EdgeOptions()
            edge_profile = profile_dir + "_edge"
            os.makedirs(edge_profile, exist_ok=True)
            edge_options.add_argument(f"--user-data-dir={os.path.abspath(edge_profile)}")
            
            if headless:
                # True headless breaks enterprise portals. Move window off-screen instead to hide it.
                edge_options.add_argument("--window-position=-32000,-32000")
            
            # Edge specific flags
            edge_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
            edge_options.add_experimental_option("useAutomationExtension", False)
            edge_options.add_argument("--window-size=1920,1080")
            edge_options.add_argument("--disable-popup-blocking")

            try:
                driver = webdriver.Edge(options=edge_options)
            except WebDriverException:
                try:
                    from webdriver_manager.microsoft import EdgeChromiumDriverManager
                    service = EdgeService(EdgeChromiumDriverManager().install())
                    driver = webdriver.Edge(service=service, options=edge_options)
                except Exception:
                    logger.info("Edge not found. Falling back to Safari...")
                    try:
                        driver = webdriver.Safari()
                        logger.info("Safari driver started.")
                    except Exception as e:
                        raise RuntimeError(f"Could not start Chrome, Edge, or Safari. Please install Chrome! Error: {e}")
            
            if "edge_profile" in locals():
                logger.info(f"Edge driver started with profile: {edge_profile}")

    driver.set_page_load_timeout(_PAGE_TIMEOUT)
    driver.implicitly_wait(5)
    return driver


# ---------------------------------------------------------------------------
#  Page extraction
# ---------------------------------------------------------------------------

def _get_browser_cookies_for_requests(driver: webdriver.Chrome) -> dict:
    """Convert Selenium cookies into a dict usable by ``requests``."""
    return {c["name"]: c["value"] for c in driver.get_cookies()}


def _looks_like_attachment_url(href: str) -> bool:
    """Heuristic: does this URL look like a downloadable attachment?"""
    if not href:
        return False
    parsed = urlparse(href)
    path_lower = parsed.path.lower()
    # Direct file extension match
    if any(path_lower.endswith(ext) for ext in _ATTACHMENT_EXTENSIONS):
        return True
    # Common download-servlet patterns
    download_patterns = [
        "download", "attach", "getfile", "blob", "media",
        "upload", "file", "resource", "content",
    ]
    if any(kw in path_lower for kw in download_patterns):
        return True
    # Query-param based downloads (e.g. ?action=download&id=...)
    query_lower = parsed.query.lower()
    if any(kw in query_lower for kw in ["download", "attach", "file"]):
        return True
    return False


def extract_ftir_page(driver: webdriver.Chrome, url: str) -> Dict[str, Any]:
    """
    Navigate to an FTIR detail page and extract subject text and
    attachment URLs.

    Parameters
    ----------
    driver : webdriver.Chrome
        An active Selenium WebDriver with a logged-in session.
    url : str
        Full URL of the FTIR detail page.

    Returns
    -------
    dict
        ``{
            "page_url": str,
            "subject_text": str or None,
            "attachment_urls": List[str],
            "page_title": str,
        }``
    """
    logger.info(f"Navigating to FTIR page: {url}")
    driver.get(url)

    # Give the page time to fully render (JS-heavy portals)
    try:
        WebDriverWait(driver, _PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except TimeoutException:
        logger.warning(f"Page load timed out for {url}")

    # Allow dynamic content to settle
    time.sleep(2)

    page_title = driver.title or ""
    base_url = driver.current_url

    # ------------------------------------------------------------------
    # 1. Subject text extraction
    # ------------------------------------------------------------------
    subject_text = None

    # Try several common patterns for subject / description fields
    subject_selectors = [
        # Labelled fields (label + adjacent value)
        "//td[contains(translate(text(),'SUBJECT','subject'),'subject')]/following-sibling::td",
        "//th[contains(translate(text(),'SUBJECT','subject'),'subject')]/following-sibling::td",
        "//label[contains(translate(text(),'SUBJECT','subject'),'subject')]/following-sibling::*",
        "//span[contains(translate(text(),'SUBJECT','subject'),'subject')]/parent::*/following-sibling::*",
        # Generic description containers
        "//div[contains(@class,'subject')]",
        "//div[contains(@class,'description')]",
        "//div[contains(@id,'subject')]",
        "//div[contains(@id,'description')]",
    ]
    for xpath in subject_selectors:
        try:
            el = driver.find_element(By.XPATH, xpath)
            txt = (el.text or "").strip()
            if txt and len(txt) > 3:
                subject_text = txt
                break
        except NoSuchElementException:
            continue

    # ------------------------------------------------------------------
    # 2. Attachment URL collection
    # ------------------------------------------------------------------
    attachment_urls: List[str] = []
    seen: set = set()

    # Strategy A: find an attachments container and grab everything inside
    container_selectors = [
        "[class*='attachment']",
        "[class*='Attachment']",
        "[id*='attachment']",
        "[id*='Attachment']",
        "[class*='upload']",
        "[class*='file-list']",
        "[class*='media']",
        "[class*='gallery']",
        "[class*='document']",
    ]
    containers_found = False
    for css in container_selectors:
        try:
            containers = driver.find_elements(By.CSS_SELECTOR, css)
            for container in containers:
                containers_found = True
                # <a href="..."> links
                for a_tag in container.find_elements(By.TAG_NAME, "a"):
                    href = a_tag.get_attribute("href")
                    if href and href not in seen:
                        seen.add(href)
                        attachment_urls.append(href)
                # <img src="..."> images (sometimes full-res images are inlined)
                for img_tag in container.find_elements(By.TAG_NAME, "img"):
                    src = img_tag.get_attribute("src")
                    if src and src not in seen and not src.startswith("data:"):
                        seen.add(src)
                        attachment_urls.append(src)
        except NoSuchElementException:
            continue

    # Strategy B: Search for exact "file category" or "file sequence" labels
    if not containers_found:
        logger.info("Scanning for 'file category' or 'file sequence' labels...")
        try:
            # Find elements containing 'file sequence' or 'file category' (case insensitive)
            label_xpath = "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'file sequence') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'file category')]"
            labels = driver.find_elements(By.XPATH, label_xpath)
            if labels:
                logger.info(f"Found {len(labels)} 'file sequence'/'file category' labels.")
                for label in labels:
                    # Look at the parent container (usually a table row or div)
                    parent = label.find_element(By.XPATH, "..")
                    for tag in parent.find_elements(By.TAG_NAME, "a"):
                        href = tag.get_attribute("href")
                        if href and href not in seen:
                            seen.add(href)
                            attachment_urls.append(href)
                    for tag in parent.find_elements(By.TAG_NAME, "img"):
                        src = tag.get_attribute("src")
                        if src and src not in seen and not src.startswith("data:"):
                            seen.add(src)
                            attachment_urls.append(src)
                if attachment_urls:
                    containers_found = True
        except Exception as e:
            logger.debug(f"File sequence extraction failed: {e}")

    # Strategy C: if still no attachments, scan the whole page
    if not containers_found:
        logger.info("No attachment container or file sequences found — scanning full page for attachment-like URLs")
        for a_tag in driver.find_elements(By.TAG_NAME, "a"):
            href = a_tag.get_attribute("href")
            if href and href not in seen and _looks_like_attachment_url(href):
                seen.add(href)
                attachment_urls.append(href)
        for img_tag in driver.find_elements(By.TAG_NAME, "img"):
            src = img_tag.get_attribute("src")
            if src and src not in seen and not src.startswith("data:"):
                seen.add(src)
                attachment_urls.append(src)

    # Resolve relative URLs to absolute
    attachment_urls = [urljoin(base_url, u) for u in attachment_urls]

    logger.info(
        f"Extracted {len(attachment_urls)} attachment URL(s) from page "
        f"(subject: {'found' if subject_text else 'not found'})"
    )

    return {
        "page_url": base_url,
        "subject_text": subject_text,
        "attachment_urls": attachment_urls,
        "page_title": page_title,
    }


def search_and_extract_ftir(driver: webdriver.Chrome, ftir_no: str, portal_url: str = PORTAL_SEARCH_URL) -> Dict[str, Any]:
    """
    Fallback method: Navigates to the portal search page, enters the FTIR number
    into the first text box, hits Enter, waits for the result page to load,
    and then delegates to extract_ftir_page.
    """
    logger.info(f"Fallback Search Triggered: Navigating to portal {portal_url} for FTIR {ftir_no}")
    
    if not portal_url or portal_url == "INSERT_URL_HERE":
        logger.error("PORTAL_SEARCH_URL is not configured. Please add it to src/browser_extract.py.")
        return {"page_url": driver.current_url, "subject_text": None, "attachment_urls": [], "page_title": ""}

    driver.get(portal_url)

    try:
        # Wait for the page to load
        WebDriverWait(driver, _PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # Let JS settle

        # Navigate click-path: Click Quick Search directly
        logger.info("Attempting to click 'Quick Search' link...")
        try:
            quick_search = driver.find_element(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'quick search')]")
            quick_search.click()
            time.sleep(2)
        except NoSuchElementException:
            logger.warning("Could not find 'Quick Search' link. Trying to find search box anyway.")

        # Find the first visible text input box
        text_boxes = driver.find_elements(By.XPATH, "//input[@type='text' or not(@type)]")
        search_box = None
        for box in text_boxes:
            if box.is_displayed():
                search_box = box
                break
        
        if not search_box:
            logger.error("Fallback Search Failed: Could not find any text input box on the portal page.")
            return {"page_url": driver.current_url, "subject_text": None, "attachment_urls": [], "page_title": ""}

        # Type the FTIR number and hit Enter
        logger.info(f"Found search box. Entering FTIR {ftir_no}...")
        search_box.clear()
        search_box.send_keys(ftir_no)
        
        from selenium.webdriver.common.keys import Keys
        search_box.send_keys(Keys.RETURN)
        
        # Wait for the resulting page to load (wait for URL to change or body to refresh)
        time.sleep(3)
        WebDriverWait(driver, _PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)
        
        logger.info(f"Search submitted. Now extracting from resulting page: {driver.current_url}")
        return extract_ftir_page(driver, driver.current_url)

    except Exception as e:
        logger.error(f"Fallback Search Failed during execution: {e}")
        return {"page_url": driver.current_url, "subject_text": None, "attachment_urls": [], "page_title": ""}


# ---------------------------------------------------------------------------
#  Attachment download
# ---------------------------------------------------------------------------

def _filename_from_response(response: requests.Response, url: str) -> str:
    """
    Derive a sensible filename from the HTTP response headers or URL.

    Priority:
    1. Content-Disposition header (``filename=...``)
    2. Last segment of the URL path
    3. Fallback to ``attachment`` + extension from Content-Type
    """
    # 1. Content-Disposition
    cd = response.headers.get("Content-Disposition", "")
    if "filename" in cd:
        # Handle both filename="name.ext" and filename*=UTF-8''name.ext
        match = re.search(r"filename\*?=['\"]?(?:UTF-8'')?([^'\";]+)", cd, re.IGNORECASE)
        if match:
            name = unquote(match.group(1)).strip()
            if name:
                return name

    # 2. URL path
    parsed = urlparse(url)
    path_segment = os.path.basename(parsed.path)
    if path_segment and "." in path_segment:
        return unquote(path_segment)

    # 3. Fallback: content-type → extension
    ct = response.headers.get("Content-Type", "application/octet-stream")
    ct = ct.split(";")[0].strip()
    ext = mimetypes.guess_extension(ct) or ".bin"
    return f"attachment{ext}"


def download_attachment(
    cookies: dict,
    url: str,
    save_dir: str,
    timeout: int = 60,
) -> Optional[str]:
    """
    Download a single attachment using the browser session's cookies.

    Parameters
    ----------
    cookies : dict
        Cookie dict extracted from the Selenium session via
        ``_get_browser_cookies_for_requests(driver)``.
    url : str
        Direct URL of the attachment to download.
    save_dir : str
        Local directory to save the file into.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    str or None
        Absolute path of the saved file, or None if download failed.
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        resp = requests.get(url, cookies=cookies, stream=True, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        return None

    filename = _filename_from_response(resp, url)
    # Sanitize filename
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    save_path = os.path.join(save_dir, filename)

    # Avoid overwriting — append counter if file exists
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size = os.path.getsize(save_path)
    logger.info(f"  Downloaded: {filename} ({file_size:,} bytes)")
    return save_path


# ---------------------------------------------------------------------------
#  End-to-end single-FTIR processor
# ---------------------------------------------------------------------------

def process_ftir(
    driver: webdriver.Chrome,
    ftir_no: str,
    url: str,
    base_save_dir: str = "ftir_records",
) -> Dict[str, Any]:
    """
    Extract and download all attachments for a single FTIR record.

    Creates ``<base_save_dir>/<ftir_no>/`` and downloads every attachment
    found on the FTIR detail page into it.

    **Resumability**: if the target folder already exists and contains at
    least one file, the FTIR is skipped entirely (returns cached info).

    Parameters
    ----------
    driver : webdriver.Chrome
        Active Selenium WebDriver with a logged-in session.
    ftir_no : str
        Unique FTIR record identifier (used as folder name).
    url : str
        Full URL of the FTIR detail page.
    base_save_dir : str
        Root directory under which per-FTIR folders are created.

    Returns
    -------
    dict
        ``{
            "ftir_no": str,
            "url": str,
            "save_dir": str,
            "subject_text": str or None,
            "downloaded_files": List[str],
            "attachment_urls": List[str],
            "skipped": bool,
        }``
    """
    save_dir = os.path.join(base_save_dir, str(ftir_no))

    import shutil
    # ── Force Fresh Download ───────────────────────────────────────────
    # The user requested to fix the issue from the root to ensure it ALWAYS
    # downloads automatically, avoiding confusion from aggressive local caching.
    if os.path.isdir(save_dir):
        logger.info(f"FTIR {ftir_no}: Clearing existing cached files for fresh download")
        for f in os.listdir(save_dir):
            file_path = os.path.join(save_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # ── Extract page ───────────────────────────────────────────────────
    try:
        logger.info(f"FTIR {ftir_no}: [EXTRACTION SOURCE: EXCEL HYPERLINK] Trying direct URL.")
        page_info = extract_ftir_page(driver, url)
        attachment_urls = page_info["attachment_urls"]
        if attachment_urls:
            logger.info(f"FTIR {ftir_no}: Successfully found attachments via Excel hyperlink.")
    except Exception as e:
        logger.warning(f"FTIR {ftir_no}: Initial extraction failed ({e}). Triggering fallback search.")
        attachment_urls = []
        page_info = {"subject_text": None}

    if not attachment_urls:
        logger.warning(f"FTIR {ftir_no}: [EXTRACTION SOURCE: QUICK SEARCH] Attempting portal search fallback.")
        fallback_info = search_and_extract_ftir(driver, ftir_no)
        attachment_urls = fallback_info["attachment_urls"]
        if attachment_urls:
            logger.info(f"FTIR {ftir_no}: Successfully found attachments via Quick Search fallback.")
        
        # If the fallback found a subject text, use it
        if fallback_info.get("subject_text"):
            page_info["subject_text"] = fallback_info["subject_text"]
            
        if not attachment_urls:
            logger.warning(f"FTIR {ftir_no}: Fallback search also failed to find attachments.")

    # ── Download attachments ───────────────────────────────────────────
    cookies = _get_browser_cookies_for_requests(driver)
    downloaded: List[str] = []

    for i, att_url in enumerate(attachment_urls):
        logger.info(f"FTIR {ftir_no}: downloading attachment {i + 1}/{len(attachment_urls)}")
        saved = download_attachment(cookies, att_url, save_dir)
        if saved:
            downloaded.append(saved)

        # Polite delay between requests
        if i < len(attachment_urls) - 1:
            time.sleep(_REQUEST_DELAY)

    logger.info(
        f"FTIR {ftir_no}: {len(downloaded)}/{len(attachment_urls)} "
        f"attachments saved to {save_dir}"
    )

    return {
        "ftir_no": ftir_no,
        "url": url,
        "save_dir": save_dir,
        "subject_text": page_info["subject_text"],
        "downloaded_files": downloaded,
        "attachment_urls": attachment_urls,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test FTIR attachment extraction against a single page URL.",
    )
    parser.add_argument(
        "url",
        help="Full URL of an FTIR detail page to extract attachments from.",
    )
    parser.add_argument(
        "--ftir-no",
        default="TEST_FTIR",
        help="FTIR record number / folder name (default: TEST_FTIR).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=f"Chrome user-data-dir profile path (default: {_DEFAULT_PROFILE_DIR}).",
    )
    parser.add_argument(
        "--save-dir",
        default="ftir_records",
        help="Root directory for downloaded attachments (default: ftir_records).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (no visible window).",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  FTIR Browser Extraction — Single-Page Test")
    print("=" * 60)
    print(f"  URL        : {args.url}")
    print(f"  FTIR No    : {args.ftir_no}")
    print(f"  Profile    : {args.profile or _DEFAULT_PROFILE_DIR}")
    print(f"  Save Dir   : {args.save_dir}")
    print(f"  Headless   : {args.headless}")
    print("=" * 60)

    drv = get_driver(profile_dir=args.profile, headless=args.headless)

    try:
        result = process_ftir(
            driver=drv,
            ftir_no=args.ftir_no,
            url=args.url,
            base_save_dir=args.save_dir,
        )

        print("\n" + "=" * 60)
        print("  Results")
        print("=" * 60)
        print(f"  Skipped (cached)  : {result['skipped']}")
        print(f"  Subject text      : {result['subject_text'] or '(not found on page)'}")
        print(f"  Attachment URLs   : {len(result['attachment_urls'])}")
        for u in result["attachment_urls"]:
            print(f"    → {u}")
        print(f"  Downloaded files  : {len(result['downloaded_files'])}")
        for f in result["downloaded_files"]:
            print(f"    → {f}")
        print(f"  Save directory    : {result['save_dir']}")
        print("=" * 60)

    finally:
        drv.quit()
        print("\nBrowser closed.")
