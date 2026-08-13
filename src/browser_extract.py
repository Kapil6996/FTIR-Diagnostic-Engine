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
# Portal URL is read from config/portal_url.txt (gitignored so git pull
# never overwrites your URL).  Falls back to this hardcoded default.
def _load_portal_url() -> str:
    """Read portal URL from config/portal_url.txt, surviving git pull."""
    # Look relative to THIS file (src/) → go one level up to project root
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(_project_root, "config", "portal_url.txt")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            url = f.read().strip()
        if url and url != "INSERT_URL_HERE":
            return url
    # Also try CWD-relative path (for when run from project root)
    if os.path.isfile("config/portal_url.txt"):
        with open("config/portal_url.txt", "r") as f:
            url = f.read().strip()
        if url and url != "INSERT_URL_HERE":
            return url
    return "INSERT_URL_HERE"


def _load_attachment_template() -> str:
    """Read exact attachment URL template from config/attachment_url_template.txt."""
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(_project_root, "config", "attachment_url_template.txt")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "http" in line and "{FTIR_NO}" in line:
                        return line
    return ""


PORTAL_SEARCH_URL = _load_portal_url()
if PORTAL_SEARCH_URL == "INSERT_URL_HERE":
    logger.warning("⚠️ Portal URL not configured! Edit config/portal_url.txt with your portal URL.")
else:
    logger.info(f"✓ Portal URL loaded from config: {PORTAL_SEARCH_URL[:40]}...")

ATTACHMENT_URL_TEMPLATE = _load_attachment_template()
if ATTACHMENT_URL_TEMPLATE:
    logger.info("✓ Exact Attachment URL Template loaded from config!")

# Delay (seconds) between consecutive network requests to avoid
# hammering the internal server.
_REQUEST_DELAY = 1.5

# Page-load timeout for Selenium waits (seconds).
# Set extremely high (5 minutes) so that if the user hits a login screen
# and has to type passwords/OTP, the script doesn't time out and crash
# while they are still typing!
_PAGE_TIMEOUT = 300

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
) -> webdriver.Edge:
    """
    Initialise the Selenium WebDriver.
    Strictly launches Microsoft Edge for enterprise compatibility.

    Parameters
    ----------
    profile_dir : str, optional
        Filesystem path to the Edge user-data directory.
        Defaults to ``~/.ftir_sbpr_tool/browser_profile``.
    headless : bool
        If True, launch Edge in headless mode (no visible window).
        Defaults to False so the user can see the browser.

    Returns
    -------
    webdriver.Edge
    """
    if profile_dir is None:
        profile_dir = _DEFAULT_PROFILE_DIR + "_edge"

    os.makedirs(profile_dir, exist_ok=True)

    edge_options = EdgeOptions()

    # Persistent profile — preserves login sessions across runs
    edge_options.add_argument(f"--user-data-dir={os.path.abspath(profile_dir)}")

    if headless:
        # True headless breaks enterprise portals. Move window off-screen instead to hide it.
        edge_options.add_argument("--window-position=-32000,-32000")

    # Suppress noisy DevTools logging
    edge_options.add_argument("--log-level=3")
    edge_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    edge_options.add_experimental_option("useAutomationExtension", False)

    # Reasonable window size for element visibility
    edge_options.add_argument("--window-size=1920,1080")

    # Disable pop-up blocker so download dialogs don't interfere
    edge_options.add_argument("--disable-popup-blocking")

    try:
        driver = webdriver.Edge(options=edge_options)
        logger.info(f"Edge driver started with profile: {profile_dir}")
    except WebDriverException as e:
        raise RuntimeError(
            f"Could not start Microsoft Edge browser.\n"
            f"Make sure Edge is installed and up to date.\n"
            f"Error: {e}"
        )

    driver.set_page_load_timeout(_PAGE_TIMEOUT)
    driver.implicitly_wait(5)
    return driver


# ---------------------------------------------------------------------------
#  Page extraction
# ---------------------------------------------------------------------------

def _get_browser_cookies_for_requests(driver: webdriver.Edge) -> dict:
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


def extract_ftir_page(driver: webdriver.Edge, url: str, ftir_no: str = None) -> Dict[str, Any]:
    """
    Navigate to an FTIR detail page and extract subject text and
    attachment URLs.

    Parameters
    ----------
    driver : webdriver.Edge
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

    # FIX: Scroll the ENTIRE page to force lazy-loaded content (like attachments
    # below the specification block) to render. The user confirmed that photos
    # appear only after scrolling past ~12 rows of incident description.
    try:
        # Scroll to absolute bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        logger.info("Page scrolled to bottom and back to trigger lazy-loading.")
    except Exception:
        pass

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
    # 2. Attachment URL collection — AGGRESSIVE MULTI-STRATEGY SCAN
    # ------------------------------------------------------------------
    attachment_urls: List[str] = []
    seen: set = set()

    # STRATEGY 0 (HIGHEST PRIORITY): Scan ALL links on page for ftirWebFile.do
    # The user confirmed that clickable photo links (e.g., "cyxs.jpeg") on the
    # FTIR page point to URLs containing "ftirWebFile.do". This is the most
    # reliable indicator, so check it FIRST and across the ENTIRE page.
    logger.info("Strategy 0: Scanning entire page for ftirWebFile/ftirWeb links...")
    for a_tag in driver.find_elements(By.TAG_NAME, "a"):
        href = a_tag.get_attribute("href")
        if href and href not in seen:
            href_lower = href.lower()
            if "ftirweb" in href_lower or "ftirfile" in href_lower or "webfile" in href_lower:
                seen.add(href)
                attachment_urls.append(href)
                logger.info(f"  Found ftirWeb link: {href[:80]}...")
    if attachment_urls:
        logger.info(f"Strategy 0: Found {len(attachment_urls)} ftirWebFile link(s)!")

    # STRATEGY 1: Scan ALL links on full page for image-like file names
    # Look for <a> tags whose visible text or href ends with image extensions
    logger.info("Strategy 1: Scanning all links for image file names...")
    for a_tag in driver.find_elements(By.TAG_NAME, "a"):
        href = a_tag.get_attribute("href")
        link_text = (a_tag.text or "").strip().lower()
        if href and href not in seen:
            # Check if link text looks like a filename (e.g., "cyxs.jpeg")
            if any(link_text.endswith(ext) for ext in _ATTACHMENT_EXTENSIONS):
                seen.add(href)
                attachment_urls.append(href)
                logger.info(f"  Found image link by name: {link_text} -> {href[:80]}...")
            elif _looks_like_attachment_url(href):
                seen.add(href)
                attachment_urls.append(href)

    # STRATEGY 2: Find attachment containers and grab everything inside
    container_selectors = [
        "[class*='attachment']", "[class*='Attachment']",
        "[id*='attachment']", "[id*='Attachment']",
        "[class*='upload']", "[class*='file-list']",
        "[class*='gallery']",
    ]
    for css in container_selectors:
        try:
            containers = driver.find_elements(By.CSS_SELECTOR, css)
            for container in containers:
                for a_tag in container.find_elements(By.TAG_NAME, "a"):
                    href = a_tag.get_attribute("href")
                    if href and href not in seen:
                        seen.add(href)
                        attachment_urls.append(href)
                for img_tag in container.find_elements(By.TAG_NAME, "img"):
                    src = img_tag.get_attribute("src")
                    if src and src not in seen and not src.startswith("data:"):
                        seen.add(src)
                        attachment_urls.append(src)
        except NoSuchElementException:
            continue

    # STRATEGY 3: Grab ALL <img> tags on the page (skip tiny icons, EXCEPT ftirWeb links)
    logger.info("Strategy 3: Scanning all <img> tags on page...")
    for img_tag in driver.find_elements(By.TAG_NAME, "img"):
        src = img_tag.get_attribute("src")
        if src and src not in seen and not src.startswith("data:"):
            # If it's a known thumbnail endpoint, rewrite it to get the full file!
            if "ftirwebthumbnail.do" in src.lower():
                full_src = re.sub(r'ftirWebThumbnail\.do', 'ftirWebFile.do', src, flags=re.IGNORECASE)
                if full_src not in seen:
                    seen.add(full_src)
                    attachment_urls.append(full_src)
                    logger.info(f"  Upgraded thumbnail to full file: {full_src[:80]}...")
                continue
            elif "ftirweb" in src.lower() or "ftirfile" in src.lower():
                seen.add(src)
                attachment_urls.append(src)
                continue

            # Skip tiny icons (< 50px) by checking natural dimensions
            try:
                w = img_tag.get_attribute("naturalWidth")
                h = img_tag.get_attribute("naturalHeight")
                if w and h and int(w) > 50 and int(h) > 50:
                    seen.add(src)
                    attachment_urls.append(src)
            except Exception:
                seen.add(src)
                attachment_urls.append(src)

    # Strategy F: Iframe Piercing
    # Enterprise portals often bury the FTIR details inside a <iframe>.
    # We must switch into every iframe and scan for attachments!
    logger.info("Piercing iframes to search for hidden attachments...")
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    for frame in iframes:
        try:
            driver.switch_to.frame(frame)
            for a_tag in driver.find_elements(By.TAG_NAME, "a"):
                href = a_tag.get_attribute("href")
                if href and href not in seen and _looks_like_attachment_url(href):
                    seen.add(href)
                    attachment_urls.append(href)
            for img_tag in driver.find_elements(By.TAG_NAME, "img"):
                src = img_tag.get_attribute("src")
                if src and src not in seen and not src.startswith("data:"):
                    if "ftirwebthumbnail.do" in src.lower():
                        full_src = re.sub(r'ftirWebThumbnail\.do', 'ftirWebFile.do', src, flags=re.IGNORECASE)
                        if full_src not in seen:
                            seen.add(full_src)
                            attachment_urls.append(full_src)
                            logger.info(f"  [Iframe] Upgraded thumbnail to full file: {full_src[:80]}...")
                        continue
                    elif "ftirweb" in src.lower() or "ftirfile" in src.lower():
                        seen.add(src)
                        attachment_urls.append(src)
                        continue
                    seen.add(src)
                    attachment_urls.append(src)
        except Exception:
            pass
        finally:
            driver.switch_to.default_content()

    guessed_urls = []
    
    # Strategy E: The Hardcoded Portal Pattern Guesser (The Ultimate Fallback)
    # If the scraper completely fails to find any images in the DOM, we can blindly 
    # brute-force the server based on the known portal URL and FTIR number!
    if ftir_no:
        if ATTACHMENT_URL_TEMPLATE:
            logger.info(f"Strategy E: Generating exact URLs using configured template for {ftir_no}")
            for category in range(1, 4):
                for sequence in range(1, 11):
                    hardcoded_url = ATTACHMENT_URL_TEMPLATE.replace("{FTIR_NO}", ftir_no).replace("{CATEGORY}", str(category)).replace("{SEQUENCE}", str(sequence))
                    if hardcoded_url not in attachment_urls and hardcoded_url not in guessed_urls:
                        guessed_urls.append(hardcoded_url)
        elif PORTAL_SEARCH_URL and PORTAL_SEARCH_URL != "INSERT_URL_HERE":
            if "/sift" in PORTAL_SEARCH_URL.lower():
                # e.g., https://maruti.com/sift/search.do -> https://maruti.com/sift
                # But handle case insensitivity
                idx = PORTAL_SEARCH_URL.lower().find("/sift")
                base_sift = PORTAL_SEARCH_URL[:idx] + "/sift"
                logger.info(f"Strategy E: Generating hardcoded URLs for {ftir_no} using base {base_sift}")
                for category in range(1, 4):
                    for sequence in range(1, 11):
                        hardcoded_url = f"{base_sift}/ftirWebFile.do?documentid={ftir_no}&fileCategory={category}&fileSequence={sequence}"
                        if hardcoded_url not in attachment_urls and hardcoded_url not in guessed_urls:
                            guessed_urls.append(hardcoded_url)

    # Strategy D: The URL Sequence Guesser
    # The user noted images follow a strict pattern: ...&fileCategory=1&fileSequence=1&...
    # If we find EVEN ONE URL matching this, we can mathematically generate the rest!
    
    # We will search both the already found attachments and the current page URL for hints
    search_pool = attachment_urls + [driver.current_url]
    
    for url in search_pool:
        if not url: continue
        # Look for the parameter pattern in the URL
        if re.search(r'fileSequence=\d+', url, re.IGNORECASE):
            logger.info(f"Found predictable URL pattern in: {url[:60]}...")
            # Generate combinations for fileCategory (1 to 3) and fileSequence (1 to 10)
            for category in range(1, 4):
                for sequence in range(1, 11):
                    # Replace category if it exists
                    new_url = re.sub(r'(fileCategory=)\d+', rf'\g<1>{category}', url, flags=re.IGNORECASE)
                    # Replace sequence
                    new_url = re.sub(r'(fileSequence=)\d+', rf'\g<1>{sequence}', new_url, flags=re.IGNORECASE)
                    
                    if new_url not in seen:
                        seen.add(new_url)
                        guessed_urls.append(new_url)
            
            # Once we've guessed based on a template, break out so we don't duplicate guesses
            break
            
    if guessed_urls:
        logger.info(f"Generated {len(guessed_urls)} predicted attachment URLs to attempt downloading.")
        attachment_urls.extend(guessed_urls)

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


def search_and_extract_ftir(driver: webdriver.Edge, ftir_no: str, portal_url: str = PORTAL_SEARCH_URL) -> Dict[str, Any]:
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

        # Type the FTIR number
        logger.info(f"Found search box. Entering FTIR {ftir_no}...")
        search_box.clear()
        search_box.send_keys(ftir_no)
        
        # Try to find and click a 'Search' button instead of just hitting Enter
        try:
            logger.info("Attempting to find and click 'Search' button...")
            search_btn_xpath = (
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')] | "
                "//input[(@type='submit' or @type='button') and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')] | "
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')]"
            )
            # Find all matching elements and click the first visible one
            search_btns = driver.find_elements(By.XPATH, search_btn_xpath)
            clicked = False
            for btn in search_btns:
                if btn.is_displayed():
                    btn.click()
                    clicked = True
                    break
            
            if not clicked:
                raise NoSuchElementException("No visible search button found")
        except Exception:
            logger.warning("Could not click a 'Search' button. Falling back to hitting ENTER key.")
            from selenium.webdriver.common.keys import Keys
            search_box.send_keys(Keys.RETURN)
        
        # Wait for the resulting page to load (wait for URL to change or body to refresh)
        time.sleep(3)
        WebDriverWait(driver, _PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)
        
        logger.info(f"Search submitted. Now extracting from resulting page: {driver.current_url}")
        return extract_ftir_page(driver, driver.current_url, ftir_no=ftir_no)

    except Exception as e:
        logger.error(f"Fallback Search Failed during execution: {e}")
        return {"page_url": driver.current_url, "subject_text": None, "attachment_urls": [], "page_title": ""}


# ---------------------------------------------------------------------------
#  FTIR Response Form Excel Download & Image Extraction
# ---------------------------------------------------------------------------

def _get_edge_download_dir(driver: webdriver.Edge) -> str:
    """Get or create a known download directory for Edge."""
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_dir = os.path.join(_project_root, "temp_downloads")
    os.makedirs(download_dir, exist_ok=True)
    return download_dir


def _wait_for_download(download_dir: str, timeout: int = 120) -> Optional[str]:
    """
    Wait for a new file to appear in download_dir and finish downloading.
    Returns the path of the downloaded file, or None on timeout.
    """
    # Record existing files
    existing = set(os.listdir(download_dir))
    
    start = time.time()
    new_file = None
    while time.time() - start < timeout:
        current = set(os.listdir(download_dir))
        new_files = current - existing
        
        # Filter out partial download files (.crdownload, .part, .tmp)
        completed = [
            f for f in new_files
            if not f.endswith(('.crdownload', '.part', '.tmp', '.download'))
        ]
        
        if completed:
            # Pick the newest completed file
            new_file = max(
                completed,
                key=lambda f: os.path.getmtime(os.path.join(download_dir, f))
            )
            # Wait a moment to ensure writing is complete
            path = os.path.join(download_dir, new_file)
            prev_size = -1
            for _ in range(5):
                time.sleep(1)
                curr_size = os.path.getsize(path)
                if curr_size == prev_size and curr_size > 0:
                    logger.info(f"Download complete: {new_file} ({curr_size:,} bytes)")
                    return path
                prev_size = curr_size
            # File seems stable
            return path
        
        time.sleep(2)
    
    logger.warning(f"Download timed out after {timeout}s")
    return None


def _extract_images_from_xlsx(xlsx_path: str, output_dir: str) -> List[str]:
    """
    Extract embedded images from an Excel .xlsx file.
    
    .xlsx files are ZIP archives. Images are stored in xl/media/.
    This approach requires NO extra packages — just the built-in zipfile module.
    """
    import zipfile
    
    extracted = []
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(xlsx_path, 'r') as zf:
            media_files = [
                f for f in zf.namelist()
                if f.startswith('xl/media/') and not f.endswith('/')
            ]
            
            if not media_files:
                logger.warning(f"No images found in xl/media/ of {xlsx_path}")
                return []
            
            logger.info(f"Found {len(media_files)} embedded images in Excel response form")
            
            for media_file in media_files:
                # Extract just the filename (e.g., "image1.png")
                filename = os.path.basename(media_file)
                save_path = os.path.join(output_dir, filename)
                
                # Avoid overwriting
                base, ext = os.path.splitext(save_path)
                counter = 1
                while os.path.exists(save_path):
                    save_path = f"{base}_{counter}{ext}"
                    counter += 1
                
                # Read from zip and write to disk
                with zf.open(media_file) as src, open(save_path, 'wb') as dst:
                    dst.write(src.read())
                
                file_size = os.path.getsize(save_path)
                logger.info(f"  Extracted from Excel: {filename} ({file_size:,} bytes)")
                extracted.append(save_path)
    except zipfile.BadZipFile:
        logger.error(f"Downloaded file is not a valid Excel/ZIP file: {xlsx_path}")
    except Exception as e:
        logger.error(f"Error extracting images from Excel: {e}")
    
    return extracted


def extract_via_response_form(
    driver: webdriver.Edge,
    ftir_no: str,
    save_dir: str,
) -> List[str]:
    """
    Strategy: Click 'Go to FTIR Response Form' button on the current FTIR page,
    which downloads an Excel file containing the attachments. Then extract
    embedded images from the downloaded Excel.
    
    This is the MOST RELIABLE extraction method because it uses the portal's
    own export functionality rather than scraping the DOM.
    
    Parameters
    ----------
    driver : webdriver.Edge
        Active Selenium WebDriver positioned on an FTIR detail page.
    ftir_no : str
        FTIR number for logging.
    save_dir : str
        Directory to save extracted images into.
    
    Returns
    -------
    List[str]
        List of file paths of extracted images.
    """
    logger.info(f"FTIR {ftir_no}: [STRATEGY: RESPONSE FORM] Attempting to find 'Go to FTIR Response Form' button...")
    
    # Set up a known download directory via Edge preferences
    download_dir = _get_edge_download_dir(driver)
    
    # Configure Edge to download to our known directory (via CDP command)
    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": os.path.abspath(download_dir),
        })
        logger.info(f"Download directory set to: {download_dir}")
    except Exception as e:
        logger.warning(f"Could not set download directory via CDP: {e}")
    
    # Search for the "Go to FTIR Response Form" button using multiple strategies
    button_strategies = [
        # Strategy 1: Exact text match (case-insensitive)
        "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ftir response form')]",
        "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ftir response form')]",
        "//input[contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ftir response form')]",
        # Strategy 2: Partial text "response form"
        "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'response form')]",
        "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'response form')]",
        "//input[contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'response form')]",
        # Strategy 3: Any element containing the text
        "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'go to ftir')]",
        "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'response form')]",
    ]
    
    def _find_button_in_context():
        for xpath in button_strategies:
            try:
                elements = driver.find_elements(By.XPATH, xpath)
                for el in elements:
                    if el.is_displayed():
                        logger.info(f"Found button with text: '{el.text.strip()}'")
                        return el
            except Exception:
                continue
        return None

    def _recursive_iframe_search_and_click() -> bool:
        # Search current context
        btn = _find_button_in_context()
        if btn:
            try:
                btn.click()
                logger.info(f"FTIR {ftir_no}: Clicked 'FTIR Response Form' button. Waiting for download...")
                return True
            except Exception as e:
                logger.warning(f"Found button but failed to click: {e}")
        
        # Search child iframes
        try:
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
        except Exception:
            return False
            
        for i in range(len(iframes)):
            try:
                frames = driver.find_elements(By.TAG_NAME, "iframe")
                if i >= len(frames):
                    continue
                driver.switch_to.frame(frames[i])
                if _recursive_iframe_search_and_click():
                    return True
                driver.switch_to.parent_frame()
            except Exception:
                try:
                    driver.switch_to.default_content()
                except:
                    pass
        return False
    
    # Record existing files in download dir before clicking
    existing_files = set(os.listdir(download_dir))
    
    driver.switch_to.default_content()
    clicked = _recursive_iframe_search_and_click()
    driver.switch_to.default_content()
    
    if not clicked:
        logger.warning(f"FTIR {ftir_no}: Could not find or click 'FTIR Response Form' button on the page or any iframes.")
        return []
    
    # Wait for the Excel file to download
    downloaded_file = _wait_for_download(download_dir, timeout=120)
    
    if not downloaded_file:
        logger.warning(f"FTIR {ftir_no}: Response Form Excel download failed or timed out.")
        return []
    
    # Check if it's an Excel file
    if not downloaded_file.lower().endswith(('.xlsx', '.xls')):
        logger.warning(f"FTIR {ftir_no}: Downloaded file is not Excel: {downloaded_file}")
        return []
    
    logger.info(f"FTIR {ftir_no}: Response Form Excel downloaded: {downloaded_file}")
    
    # Extract images from the Excel file
    extracted_images = _extract_images_from_xlsx(downloaded_file, save_dir)
    
    # Clean up the downloaded Excel file
    try:
        os.remove(downloaded_file)
        logger.info(f"Cleaned up temp Excel file: {downloaded_file}")
    except OSError:
        pass
    
    if extracted_images:
        logger.info(f"FTIR {ftir_no}: ✓ Successfully extracted {len(extracted_images)} images from Response Form Excel!")
    else:
        logger.warning(f"FTIR {ftir_no}: Response Form Excel contained no extractable images.")
    
    return extracted_images


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
    driver = None,
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

    ct = resp.headers.get("Content-Type", "").lower()
    if "text/html" in ct and driver is not None:
        logger.info(f"URL returned an HTML viewer page. Using Selenium element-screenshot fallback for: {url}")
        original_window = driver.current_window_handle
        try:
            driver.switch_to.new_window('tab')
            driver.get(url)
            import time
            time.sleep(3) # Wait for viewer to render image
            
            # Find the largest image on the page
            from selenium.webdriver.common.by import By
            imgs = driver.find_elements(By.TAG_NAME, "img")
            largest_img = None
            max_area = 0
            for img in imgs:
                try:
                    size = img.size
                    area = size["width"] * size["height"]
                    if area > max_area:
                        max_area = area
                        largest_img = img
                except Exception:
                    pass
            
            if largest_img and max_area > 5000:
                filename = _filename_from_response(resp, url)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filename += ".png"
                import re, os
                filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
                save_path = os.path.join(save_dir, filename)
                
                # Avoid overwriting
                base, ext = os.path.splitext(save_path)
                counter = 1
                while os.path.exists(save_path):
                    save_path = f"{base}_{counter}{ext}"
                    counter += 1
                
                largest_img.screenshot(save_path)
                logger.info(f"  Screenshot extracted from viewer: {filename}")
                return save_path
            else:
                logger.warning("Could not find a prominent image in the viewer page.")
        except Exception as e:
            logger.error(f"Failed to screenshot viewer page: {e}")
        finally:
            driver.close()
            driver.switch_to.window(original_window)

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
    driver: webdriver.Edge,
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
    driver : webdriver.Edge
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
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.warning(f"Could not remove cached file {file_path}: {e}")

    extraction_source = "None"
    attachment_urls = []
    page_info = {"subject_text": None}

    # ── Extract page ───────────────────────────────────────────────────
    if url and url.startswith("http"):
        try:
            logger.info(f"FTIR {ftir_no}: [EXTRACTION SOURCE: EXCEL HYPERLINK] Trying direct URL.")
            page_info = extract_ftir_page(driver, url, ftir_no=ftir_no)
            attachment_urls = page_info["attachment_urls"]
            if attachment_urls:
                extraction_source = "Hyperlink"
                logger.info(f"FTIR {ftir_no}: Successfully found attachments via Excel hyperlink.")
        except Exception as e:
            logger.warning(f"FTIR {ftir_no}: Initial extraction failed ({e}). Triggering fallback search.")
            attachment_urls = []
    else:
        logger.info(f"FTIR {ftir_no}: No hyperlink URL in Excel. Going directly to Quick Search.")

    if not attachment_urls:
        logger.warning(f"FTIR {ftir_no}: [EXTRACTION SOURCE: QUICK SEARCH] Attempting portal search fallback.")
        fallback_info = search_and_extract_ftir(driver, ftir_no)
        attachment_urls = fallback_info["attachment_urls"]
        if attachment_urls:
            extraction_source = "Quick Search"
            logger.info(f"FTIR {ftir_no}: Successfully found attachments via Quick Search fallback.")
        
        # If the fallback found a subject text, use it
        if fallback_info.get("subject_text"):
            page_info["subject_text"] = fallback_info["subject_text"]
            
        if not attachment_urls:
            logger.warning(f"FTIR {ftir_no}: Fallback search also failed to find attachments.")

    # ── Try Excel Response Form Strategy First ─────────────────────────
    extracted_images = extract_via_response_form(driver, ftir_no, save_dir)
    if extracted_images:
        extraction_source = "Excel Response Form"
        downloaded = extracted_images
    else:
        # ── Fallback to HTTP Download of DOM scraped URLs ──────────────
        cookies = _get_browser_cookies_for_requests(driver)
        downloaded: List[str] = []

        for i, att_url in enumerate(attachment_urls):
            logger.info(f"FTIR {ftir_no}: downloading attachment {i + 1}/{len(attachment_urls)}")
            try:
                saved = download_attachment(cookies, att_url, save_dir, driver=driver)
                if saved:
                    downloaded.append(saved)
            except Exception as e:
                logger.error(f"Error downloading attachment {att_url}: {e}")

            # Polite delay between requests
            if i < len(attachment_urls) - 1:
                time.sleep(_REQUEST_DELAY)

    logger.info(
        f"FTIR {ftir_no}: {len(downloaded)} attachments saved to {save_dir}"
    )

    return {
        "ftir_no": ftir_no,
        "url": url,
        "save_dir": save_dir,
        "subject_text": page_info["subject_text"],
        "downloaded_files": downloaded,
        "attachment_urls": attachment_urls,
        "skipped": False,
        "extraction_source": extraction_source,
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
