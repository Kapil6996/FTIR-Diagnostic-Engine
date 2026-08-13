import re

with open("src/browser_extract.py", "r") as f:
    code = f.read()

# 1. Capture extraction source
target_process = """    # ── Extract page ───────────────────────────────────────────────────
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
            logger.info(f"FTIR {ftir_no}: Successfully found attachments via Quick Search fallback.")"""

replacement_process = """    extraction_source = "None"
    # ── Extract page ───────────────────────────────────────────────────
    try:
        logger.info(f"FTIR {ftir_no}: [EXTRACTION SOURCE: EXCEL HYPERLINK] Trying direct URL.")
        page_info = extract_ftir_page(driver, url)
        attachment_urls = page_info["attachment_urls"]
        if attachment_urls:
            extraction_source = "Hyperlink"
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
            extraction_source = "Quick Search"
            logger.info(f"FTIR {ftir_no}: Successfully found attachments via Quick Search fallback.")"""

code = code.replace(target_process, replacement_process)

target_return = """    return {
        "ftir_no": ftir_no,
        "url": url,
        "save_dir": save_dir,
        "subject_text": page_info["subject_text"],
        "downloaded_files": downloaded,
        "attachment_urls": attachment_urls,
        "skipped": False,
    }"""

replacement_return = """    return {
        "ftir_no": ftir_no,
        "url": url,
        "save_dir": save_dir,
        "subject_text": page_info["subject_text"],
        "downloaded_files": downloaded,
        "attachment_urls": attachment_urls,
        "skipped": False,
        "extraction_source": extraction_source,
    }"""

code = code.replace(target_return, replacement_return)

with open("src/browser_extract.py", "w") as f:
    f.write(code)

print("browser_extract.py updated")
