import re

with open("src/browser_extract.py", "r") as f:
    code = f.read()

# 1. Update extract_ftir_page to look for file category / file sequence
extract_target = """    # Strategy B: if no dedicated container found, scan the whole page
    # for links and images that look like downloadable attachments
    if not containers_found:
        logger.info("No attachment container found — scanning full page for attachment-like URLs")

        for a_tag in driver.find_elements(By.TAG_NAME, "a"):
            href = a_tag.get_attribute("href")
            if href and href not in seen and _looks_like_attachment_url(href):
                seen.add(href)
                attachment_urls.append(href)

        for img_tag in driver.find_elements(By.TAG_NAME, "img"):
            src = img_tag.get_attribute("src")
            if src and src not in seen and not src.startswith("data:"):
                # Only include images that are reasonably large (skip icons)
                try:
                    w = img_tag.get_attribute("naturalWidth")
                    h = img_tag.get_attribute("naturalHeight")
                    if w and h and int(w) > 100 and int(h) > 100:
                        seen.add(src)
                        attachment_urls.append(src)
                except (ValueError, TypeError):
                    # Can't determine size — include it anyway
                    if _looks_like_attachment_url(src):
                        seen.add(src)
                        attachment_urls.append(src)"""

extract_replacement = """    # Strategy B: Search for exact "file category" or "file sequence" labels
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
                attachment_urls.append(src)"""

code = code.replace(extract_target, extract_replacement)

# 2. Update search_and_extract_ftir for complex navigation
search_target = """    try:
        # Wait for the page to load
        WebDriverWait(driver, _PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # Let JS settle

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
        logger.info(f"Found search box. Searching for FTIR {ftir_no}...")
        search_box.clear()
        search_box.send_keys(ftir_no)
        
        from selenium.webdriver.common.keys import Keys
        search_box.send_keys(Keys.RETURN)"""

search_replacement = """    try:
        # Wait for the page to load
        WebDriverWait(driver, _PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # Let JS settle

        # Navigate click-path: Search Menu -> Quick Search
        logger.info("Attempting to navigate through Search -> Quick Search menus...")
        try:
            # 1. Look for 'search' menu bar
            search_menu = driver.find_element(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search') and not(self::input)]")
            search_menu.click()
            time.sleep(1)
            # 2. Look for 'quick search' link
            quick_search = driver.find_element(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'quick search')]")
            quick_search.click()
            time.sleep(2)
        except NoSuchElementException:
            logger.warning("Could not find exact 'Search' -> 'Quick Search' menu path. Trying to find search box anyway.")

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
        search_box.send_keys(Keys.RETURN)"""

code = code.replace(search_target, search_replacement)

# 3. Update process_ftir logging
process_target = """    # ── Extract page ───────────────────────────────────────────────────
    try:
        page_info = extract_ftir_page(driver, url)
        attachment_urls = page_info["attachment_urls"]
    except Exception as e:
        logger.warning(f"FTIR {ftir_no}: Initial extraction failed ({e}). Triggering fallback search.")
        attachment_urls = []
        page_info = {"subject_text": None}

    if not attachment_urls:
        logger.warning(f"FTIR {ftir_no}: No attachments found via direct URL. Attempting portal search fallback.")
        fallback_info = search_and_extract_ftir(driver, ftir_no)
        attachment_urls = fallback_info["attachment_urls"]"""

process_replacement = """    # ── Extract page ───────────────────────────────────────────────────
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

code = code.replace(process_target, process_replacement)

with open("src/browser_extract.py", "w") as f:
    f.write(code)

print("Patch applied to src/browser_extract.py")
