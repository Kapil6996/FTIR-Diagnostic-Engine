import re

with open("src/browser_extract.py", "r") as f:
    code = f.read()

target = """    # Resolve relative URLs to absolute
    attachment_urls = [urljoin(base_url, u) for u in attachment_urls]"""

replacement = """    # Strategy D: The URL Sequence Guesser
    # The user noted images follow a strict pattern: ...&fileCategory=1&fileSequence=1&...
    # If we find EVEN ONE URL matching this, we can mathematically generate the rest!
    guessed_urls = []
    
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
    attachment_urls = [urljoin(base_url, u) for u in attachment_urls]"""

code = code.replace(target, replacement)

with open("src/browser_extract.py", "w") as f:
    f.write(code)

print("Patch applied to src/browser_extract.py")
