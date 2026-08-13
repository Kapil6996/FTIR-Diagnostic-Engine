import re

with open("src/pipeline.py", "r") as f:
    code = f.read()

# 1. Remove Extraction_Source from result_cols
target_cols = """        "Failure_Stage":        [],
        "Failure_Diagnostics":  [],
        "Extraction_Source":    [],
    }"""
replacement_cols = """        "Failure_Stage":        [],
        "Failure_Diagnostics":  [],
    }"""
code = code.replace(target_cols, replacement_cols)

# 2. Remove Extraction_Source append
target_append = """        result_cols["Pipeline_Status"].append(pipeline_status)
        result_cols["Failure_Stage"].append(failure_stage)
        result_cols["Failure_Diagnostics"].append(failure_diagnostics)
        result_cols["Extraction_Source"].append(extraction_source)"""
replacement_append = """        result_cols["Pipeline_Status"].append(pipeline_status)
        result_cols["Failure_Stage"].append(failure_stage)
        result_cols["Failure_Diagnostics"].append(failure_diagnostics)"""
code = code.replace(target_append, replacement_append)

# 3. Remove Extraction_Source append in override
target_override = """            result_cols["Failure_Diagnostics"].append(
                f"Human correction applied. Original error type: {ctype}"
            )
            result_cols["Extraction_Source"].append(extraction_source)"""
replacement_override = """            result_cols["Failure_Diagnostics"].append(
                f"Human correction applied. Original error type: {ctype}"
            )"""
code = code.replace(target_override, replacement_override)

target_override2 = """            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(f"Human correction applied via similarity. Error type: {ctype}")
            result_cols["Extraction_Source"].append(extraction_source)"""
replacement_override2 = """            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(f"Human correction applied via similarity. Error type: {ctype}")"""
code = code.replace(target_override2, replacement_override2)

target_nonrust = """            result_cols["Failure_Stage"].append(failure_stage)
            result_cols["Failure_Diagnostics"].append(failure_diagnostics)
            result_cols["Extraction_Source"].append(extraction_source)
            logger.info(f"{row_label}: → NON-RUST ({rust_confidence:.0%}) — skipping SBPR")"""
replacement_nonrust = """            result_cols["Failure_Stage"].append(failure_stage)
            result_cols["Failure_Diagnostics"].append(failure_diagnostics)
            logger.info(f"{row_label}: → NON-RUST ({rust_confidence:.0%}) — skipping SBPR")"""
code = code.replace(target_nonrust, replacement_nonrust)

# 4. Change headless=True to headless=False so the browser is visible
target_driver = """            from src.browser_extract import get_driver, process_ftir
            browser_driver = get_driver(profile_dir=profile_dir, headless=True)
            logger.info("  ✓ Browser driver initialised")"""
replacement_driver = """            from src.browser_extract import get_driver, process_ftir
            # Set headless=False so the user can visually confirm the browser is opening the FTIR
            browser_driver = get_driver(profile_dir=profile_dir, headless=False)
            logger.info("  ✓ Browser driver initialised (Visible Mode)")"""
code = code.replace(target_driver, replacement_driver)

with open("src/pipeline.py", "w") as f:
    f.write(code)

print("pipeline.py updated for visible browser and removed excel column")
