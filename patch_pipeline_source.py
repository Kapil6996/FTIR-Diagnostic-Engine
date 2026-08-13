import re

with open("src/pipeline.py", "r") as f:
    code = f.read()

# Add to result_cols
target_cols = """    result_cols = {
        "Defect_Type":          [],
        "SBPR_Number":          [],
        "Reason":               [],
        "Metadata_Confidence":  [],
        "Image_Confidence":     [],
        "Flag_For_Review":      [],
        "Pipeline_Status":      [],
        "Failure_Stage":        [],
        "Failure_Diagnostics":  [],
    }"""
replacement_cols = """    result_cols = {
        "Defect_Type":          [],
        "SBPR_Number":          [],
        "Reason":               [],
        "Metadata_Confidence":  [],
        "Image_Confidence":     [],
        "Flag_For_Review":      [],
        "Pipeline_Status":      [],
        "Failure_Stage":        [],
        "Failure_Diagnostics":  [],
        "Extraction_Source":    [],
    }"""
code = code.replace(target_cols, replacement_cols)

# Extract from browser_extract
target_extract = """                if extract_result.get("skipped"):
                    logger.debug(f"{row_label}: Attachments already downloaded (resumability skip)")
                else:
                    n_files = len(extract_result.get("downloaded_files", []))
                    logger.info(f"{row_label}: Downloaded {n_files} attachment(s)")"""
replacement_extract = """                extraction_source = extract_result.get("extraction_source", "Unknown")
                if extract_result.get("skipped"):
                    logger.debug(f"{row_label}: Attachments already downloaded (resumability skip)")
                    extraction_source = "Local Cache"
                else:
                    n_files = len(extract_result.get("downloaded_files", []))
                    logger.info(f"{row_label}: Downloaded {n_files} attachment(s)")"""
code = code.replace(target_extract, replacement_extract)

# Default extraction source at the start of loop
target_loop_start = """        pipeline_status = "SUCCESS"
        failure_stage = "None"
        failure_diagnostics = "All pipeline stages executed cleanly."

        # ── 4a. Browser extraction (with resumability) ─────────────────"""
replacement_loop_start = """        pipeline_status = "SUCCESS"
        failure_stage = "None"
        failure_diagnostics = "All pipeline stages executed cleanly."
        extraction_source = "None"

        # ── 4a. Browser extraction (with resumability) ─────────────────"""
code = code.replace(target_loop_start, replacement_loop_start)

# Append to result_cols
target_append = """        result_cols["Defect_Type"].append(defect_type)
        result_cols["SBPR_Number"].append(fused["final_sbpr"])
        result_cols["Reason"].append(fused["reason"])
        result_cols["Metadata_Confidence"].append(fused["metadata_confidence"])
        result_cols["Image_Confidence"].append(fused["image_confidence"])
        result_cols["Flag_For_Review"].append(fused["flag_for_review"])
        result_cols["Pipeline_Status"].append(pipeline_status)
        result_cols["Failure_Stage"].append(failure_stage)
        result_cols["Failure_Diagnostics"].append(failure_diagnostics)"""
replacement_append = """        result_cols["Defect_Type"].append(defect_type)
        result_cols["SBPR_Number"].append(fused["final_sbpr"])
        result_cols["Reason"].append(fused["reason"])
        result_cols["Metadata_Confidence"].append(fused["metadata_confidence"])
        result_cols["Image_Confidence"].append(fused["image_confidence"])
        result_cols["Flag_For_Review"].append(fused["flag_for_review"])
        result_cols["Pipeline_Status"].append(pipeline_status)
        result_cols["Failure_Stage"].append(failure_stage)
        result_cols["Failure_Diagnostics"].append(failure_diagnostics)
        result_cols["Extraction_Source"].append(extraction_source)"""
code = code.replace(target_append, replacement_append)

# Add to early returns for override/non_rust
target_override1 = """            result_cols["Pipeline_Status"].append("LEARNING_OVERRIDE")
            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(
                f"Human correction applied. Original error type: {ctype}"
            )"""
replacement_override1 = """            result_cols["Pipeline_Status"].append("LEARNING_OVERRIDE")
            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(
                f"Human correction applied. Original error type: {ctype}"
            )
            result_cols["Extraction_Source"].append(extraction_source)"""
code = code.replace(target_override1, replacement_override1)

target_override2 = """            result_cols["Pipeline_Status"].append("LEARNING_OVERRIDE")
            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(f"Human correction applied via similarity. Error type: {ctype}")"""
replacement_override2 = """            result_cols["Pipeline_Status"].append("LEARNING_OVERRIDE")
            result_cols["Failure_Stage"].append("None")
            result_cols["Failure_Diagnostics"].append(f"Human correction applied via similarity. Error type: {ctype}")
            result_cols["Extraction_Source"].append(extraction_source)"""
code = code.replace(target_override2, replacement_override2)

target_nonrust = """            result_cols["Pipeline_Status"].append(pipeline_status)
            result_cols["Failure_Stage"].append(failure_stage)
            result_cols["Failure_Diagnostics"].append(failure_diagnostics)
            logger.info(f"{row_label}: → NON-RUST ({rust_confidence:.0%}) — skipping SBPR")"""
replacement_nonrust = """            result_cols["Pipeline_Status"].append(pipeline_status)
            result_cols["Failure_Stage"].append(failure_stage)
            result_cols["Failure_Diagnostics"].append(failure_diagnostics)
            result_cols["Extraction_Source"].append(extraction_source)
            logger.info(f"{row_label}: → NON-RUST ({rust_confidence:.0%}) — skipping SBPR")"""
code = code.replace(target_nonrust, replacement_nonrust)

# Pass to progress callback
target_callback = """                "failure_diagnostics": failure_diagnostics, "ftir_no": ftir_no
            })"""
replacement_callback = """                "failure_diagnostics": failure_diagnostics, "ftir_no": ftir_no,
                "extraction_source": extraction_source
            })"""
code = code.replace(target_callback, replacement_callback)

with open("src/pipeline.py", "w") as f:
    f.write(code)

print("pipeline.py updated")
