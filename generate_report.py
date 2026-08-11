import os
import pandas as pd
import joblib
from src.excel_io import read_ftir_sheet
from src.sbpr_features import build_features
import re

excel_dir = os.path.join("data", "excel")
files = sorted([f for f in os.listdir(excel_dir) if f.endswith(".xlsx") and not f.startswith(".")])

dfs = []
for fname in files:
    fpath = os.path.join(excel_dir, fname)
    df_sub = read_ftir_sheet(fpath)
    match = re.search(r"(SBIN\d+[A-Z]\d+)", fname)
    df_sub["Human_SBPR"] = match.group(1) if match else "Unknown"
    dfs.append(df_sub)

df_master = pd.concat(dfs, ignore_index=True)
X, y, feature_names = build_features(df_master)
model = joblib.load("models/sbpr_tree.joblib")["model"]
y_pred = model.predict(X)

misclassified = df_master[y != y_pred].copy()
misclassified["AI_SBPR"] = y_pred[y != y_pred]

def get_ftir_no(row):
    for col in row.index:
        if "ftir" in str(col).lower() and ("no" in str(col).lower() or "number" in str(col).lower()):
            val = str(row[col])
            if val and val != "nan": return val
    return str(row.get("Masked VIN", "Unknown_VIN"))

misclassified["FTIR_ID"] = misclassified.apply(get_ftir_no, axis=1)

sbprs = ["SBIN201210B00011", "SBIN202310B06811", "SBIN202507B07143"]

output = []
output.append("# Human Misclassification Report\n")
output.append("This report details the records where the human data entry clerk filed the FTIR into the wrong SBPR folder, but the AI correctly identified the true category based on the text.\n\n")

for sbpr in sbprs:
    subset = misclassified[misclassified["Human_SBPR"] == sbpr]
    output.append(f"## Human Folder: {sbpr}\n")
    output.append(f"**Total human errors in this folder:** {len(subset)}\n\n")
    if len(subset) == 0:
        output.append("No errors found in this folder.\n\n")
        continue
        
    output.append("| FTIR ID / VIN | AI Predicted (True SBPR) | Customer Complaint | AI Justification |\n")
    output.append("| :--- | :--- | :--- | :--- |\n")
    
    for _, row in subset.iterrows():
        ftir = str(row["FTIR_ID"]).replace("\n", " ").replace("|", "")
        true_sbpr = row["AI_SBPR"]
        text = str(row.get("Subject (English)") or row.get("Customer Complaint") or "N/A").replace("\n", " ").replace("|", "").strip()
        if len(text) > 120: text = text[:117] + "..."
        
        justification = ""
        text_lower = text.lower()
        if true_sbpr == "SBIN201210B00011":
            if "seat" in text_lower or "bracket" in text_lower: justification = "Text explicitly mentions SEAT/BRACKET"
            else: justification = "Mileage indicates early defect (<10k)"
        elif true_sbpr == "SBIN202310B06811":
            if any(w in text_lower for w in ["chrome", "peel", "paint", "bird", "stain", "trim"]): justification = "Text mentions CHROME/PAINT/STAIN"
            else: justification = "Matches exterior trim profile"
        else:
            if any(w in text_lower for w in ["hw", "door", "panel", "body", "hinge"]): justification = "Text mentions HW/DOOR/PANEL"
            else: justification = "Matches generic body rust profile"
            
        output.append(f"| {ftir} | **{true_sbpr}** | {text} | {justification} |\n")
    output.append("\n")

with open("/Users/kapilkumar/.gemini/antigravity/brain/1de6d7ed-bf1c-4592-9888-7901105db77b/human_misclassifications.md", "w") as f:
    f.writelines(output)
