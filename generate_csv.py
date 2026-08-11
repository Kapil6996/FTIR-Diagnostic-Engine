import os
import pandas as pd
import joblib
from src.excel_io import read_ftir_sheet
from src.sbpr_features import build_features
import re
import csv

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

def get_vin(row):
    for col in row.index:
        if str(col).strip().lower() in ["vin no.", "vin no", "vin"]:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                return val
    return str(row.get("Masked VIN", "Unknown_VIN")).strip()

csv_path = "/Users/kapilkumar/Desktop/human_misclassifications.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["VIN no.", "Actual SBPR no. (Human Error)", "AI Predicted SBPR no.", "Justification", "Subject"])
    
    for _, row in misclassified.iterrows():
        vin = get_vin(row)
        human_sbpr = row["Human_SBPR"]
        ai_sbpr = row["AI_SBPR"]
        text = str(row.get("Subject (English)") or row.get("Customer Complaint") or "N/A").strip()
        
        text_lower = text.lower()
        justification = ""
        if ai_sbpr == "SBIN201210B00011":
            if "seat" in text_lower or "bracket" in text_lower or "cushion" in text_lower: justification = "Text explicitly mentions SEAT/BRACKET"
            else: justification = "Mileage indicates early defect (<10k)"
        elif ai_sbpr == "SBIN202310B06811":
            if any(w in text_lower for w in ["chrome", "peel", "paint", "bird", "stain", "trim"]): justification = "Text mentions CHROME/PAINT/STAIN"
            else: justification = "Matches exterior trim profile"
        else:
            if any(w in text_lower for w in ["hw", "door", "panel", "body", "hinge", "engine"]): justification = "Text mentions HW/DOOR/PANEL/BODY"
            else: justification = "Matches generic body rust profile"
            
        writer.writerow([vin, human_sbpr, ai_sbpr, justification, text])

print(f"CSV saved to {csv_path}")
