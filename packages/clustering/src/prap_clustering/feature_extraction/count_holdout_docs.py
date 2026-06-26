#!/usr/bin/env python3
"""
Count total and OCR-having documents for each holdout agency.
Helps verify what CORPUS_TOTAL_DOCS should be for cost extrapolation.
"""

from pathlib import Path

import pandas as pd

INPUT_DIR = Path("../../data/input")

HOLDOUT_AGENCIES = {
    "Folsom_Police_Department",
    "San_Leandro_Police_Department",
    "Santa_Clara_Police_Department",
    "Hayward_Police_Department",
    "Vallejo_Police_Department",
    "Santa_Ana_Police_Department",
    "Chula_Vista_Police_Department",
    "Irvine_Police_Department",
    "Pasadena_Police_Department",
    "San_Diego_County_Medical_Examiner",
    "Fresno_County_Sheriff",
    "Sacramento_County_Sheriff",
    "San_Francisco_County_Sheriff",
    "Richmond_Police_Department",
    "Los_Angeles_District_Attorney",
    "San_Francisco_Police_Commission",
    "Riverside_County_Department_of_Public_Social_Services",
    "Cal_State_East_Bay_University_Police_Department",
    "San_Joaquin_County_Medical_Examiner",
    "Santa_Monica_Police_Department",
    "Kern_County_Sheriff",
    "Santa_Clara_County_Sheriff",
    "Shasta_County_District_Attorney",
    "Contra_Costa_County_Sheriff",
    "Contra_Costa_County_District_Attorney",
    "UC_Davis_Police_Department",
    "Seal_Beach_Police_Department",
    "Office_of_Inspector_General_for_Prisons",
    "Bakersfield_Police_Department",
    "California_Department_of_Corrections_and_Rehabilitation",
    "California_Department_of_Justice",
}

all_csvs = list(INPUT_DIR.glob("*.csv"))

print(f"{'Agency':<55} {'Total rows':>10}  {'With OCR':>10}")
print("-" * 80)

total_rows = 0
total_ocr = 0
not_found = []

for agency_key in sorted(HOLDOUT_AGENCIES):
    agency_name = agency_key.replace("_", " ")
    matches = [f for f in all_csvs if agency_name in f.name]

    if not matches:
        not_found.append(agency_name)
        print(f"{'  NOT FOUND: ' + agency_name:<55}")
        continue

    csv_path = matches[0]  # take first match if multiple
    df = pd.read_csv(csv_path, low_memory=False)

    n_total = len(df)
    if "ocr_text_per_page" in df.columns:
        n_ocr = (~(df["ocr_text_per_page"].fillna("") == "")).sum()
    else:
        n_ocr = 0
        print(f"  WARNING: no ocr_text_per_page column in {csv_path.name}")

    total_rows += n_total
    total_ocr += n_ocr
    print(f"{agency_name:<55} {n_total:>10,}  {n_ocr:>10,}")

print("-" * 80)
print(f"{'TOTAL':<55} {total_rows:>10,}  {total_ocr:>10,}")

if not_found:
    print(f"\nNot found ({len(not_found)}): {not_found}")
