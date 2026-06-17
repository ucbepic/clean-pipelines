"""
Quick script to inspect what case IDs are being extracted from filepaths.
"""
from collections import Counter

import pandas as pd

# Load the new extraction results
df = pd.read_csv("../../data/output/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36_with_extracted_features_improved_regex.csv")

# Get all case IDs from filepath
all_case_ids_fp = []
for val in df['extracted_case_ids_fp'].dropna():
    if str(val).strip():
        ids = str(val).split(',')
        all_case_ids_fp.extend([id.strip() for id in ids if id.strip()])

# Count frequencies
counter = Counter(all_case_ids_fp)

print("TOP 50 MOST COMMON CASE IDs FROM FILEPATH:")
print("="*80)
for case_id, count in counter.most_common(50):
    print(f"{case_id:30s} | {count:5d} occurrences")

print("\n" + "="*80)
print(f"Total unique case IDs: {len(counter)}")
print(f"Total case ID extractions: {sum(counter.values())}")

# Look for date-like patterns
print("\n" + "="*80)
print("LIKELY DATE PATTERNS (YYYY-MM-DD):")
print("="*80)
import re

date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
date_like = [case_id for case_id in counter.keys() if date_pattern.match(case_id)]
for case_id in sorted(date_like)[:30]:
    print(f"{case_id} | {counter[case_id]} occurrences")

print(f"\nTotal date-like extractions: {len(date_like)}")
print(f"Total count: {sum(counter[c] for c in date_like)}")
