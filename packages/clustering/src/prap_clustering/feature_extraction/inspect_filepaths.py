"""
Show actual filepaths where case IDs are being extracted.
"""
import pandas as pd

df = pd.read_csv("../../data/output/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36_with_extracted_features_improved_regex.csv")

# Find examples of the most common case ID
target_ids = ['17-0132', '18-0249', '15-0436']

print("EXAMPLES OF FILEPATHS WITH EXTRACTED CASE IDs:")
print("="*80)

for target_id in target_ids:
    print(f"\nCase ID: {target_id}")
    print("-"*80)

    # Find rows with this case ID in filepath
    rows = df[df['extracted_case_ids_fp'].str.contains(target_id, na=False, regex=False)]

    # Show first 5 examples
    for _idx, row in rows.head(5).iterrows():
        filepath = row.get('gdrive_path', '')
        filename = row.get('gdrive_name', '')
        print(f"Path: {filepath}")
        print(f"File: {filename}")
        print()
