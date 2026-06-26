#!/usr/bin/env python3
"""
Quick EDA to diagnose embeddings generation issue
"""

import pandas as pd

INPUT_CSV = "../data/input/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36.csv"
OUTPUT_CSV = "../data/output/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36_embeddings.csv"

print("=" * 80)
print("INPUT CSV ANALYSIS")
print("=" * 80)
df_input = pd.read_csv(INPUT_CSV)
print(f"Total rows: {len(df_input)}")
print(f"Unique sha1s: {df_input['sha1'].nunique()}")
print(f"Duplicate rows by sha1: {len(df_input) - df_input['sha1'].nunique()}")
print(f"\nColumns: {list(df_input.columns)}")

# Check for PDFs vs non-PDFs
pdf_mask = df_input["gdrive_name"].str.endswith(".pdf", na=False)
print(f"\nPDF files: {pdf_mask.sum()}")
print(f"Non-PDF files: {(~pdf_mask).sum()}")

# Check OCR text availability
has_ocr = ~(df_input["ocr_text_per_page"].fillna("") == "")
print(f"Has OCR text: {has_ocr.sum()}")
print(f"Missing OCR text: {(~has_ocr).sum()}")

print("\n" + "=" * 80)
print("OUTPUT CSV ANALYSIS")
print("=" * 80)
df_output = pd.read_csv(OUTPUT_CSV)
print(f"Total rows: {len(df_output)}")
print(f"Unique sha1s: {df_output['sha1'].nunique()}")
print(f"Duplicate rows by sha1: {len(df_output) - df_output['sha1'].nunique()}")
print(f"\nColumns: {list(df_output.columns)}")

if "embeddings_generated" in df_output.columns:
    print(f"\nembeddings_generated = True: {(df_output['embeddings_generated']).sum()}")
    print(f"embeddings_generated = False: {(not df_output['embeddings_generated']).sum()}")

# Check embedding validity
print("\n" + "=" * 80)
print("EMBEDDING VALIDITY CHECK")
print("=" * 80)

if "embedding" in df_output.columns:
    # Sample 10 rows with PDFs
    df_pdfs = df_output[df_output["gdrive_name"].str.endswith(".pdf", na=False)]
    sample_pdfs = df_pdfs.head(10)

    print(f"\nChecking {len(sample_pdfs)} PDF documents:")
    for _idx, row in sample_pdfs.iterrows():
        embedding_str = row["embedding"]
        try:
            # Try to parse embedding
            if pd.notna(embedding_str):
                # Convert string representation to array
                if isinstance(embedding_str, str):
                    # Remove brackets and split
                    embedding_str = embedding_str.strip("[]")
                    values = [float(x) for x in embedding_str.split()[:5]]  # First 5 values
                    is_zero = all(v == 0 for v in values)
                    print(f"  {row['gdrive_name'][:50]}: {values[:5]}... (all zeros: {is_zero})")
                else:
                    print(f"  {row['gdrive_name'][:50]}: Embedding is not a string")
            else:
                print(f"  {row['gdrive_name'][:50]}: Embedding is null")
        except Exception as e:
            print(f"  {row['gdrive_name'][:50]}: Error parsing - {e}")

# Check for duplicate sha1s in output
print("\n" + "=" * 80)
print("DUPLICATE SHA1 CHECK")
print("=" * 80)
duplicate_sha1s = df_output[df_output.duplicated(subset=["sha1"], keep=False)]
if len(duplicate_sha1s) > 0:
    print(f"Found {len(duplicate_sha1s)} rows with duplicate sha1s!")
    print(f"Unique duplicated sha1s: {duplicate_sha1s['sha1'].nunique()}")
    print("\nExample duplicates:")
    example_sha1 = duplicate_sha1s["sha1"].iloc[0]
    print(
        df_output[df_output["sha1"] == example_sha1][
            ["sha1", "gdrive_name", "embeddings_generated"]
        ]
    )
else:
    print("No duplicate sha1s found - this is good!")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
if len(df_output) > len(df_input):
    print(f"⚠️  OUTPUT has MORE rows ({len(df_output)}) than INPUT ({len(df_input)})")
    print("   This suggests the resume logic created duplicates!")
elif len(df_output) == len(df_input):
    print(f"✓  Row count matches: {len(df_output)} rows")
else:
    print(f"⚠️  OUTPUT has FEWER rows ({len(df_output)}) than INPUT ({len(df_input)})")

if (df_output["embeddings_generated"]).sum() == 10:
    print("⚠️  Only 10 documents marked as processed (from test run)")
    print("   Need to regenerate embeddings for all documents!")
elif (df_output["embeddings_generated"]).sum() == len(df_input):
    print("✓  All input documents marked as processed")
