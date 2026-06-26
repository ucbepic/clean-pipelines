#!/usr/bin/env python3
"""
Filename feature extraction (baseline comparison).

Extracts features from filepath only using regex patterns.
Uses regex_extract_fp_fn.py functions for dates, case IDs, and names.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .regex_extract_fp_fn import (
    extract_date_from_metadata,
    extract_ids_from_metadata,
    extract_names_from_metadata,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = "../../data/input/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36.csv"
# OUTPUT_CSV = "../../data/output/test.csv"

OUTPUT_CSV = "../../data/output/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36_metadata.csv"
FORCE_REGENERATE = False
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extract_filename_features.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def extract_features_from_path(gdrive_path: str, gdrive_name: str) -> dict:
    """
    Extract features from directory path and filename separately using regex patterns.

    Args:
        gdrive_path: Full file path (e.g., "Oakland Police Department/.../01-20621 OIS Offense Report.pdf")
        gdrive_name: Filename (e.g., "01-20621 OIS Offense Report.pdf")

    Returns:
        Dictionary with extracted features from both filepath (fp) and filename (fn)
    """
    features = {
        "extracted_dates_fp": None,
        "extracted_dates_fn": None,
        "extracted_case_ids_fp": None,
        "extracted_case_ids_fn": None,
        "extracted_names_fp": None,
        "extracted_names_fn": None,
    }

    # Split gdrive_path into directory path (everything except filename)
    if gdrive_path:
        # Get directory path (everything before the last slash)
        last_slash = gdrive_path.rfind("/")
        if last_slash != -1:
            directory_path = gdrive_path[:last_slash]  # Everything before last slash
        else:
            directory_path = ""  # No slash found, no directory
    else:
        directory_path = ""

    # Extract from directory path (fp)
    if directory_path:
        try:
            dates_fp = extract_date_from_metadata(directory_path)
            features["extracted_dates_fp"] = str(dates_fp) if dates_fp else None

            case_ids_fp = extract_ids_from_metadata(directory_path)
            features["extracted_case_ids_fp"] = str(case_ids_fp) if case_ids_fp else None

            names_fp = extract_names_from_metadata(directory_path)
            features["extracted_names_fp"] = str(names_fp) if names_fp else None
        except Exception as e:
            logger.error(f"Error extracting features from directory path '{directory_path}': {e}")

    # Extract from filename (fn)
    if gdrive_name:
        try:
            dates_fn = extract_date_from_metadata(gdrive_name)
            features["extracted_dates_fn"] = str(dates_fn) if dates_fn else None

            case_ids_fn = extract_ids_from_metadata(gdrive_name)
            features["extracted_case_ids_fn"] = str(case_ids_fn) if case_ids_fn else None

            names_fn = extract_names_from_metadata(gdrive_name)
            features["extracted_names_fn"] = str(names_fn) if names_fn else None
        except Exception as e:
            logger.error(f"Error extracting features from filename '{gdrive_name}': {e}")

    return features


def calculate_coverage_stats(df: pd.DataFrame) -> dict:
    """Calculate feature coverage statistics."""
    total = len(df)
    stats = {
        "total_documents": total,
        "documents_with_dates_fp": 0,
        "documents_with_dates_fn": 0,
        "documents_with_dates_any": 0,
        "documents_with_case_ids_fp": 0,
        "documents_with_case_ids_fn": 0,
        "documents_with_case_ids_any": 0,
        "documents_with_names_fp": 0,
        "documents_with_names_fn": 0,
        "documents_with_names_any": 0,
        "documents_with_any_feature": 0,
        "documents_with_no_features": 0,
    }

    if total == 0:
        return stats

    for _idx, row in df.iterrows():
        # Check for dates
        has_dates_fp = row.get("extracted_dates_fp") not in [None, "None", "[]", ""]
        has_dates_fn = row.get("extracted_dates_fn") not in [None, "None", "[]", ""]
        has_dates_any = has_dates_fp or has_dates_fn

        # Check for case IDs
        has_case_ids_fp = row.get("extracted_case_ids_fp") not in [None, "None", "[]", ""]
        has_case_ids_fn = row.get("extracted_case_ids_fn") not in [None, "None", "[]", ""]
        has_case_ids_any = has_case_ids_fp or has_case_ids_fn

        # Check for names
        has_names_fp = row.get("extracted_names_fp") not in [None, "None", "[]", ""]
        has_names_fn = row.get("extracted_names_fn") not in [None, "None", "[]", ""]
        has_names_any = has_names_fp or has_names_fn

        if has_dates_fp:
            stats["documents_with_dates_fp"] += 1
        if has_dates_fn:
            stats["documents_with_dates_fn"] += 1
        if has_dates_any:
            stats["documents_with_dates_any"] += 1

        if has_case_ids_fp:
            stats["documents_with_case_ids_fp"] += 1
        if has_case_ids_fn:
            stats["documents_with_case_ids_fn"] += 1
        if has_case_ids_any:
            stats["documents_with_case_ids_any"] += 1

        if has_names_fp:
            stats["documents_with_names_fp"] += 1
        if has_names_fn:
            stats["documents_with_names_fn"] += 1
        if has_names_any:
            stats["documents_with_names_any"] += 1

        if has_dates_any or has_case_ids_any or has_names_any:
            stats["documents_with_any_feature"] += 1
        else:
            stats["documents_with_no_features"] += 1

    return stats


def print_coverage_stats(stats: dict):
    """Print coverage statistics in a readable format."""
    total = stats["total_documents"]
    if total == 0:
        logger.info("No documents processed")
        return

    logger.info(f"\n{'=' * 80}")
    logger.info("FEATURE COVERAGE STATISTICS")
    logger.info(f"Total documents: {total}")
    logger.info("\nDates:")
    logger.info(
        f"  From directory path (fp): {stats['documents_with_dates_fp']} ({stats['documents_with_dates_fp'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  From filename (fn): {stats['documents_with_dates_fn']} ({stats['documents_with_dates_fn'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  From either: {stats['documents_with_dates_any']} ({stats['documents_with_dates_any'] / total * 100:.1f}%)"
    )
    logger.info("\nCase IDs:")
    logger.info(
        f"  From directory path (fp): {stats['documents_with_case_ids_fp']} ({stats['documents_with_case_ids_fp'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  From filename (fn): {stats['documents_with_case_ids_fn']} ({stats['documents_with_case_ids_fn'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  From either: {stats['documents_with_case_ids_any']} ({stats['documents_with_case_ids_any'] / total * 100:.1f}%)"
    )
    logger.info("\nNames:")
    logger.info(
        f"  From directory path (fp): {stats['documents_with_names_fp']} ({stats['documents_with_names_fp'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  From filename (fn): {stats['documents_with_names_fn']} ({stats['documents_with_names_fn'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  From either: {stats['documents_with_names_any']} ({stats['documents_with_names_any'] / total * 100:.1f}%)"
    )
    logger.info("\nOverall:")
    logger.info(
        f"  Documents with ANY feature: {stats['documents_with_any_feature']} ({stats['documents_with_any_feature'] / total * 100:.1f}%)"
    )
    logger.info(
        f"  Documents with NO features: {stats['documents_with_no_features']} ({stats['documents_with_no_features'] / total * 100:.1f}%)"
    )
    logger.info(f"{'=' * 80}")


def main():
    """Main entry point for filename feature extraction."""
    logger.info("=" * 80)
    logger.info("FILENAME FEATURE EXTRACTION - Phase 2b")
    logger.info("=" * 80)
    logger.info(f"Input CSV: {INPUT_CSV}")
    logger.info(f"Output CSV: {OUTPUT_CSV}")
    logger.info(f"Force regeneration: {FORCE_REGENERATE}")

    # Load input CSV
    logger.info("Loading input CSV...")
    try:
        df = pd.read_csv(INPUT_CSV)

        if "ocr_text_per_page" in df.columns:
            df = df.drop(columns=["ocr_text_per_page"])

        # Filter out rows without OCR text
        # df = df[~((df.ocr_text_per_page.fillna("") == ""))]
        # df = df.tail(25)
        logger.info(f"Loaded {len(df)} documents with OCR text")
    except Exception as e:
        logger.error(f"Failed to load input CSV: {e}")
        sys.exit(1)

    # Check if feature columns already exist
    feature_cols = [
        "extracted_dates_fp",
        "extracted_dates_fn",
        "extracted_case_ids_fp",
        "extracted_case_ids_fn",
        "extracted_names_fp",
        "extracted_names_fn",
    ]
    if all(col in df.columns for col in feature_cols) and not FORCE_REGENERATE:
        logger.info("Feature columns already exist. Set FORCE_REGENERATE=True to re-extract.")
        sys.exit(0)

    # Process documents
    start_time = time.time()
    errors = 0

    logger.info(f"Processing {len(df)} documents...")

    # Extract features for all documents
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            # Get gdrive_path and gdrive_name
            gdrive_path = row.get("gdrive_path", "")
            gdrive_name = row.get("gdrive_name", "")

            # Extract features from both directory path and filename
            features = extract_features_from_path(gdrive_path, gdrive_name)

            # Update dataframe
            for key, value in features.items():
                df.at[idx, key] = value

        except Exception as e:
            logger.error(f"Error processing document {idx}: {e}")
            errors += 1

    # Add extraction flag
    df["filename_features_extracted"] = True

    # Calculate coverage statistics
    stats = calculate_coverage_stats(df)

    # Filter output to only relevant columns
    output_columns = [
        # Ground truth and identifiers
        "provisional_case_name",
        "gdrive_id",
        "sha1",
        # Path information
        "gdrive_path",
        "gdrive_name",
        # Extracted features from directory path (fp)
        "extracted_dates_fp",
        "extracted_case_ids_fp",
        "extracted_names_fp",
        # Extracted features from filename (fn)
        "extracted_dates_fn",
        "extracted_case_ids_fn",
        "extracted_names_fn",
        # Extraction flag
        "filename_features_extracted",
    ]

    # Only keep columns that exist in the dataframe
    existing_output_columns = [col for col in output_columns if col in df.columns]
    df_output = df[existing_output_columns]

    # Save output
    logger.info("Saving output...")
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_CSV, index=False)

    # Summary statistics
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'=' * 80}")
    logger.info("FILENAME FEATURE EXTRACTION COMPLETE")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    if len(df) > 0:
        logger.info(f"Average time per document: {elapsed_time / len(df):.4f} seconds")
    logger.info(f"Errors encountered: {errors}")
    logger.info(f"Output saved to: {OUTPUT_CSV}")

    # Print coverage statistics
    print_coverage_stats(stats)
