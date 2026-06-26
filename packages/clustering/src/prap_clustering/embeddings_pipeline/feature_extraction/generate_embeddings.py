#!/usr/bin/env python3
"""
Embedding generation (baseline comparison).

Generates document embeddings from OCR text using sentence-transformers.
Model: all-MiniLM-L6-v2 (384 dimensions)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from prap_clustering.embeddings import _get_embedding_model

# List of all CSV files to process


AGENCIES = [
    ### hold out
    ## run 1
    # {
    #     'name': 'Ontario Police Department',
    #     'csv_path': '../data/input/autofolio_1.2.0_output--Ontario Police Department--2025-04-24_16-59-27 - autofolio_1.2.0_output--Ontario Police Department--2025-04-24_16-59-27.csv',
    # },
    {
        "name": "Bakersfield Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Bakersfield Police Department--2025-04-09_22-08-24 - autofolio_1.2.0_output--Bakersfield Police Department--2025-04-09_22-08-24.csv",
    },
    {
        "name": "Santa Monica Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Monica Police Department--2024-12-21_02-18-20 - autofolio_1.1.0_output--Santa Monica Police Department--2024-12-21_02-18-20.csv",
    },
    {
        "name": "Richmond Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Richmond Police Department--2025-04-09_20-59-59 - autofolio_1.2.0_output--Richmond Police Department--2025-04-09_20-59-59.csv",
    },
    {
        "name": "Los Angeles District Attorney",
        "csv_path": "../data/input/autofolio_1.1.0_output--Los Angeles District Attorney--2024-11-27_05-47-23 - autofolio_1.1.0_output--Los Angeles District Attorney--2024-11-27_05-47-23.csv",
    },
    {
        "name": "California Department of Justice",
        "csv_path": "../data/input/autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52 - autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52.csv",
    },
    {
        "name": "Office of Inspector General for Prisons",
        "csv_path": "../data/input/autofolio_1.2.0_output--Office of Inspector General for Prisons--2025-04-27_21-48-50 - autofolio_1.2.0_output--Office of Inspector General for Prisons--2025-04-27_21-48-50.csv",
    },
    {
        "name": "Santa Ana Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Ana Police Department--2025-02-13_01-55-05 - autofolio_1.1.0_output--Santa Ana Police Department--2025-02-13_01-55-05.csv",
    },
    {
        "name": "San Francisco Police Commission",
        "csv_path": "../data/input/autofolio_1.2.0_output--San Francisco Police Commission--2025-04-09_21-20-14 - autofolio_1.2.0_output--San Francisco Police Commission--2025-04-09_21-20-14.csv",
    },
    {
        "name": "Kern County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Kern County Sheriff--2024-07-15_23-59-08 - autofolio_1.1.0_output--Kern County Sheriff--2024-07-15_23-59-08.csv",
    },
    {
        "name": "Santa Clara County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Clara County Sheriff--2024-12-13_15-09-21 - autofolio_1.1.0_output--Santa Clara County Sheriff--2024-12-13_15-09-21.csv",
    },
    {
        "name": "Fresno County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42 - autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42.csv",
    },
    {
        "name": "Sacramento County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Sacramento County Sheriff--2025-03-04_10-11-24 - autofolio_1.1.0_output--Sacramento County Sheriff--2025-03-04_10-11-24.csv",
    },
    {
        "name": "San Francisco County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Francisco County Sheriff--2024-07-20_01-57-38 - autofolio_1.1.0_output--San Francisco County Sheriff--2024-07-20_01-57-38.csv",
    },
    {
        "name": "California Department of Corrections and Rehabilitation",
        "csv_path": "../data/input/OLDautofolio_1.2.0_output--California Department of Corrections and Rehabilitation--2025-03-24_23-57-31 - autofolio_1.2.0_output--California Department of Corrections and Rehabilitation--2025-03-24_23-57-31.csv",
    },
    # {
    #     'name': 'Orange County District Attorney',
    #     'csv_path': '../data/input/autofolio_1.2.0_output--Orange County District Attorney--2025-03-13_22-40-21 - autofolio_1.2.0_output--Orange County District Attorney--2025-03-13_22-40-21.csv',
    # },
    # run 2
    {
        "name": "Folsom Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Folsom Police Department--2025-02-27_23-14-01 - autofolio_1.1.0_output--Folsom Police Department--2025-02-27_23-14-01.csv",
    },
    # {
    #     'name': 'Brentwood Police Department',
    #     'csv_path': '../data/input/autofolio_1.1.0_output--Brentwood Police Department--2024-09-30_20-03-15 - autofolio_1.1.0_output--Brentwood Police Department--2024-09-30_20-03-15.csv',
    # },
    {
        "name": "UC Davis Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--UC Davis Police Department--2024-07-20_08-53-00 - autofolio_1.1.0_output--UC Davis Police Department--2024-07-20_08-53-00.csv",
    },
    # {
    #     'name': 'San Ramon Police Department',
    #     'csv_path': '../data/input/OLDautofolio_1.1.0_output--San Ramon Police Department--2024-07-16_01-32-12 - autofolio_1.1.0_output--San Ramon Police Department--2024-07-16_01-32-12.csv',
    # },
    {
        "name": "Seal Beach Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Seal Beach Police Department--2025-02-13_08-16-22 - autofolio_1.1.0_output--Seal Beach Police Department--2025-02-13_08-16-22.csv",
    },
    {
        "name": "Contra Costa County District Attorney",
        "csv_path": "../data/input/autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04 - autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04.csv",
    },
    {
        "name": "Contra Costa County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Contra Costa County Sheriff--2024-12-21_01-47-17 - autofolio_1.1.0_output--Contra Costa County Sheriff--2024-12-21_01-47-17.csv",
    },
    # {
    #     'name': 'Salinas Police Department',
    #     'csv_path': '../data/input/autofolio_1.1.0_output--Salinas Police Department--2024-12-05_13-18-13 - autofolio_1.1.0_output--Salinas Police Department--2024-12-05_13-18-13.csv',
    # },
    {
        "name": "Shasta County District Attorney",
        "csv_path": "../data/input/autofolio_1.1.0_output--Shasta County District Attorney--2024-12-18_05-02-13 - autofolio_1.1.0_output--Shasta County District Attorney--2024-12-18_05-02-13.csv",
    },
    {
        "name": "Riverside County Department of Public Social Services",
        "csv_path": "../data/input/autofolio_1.1.0_output--Riverside County Department of Public Social Services--2024-07-05_20-47-44 - autofolio_1.1.0_output--Riverside County Department of Public Social Services--2024-07-05_20-47-44.csv",
    },
    {
        "name": "Cal State East Bay University Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Cal State East Bay University Police Department--2025-04-09_20-14-32 - autofolio_1.2.0_output--Cal State East Bay University Police Department--2025-04-09_20-14-32.csv",
    },
    {
        "name": "San Joaquin County Medical Examiner",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44 - autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44.csv",
    },
    # {
    #     'name': 'CSU San Jose Police Department',
    #     'csv_path': '../data/input/autofolio_1.1.0_output--CSU San Jose Police Department--2024-05-31_13-46-58 - autofolio_1.1.0_output--CSU San Jose Police Department--2024-05-31_13-46-58.csv',
    # },
    {
        "name": "Pasadena Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Pasadena Police Department--2025-01-24_06-19-29 - autofolio_1.1.0_output--Pasadena Police Department--2025-01-24_06-19-29.csv",
    },
    {
        "name": "Irvine Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Irvine Police Department--2024-07-09_01-45-57 - autofolio_1.1.0_output--Irvine Police Department--2024-07-09_01-45-57.csv",
    },
    {
        "name": "San Diego County Medical Examiner",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Diego County Medical Examiner--2025-01-07_23-53-09 - autofolio_1.1.0_output--San Diego County Medical Examiner--2025-01-07_23-53-09.csv",
    },
    {
        "name": "San Leandro Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Leandro Police Department--2024-08-15_22-50-51 - autofolio_1.1.0_output--San Leandro Police Department--2024-08-15_22-50-51.csv",
    },
    {
        "name": "Santa Clara Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Clara Police Department--2024-06-01_21-38-12 - autofolio_1.1.0_output--Santa Clara Police Department--2024-06-01_21-38-12.csv",
    },
    {
        "name": "Hayward Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37 - autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37.csv",
    },
    {
        "name": "Vallejo Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Vallejo Police Department--2025-03-25_00-20-52 - autofolio_1.2.0_output--Vallejo Police Department--2025-03-25_00-20-52.csv",
    },
    {
        "name": "Chula Vista Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Chula Vista Police Department--2025-02-20_07-43-22 - autofolio_1.1.0_output--Chula Vista Police Department--2025-02-20_07-43-22.csv",
    },
    # {
    #     'name': 'Whittier Police Department',
    #     'csv_path': '../data/input/autofolio_1.1.0_output--Whittier Police Department--2024-08-05_15-28-07 - autofolio_1.1.0_output--Whittier Police Department--2024-08-05_15-28-07.csv',
    # },
    ## hold out
]


OUTPUT_DIR = "../data/output/"
MODEL_NAME = "all-MiniLM-L6-v2"
FORCE_REGENERATE = True
CHECKPOINT_INTERVAL = 100
BATCH_SIZE = 256  # Optimized for RTX 4090 24GB VRAM
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("generate_embeddings.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_ocr_text(ocr_data) -> str:
    """
    Parse OCR text from JSON format to plain text.

    Args:
        ocr_data: String containing JSON array [{"page_number": 1, "text": "..."}, ...]

    Returns:
        Combined text from all pages
    """
    if pd.isna(ocr_data) or not ocr_data:
        return ""

    try:
        if isinstance(ocr_data, str):
            pages = json.loads(ocr_data)
        else:
            pages = ocr_data

        if isinstance(pages, list):
            all_text = []
            for page in pages:
                if isinstance(page, dict) and "text" in page:
                    all_text.append(page["text"])
            return "\n\n".join(all_text)
        else:
            return str(ocr_data)

    except json.JSONDecodeError:
        logger.debug("OCR data is not valid JSON, treating as plain text")
        return str(ocr_data)
    except Exception as e:
        logger.warning(f"Error parsing OCR text: {e}")
        return ""


def process_single_csv(input_csv: str, output_csv: str, model):
    """Process a single CSV file to generate embeddings.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        model: Loaded SentenceTransformer model
    """
    logger.info("=" * 80)
    logger.info(f"Processing: {Path(input_csv).name}")
    logger.info("=" * 80)
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Force regeneration: {FORCE_REGENERATE}")

    # Load input CSV
    logger.info("Loading input CSV...")
    try:
        df = pd.read_csv(input_csv)
        # Process ALL documents (zero vectors for missing OCR)
        # df = df[~((df.ocr_text_per_page.fillna("") == ""))]  # TEST: filter to docs with OCR
        # df = df.tail(10)  # TEST: process only 10 documents
        logger.info(f"Loaded {len(df)} documents")
    except Exception as e:
        logger.error(f"Failed to load input CSV: {e}")
        return False

    # Check if output CSV exists (resume functionality)
    output_path = Path(output_csv)
    if output_path.exists() and not FORCE_REGENERATE:
        logger.info(f"Found existing output file: {output_csv}")
        logger.info("Loading existing results to resume from where we left off...")
        try:
            df_existing = pd.read_csv(output_csv)

            # Identify which rows have been processed
            # Use sha1 as unique identifier
            if "sha1" in df.columns and "sha1" in df_existing.columns:
                processed_sha1s = set(
                    df_existing[df_existing["embeddings_generated"]]["sha1"].dropna()
                )
                logger.info(f"Found {len(processed_sha1s)} already processed documents")

                # Merge existing embeddings back into input df
                if "embedding" in df_existing.columns:
                    # Keep only unprocessed rows + merge processed ones
                    df_unprocessed = df[~df["sha1"].isin(processed_sha1s)]
                    df_processed = df_existing[df_existing["sha1"].isin(df["sha1"])]

                    # Combine them
                    df = pd.concat([df_processed, df_unprocessed], ignore_index=True)
                    logger.info(f"Resuming with {len(df_unprocessed)} documents left to process")
            else:
                logger.warning("Cannot resume: 'sha1' column not found in input or output")
        except Exception as e:
            logger.warning(f"Failed to load existing output for resume: {e}")
            logger.warning("Starting fresh extraction...")

    # Check if all embeddings already exist
    if "embedding" in df.columns and "embeddings_generated" in df.columns:
        already_done = (df["embeddings_generated"]).sum()
        if already_done == len(df):
            logger.info(f"All {len(df)} documents already have embeddings. Nothing to do!")
            logger.info("Set FORCE_REGENERATE=True to regenerate.")
            return True
        elif already_done > 0:
            logger.info(f"{already_done}/{len(df)} documents already have embeddings")

    # Process documents
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    zero_vectors = 0
    errors = 0

    # Initialize embedding list if not already present
    if "embedding" not in df.columns:
        df["embedding"] = [None] * len(df)
    if "embeddings_generated" not in df.columns:
        df["embeddings_generated"] = False

    # Identify unprocessed documents
    unprocessed_mask = not df["embeddings_generated"]
    unprocessed_indices = df[unprocessed_mask].index.tolist()

    if len(unprocessed_indices) == 0:
        logger.info("No documents to process!")
        skipped_count = len(df)
    else:
        logger.info(
            f"Processing {len(unprocessed_indices)} documents in batches of {BATCH_SIZE}..."
        )

        # Parse OCR text for all unprocessed documents
        logger.info("Parsing OCR text for all unprocessed documents...")
        unprocessed_texts = []
        for idx in tqdm(unprocessed_indices, desc="Parsing OCR text"):
            ocr_data = df.at[idx, "ocr_text_per_page"]
            ocr_text = parse_ocr_text(ocr_data)
            unprocessed_texts.append(ocr_text)

        # Process in batches
        num_batches = (len(unprocessed_indices) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Processing {num_batches} batches...")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(unprocessed_indices))

            batch_indices = unprocessed_indices[batch_start:batch_end]
            batch_texts = unprocessed_texts[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_indices)} documents)"
            )

            try:
                # Identify which docs in this batch have actual OCR text
                has_ocr = [bool(t.strip()) for t in batch_texts]
                texts_to_encode = [t for t, has in zip(batch_texts, has_ocr, strict=False) if has]

                # Only encode docs that have OCR text; assign zero vectors to the rest
                if texts_to_encode:
                    encoded = model.encode(
                        texts_to_encode,
                        batch_size=BATCH_SIZE,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                    )
                    encode_iter = iter(encoded)
                else:
                    encode_iter = iter([])

                batch_embeddings = [next(encode_iter) if has else np.zeros(384) for has in has_ocr]

                # Store embeddings back into dataframe
                for i, idx in enumerate(batch_indices):
                    embedding = batch_embeddings[i]

                    # Track statistics
                    if np.allclose(embedding, 0):
                        zero_vectors += 1

                    df.at[idx, "embedding"] = embedding
                    df.at[idx, "embeddings_generated"] = True
                    processed_count += 1

                # Save checkpoint after each batch
                logger.info(f"Batch {batch_idx + 1} complete. Saving checkpoint...")

                checkpoint_columns = [
                    "provisional_case_name",
                    "gdrive_id",
                    "sha1",
                    "gdrive_path",
                    "gdrive_name",
                    "ocr_text_per_page",
                    "embedding",
                    "embeddings_generated",
                ]
                existing_checkpoint_columns = [
                    col for col in checkpoint_columns if col in df.columns
                ]
                df_checkpoint = df[existing_checkpoint_columns]

                output_path = Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_checkpoint.to_csv(output_csv, index=False)
                logger.info(
                    f"Checkpoint saved: {processed_count}/{len(unprocessed_indices)} documents processed"
                )

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                errors += len(batch_indices)

                # Set zero vectors for entire batch on error
                for idx in batch_indices:
                    df.at[idx, "embedding"] = np.zeros(384)
                    df.at[idx, "embeddings_generated"] = True

        skipped_count = len(df) - len(unprocessed_indices)

    # Filter output to only relevant columns
    output_columns = [
        # Ground truth and identifiers
        "provisional_case_name",
        "gdrive_id",
        "sha1",
        # Path information
        "gdrive_path",
        "gdrive_name",
        # OCR text (needed for verification)
        "ocr_text_per_page",
        # Embedding
        "embedding",
        "embeddings_generated",
    ]

    # Only keep columns that exist in the dataframe
    existing_output_columns = [col for col in output_columns if col in df.columns]
    df_output = df[existing_output_columns]

    # Save final output
    logger.info("Saving final output...")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv, index=False)

    # Summary statistics
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'=' * 80}")
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info(f"Total documents: {len(df)}")
    logger.info(f"New embeddings generated: {processed_count}")
    logger.info(f"Already processed (skipped): {skipped_count}")
    logger.info(f"Zero vectors (missing OCR): {zero_vectors}")
    logger.info(f"Errors encountered: {errors}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    if processed_count > 0:
        logger.info(f"Average time per document: {elapsed_time / processed_count:.2f} seconds")
    logger.info(f"Output saved to: {output_csv}")
    logger.info(f"{'=' * 80}\n")
    return True


def main():
    """Main entry point that processes all agencies."""
    script_dir = Path(__file__).parent
    logger.info("\n" + "=" * 80)
    logger.info("BATCH EMBEDDING GENERATION - Phase 2a")
    logger.info("=" * 80)
    logger.info(f"Total agencies to process: {len(AGENCIES)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("=" * 80 + "\n")

    # Load model once for all CSVs (lazy-loaded via prap_clustering.embeddings)
    logger.info(f"Loading sentence-transformers model: {MODEL_NAME}")
    try:
        model = _get_embedding_model()
        logger.info("Model loaded successfully\n")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track overall progress
    total_start_time = time.time()
    successful = 0
    failed = 0

    # Process each agency
    for idx, agency in enumerate(AGENCIES, 1):
        agency_name = agency["name"]
        csv_path = str(script_dir / agency["csv_path"])

        logger.info(f"\n{'#' * 80}")
        logger.info(f"[{idx}/{len(AGENCIES)}] {agency_name}")
        logger.info(f"{'#' * 80}\n")

        # Check if input file exists
        input_path = Path(csv_path)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {csv_path}")
            failed += 1
            continue

        # Generate output filename: add _embeddings suffix before .csv extension
        input_filename = input_path.name
        if input_filename.endswith(".csv"):
            output_filename = input_filename[:-4] + "_embeddings.csv"
        else:
            output_filename = input_filename + "_embeddings.csv"

        output_csv = str(output_dir / output_filename)

        # Process the CSV
        try:
            success = process_single_csv(csv_path, output_csv, model)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {agency_name}: {e}")
            failed += 1

    # Final summary
    total_elapsed = time.time() - total_start_time
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total agencies processed: {len(AGENCIES)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(
        f"Total elapsed time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)"
    )
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80 + "\n")
