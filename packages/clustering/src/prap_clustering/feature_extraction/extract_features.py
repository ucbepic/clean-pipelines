#!/usr/bin/env python3
"""
Feature-extraction orchestrator.

Extracts features from filepath, filename, and OCR text in priority order:
1. Filepath regex (Priority 1)
2. Filename regex (Priority 2)
3. LLM extraction from OCR (Priority 3)

Output: CSV with pre-extracted features, consumed by the clustering stage.
"""

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock

import pandas as pd
from jinja2 import Template

from prap_clustering._llm import get_llm

# Import pipelines
from .extract_pipeline import extract_and_convert
from .prompts.case_ids import PROMPTS as CASE_ID_PROMPTS

# Import feature-specific prompts
from .prompts.dates import PROMPTS as DATE_PROMPTS
from .prompts.officer_names import PROMPTS as OFFICER_NAME_PROMPTS
from .prompts.structured import PROMPTS as STRUCTURED_PROMPTS
from .prompts.subject_names import PROMPTS as SUBJECT_NAME_PROMPTS

# Import regex extraction functions
from .regex_extract_fp_fn import (
    extract_date_from_metadata,
    extract_ids_from_metadata,
    extract_names_from_metadata,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_features.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_ocr_text(ocr_data) -> str:
    """
    Parse OCR text from JSON format to plain text.

    The OCR data is expected to be a JSON array of objects like:
    [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}]

    Args:
        ocr_data: String containing JSON array or already parsed list

    Returns:
        Combined text from all pages
    """
    if pd.isna(ocr_data) or not ocr_data:
        return ""

    try:
        # If it's already a string, try to parse it as JSON
        if isinstance(ocr_data, str):
            pages = json.loads(ocr_data)
        else:
            # If it's already parsed (shouldn't happen but handle it)
            pages = ocr_data

        # Combine all page texts
        if isinstance(pages, list):
            all_text = []
            for page in pages:
                if isinstance(page, dict) and 'text' in page:
                    all_text.append(page['text'])
            return "\n\n".join(all_text)
        else:
            # If it's not a list, just return it as string
            return str(ocr_data)

    except json.JSONDecodeError:
        # If JSON parsing fails, treat it as plain text
        logger.debug("OCR data is not valid JSON, treating as plain text")
        return str(ocr_data)
    except Exception as e:
        logger.warning(f"Error parsing OCR text: {e}")
        return ""


def filter_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include columns needed for clustering.

    Args:
        df: Full DataFrame with all columns

    Returns:
        DataFrame with only the specified columns
    """
    # Define the columns to keep (in order)
    desired_columns = [
        # Original metadata columns
        'provisional_case_name',
        'gdrive_path',
        'gdrive_name',
        'mimeType',
        'gdrive_id',
        'first_look_summary',
        'file_name_from_json',
        'ocr_text_per_page',

        # Regex extracted features
        'extracted_case_ids_fp',
        'extracted_case_ids_fn',
        'extracted_dates_fp',
        'extracted_dates_fn',
        'extracted_names_fp',
        'extracted_names_fn',

        # LLM extracted features (string outputs)
        'extracted_dates_llm',
        'extracted_case_ids_llm',
        'extracted_subject_names_llm',
        'extracted_officer_names_llm',

        # LLM extracted features (structured JSON outputs)
        'extracted_dates_llm_structured',
        'extracted_case_ids_llm_structured',
        'extracted_subject_names_llm_structured',
        'extracted_officer_names_llm_structured',

        # Feature extraction flag
        'features_extracted',

        # Summaries (at the end so they don't get in the way)
        'dates_summary',
        'case_ids_summary',
        'subject_names_summary',
        'officer_names_summary',
    ]

    # Only keep columns that exist in the dataframe
    existing_columns = [col for col in desired_columns if col in df.columns]

    # Warn about missing columns
    missing_columns = [col for col in desired_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Some desired columns are missing from DataFrame: {missing_columns}")

    logger.info(f"Filtering output to {len(existing_columns)} columns")
    return df[existing_columns]


def setup_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add required extraction columns to DataFrame if they don't exist.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with all required columns
    """
    required_columns = [
        'extracted_dates_fp',
        'extracted_dates_fn',
        'extracted_dates_llm',
        'extracted_dates_llm_structured',
        'extracted_case_ids_fp',
        'extracted_case_ids_fn',
        'extracted_case_ids_llm',
        'extracted_case_ids_llm_structured',
        'extracted_names_fp',
        'extracted_names_fn',
        'extracted_subject_names_llm',
        'extracted_subject_names_llm_structured',
        'extracted_officer_names_llm',
        'extracted_officer_names_llm_structured',
        'dates_summary',
        'case_ids_summary',
        'subject_names_summary',
        'officer_names_summary',
        'features_extracted'
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Ensure features_extracted is boolean
    if 'features_extracted' in df.columns:
        df['features_extracted'] = df['features_extracted'].fillna(False).astype(bool)
    else:
        df['features_extracted'] = False

    return df


def extract_regex_features(directory_path: str, filename: str) -> dict:
    """
    Extract features using regex from directory path and filename separately.

    Args:
        directory_path: Directory path only (fp) - everything before the last slash
                       e.g., "Oakland Police Department/20241018_Scraped_Oakland Police Department/assets/01-20621_OIS"
        filename: Just the filename (fn) - e.g., "01-20621 OIS Offense Report.pdf"

    Returns:
        Dictionary with extracted features from both fp and fn
    """
    features = {}

    # Extract dates from directory path (fp) and filename (fn)
    features['dates_fp'] = extract_date_from_metadata(directory_path) if directory_path else []
    features['dates_fn'] = extract_date_from_metadata(filename) if filename else []

    # Extract case IDs from directory path (fp) and filename (fn)
    features['case_ids_fp'] = extract_ids_from_metadata(directory_path) if directory_path else []
    features['case_ids_fn'] = extract_ids_from_metadata(filename) if filename else []

    # Extract names from directory path (fp) and filename (fn)
    features['names_fp'] = extract_names_from_metadata(directory_path) if directory_path else []
    features['names_fn'] = extract_names_from_metadata(filename) if filename else []

    return features


def structure_extraction(extraction_result: str, feature_type: str) -> str:
    """
    Convert extraction result to structured JSON format.

    Args:
        extraction_result: String output from LLM extraction pipeline
        feature_type: One of 'dates', 'case_ids', 'subject_names', 'officer_names'

    Returns:
        JSON string with structured data
    """
    if not extraction_result or not extraction_result.strip():
        logger.warning(f"Empty extraction result for {feature_type}")
        # Return appropriate empty structure
        if feature_type == 'dates':
            return json.dumps({"incident_date": None})
        else:
            return json.dumps([])

    try:
        logger.info(f"Structuring {feature_type} extraction result...")

        # Get the appropriate structuring prompt
        prompt_template = STRUCTURED_PROMPTS[feature_type]
        template = Template(prompt_template)
        prompt = template.render(source_text=extraction_result)

        # Call LLM to structure the output
        structured_result = get_llm().complete(prompt).text

        # Validate it's valid JSON
        parsed = json.loads(structured_result.strip())
        logger.info(f"Structured {feature_type}: {structured_result[:200]}...")

        return json.dumps(parsed)  # Return as JSON string

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from structuring prompt for {feature_type}: {e}")
        logger.error(f"Raw output: {structured_result}")
        # Return appropriate empty structure
        if feature_type == 'dates':
            return json.dumps({"incident_date": None})
        else:
            return json.dumps([])
    except Exception as e:
        logger.error(f"Error structuring {feature_type}: {e}", exc_info=True)
        # Return appropriate empty structure
        if feature_type == 'dates':
            return json.dumps({"incident_date": None})
        else:
            return json.dumps([])


def extract_single_feature(args):
    """Wrapper for parallel feature extraction. Returns extraction result AND summary."""
    feature_name, ocr_text, prompts = args
    try:
        logger.info(f"Extracting {feature_name} via LLM...")

        # Step 1: Generate summary
        from summarize_pipeline import summarize_document
        summary = summarize_document(ocr_text, prompts)

        # Step 2: Extract from summary
        if not summary:
            logger.warning(f"No summary generated for {feature_name}")
            return feature_name, None, None

        result = extract_and_convert(summary, prompts)

        logger.info(f"{feature_name} extracted: {result}")
        return feature_name, result, summary
    except Exception as e:
        logger.error(f"Error extracting {feature_name}: {e}", exc_info=True)
        return feature_name, None, None


def extract_llm_features(ocr_text: str) -> dict:
    """
    Extract features using LLM from OCR text in parallel.

    Args:
        ocr_text: Full OCR text of document

    Returns:
        Dictionary with extracted features (string, structured, and summaries)
    """
    features = {}
    summaries = {}

    logger.info("Starting LLM feature extraction (parallel)")

    try:
        # Define all feature extraction tasks
        extraction_tasks = [
            ('dates_llm', ocr_text, DATE_PROMPTS),
            ('subject_names_llm', ocr_text, SUBJECT_NAME_PROMPTS),
            ('officer_names_llm', ocr_text, OFFICER_NAME_PROMPTS),
            ('case_ids_llm', ocr_text, CASE_ID_PROMPTS),
        ]

        # Process all extractions in parallel
        max_workers = 1  # One worker per feature
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_single_feature, task) for task in extraction_tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    feature_name, result, summary = future.result()
                    features[feature_name] = result
                    summaries[feature_name] = summary
                except Exception as e:
                    logger.error(f"Feature extraction failed: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during LLM extraction: {e}", exc_info=True)
        # Set all to None if extraction fails
        features['dates_llm'] = None
        features['subject_names_llm'] = None
        features['officer_names_llm'] = None
        features['case_ids_llm'] = None
        summaries['dates_llm'] = None
        summaries['subject_names_llm'] = None
        summaries['officer_names_llm'] = None
        summaries['case_ids_llm'] = None

    # Step 2: Structure the extraction results
    logger.info("Starting structured output generation")

    try:
        # Structure each feature extraction result
        features['dates_llm_structured'] = structure_extraction(
            features.get('dates_llm', ''), 'dates'
        )
        features['case_ids_llm_structured'] = structure_extraction(
            features.get('case_ids_llm', ''), 'case_ids'
        )
        features['subject_names_llm_structured'] = structure_extraction(
            features.get('subject_names_llm', ''), 'subject_names'
        )
        features['officer_names_llm_structured'] = structure_extraction(
            features.get('officer_names_llm', ''), 'officer_names'
        )

        logger.info("Structured output generation complete")

    except Exception as e:
        logger.error(f"Error during structured output generation: {e}", exc_info=True)
        # Set structured outputs to empty/null
        features['dates_llm_structured'] = json.dumps({"incident_date": None})
        features['case_ids_llm_structured'] = json.dumps([])
        features['subject_names_llm_structured'] = json.dumps([])
        features['officer_names_llm_structured'] = json.dumps([])

    # Step 3: Add summaries to the features dictionary
    features['dates_summary'] = summaries.get('dates_llm')
    features['case_ids_summary'] = summaries.get('case_ids_llm')
    features['subject_names_summary'] = summaries.get('subject_names_llm')
    features['officer_names_summary'] = summaries.get('officer_names_llm')

    return features


def process_document(row: pd.Series, force: bool = False) -> dict:
    """
    Process a single document row and extract all features.

    Args:
        row: DataFrame row with document data
        force: If True, re-extract even if already extracted

    Returns:
        Dictionary with all extracted features
    """
    # Check idempotency
    if not force and row.get('features_extracted', False):
        logger.debug(f"Skipping already extracted document: {row.get('filepath', 'unknown')}")
        return None

    logger.info(f"Processing document: {row.get('filepath', 'unknown')}")

    extracted = {}

    # Get full path and filename
    full_path = row.get('filepath', row.get('file_path', row.get('gdrive_path', '')))
    filename = row.get('gdrive_name', '')

    # If filename not in row, extract from path
    if not filename and full_path:
        filename = Path(full_path).name

    # Get directory path (remove filename from full path)
    if full_path:
        last_slash = full_path.rfind('/')
        if last_slash != -1:
            directory_path = full_path[:last_slash]  # Everything before last slash
        else:
            directory_path = ""  # No slash found, no directory
    else:
        directory_path = ""

    # Step 1: Regex extraction (Priority 1-2)
    logger.info("Step 1: Regex extraction from directory path (fp) and filename (fn)")
    regex_features = extract_regex_features(directory_path, filename)

    extracted['extracted_dates_fp'] = str(regex_features['dates_fp']) if regex_features['dates_fp'] else None
    extracted['extracted_dates_fn'] = str(regex_features['dates_fn']) if regex_features['dates_fn'] else None
    extracted['extracted_case_ids_fp'] = str(regex_features['case_ids_fp']) if regex_features['case_ids_fp'] else None
    extracted['extracted_case_ids_fn'] = str(regex_features['case_ids_fn']) if regex_features['case_ids_fn'] else None
    extracted['extracted_names_fp'] = str(regex_features['names_fp']) if regex_features['names_fp'] else None
    extracted['extracted_names_fn'] = str(regex_features['names_fn']) if regex_features['names_fn'] else None

    # Step 2: LLM extraction (Priority 3)
    ocr_data = row.get('ocr_text', row.get('text', row.get('ocr_text_per_page', '')))

    # Parse OCR text from JSON format if needed
    ocr_text = parse_ocr_text(ocr_data)

    if ocr_text and ocr_text.strip():
        logger.info(f"Step 2: LLM extraction from OCR text ({len(ocr_text)} chars)")
        llm_features = extract_llm_features(ocr_text)

        # Store string outputs (human-readable)
        extracted['extracted_dates_llm'] = str(llm_features['dates_llm']) if llm_features['dates_llm'] else None
        extracted['extracted_subject_names_llm'] = str(llm_features['subject_names_llm']) if llm_features['subject_names_llm'] else None
        extracted['extracted_officer_names_llm'] = str(llm_features['officer_names_llm']) if llm_features['officer_names_llm'] else None
        extracted['extracted_case_ids_llm'] = str(llm_features['case_ids_llm']) if llm_features['case_ids_llm'] else None

        # Store structured outputs (JSON strings)
        extracted['extracted_dates_llm_structured'] = llm_features['dates_llm_structured']
        extracted['extracted_case_ids_llm_structured'] = llm_features['case_ids_llm_structured']
        extracted['extracted_subject_names_llm_structured'] = llm_features['subject_names_llm_structured']
        extracted['extracted_officer_names_llm_structured'] = llm_features['officer_names_llm_structured']

        # Store summaries
        extracted['dates_summary'] = llm_features['dates_summary']
        extracted['case_ids_summary'] = llm_features['case_ids_summary']
        extracted['subject_names_summary'] = llm_features['subject_names_summary']
        extracted['officer_names_summary'] = llm_features['officer_names_summary']
    else:
        logger.warning(f"No OCR text available for document: {filepath}")
        # String outputs
        extracted['extracted_dates_llm'] = None
        extracted['extracted_subject_names_llm'] = None
        extracted['extracted_officer_names_llm'] = None
        extracted['extracted_case_ids_llm'] = None
        # Structured outputs (empty JSON)
        extracted['extracted_dates_llm_structured'] = json.dumps({"incident_date": None})
        extracted['extracted_case_ids_llm_structured'] = json.dumps([])
        extracted['extracted_subject_names_llm_structured'] = json.dumps([])
        extracted['extracted_officer_names_llm_structured'] = json.dumps([])
        # Summaries
        extracted['dates_summary'] = None
        extracted['case_ids_summary'] = None
        extracted['subject_names_summary'] = None
        extracted['officer_names_summary'] = None

    # Mark as extracted
    extracted['features_extracted'] = True

    return extracted


def process_document_wrapper(args):
    """Wrapper for parallel document processing."""
    idx, row, force = args
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing document index {idx}")

        extracted = process_document(row, force=force)
        return idx, extracted, None
    except Exception as e:
        logger.error(f"Failed to process document {idx}: {e}", exc_info=True)
        return idx, None, str(e)


def main():
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(
        description='Extract features from documents using regex and LLM'
    )
    parser.add_argument(
        'input_csv',
        help='Path to input CSV with OCR text and metadata'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output CSV (default: input_with_extracted_features.csv)',
        default=None
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-extract features even if already extracted'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit processing to first N documents (for testing)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default output path (env-var driven for the package; falls back to CWD).
    if args.output is None:
        input_path = Path(args.input_csv)
        import os
        base = os.environ.get("PRAP_CLUSTERING_FEATURES_DIR", ".")
        args.output = str(Path(base) / f"{input_path.stem}_with_extracted_features.csv")

    logger.info("Starting feature extraction")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Output CSV: {args.output}")
    logger.info(f"Force re-extraction: {args.force}")

    # Load input CSV
    logger.info("Loading input CSV...")
    try:
        df = pd.read_csv(args.input_csv)
        # df = df.head(10)
        df = df[~(df.ocr_text_per_page.fillna("") == "")]
        # df = df.tail(1)
        logger.info(f"Loaded {len(df)} documents")
    except Exception as e:
        logger.error(f"Failed to load input CSV: {e}")
        sys.exit(1)

    # Check if output CSV exists (resume functionality)
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.info(f"Found existing output file: {args.output}")
        logger.info("Loading existing results to resume from where we left off...")
        try:
            df_existing = pd.read_csv(args.output)

            # Identify which rows have been processed
            # We'll use sha1 as the unique identifier to match rows
            if 'sha1' in df.columns and 'sha1' in df_existing.columns:
                processed_sha1s = set(df_existing[df_existing['features_extracted']]['sha1'].dropna())
                logger.info(f"Found {len(processed_sha1s)} already processed documents")

                # Mark already processed rows in the input dataframe
                df['features_extracted'] = df['sha1'].isin(processed_sha1s)

                # Merge existing extracted features back into input df
                merge_cols = [col for col in df_existing.columns if col.startswith('extracted_') or col in ['dates_summary', 'case_ids_summary', 'subject_names_summary', 'officer_names_summary', 'features_extracted']]
                merge_cols = [col for col in merge_cols if col in df_existing.columns]

                if merge_cols:
                    # Merge on sha1
                    df = df.drop(columns=[col for col in merge_cols if col in df.columns], errors='ignore')
                    df = df.merge(df_existing[['sha1'] + merge_cols], on='sha1', how='left', suffixes=('', '_existing'))
                    logger.info(f"Merged {len(merge_cols)} feature columns from existing results")
            else:
                logger.warning("Cannot resume: 'sha1' column not found in input or output")
        except Exception as e:
            logger.warning(f"Failed to load existing output for resume: {e}")
            logger.warning("Starting fresh extraction...")

    # Setup columns
    df = setup_csv_columns(df)

    # Apply limit if specified
    if args.limit:
        logger.info(f"Limiting to first {args.limit} documents")
        df = df.head(args.limit)

    # Count documents to process
    if args.force:
        docs_to_process = len(df)
    else:
        docs_to_process = (~df['features_extracted']).sum()

    logger.info(f"Documents to process: {docs_to_process}")

    # Process documents in parallel
    start_time = time.time()
    processed = 0
    failed = 0
    skipped = 0

    # Prepare arguments for parallel processing
    doc_args = [(idx, row, args.force) for idx, row in df.iterrows()]

    # Use ThreadPoolExecutor for parallel document processing
    max_workers = 5  # Process 5 documents in parallel (balance API rate limits)
    logger.info(f"Processing documents in parallel with {max_workers} workers")

    # Create a lock for thread-safe DataFrame updates
    df_lock = Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all document processing jobs
        futures = [executor.submit(process_document_wrapper, args) for args in doc_args]
        logger.info(f"Submitted {len(futures)} document processing jobs")

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, extracted, error = future.result()
                completed_count += 1

                if error:
                    logger.error(f"Document {idx} failed: {error}")
                    failed += 1
                elif extracted is not None:
                    # Update DataFrame (thread-safe)
                    with df_lock:
                        for key, value in extracted.items():
                            df.at[idx, key] = value
                    processed += 1
                    logger.info(f"✓ Document {idx} completed ({completed_count}/{len(futures)})")
                else:
                    skipped += 1
                    logger.debug(f"Document {idx} skipped (already extracted)")

                # Save checkpoint every 50 processed documents
                if processed % 50 == 0 and processed > 0:
                    logger.info(f"Checkpoint: Saving progress ({processed} documents processed)...")
                    with df_lock:
                        df_filtered = filter_output_columns(df)
                        df_filtered.to_csv(args.output, index=False)
                    logger.info(f"Progress saved to {args.output}")

            except Exception as e:
                logger.error(f"Error processing future: {e}", exc_info=True)
                failed += 1

    # Save final output
    logger.info(f"\n{'='*80}")
    logger.info("Saving final output...")
    with df_lock:
        # Filter to only include desired columns
        df_filtered = filter_output_columns(df)
        df_filtered.to_csv(args.output, index=False)

    # Summary statistics
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Total documents: {len(df)}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already extracted): {skipped}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    if processed > 0:
        logger.info(f"Average time per document: {elapsed_time/processed:.2f} seconds")
        logger.info(f"Success rate: {(processed/(processed+failed)*100):.1f}%")
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
