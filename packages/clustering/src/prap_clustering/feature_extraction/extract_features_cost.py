#!/usr/bin/env python3
"""
Feature-extraction orchestrator (cost-tracking variant).

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
import re
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


CORPUS_TOTAL_DOCS = 29824  # Docs with OCR text across all 31 holdout agencies (69,767 total rows, 29,824 with OCR)

# Representative sample of 5 agencies covering different types and sizes:
# medical examiner (small), state agency (medium), police (medium),
# DA (medium-large), sheriff (large)
DEFAULT_AGENCY_CSVS = [
    "../../data/input/autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44 - autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44.csv",
    "../../data/input/autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52 - autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52.csv",
    "../../data/input/autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37 - autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37.csv",
    "../../data/input/autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04 - autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04.csv",
    "../../data/input/autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42 - autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42.csv",
]

INPUT_COST_PER_TOKEN = 0.80 / 1_000_000   # $0.80 / 1M tokens
OUTPUT_COST_PER_TOKEN = 3.20 / 1_000_000  # $3.20 / 1M tokens


def _agency_name_from_path(csv_path: str) -> str:
    """Extract human-readable agency name from autofolio CSV filename."""
    m = re.search(r'output--(.+?)--\d{4}-\d{2}-\d{2}', Path(csv_path).name)
    return m.group(1) if m else Path(csv_path).stem


def write_cost_report(report_path: str, agency_results: list):
    """
    Write per-agency and aggregate token usage / cost report.

    Args:
        report_path: Path to write the report text file
        agency_results: List of dicts, one per agency, with keys:
            agency, input_csv, docs_processed,
            prompt_tokens, completion_tokens, call_count
    """
    total_docs = sum(r['docs_processed'] for r in agency_results)
    total_prompt = sum(r['prompt_tokens'] for r in agency_results)
    total_completion = sum(r['completion_tokens'] for r in agency_results)
    total_calls = sum(r['call_count'] for r in agency_results)

    total_input_cost = total_prompt * INPUT_COST_PER_TOKEN
    total_output_cost = total_completion * OUTPUT_COST_PER_TOKEN
    total_cost = total_input_cost + total_output_cost

    avg_prompt = total_prompt / total_docs if total_docs > 0 else 0
    avg_completion = total_completion / total_docs if total_docs > 0 else 0
    avg_cost = total_cost / total_docs if total_docs > 0 else 0

    extrap_input = avg_prompt * CORPUS_TOTAL_DOCS * INPUT_COST_PER_TOKEN
    extrap_output = avg_completion * CORPUS_TOTAL_DOCS * OUTPUT_COST_PER_TOKEN
    extrap_total = extrap_input + extrap_output

    W = 75
    lines = [
        "FEATURE EXTRACTION COST REPORT",
        "=" * W,
        "",
        "PER-AGENCY BREAKDOWN",
        "-" * W,
        f"{'Agency':<38} {'Docs':>6}  {'Input tok':>11}  {'Output tok':>11}  {'Cost':>8}",
        "-" * W,
    ]
    for r in agency_results:
        ic = r['prompt_tokens'] * INPUT_COST_PER_TOKEN
        oc = r['completion_tokens'] * OUTPUT_COST_PER_TOKEN
        tc = ic + oc
        lines.append(
            f"{r['agency']:<38} {r['docs_processed']:>6,}"
            f"  {r['prompt_tokens']:>11,}  {r['completion_tokens']:>11,}  ${tc:>7.2f}"
        )
    lines += [
        "-" * W,
        f"{'TOTAL':<38} {total_docs:>6,}  {total_prompt:>11,}  {total_completion:>11,}  ${total_cost:>7.2f}",
        "",
        "AGGREGATE TOKEN USAGE",
        "-" * W,
        f"Total input tokens:    {total_prompt:,}",
        f"Total output tokens:   {total_completion:,}",
        f"Total LLM calls:       {total_calls:,}",
        f"LLM calls/doc:         {total_calls / total_docs:.1f}" if total_docs > 0 else "LLM calls/doc:  N/A",
        "",
        "AGGREGATE COSTS  (gpt-4.1-mini: input $0.80/1M tokens, output $3.20/1M tokens)",
        "-" * W,
        f"Input cost:    ${total_input_cost:.4f}",
        f"Output cost:   ${total_output_cost:.4f}",
        f"Total cost:    ${total_cost:.4f}",
        "",
        "WEIGHTED PER-DOCUMENT AVERAGES  (weighted by doc count across sampled agencies)",
        "-" * W,
        f"Avg input tokens/doc:    {avg_prompt:,.1f}",
        f"Avg output tokens/doc:   {avg_completion:,.1f}",
        f"Avg cost/doc:            ${avg_cost:.6f}",
        "",
        f"CORPUS EXTRAPOLATION  ({CORPUS_TOTAL_DOCS:,} total holdout docs across 31 agencies)",
        "-" * W,
        f"Extrapolated input cost:    ${extrap_input:.2f}",
        f"Extrapolated output cost:   ${extrap_output:.2f}",
        f"Extrapolated total cost:    ${extrap_total:.2f}",
        "",
        f"Sample coverage: {total_docs:,} of {CORPUS_TOTAL_DOCS:,} docs "
        f"({100 * total_docs / CORPUS_TOTAL_DOCS:.1f}% of corpus)",
        "Extrapolation uses doc-weighted avg cost/doc across sampled agencies.",
    ]

    report_text = "\n".join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Cost report written to: {report_path}")
    logger.info(f"Total cost for {total_docs:,} docs across {len(agency_results)} agencies: ${total_cost:.4f}")
    logger.info(f"Extrapolated corpus cost ({CORPUS_TOTAL_DOCS:,} docs): ${extrap_total:.2f}")


def process_csv(input_csv: str, output_path: str, force: bool, limit: int | None) -> dict:
    """
    Process a single agency CSV for cost estimation.

    Returns a dict with agency name, doc count, and token usage.
    """
    agency = _agency_name_from_path(input_csv)

    logger.info(f"\n{'='*80}")
    logger.info(f"AGENCY: {agency}")
    logger.info(f"Input:  {input_csv}")
    logger.info(f"Output: {output_path}")

    # Reset token counters for this agency
    _llm = get_llm()
    _llm.usage.prompt_tokens = 0
    _llm.usage.completion_tokens = 0
    _llm.usage.total_tokens = 0
    _llm.usage.cost_usd = 0.0

    empty_result = {
        'agency': agency, 'input_csv': input_csv, 'docs_processed': 0,
        'prompt_tokens': 0, 'completion_tokens': 0, 'call_count': 0,
    }

    # Load input CSV
    try:
        df = pd.read_csv(input_csv)
        df = df[~(df.ocr_text_per_page.fillna("") == "")]
        logger.info(f"Loaded {len(df)} documents with OCR text")
    except Exception as e:
        logger.error(f"Failed to load {input_csv}: {e}")
        return empty_result

    # Resume from existing output if present
    output_path_obj = Path(output_path)
    if output_path_obj.exists() and not force:
        logger.info("Found existing output, resuming...")
        try:
            df_existing = pd.read_csv(output_path)
            if 'sha1' in df.columns and 'sha1' in df_existing.columns:
                processed_sha1s = set(df_existing[df_existing['features_extracted']]['sha1'].dropna())
                logger.info(f"Found {len(processed_sha1s)} already processed documents")
                df['features_extracted'] = df['sha1'].isin(processed_sha1s)
                merge_cols = [
                    col for col in df_existing.columns
                    if col.startswith('extracted_') or col in [
                        'dates_summary', 'case_ids_summary',
                        'subject_names_summary', 'officer_names_summary', 'features_extracted'
                    ]
                    if col in df_existing.columns
                ]
                if merge_cols:
                    df = df.drop(columns=[c for c in merge_cols if c in df.columns], errors='ignore')
                    df = df.merge(df_existing[['sha1'] + merge_cols], on='sha1', how='left')
            else:
                logger.warning("Cannot resume: 'sha1' column not found")
        except Exception as e:
            logger.warning(f"Resume failed ({e}), starting fresh")

    df = setup_csv_columns(df)

    if limit:
        logger.info(f"Limiting to first {limit} documents")
        df = df.head(limit)

    docs_to_process = len(df) if force else (~df['features_extracted']).sum()
    logger.info(f"Documents to process: {docs_to_process}")

    start_time = time.time()
    processed = 0
    failed = 0
    skipped = 0
    df_lock = Lock()

    doc_args = [(idx, row, force) for idx, row in df.iterrows()]

    max_workers = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_document_wrapper, a) for a in doc_args]
        logger.info(f"Submitted {len(futures)} jobs")

        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, extracted, error = future.result()
                completed_count += 1
                if error:
                    logger.error(f"Document {idx} failed: {error}")
                    failed += 1
                elif extracted is not None:
                    with df_lock:
                        for key, value in extracted.items():
                            df.at[idx, key] = value
                    processed += 1
                    logger.info(f"✓ Document {idx} completed ({completed_count}/{len(futures)})")
                else:
                    skipped += 1

                if processed % 50 == 0 and processed > 0:
                    with df_lock:
                        filter_output_columns(df).to_csv(output_path, index=False)
                    logger.info(f"Checkpoint saved ({processed} docs)")
            except Exception as e:
                logger.error(f"Error processing future: {e}", exc_info=True)
                failed += 1

    with df_lock:
        filter_output_columns(df).to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(f"Agency complete: {processed} processed, {failed} failed, {skipped} skipped in {elapsed:.1f}s")
    logger.info(f"Output saved to: {output_path}")

    usage = get_llm().usage
    return {
        'agency': agency,
        'input_csv': input_csv,
        'docs_processed': processed,
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
        'call_count': 0,
    }


def main():
    """Main entry point for multi-agency cost estimation."""
    parser = argparse.ArgumentParser(
        description='Extract features and measure token costs across one or more agency CSVs'
    )
    parser.add_argument(
        'input_csvs',
        nargs='*',
        help='One or more input CSVs (one per agency). Defaults to the 5 hardcoded representative agencies.'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for output CSVs (required; or set PRAP_CLUSTERING_COSTS_DIR).',
        default=None,
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
        help='Limit each agency to first N documents (for testing)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--cost-report',
        help='Path to write the cost report (default: <output-dir>/cost_report.txt)',
        default=None
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    input_csvs = args.input_csvs if args.input_csvs else DEFAULT_AGENCY_CSVS

    import os
    output_dir_val = args.output_dir or os.environ.get("PRAP_CLUSTERING_COSTS_DIR")
    if not output_dir_val:
        parser.error("--output-dir is required (or set PRAP_CLUSTERING_COSTS_DIR).")
    output_dir = Path(output_dir_val)
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_report_path = args.cost_report or str(output_dir / 'cost_report.txt')

    logger.info("Starting cost estimation run")
    logger.info(f"Agencies to process: {len(input_csvs)}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Cost report: {cost_report_path}")

    agency_results = []
    for input_csv in input_csvs:
        output_path = str(output_dir / f"{Path(input_csv).stem}_with_extracted_features.csv")
        result = process_csv(input_csv, output_path, args.force, args.limit)
        agency_results.append(result)

    write_cost_report(cost_report_path, agency_results)


if __name__ == '__main__':
    main()
