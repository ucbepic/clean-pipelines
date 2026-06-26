"""Redaction identification pipeline.

- LLM is threaded in via prap_core.llm.LLM
- Azure Content Safety / Blob credentials come from prap_core.config.Settings
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from prap_core.config import Settings
from prap_core.llm import LLM

from .classifiers.cards import classify_pages_with_cards
from .classifiers.dob import classify_pages_with_dob
from .classifiers.graphic_imagery import classify_pdf_for_graphic_imagery
from .classifiers.sensitive_persons import extract_names
from .classifiers.ssn import classify_pages_with_ssn
from .helpers import has_low_word_count, has_penal_code_mentions
from .schemas import RunResult

logger = logging.getLogger("prap.redactions")


def load_case_file(json_path: Path) -> dict:
    """Load a case file JSON."""
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return None


def prepare_ocr_pages(case_data: dict, case_name: str) -> list[dict]:
    """
    Convert case file data into the format expected by extract_names.
    Returns a list of pages with their metadata.
    """
    ocr_pages = []

    case_files = case_data.get("case_files", [])

    for file_data in case_files:
        sha1 = file_data.get("sha1", "")
        file_name = file_data.get("file_name", "Unknown")

        # Get gdrive_id from page_range if available
        page_range = file_data.get("page_range", {})
        gdrive_id = page_range.get("gdrive_id", "")

        # Get OCR data from the new structure
        ocr_doc = file_data.get("ocr_doc_text_per_page", {})
        page_texts = ocr_doc.get("page_texts", [])

        # Process each page
        for page in page_texts:
            page_number = page.get("page_number", 0)
            page_content = page.get("text", "")

            ocr_pages.append(
                {
                    "text": page_content,
                    "page_number": page_number,
                    "document_name": file_name,
                    "gdrive_id": gdrive_id,
                    "sha1": sha1,
                    "provisional_case_name": case_name,
                }
            )

    return ocr_pages


def load_processed_cases(checkpoint_file: Path) -> set:
    """Load set of already-processed case names from checkpoint file."""
    if not checkpoint_file.exists():
        return set()

    try:
        with open(checkpoint_file) as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        logger.warning(f"Failed to load checkpoint file: {e}")
        return set()


def save_processed_case(checkpoint_file: Path, case_name: str):
    """Append a case name to the checkpoint file."""
    try:
        with open(checkpoint_file, "a") as f:
            f.write(f"{case_name}\n")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for {case_name}: {e}")


def append_results_to_csv(results_df: pd.DataFrame, output_path: Path):
    """Append results to CSV file (create if doesn't exist)."""
    if results_df.empty:
        return

    try:
        if output_path.exists():
            # Append to existing file
            results_df.to_csv(output_path, mode="a", header=False, index=False)
            logger.info(f"Appended {len(results_df)} rows to {output_path}")
        else:
            # Create new file with header
            results_df.to_csv(output_path, index=False)
            logger.info(f"Created {output_path} with {len(results_df)} rows")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")
        raise


def process_case(json_path: Path, llm: LLM, settings: Settings) -> pd.DataFrame:
    """
    Process a single case file and return DataFrame with redaction info.
    """
    case_name = json_path.stem.replace("agency_case_file_bundle-", "")
    logger.info(f"Processing case: {case_name}")

    # Load case data
    case_data = load_case_file(json_path)
    if not case_data:
        logger.warning(f"Skipping {case_name} - failed to load")
        return pd.DataFrame()

    # Prepare OCR pages
    ocr_pages = prepare_ocr_pages(case_data, case_name)

    if not ocr_pages:
        logger.warning(f"No OCR pages found in {case_name}")
        return pd.DataFrame()

    logger.info(f"Extracted {len(ocr_pages)} pages from case {case_name}")

    # Store all redaction results
    all_redaction_data = []

    # Get unique SHA1s from this case
    sha1_to_pages = {}
    for page in ocr_pages:
        sha1 = page["sha1"]
        if sha1 not in sha1_to_pages:
            sha1_to_pages[sha1] = []
        sha1_to_pages[sha1].append(page)

    # Process each file
    for sha1, file_pages in sha1_to_pages.items():
        # Track pages needing redaction with reasons
        # Structure: {page_number: [list of classifier names]}
        redaction_pages_with_reasons = {}

        # Step 1: Check if penal codes are present
        logger.info(f"Checking {sha1} for penal code mentions...")
        has_penal_codes = has_penal_code_mentions(file_pages)

        # Step 2: Run sensitive persons classifier only if penal codes found
        if has_penal_codes:
            logger.info(f"Penal codes found in {sha1}, running sensitive persons classifier...")
            try:
                results_df = extract_names(llm, file_pages)

                if not results_df.empty:
                    # Get page numbers for this SHA1
                    file_results = results_df[results_df["sha1"] == sha1]
                    if not file_results.empty:
                        sensitive_persons_pages = set()
                        for _, row in file_results.iterrows():
                            page_num = row.get("page_number")
                            if pd.notna(page_num):
                                page_num = int(page_num)
                                sensitive_persons_pages.add(page_num)
                                # Add classifier to this page's reasons
                                if page_num not in redaction_pages_with_reasons:
                                    redaction_pages_with_reasons[page_num] = []
                                redaction_pages_with_reasons[page_num].append(
                                    "sensitive_persons_classifier"
                                )
                        logger.info(
                            f"Found {len(sensitive_persons_pages)} pages with "
                            f"sensitive persons in {sha1}"
                        )

            except Exception as e:
                logger.error(f"Error extracting names from {sha1}: {e}", exc_info=True)
        else:
            logger.info(f"No penal codes found in {sha1}, skipping sensitive persons classifier")

        # Step 3: Run graphic imagery classifier (only on pages with low word count)
        # Filter pages to only those with <=20 words (likely image pages)
        low_word_count_pages = []
        for page in file_pages:
            if has_low_word_count(page.get("text", ""), threshold=20):
                low_word_count_pages.append(page["page_number"])

        if low_word_count_pages:
            logger.info(
                f"Running graphic imagery classifier on {sha1} "
                f"({len(low_word_count_pages)}/{len(file_pages)} pages with <=20 words)..."
            )
            try:
                graphic_result = classify_pdf_for_graphic_imagery(
                    sha1,
                    violence_threshold=4,
                    page_numbers=low_word_count_pages,
                    settings=settings,
                )

                if graphic_result["success"]:
                    graphic_pages = graphic_result.get("pages_with_graphic_imagery", [])
                    if graphic_pages:
                        for page_num in graphic_pages:
                            if page_num not in redaction_pages_with_reasons:
                                redaction_pages_with_reasons[page_num] = []
                            redaction_pages_with_reasons[page_num].append(
                                "graphic_imagery_classifier"
                            )
                        logger.info(
                            f"Found {len(graphic_pages)} pages with graphic imagery in {sha1}"
                        )
                else:
                    logger.warning(
                        f"Graphic imagery classification failed for {sha1}: "
                        f"{graphic_result.get('error')}"
                    )

            except Exception as e:
                logger.error(f"Error classifying graphic imagery for {sha1}: {e}", exc_info=True)
        else:
            logger.info(
                f"Skipping graphic imagery classifier for {sha1} - "
                f"all {len(file_pages)} pages have >20 words"
            )

        # Step 4: Run DOB classifier on all pages
        logger.info(f"Running DOB classifier on {sha1}...")
        try:
            dob_pages = classify_pages_with_dob(file_pages)
            if dob_pages:
                for page_num in dob_pages:
                    if page_num not in redaction_pages_with_reasons:
                        redaction_pages_with_reasons[page_num] = []
                    redaction_pages_with_reasons[page_num].append("dob_classifier")
                logger.info(f"Found {len(dob_pages)} pages with DOB references in {sha1}")
        except Exception as e:
            logger.error(f"Error classifying DOB for {sha1}: {e}", exc_info=True)

        # Step 5: Run card classifier on all pages
        logger.info(f"Running card classifier on {sha1}...")
        try:
            card_pages = classify_pages_with_cards(file_pages)
            if card_pages:
                for page_num in card_pages:
                    if page_num not in redaction_pages_with_reasons:
                        redaction_pages_with_reasons[page_num] = []
                    redaction_pages_with_reasons[page_num].append("card_classifier")
                logger.info(f"Found {len(card_pages)} pages with card numbers in {sha1}")
        except Exception as e:
            logger.error(f"Error classifying cards for {sha1}: {e}", exc_info=True)

        # Step 6: Run SSN classifier on all pages
        logger.info(f"Running SSN classifier on {sha1}...")
        try:
            ssn_pages = classify_pages_with_ssn(file_pages)
            if ssn_pages:
                for page_num in ssn_pages:
                    if page_num not in redaction_pages_with_reasons:
                        redaction_pages_with_reasons[page_num] = []
                    redaction_pages_with_reasons[page_num].append("ssn_classifier")
                logger.info(f"Found {len(ssn_pages)} pages with SSN in {sha1}")
        except Exception as e:
            logger.error(f"Error classifying SSN for {sha1}: {e}", exc_info=True)

        # Step 7: Add to results if any pages need redaction
        if redaction_pages_with_reasons:
            # Sort page numbers
            sorted_pages = sorted(redaction_pages_with_reasons.keys())

            # Create structured data: list of {"page": X, "classifiers": [list]}
            page_numbers_structured = [
                {"page": page_num, "classifiers": redaction_pages_with_reasons[page_num]}
                for page_num in sorted_pages
            ]

            all_redaction_data.append(
                {
                    "sha1": sha1,
                    "provisional_case_name": case_name,
                    "page_numbers": ", ".join(str(p) for p in sorted_pages),
                    "page_numbers_structured": json.dumps(page_numbers_structured),
                }
            )
            logger.info(f"{sha1} needs redaction on {len(redaction_pages_with_reasons)} pages")
        else:
            logger.info(f"No redactions needed for {sha1}")

    # Convert to DataFrame
    if all_redaction_data:
        return pd.DataFrame(all_redaction_data)
    else:
        return pd.DataFrame()


def process_case_wrapper(args: tuple) -> tuple[str, str, int, object]:
    """
    Wrapper function for parallel case processing.

    Args:
        args: Tuple of (idx, json_path, total_cases, llm, settings)

    Returns:
        Tuple of (status, case_name, idx, data)
        - status: 'success' or 'error'
        - case_name: Name of the case
        - idx: Case index number
        - data: Either results DataFrame or Exception
    """
    idx, json_path, total_cases, llm, settings = args
    case_name = json_path.stem.replace("agency_case_file_bundle-", "")

    try:
        logger.info(f"{'=' * 80}")
        logger.info(f"Case {idx}/{total_cases}: {case_name}")
        logger.info(f"{'=' * 80}")

        results_df = process_case(json_path, llm, settings)

        if not results_df.empty:
            logger.info(f"Case {case_name}: Found redactions needed ({len(results_df)} files)")
            return ("success", case_name, idx, results_df)
        else:
            logger.info(f"Case {case_name}: No redactions needed")
            return ("success", case_name, idx, pd.DataFrame())

    except Exception as e:
        logger.error(f"Error processing case {case_name}: {e}", exc_info=True)
        return ("error", case_name, idx, e)


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> RunResult:
    """
    Main function to process all cases and generate redaction CSV.
    Uses parallel processing with checkpointing and resume capability.
    """
    logger.info("=" * 80)
    logger.info("REDACTION IDENTIFICATION PIPELINE")
    logger.info("=" * 80)

    input_dir_p = Path(input_dir)
    output_dir_p = Path(output_dir)

    # Create output directory
    output_dir_p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input directory: {input_dir_p}")
    logger.info(f"Output directory: {output_dir_p}")

    if settings is None:
        settings = Settings()
    if llm is None:
        llm = LLM()

    # Setup checkpoint and output files
    checkpoint_file = output_dir_p / "processed_cases.txt"
    output_path = output_dir_p / "redaction_list.csv"
    error_log_path = output_dir_p / "redaction_errors.txt"

    # Find all case JSON files
    json_files = list(input_dir_p.glob("agency_case_file_bundle-*.json"))
    logger.info(f"Found {len(json_files)} case files to process")

    if not json_files:
        logger.error("No case JSON files found!")
        return RunResult(
            n_cases_attempted=0,
            n_cases_processed=0,
            n_cases_errored=0,
            n_files_with_redactions=0,
            output_path=str(output_path),
        )

    # Load already-processed cases
    processed_cases = load_processed_cases(checkpoint_file)
    if processed_cases:
        logger.info(f"Resuming: Found {len(processed_cases)} already-processed cases")

    # Filter out already-processed cases
    remaining_files = [
        f
        for f in json_files
        if f.stem.replace("agency_case_file_bundle-", "") not in processed_cases
    ]

    if not remaining_files:
        logger.info("All cases already processed!")
        logger.info(f"Results available at: {output_path}")
        return RunResult(
            n_cases_attempted=len(json_files),
            n_cases_processed=len(processed_cases),
            n_cases_errored=0,
            n_files_with_redactions=0,
            output_path=str(output_path),
        )

    logger.info(f"Processing {len(remaining_files)} remaining cases")

    # Prepare arguments for parallel processing
    case_args = [
        (idx, json_path, len(json_files), llm, settings)
        for idx, json_path in enumerate(remaining_files, len(processed_cases) + 1)
    ]

    # Process cases in parallel
    case_workers = 15  # Match incident_date pipeline
    logger.info(f"Processing cases in parallel with {case_workers} workers")

    # Tracking variables
    pending_results = []  # Buffer for results between checkpoints
    total_files_with_redactions = 0
    cases_with_errors = 0
    error_details = []
    cases_processed = len(processed_cases)

    with ThreadPoolExecutor(max_workers=case_workers) as executor:
        futures = [executor.submit(process_case_wrapper, args) for args in case_args]

        for future in as_completed(futures):
            try:
                status, case_name, idx, data = future.result()

                if status == "error":
                    # data is the exception
                    cases_with_errors += 1
                    error_details.append(
                        {
                            "case_name": case_name,
                            "idx": idx,
                            "error": str(data),
                            "error_type": type(data).__name__,
                        }
                    )
                else:
                    # data is the results DataFrame
                    results_df = data

                    if not results_df.empty:
                        pending_results.append(results_df)
                        total_files_with_redactions += len(results_df)

                    # Mark case as processed
                    cases_processed += 1
                    save_processed_case(checkpoint_file, case_name)

                    # Checkpoint every 10 cases
                    if len(pending_results) >= 10:
                        batch_df = pd.concat(pending_results, ignore_index=True)
                        append_results_to_csv(batch_df, output_path)
                        pending_results = []
                        logger.info(f"{'=' * 60}")
                        logger.info(
                            f"CHECKPOINT: {cases_processed}/{len(json_files)} cases processed"
                        )
                        logger.info(f"{'=' * 60}")

            except Exception as e:
                cases_with_errors += 1
                logger.error(f"Case processing exception: {e}", exc_info=True)
                error_details.append(
                    {
                        "case_name": "unknown",
                        "idx": 0,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

    # Save any remaining results
    if pending_results:
        batch_df = pd.concat(pending_results, ignore_index=True)
        append_results_to_csv(batch_df, output_path)
        logger.info("Saved final batch of results")

    # Write error log if there were any errors
    if error_details:
        with open(error_log_path, "w") as error_log:
            error_log.write("REDACTION IDENTIFICATION ERRORS\n")
            error_log.write("=" * 80 + "\n")
            error_log.write(f"Generated: {pd.Timestamp.now()}\n")
            error_log.write("=" * 80 + "\n\n")

            for error in error_details:
                error_log.write(f"Case: {error['case_name']}\n")
                error_log.write(f"Index: {error['idx']}\n")
                error_log.write(f"Error: {error['error']}\n")
                error_log.write(f"Error type: {error['error_type']}\n")
                error_log.write("-" * 80 + "\n\n")

            # Write summary to error log
            error_log.write("\n" + "=" * 80 + "\n")
            error_log.write("SUMMARY\n")
            error_log.write("=" * 80 + "\n")
            error_log.write(f"Total cases processed: {len(json_files)}\n")
            error_log.write(f"Cases with errors: {cases_with_errors}\n")
            error_log.write(
                f"Cases successfully processed: {len(json_files) - cases_with_errors}\n"
            )
            error_log.write("=" * 80 + "\n")

        logger.info(f"Error log saved to {error_log_path}")
    else:
        logger.info("No errors encountered")

    # Create empty CSV if no results
    if not output_path.exists():
        empty_df = pd.DataFrame(
            columns=["sha1", "provisional_case_name", "page_numbers", "page_numbers_structured"]
        )
        empty_df.to_csv(output_path, index=False)
        logger.info(f"No redactions needed - created empty file at {output_path}")

    # Final summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total cases attempted: {len(json_files)}")
    logger.info(f"Cases successfully processed: {cases_processed}")
    logger.info(f"Cases with errors: {cases_with_errors}")
    logger.info(f"Files requiring redaction: {total_files_with_redactions}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)

    return RunResult(
        n_cases_attempted=len(json_files),
        n_cases_processed=cases_processed,
        n_cases_errored=cases_with_errors,
        n_files_with_redactions=total_files_with_redactions,
        output_path=str(output_path),
    )
