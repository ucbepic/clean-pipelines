"""Involved-agency extraction pipeline: extract → verify → cite."""

from __future__ import annotations

import concurrent.futures
import json
import logging
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from jinja2 import Template

from .citations import find_agency_citations
from .helpers import clean_summaries, filter_important_summaries
from .schemas import (
    Agency,
    AgencyExtraction,
    RunResult,
    SingleAgencyVerification,
)

if TYPE_CHECKING:
    from prap_core.llm import LLM

logger = logging.getLogger("prap.involved_agency")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_involved_agency.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


# ============================================================================
# LLM steps
# ============================================================================


def extract_agencies(
    llm: LLM,
    first_look_summaries: dict[str, str],
    case_name: str | None = None,
) -> AgencyExtraction:
    """Extract investigating and responding agencies from document summaries."""
    first_look_summaries = clean_summaries(first_look_summaries)

    summaries_list = [
        f"# {slug}\n\n{summary}" for slug, summary in sorted(first_look_summaries.items())
    ]

    filtered_summaries = filter_important_summaries(llm, summaries_list)
    concatenated_summary = "\n\n".join(filtered_summaries)

    prompt_template = Template(_load_prompt("extract_agencies"))
    initial_prompt = prompt_template.render(source_text=concatenated_summary)

    agency_extraction = llm.complete(initial_prompt, response_format=AgencyExtraction)
    return agency_extraction


def verify_agency(
    llm: LLM,
    agency: Agency,
    agency_type: str,
    source_text: str,
    all_extraction_context: str,
    case_name: str | None = None,
) -> dict[str, Any]:
    """Verify a single agency extraction against source documents."""
    verification_template = Template(_load_prompt("verify_agency"))

    verification_prompt = verification_template.render(
        agency=agency,
        agency_type=agency_type,
        source_text=source_text,
        all_extraction_context=all_extraction_context,
    )

    verification_result = llm.complete(
        verification_prompt, response_format=SingleAgencyVerification
    )
    return verification_result.model_dump()


# ============================================================================
# Per-case worker
# ============================================================================


def format_case_summaries_for_filtering(case_files: list[dict]) -> dict[str, str]:
    """Convert case_files list into {slug: summary} format."""
    summaries_dict = {}
    for file_data in case_files:
        file_name = file_data.get("file_name", "unknown")
        summary = file_data.get("summary", "")

        if summary and summary.strip():
            summaries_dict[file_name] = summary

    return summaries_dict


def process_case(
    llm: LLM,
    case_data: dict[str, Any],
    case_name: str,
) -> list[dict[str, Any]]:
    """Process a single case to extract agencies and find citations.

    `case_data` is the full case bundle dict (an `agency_case_file_bundle-*.json`
    structure). Returns one row per agency-role combination, with at least one
    row per case.
    """
    logger.info(f"Processing case: {case_name}")

    case_files = case_data.get("case_files", [])

    # Extract provisional_case_name (falls back to the case name if absent)
    provisional_case_name = case_data.get("provisional_case_name", case_name)

    case_url = (
        f"https://clean.calmatters.org/cases/{provisional_case_name}"
        if provisional_case_name
        else ""
    )

    if not case_files:
        logger.warning(f"No files found in case {case_name}")
        return [
            {
                "case_name": case_name,
                "provisional_case_name": provisional_case_name,
                "case_url": case_url,
                "agency_name": None,
                "agency_type": None,
                "agency_found": False,
                "verified": False,
                "verification_status": None,
                "confidence_level": None,
                "llm_reasoning": "No files found in case",
                "role_description": None,
                "evidence": None,
                "has_dual_role": False,
                "dual_role_note": None,
                "num_citations": 0,
                "citations": None,
                "full_extraction_response": "No files found in case",
            }
        ]

    summaries_dict = format_case_summaries_for_filtering(case_files)

    if not summaries_dict:
        logger.warning(f"No summaries found in case {case_name}")
        return [
            {
                "case_name": case_name,
                "provisional_case_name": provisional_case_name,
                "case_url": case_url,
                "agency_name": None,
                "agency_type": None,
                "agency_found": False,
                "verified": False,
                "verification_status": None,
                "confidence_level": None,
                "llm_reasoning": "No summaries found in case files",
                "role_description": None,
                "evidence": None,
                "has_dual_role": False,
                "dual_role_note": None,
                "num_citations": 0,
                "citations": None,
                "full_extraction_response": "No summaries found in case files",
            }
        ]

    logger.info(f"Extracting agencies for case {case_name}")
    extraction_result = extract_agencies(llm, summaries_dict, case_name=case_name)

    if not extraction_result.responding_agencies and not extraction_result.investigating_agencies:
        logger.warning(f"No agencies extracted for case {case_name}")
        return [
            {
                "case_name": case_name,
                "provisional_case_name": provisional_case_name,
                "case_url": case_url,
                "agency_name": None,
                "agency_type": None,
                "agency_found": False,
                "verified": False,
                "verification_status": None,
                "confidence_level": None,
                "llm_reasoning": "No agencies extracted from summaries",
                "role_description": None,
                "evidence": None,
                "has_dual_role": False,
                "dual_role_note": None,
                "num_citations": 0,
                "citations": None,
                "full_extraction_response": str(extraction_result.model_dump())
                if extraction_result
                else "No extraction result",
            }
        ]

    extraction_context = (
        f"RESPONDING AGENCIES: {[a.agency_name for a in extraction_result.responding_agencies]}\n"
        f"INVESTIGATING AGENCIES: "
        f"{[a.agency_name for a in extraction_result.investigating_agencies]}"
    )

    summaries_list = [f"# {slug}\n\n{summary}" for slug, summary in sorted(summaries_dict.items())]
    filtered_summaries = filter_important_summaries(llm, summaries_list)
    source_text = "\n\n".join(filtered_summaries)

    results: list[dict[str, Any]] = []

    # Process responding agencies
    for agency in extraction_result.responding_agencies:
        logger.info(f"Verifying RESPONDING agency: {agency.agency_name}")

        verification = verify_agency(
            llm=llm,
            agency=agency,
            agency_type="RESPONDING",
            source_text=source_text,
            all_extraction_context=extraction_context,
            case_name=case_name,
        )

        if verification["recommendation"] != "INCLUDE":
            logger.warning(
                f"Agency {agency.agency_name} excluded: {verification['verification_reasoning']}"
            )
            continue

        if verification["has_dual_role"]:
            logger.info(
                f"Agency {verification['verified_agency_name']} has dual role - creating two rows"
            )

            citations_responding = find_agency_citations(
                llm=llm,
                case_files=case_files,
                agency_name=verification["verified_agency_name"],
                agency_type="RESPONDING",
                extraction_context=extraction_context,
                max_citations=3,
            )

            results.append(
                _row(
                    case_name,
                    provisional_case_name,
                    case_url,
                    verification,
                    "RESPONDING",
                    citations_responding,
                    extraction_context,
                    dual=True,
                )
            )

            citations_investigating = find_agency_citations(
                llm=llm,
                case_files=case_files,
                agency_name=verification["verified_agency_name"],
                agency_type="INVESTIGATING",
                extraction_context=extraction_context,
                max_citations=3,
            )

            results.append(
                _row(
                    case_name,
                    provisional_case_name,
                    case_url,
                    verification,
                    "INVESTIGATING",
                    citations_investigating,
                    extraction_context,
                    dual=True,
                )
            )

        else:
            citations = find_agency_citations(
                llm=llm,
                case_files=case_files,
                agency_name=verification["verified_agency_name"],
                agency_type="RESPONDING",
                extraction_context=extraction_context,
                max_citations=3,
            )

            results.append(
                _row(
                    case_name,
                    provisional_case_name,
                    case_url,
                    verification,
                    "RESPONDING",
                    citations,
                    extraction_context,
                    dual=False,
                )
            )

    # Process investigating agencies
    for agency in extraction_result.investigating_agencies:
        logger.info(f"Verifying INVESTIGATING agency: {agency.agency_name}")

        verification = verify_agency(
            llm=llm,
            agency=agency,
            agency_type="INVESTIGATING",
            source_text=source_text,
            all_extraction_context=extraction_context,
            case_name=case_name,
        )

        if verification["recommendation"] != "INCLUDE":
            logger.warning(
                f"Agency {agency.agency_name} excluded: {verification['verification_reasoning']}"
            )
            continue

        if verification["has_dual_role"]:
            agency_already_added = any(
                r["agency_name"] == verification["verified_agency_name"] and r["has_dual_role"]
                for r in results
            )

            if agency_already_added:
                logger.info(
                    f"Dual-role agency {verification['verified_agency_name']} "
                    f"already processed - skipping"
                )
                continue

            logger.info(
                f"Agency {verification['verified_agency_name']} has dual role - creating two rows"
            )

            citations_investigating = find_agency_citations(
                llm=llm,
                case_files=case_files,
                agency_name=verification["verified_agency_name"],
                agency_type="INVESTIGATING",
                extraction_context=extraction_context,
                max_citations=3,
            )

            results.append(
                _row(
                    case_name,
                    provisional_case_name,
                    case_url,
                    verification,
                    "INVESTIGATING",
                    citations_investigating,
                    extraction_context,
                    dual=True,
                )
            )

            citations_responding = find_agency_citations(
                llm=llm,
                case_files=case_files,
                agency_name=verification["verified_agency_name"],
                agency_type="RESPONDING",
                extraction_context=extraction_context,
                max_citations=3,
            )

            results.append(
                _row(
                    case_name,
                    provisional_case_name,
                    case_url,
                    verification,
                    "RESPONDING",
                    citations_responding,
                    extraction_context,
                    dual=True,
                )
            )

        else:
            citations = find_agency_citations(
                llm=llm,
                case_files=case_files,
                agency_name=verification["verified_agency_name"],
                agency_type="INVESTIGATING",
                extraction_context=extraction_context,
                max_citations=3,
            )

            results.append(
                _row(
                    case_name,
                    provisional_case_name,
                    case_url,
                    verification,
                    "INVESTIGATING",
                    citations,
                    extraction_context,
                    dual=False,
                )
            )

    if not results:
        logger.warning(f"Case {case_name}: Agencies extracted but all excluded during verification")
        return [
            {
                "case_name": case_name,
                "provisional_case_name": provisional_case_name,
                "case_url": case_url,
                "agency_name": None,
                "agency_type": None,
                "agency_found": False,
                "verified": False,
                "verification_status": None,
                "confidence_level": None,
                "llm_reasoning": "All extracted agencies excluded during verification",
                "role_description": None,
                "evidence": None,
                "has_dual_role": False,
                "dual_role_note": None,
                "num_citations": 0,
                "citations": None,
                "full_extraction_response": extraction_context,
            }
        ]

    logger.info(f"Case {case_name}: Extracted {len(results)} agency-role rows total")
    return results


def _row(
    case_name: str,
    provisional_case_name: str,
    case_url: str,
    verification: dict[str, Any],
    agency_type: str,
    citations: list[dict[str, Any]],
    extraction_context: str,
    *,
    dual: bool,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "provisional_case_name": provisional_case_name,
        "case_url": case_url,
        "agency_name": verification["verified_agency_name"],
        "agency_type": agency_type,
        "agency_found": True,
        "verified": verification["verification_status"] == "CONFIRMED",
        "verification_status": verification["verification_status"],
        "confidence_level": verification["confidence_level"],
        "llm_reasoning": verification["verification_reasoning"],
        "role_description": verification["verified_role_description"],
        "evidence": json.dumps(verification["verified_evidence"]),
        "has_dual_role": dual,
        "dual_role_note": verification["dual_role_note"] if dual else None,
        "num_citations": len(citations),
        "citations": json.dumps(citations, indent=2) if citations else "[]",
        "full_extraction_response": extraction_context,
    }


# ============================================================================
# Main entrypoint
# ============================================================================


def _process_case_wrapper(args):
    """Wrapper function for parallel case processing."""
    idx, total_cases, case_record, llm = args
    case_name = case_record.get("case_name") or case_record.get(
        "provisional_case_name", f"case_{idx}"
    )

    try:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Case {idx}/{total_cases}: {case_name}")
        logger.info(f"{'=' * 80}")

        results = process_case(llm, case_record, case_name)

        agencies_count = len(results) if results else 0

        if agencies_count > 0:
            logger.info(f"✓ Case {case_name}: Extracted {agencies_count} rows")
            return ("success", case_name, idx, results)
        else:
            logger.info(f"○ Case {case_name}: No agencies extracted")
            return ("no_agencies", case_name, idx, results)

    except Exception as e:
        logger.error(f"✗ Error processing case {case_name}: {e}", exc_info=True)
        return ("error", case_name, idx, e)


def run(
    input: str | Path,
    output: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 15,
    settings: Any = None,
    resume: bool = False,
    save_every: int = 10,
) -> RunResult:
    """Run the involved-agency extraction pipeline.

    Args:
        input: jsonl file where each line is a case bundle dict (see `prepare`).
        output: CSV path to write per-(case,agency,role) rows.
        llm: optional `prap_core.llm.LLM` instance. If None, constructed from env.
        n_threads: number of cases processed in parallel.
        settings: optional `prap_core.config.Settings` (used if `llm` is None).
        resume: if True, skip cases already present in the output CSV.
        save_every: write a CSV checkpoint after every N completed cases.
    """
    from prap_core.io import read_jsonl

    input_path = Path(input)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_log_path = output_path.parent / "extraction_errors.txt"

    if llm is None:
        from prap_core.llm import LLM

        llm = LLM(settings) if settings is not None else LLM()

    logger.info("=" * 80)
    logger.info("AGENCY EXTRACTION PIPELINE")
    logger.info("=" * 80)

    case_records = list(read_jsonl(input_path))
    logger.info(f"Loaded {len(case_records)} case records from {input_path}")

    if len(case_records) == 0:
        logger.error("No case records found in input!")
        df = pd.DataFrame()
        df.to_csv(output_path, index=False)
        return RunResult(n_cases=0, n_agencies_extracted=0, output_path=str(output_path))

    # Resume support
    processed_cases: set[str] = set()
    existing_results: list = []
    if resume and output_path.exists():
        logger.info(f"Resume mode enabled - loading existing results from {output_path}")
        try:
            existing_df = pd.read_csv(output_path)
            if "case_name" in existing_df.columns:
                processed_cases = set(existing_df["case_name"].unique())
                existing_results = existing_df.to_dict("records")
                logger.info(f"Found {len(processed_cases)} already processed cases")
        except Exception as e:
            logger.warning(f"Could not load existing CSV: {e} - starting fresh")

        case_records = [
            r
            for r in case_records
            if (r.get("case_name") or r.get("provisional_case_name")) not in processed_cases
        ]
        logger.info(f"Remaining cases to process: {len(case_records)}")

        if len(case_records) == 0:
            logger.info("All cases already processed! Nothing to do.")
            return RunResult(
                n_cases=len(processed_cases),
                n_agencies_extracted=len(existing_results),
                output_path=str(output_path),
            )
    elif not resume and output_path.exists():
        logger.info(f"Resume disabled - removing existing output: {output_path}")
        output_path.unlink()

    case_args = [
        (idx, len(case_records), case_record, llm)
        for idx, case_record in enumerate(case_records, 1)
    ]

    logger.info(f"Processing cases in parallel with {n_threads} workers")
    logger.info(f"Incremental save: every {save_every} cases")

    all_results: list = existing_results.copy() if resume else []
    total_agencies_extracted = 0
    cases_with_agencies = 0
    cases_without_agencies = 0
    cases_with_errors = 0
    error_details: list = []
    cases_completed_since_last_save = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_process_case_wrapper, args) for args in case_args]

        for future in concurrent.futures.as_completed(futures):
            try:
                status, case_name, idx, data = future.result()

                if status == "error":
                    cases_with_errors += 1
                    cases_without_agencies += 1
                    error_details.append(
                        {
                            "case_name": case_name,
                            "idx": idx,
                            "error": str(data),
                            "error_type": type(data).__name__,
                        }
                    )
                elif status == "no_agencies":
                    cases_without_agencies += 1
                    if data:
                        all_results.extend(data)
                else:
                    case_results = data
                    all_results.extend(case_results)
                    total_agencies_extracted += len(case_results)
                    cases_with_agencies += 1

                cases_completed_since_last_save += 1
                if cases_completed_since_last_save >= save_every:
                    logger.info(
                        f"Checkpoint: Saving {len(all_results)} rows to CSV "
                        f"after {cases_completed_since_last_save} cases"
                    )
                    df_checkpoint = pd.DataFrame(all_results)
                    df_checkpoint.to_csv(output_path, index=False)
                    cases_completed_since_last_save = 0

            except Exception as e:
                cases_with_errors += 1
                cases_without_agencies += 1
                logger.error(f"Case processing exception: {e}", exc_info=True)
                error_details.append(
                    {
                        "case_name": "unknown",
                        "idx": 0,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

    if error_details:
        with open(error_log_path, "w") as error_log:
            error_log.write("AGENCY EXTRACTION ERRORS\n")
            error_log.write("=" * 80 + "\n")
            error_log.write(f"Generated: {pd.Timestamp.now()}\n")
            error_log.write("=" * 80 + "\n\n")

            for error in error_details:
                error_log.write(f"Case: {error['case_name']}\n")
                error_log.write(f"Index: {error['idx']}\n")
                error_log.write(f"Error: {error['error']}\n")
                error_log.write(f"Error type: {error['error_type']}\n")
                error_log.write("-" * 80 + "\n\n")

            error_log.write("\n" + "=" * 80 + "\n")
            error_log.write("SUMMARY\n")
            error_log.write("=" * 80 + "\n")
            error_log.write(f"Total cases processed: {len(case_records)}\n")
            error_log.write(f"Cases with errors: {cases_with_errors}\n")
            error_log.write(
                f"Cases successfully processed: {len(case_records) - cases_with_errors}\n"
            )
            error_log.write("=" * 80 + "\n")

        logger.info(f"\n✓ Error log saved to {error_log_path}")
    else:
        logger.info("\n✓ No errors encountered")

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(all_results)} rows to {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total cases attempted: {len(case_records)}")
    logger.info(f"Cases successfully processed: {len(case_records) - cases_with_errors}")
    logger.info(f"Cases with errors: {cases_with_errors}")
    logger.info(f"Cases with agencies: {cases_with_agencies}")
    logger.info(f"Cases without agencies: {cases_without_agencies}")
    logger.info(f"Total agency rows extracted: {total_agencies_extracted}")
    logger.info("=" * 80)

    return RunResult(
        n_cases=len(case_records),
        n_agencies_extracted=total_agencies_extracted,
        output_path=str(output_path),
    )
