"""Citation-finding logic for verified agencies."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from jinja2 import Template

from .schemas import PrimaryCitationAnalysis, ValidatorCitationResult

if TYPE_CHECKING:
    from prap_core.llm import LLM

logger = logging.getLogger("prap.involved_agency.citations")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_involved_agency.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


def analyze_page_for_agency_citation(
    llm: "LLM",
    page_text: str,
    page_number: int,
    file_name: str,
    agency_name: str,
    agency_type: str,
    extraction_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Two-stage analysis: Primary LLM searches for agency citations,
    followed by validator LLM to verify the citation quality.
    """

    primary_prompt_template = Template(_load_prompt("primary_citation"))
    validator_prompt_template = Template(_load_prompt("validator_citation"))

    try:
        # Stage 1: Primary analysis
        primary_prompt = primary_prompt_template.render(
            page_text=page_text,
            agency_name=agency_name,
            agency_type=agency_type,
            extraction_context=extraction_context or "No additional context provided",
        )

        primary_result = llm.complete(
            primary_prompt, response_format=PrimaryCitationAnalysis
        )

        if not primary_result.has_citation:
            return {
                "page_number": page_number,
                "file_name": file_name,
                "match": False,
                "primary_analysis": primary_result.reasoning,
                "validator_analysis": "Skipped - primary analysis found no citation",
                "quote": primary_result.quote,
            }

        # Stage 2: Validator analysis
        validator_prompt = validator_prompt_template.render(
            agency_name=agency_name,
            agency_type=agency_type,
            primary_has_citation=primary_result.has_citation,
            primary_reasoning=primary_result.reasoning,
            primary_quote=primary_result.quote,
            primary_confidence=primary_result.confidence,
            page_text=page_text,
        )

        validator_result = llm.complete(
            validator_prompt, response_format=ValidatorCitationResult
        )

        return {
            "page_number": page_number,
            "file_name": file_name,
            "match": validator_result.final_decision,
            "primary_analysis": primary_result.reasoning,
            "validator_analysis": validator_result.validator_reasoning,
            "quote": validator_result.verified_quote,
            "evidence_strength": validator_result.evidence_strength,
        }

    except Exception as e:
        logger.error(f"Error analyzing page {page_number} from {file_name}: {e}")
        return {
            "page_number": page_number,
            "file_name": file_name,
            "match": False,
            "primary_analysis": f"Error: {str(e)}",
            "validator_analysis": "Error during analysis",
            "quote": "",
            "evidence_strength": "WEAK",
        }


def analyze_page_wrapper(
    llm: "LLM",
    page_data: Dict[str, Any],
    file_name: str,
    file_id: str,
    gdrive_id: str,
    agency_name: str,
    agency_type: str,
    extraction_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Wrapper function for parallel processing of page analysis."""
    page_number = page_data.get("page_number", 0)
    page_content = page_data.get("text", "")

    if not page_content or len(page_content.strip()) < 50:
        logger.debug(
            f"  Page {page_number} ({file_name}): Skipped (content too short: {len(page_content)} chars)"
        )
        return None

    logger.info(
        f"  Page {page_number} ({file_name}): Analyzing ({len(page_content)} chars)..."
    )

    result = analyze_page_for_agency_citation(
        llm=llm,
        page_text=page_content,
        page_number=page_number,
        file_name=file_name,
        agency_name=agency_name,
        agency_type=agency_type,
        extraction_context=extraction_context,
    )

    if result["match"]:
        citation = {
            "file_name": file_name,
            "file_id": file_id,
            "gdrive_id": gdrive_id,
            "gdrive_url": f"https://drive.google.com/file/d/{gdrive_id}/view"
            if gdrive_id
            else "",
            "page_number": page_number,
            "quote": result["quote"],
            "validator_reasoning": result["validator_analysis"],
            "evidence_strength": result.get("evidence_strength", "EXPLICIT"),
            "agency_name": agency_name,
            "agency_type": agency_type,
        }
        logger.info(f"    ✓✓✓ CITATION FOUND ✓✓✓")
        logger.info(f"    Agency: {agency_name} ({agency_type})")
        logger.info(f"    File: {file_name}, Page: {page_number}")
        logger.info(
            f"    Evidence Strength: {result.get('evidence_strength', 'N/A')}"
        )
        logger.info(f"    Quote preview: {result['quote'][:150]}...")
        return citation
    else:
        logger.debug(f"    ✗ No match (File: {file_name}, Page: {page_number})")
        logger.debug(f"    Primary analysis: {result['primary_analysis'][:100]}...")
        logger.debug(f"    Validator: {result['validator_analysis'][:100]}...")
        return None


def find_agency_citations(
    llm: "LLM",
    case_files: List[Dict[str, Any]],
    agency_name: str,
    agency_type: str,
    extraction_context: Optional[str] = None,
    max_citations: int = 3,
    max_workers: int = 10,
) -> List[Dict[str, Any]]:
    """Search through case files to find 2-3 citations supporting the agency extraction.

    Uses parallel processing to analyze pages concurrently.
    """
    logger.info(f"=" * 60)
    logger.info(
        f"CITATION SEARCH: Looking for up to {max_citations} citations"
    )
    logger.info(f"Agency: {agency_name} ({agency_type})")
    logger.info(f"Using parallel processing with {max_workers} workers")
    logger.info(f"=" * 60)

    citations_found: List[Dict[str, Any]] = []
    total_pages_analyzed = 0
    total_pages_skipped = 0

    page_tasks = []
    for file_idx, file_data in enumerate(case_files, 1):
        file_name = file_data.get("file_name", "unknown")
        file_id = file_data.get("sha1", f"file_{file_idx}")
        gdrive_id = file_data.get("page_range", {}).get("gdrive_id", "")

        ocr_data = file_data.get("ocr_doc_text_per_page", {})
        page_texts = ocr_data.get("page_texts", [])

        if not page_texts:
            logger.warning(
                f"File {file_idx}/{len(case_files)}: No page_texts found for {file_name}"
            )
            continue

        logger.info(f"\nFile {file_idx}/{len(case_files)}: {file_name}")
        logger.info(f"  File ID: {file_id}")
        logger.info(f"  GDrive ID: {gdrive_id}")
        logger.info(f"  Pages to search: {len(page_texts)}")

        for page_data in page_texts:
            page_tasks.append(
                {
                    "page_data": page_data,
                    "file_name": file_name,
                    "file_id": file_id,
                    "gdrive_id": gdrive_id,
                    "agency_name": agency_name,
                    "agency_type": agency_type,
                    "extraction_context": extraction_context,
                }
            )

    logger.info(f"\nTotal pages queued for analysis: {len(page_tasks)}")
    logger.info(f"Starting parallel processing...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                analyze_page_wrapper,
                llm,
                task["page_data"],
                task["file_name"],
                task["file_id"],
                task["gdrive_id"],
                task["agency_name"],
                task["agency_type"],
                task["extraction_context"],
            ): task
            for task in page_tasks
        }

        for future in as_completed(future_to_task):
            if len(citations_found) >= max_citations:
                logger.info(
                    f"\n✓ Found {max_citations} citations. Cancelling remaining tasks."
                )
                for f in future_to_task:
                    f.cancel()
                break

            try:
                result = future.result()
                if result is not None:
                    citations_found.append(result)
                    logger.info(
                        f"    Progress: {len(citations_found)}/{max_citations} citations found"
                    )
                else:
                    total_pages_skipped += 1

                total_pages_analyzed += 1

            except Exception as e:
                task = future_to_task[future]
                logger.error(
                    f"Error processing page from {task['file_name']}: {e}"
                )
                total_pages_analyzed += 1

    logger.info(f"\n" + "=" * 60)
    logger.info(f"CITATION SEARCH COMPLETE")
    logger.info(f"=" * 60)
    logger.info(f"Agency: {agency_name} ({agency_type})")
    logger.info(
        f"Citations found: {len(citations_found)}/{max_citations}"
    )
    logger.info(f"Total pages analyzed: {total_pages_analyzed}")
    logger.info(f"Total pages skipped: {total_pages_skipped}")
    logger.info(f"=" * 60)

    return citations_found


def format_citations_for_output(citations: List[Dict[str, Any]]) -> str:
    """Format citations list into a readable string for CSV output."""
    if not citations:
        return "No citations found"

    formatted = []
    for i, citation in enumerate(citations, 1):
        formatted.append(
            f"Citation {i}:\n"
            f"  Agency: {citation.get('agency_name', 'N/A')} ({citation.get('agency_type', 'N/A')})\n"
            f"  File: {citation['file_name']}\n"
            f"  File ID: {citation.get('file_id', 'N/A')}\n"
            f"  Page: {citation['page_number']}\n"
            f"  Evidence Strength: {citation.get('evidence_strength', 'N/A')}\n"
            f"  Quote: {citation['quote']}\n"
            f"  Reasoning: {citation['validator_reasoning']}"
        )

    return "\n\n".join(formatted)
