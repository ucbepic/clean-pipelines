"""Location (city) extraction pipeline.

Multi-stage flow:

  1. (Special cases only) summarize per-page OCR texts into per-doc summaries.
  2. Filter most-important summaries via the chunk-then-rerank strategy
     (chunk_size=4, final_count=2, max_allowed_summaries=5).
  3. Initial city analysis prompt.
  4. Validation prompt → JSON.
  5. Convert validated JSON into a structured "City, State" string (or None).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from string import Template

from prap_core.io import write_jsonl
from prap_core.llm import LLM
from prap_core.summary_filter import filter_important_summaries
from tqdm import tqdm

from .schemas import CaseRecord, LocationResult, RunResult, ValidationResult

logger = logging.getLogger("prap.location")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_location.prompts").joinpath(f"{name}.txt").read_text(encoding="utf-8")
    )


def _initial_location_analysis(llm: LLM, concatenated_summary: str) -> str:
    prompt = Template(_load_prompt("initial")).safe_substitute(source_text=concatenated_summary)
    return llm.complete(prompt).text


def _location_validation(
    llm: LLM, concatenated_summary: str, targeted_response: str
) -> ValidationResult:
    prompt = Template(_load_prompt("validation")).safe_substitute(
        source_text=concatenated_summary,
        targeted_response=targeted_response,
    )
    response = llm.complete(prompt).text
    try:
        return ValidationResult(**json.loads(response))
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse validation response as JSON: {response}")
        return ValidationResult(
            validation_decision="override",
            final_decision="NO",
            validator_reasoning="JSON parsing failed",
            verified_quote="",
            confidence="low",
            city_completeness="minimal",
            specificity_assessment="insufficient",
            additional_details="",
        )


def _convert_to_structured(llm: LLM, validation_result: ValidationResult) -> str | None:
    prompt = Template(_load_prompt("structured")).safe_substitute(
        final_decision=validation_result.final_decision,
        validation_decision=validation_result.validation_decision,
        validator_reasoning=validation_result.validator_reasoning,
        verified_quote=validation_result.verified_quote,
        confidence=validation_result.confidence,
        city_completeness=validation_result.city_completeness,
        specificity_assessment=validation_result.specificity_assessment,
        additional_details=validation_result.additional_details,
    )
    result = llm.complete(prompt).text.strip()
    if result.lower() == "none":
        return None
    return result


def _summarize_ocr_text(llm: LLM, ocr_text: str) -> str:
    prompt = Template(_load_prompt("ocr_summary")).safe_substitute(ocr_text=ocr_text)
    return llm.complete(prompt).text


def _summarize_ocr_texts(llm: LLM, ocr_texts: list[str], case_name: str) -> list[str]:
    summaries: list[str] = []
    for i, ocr_text in enumerate(ocr_texts):
        try:
            logger.info(
                f"Special case {case_name}: Summarizing OCR text section {i + 1}/{len(ocr_texts)}"
            )
            summaries.append(_summarize_ocr_text(llm, ocr_text))
        except Exception as e:
            logger.error(
                f"Error summarizing OCR text for special case {case_name}, section {i + 1}: {e}"
            )
            truncated = ocr_text[:2000] + "..." if len(ocr_text) > 2000 else ocr_text
            summaries.append(f"OCR Summary (truncated): {truncated}")
    logger.info(f"Special case {case_name}: Generated {len(summaries)} summaries from OCR texts")
    return summaries


# Mirrors location_city.py:765
SPECIAL_CASE_PREFIXES = [
    "1728506173798-unm",
    "1728506188915-ikx",
    "1728506198876-yxe",
    "1728506226039-het",
    "1728506250055-mpa",
    "1728506280750-mgr",
    "1725639855798-hhv",
    "1725640019247-qdl",
]


def process_case(
    llm: LLM,
    case: CaseRecord,
    filter_prompt: str,
    *,
    row_index: int = 0,
) -> LocationResult:
    case_name = case.provisional_case_name
    summaries_or_ocr_texts = list(case.summaries_or_ocr_texts)
    is_special_case = case.is_special_case

    try:
        if not summaries_or_ocr_texts:
            logger.warning(f"No summaries/OCR texts found for case {case_name}")
            return LocationResult(
                provisional_case_name=case_name,
                extracted_location=None,
                initial_analysis="No summaries/OCR texts found",
                validation_result=None,
                pipeline_stage_completed="no_content",
                is_special_case=is_special_case,
                note="No summaries/OCR texts found",
            )

        if is_special_case and any(
            case_name.startswith(prefix) for prefix in SPECIAL_CASE_PREFIXES
        ):
            logger.info(f"Special case {case_name}: Processing OCR text, creating summaries first")
            summaries = _summarize_ocr_texts(llm, summaries_or_ocr_texts, case_name)
        else:
            summaries = summaries_or_ocr_texts

        important_summaries = filter_important_summaries(
            llm,
            summaries,
            filter_prompt,
            log=logger,
            case_name=case_name,
            row_index=row_index,
            chunk_size=4,
            final_count=2,
            max_allowed_summaries=5,
        )

        concatenated_summary = "\n\n===== DOCUMENT BREAK =====\n\n".join(important_summaries)

        initial_analysis = _initial_location_analysis(llm, concatenated_summary)
        validation_result = _location_validation(llm, concatenated_summary, initial_analysis)
        extracted_location = _convert_to_structured(llm, validation_result)

        logger.info(f"Case {case_name}: Extracted location: {extracted_location}")
        logger.info(f"Case {case_name}: Validation decision: {validation_result.final_decision}")
        logger.info(f"Case {case_name}: Is special case: {is_special_case}")

        return LocationResult(
            provisional_case_name=case_name,
            extracted_location=extracted_location,
            initial_analysis=initial_analysis,
            validation_result=validation_result,
            pipeline_stage_completed="complete",
            is_special_case=is_special_case,
        )
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing case {case_name}: {error_message}")
        return LocationResult(
            provisional_case_name=case_name,
            extracted_location=None,
            initial_analysis=f"Error: {error_message}",
            validation_result=None,
            pipeline_stage_completed="error",
            is_special_case=is_special_case,
            note=f"Error: {error_message}",
        )


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 8,
    settings=None,
) -> RunResult:
    """Extract incident cities from a jsonl of `CaseRecord`s.

    Each input line is a `CaseRecord`; each output line is a `LocationResult`.
    """
    from prap_core.io import read_jsonl

    cases = [CaseRecord.model_validate(rec) for rec in read_jsonl(input_path)]
    if llm is None:
        llm = LLM(settings) if settings is not None else LLM()

    filter_prompt = _load_prompt("filter")
    results: list[LocationResult | None] = [None] * len(cases)

    special_count = sum(1 for c in cases if c.is_special_case)
    logger.info(
        f"Prepared {len(cases)} cases for processing "
        f"({special_count} special cases using OCR summarization)"
    )

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {
            ex.submit(process_case, llm, case, filter_prompt, row_index=i): i
            for i, case in enumerate(cases)
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(cases),
            desc="Processing cases with improved pipeline",
        ):
            results[futures[fut]] = fut.result()

    write_jsonl(output_path, [r.model_dump() for r in results if r is not None])
    logger.info(f"Wrote {len(cases)} results to {output_path}")
    return RunResult(n_cases=len(cases), output_path=str(output_path))
