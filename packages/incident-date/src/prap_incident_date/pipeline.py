"""Incident-date extraction pipeline: three-stage extract → verify → ISO-8601 prompt chain over a chunked summary filter."""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from string import Template

import dateparser
from prap_core.io import write_jsonl
from prap_core.llm import LLM
from prap_core.summary_filter import filter_important_summaries
from tqdm import tqdm

from .schemas import CaseRecord, IncidentDateResult, RunResult

logger = logging.getLogger("prap.incident_date")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_incident_date.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


def _prompt_for_incident_date(llm: LLM, concatenated_summary: str) -> str:
    initial = llm.complete(
        Template(_load_prompt("extract")).safe_substitute(source_text=concatenated_summary)
    ).text

    # NOTE: the verify template references `$initial_dates` (plural) but we pass
    # `initial_date` (singular). The placeholder is intentionally left
    # un-substituted — changing this changes model output.
    return llm.complete(
        Template(_load_prompt("verify")).safe_substitute(
            initial_date=initial, source_text=concatenated_summary
        )
    ).text


def _prompt_convert_nl_date_to_iso8601(llm: LLM, summary: str) -> list[str] | None:
    prompt = Template(_load_prompt("to_iso")).safe_substitute(source_text=summary)
    result = llm.complete(prompt).text.strip()
    if result.lower() == "none":
        return None
    dates: list[str] = []
    for date_str in result.split(","):
        date_str = date_str.strip()
        try:
            parsed = dateparser.parse(date_str)
            if parsed:
                dates.append(parsed.strftime("%Y-%m-%d"))
        except Exception as e:
            logger.warning(f"Error parsing date '{date_str}': {e}")
    return dates if dates else None


def _normalize_summaries(case_name: str, raw: list) -> list[str]:
    summaries: list[str] = []
    for i, s in enumerate(raw):
        if s is None:
            logger.warning(
                f"Case {case_name}: Found None summary at index {i}, replacing with 'No summary'"
            )
            summaries.append("No summary")
        elif not isinstance(s, str):
            logger.warning(
                f"Case {case_name}: Found non-string summary at index {i} "
                f"(type: {type(s)}), converting to string"
            )
            summaries.append(str(s))
        else:
            summaries.append(s)
    return summaries or ["No summary"]


def process_case(
    llm: LLM,
    case: CaseRecord,
    filter_prompt: str,
    *,
    row_index: int = 0,
) -> IncidentDateResult:
    try:
        if case.ocr_pages:
            logger.info(
                f"Case {case.provisional_case_name}: "
                "Processing OCR text directly (skipping filtering)"
            )
            concatenated = "\n\n===== DOCUMENT BREAK =====\n\n".join(case.ocr_pages)
        else:
            summaries = _normalize_summaries(case.provisional_case_name, list(case.summaries))
            important = filter_important_summaries(
                llm,
                summaries,
                filter_prompt,
                log=logger,
                case_name=case.provisional_case_name,
                row_index=row_index,
                add_missing_fallback_indices=False,
                cap_combined=True,
            )
            concatenated = "\n\n===== DOCUMENT BREAK =====\n\n".join(important)

        nl_date = _prompt_for_incident_date(llm, concatenated)
        extracted = _prompt_convert_nl_date_to_iso8601(llm, nl_date)
        logger.info(f"Case {case.provisional_case_name}: Extracted dates: {extracted}")
        return IncidentDateResult(
            provisional_case_name=case.provisional_case_name,
            extracted_date=extracted,
            nl_date=nl_date,
        )
    except Exception as e:
        logger.error(f"Error processing case {case.provisional_case_name}: {e}")
        logger.error(traceback.format_exc())
        return IncidentDateResult(
            provisional_case_name=case.provisional_case_name,
            extracted_date=None,
            nl_date=f"Error: {e}",
        )


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 20,
) -> RunResult:
    """Extract incident dates from a jsonl of case records.

    Each input line is a `CaseRecord` (see schemas.py); each output line is
    an `IncidentDateResult`. The LLM is injectable; if omitted, a default
    `LLM()` is constructed (which reads PRAP_LLM_* env vars).
    """
    from prap_core.io import read_jsonl

    cases = [CaseRecord.model_validate(rec) for rec in read_jsonl(input_path)]
    if llm is None:
        llm = LLM()

    filter_prompt = _load_prompt("filter")
    results: list[IncidentDateResult | None] = [None] * len(cases)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {
            ex.submit(process_case, llm, case, filter_prompt, row_index=i): i
            for i, case in enumerate(cases)
        }
        for fut in tqdm(as_completed(futures), total=len(cases), desc="Processing cases"):
            results[futures[fut]] = fut.result()

    write_jsonl(output_path, [r.model_dump() for r in results if r is not None])
    logger.info(f"Wrote {len(cases)} results to {output_path}")
    return RunResult(n_cases=len(cases), output_path=str(output_path))
