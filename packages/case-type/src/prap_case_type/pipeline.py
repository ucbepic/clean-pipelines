"""Case-type classification pipeline: filter → master → three boolean classifications."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from string import Template

import tiktoken
from prap_core.io import write_jsonl
from prap_core.llm import LLM
from prap_core.summary_filter import filter_important_summaries
from tqdm import tqdm

from .helpers import natural_language_to_tristate_enum
from .schemas import CaseClassifications, CaseRecord, CaseTypeResult, RunResult

logger = logging.getLogger("prap.case_type")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_case_type.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


def classify_case(llm: LLM, source_text: str) -> CaseClassifications:
    """Master analysis followed by three boolean classifications."""
    master = llm.complete(
        Template(_load_prompt("master")).safe_substitute(source_text=source_text)
    ).text

    uof = llm.complete(Template(_load_prompt("uof")).safe_substitute(source_text=master)).text
    misconduct = llm.complete(
        Template(_load_prompt("misconduct")).safe_substitute(source_text=master)
    ).text
    ois = llm.complete(Template(_load_prompt("ois")).safe_substitute(source_text=master)).text

    return CaseClassifications(
        use_of_force=natural_language_to_tristate_enum(uof),
        misconduct=natural_language_to_tristate_enum(misconduct),
        officer_involved_shooting=natural_language_to_tristate_enum(ois),
    )


def process_case(
    llm: LLM,
    case: CaseRecord,
    filter_prompt: str,
    *,
    row_index: int = 0,
) -> CaseTypeResult:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        summaries = list(case.summaries) or ["No summary"]
        total_tokens = sum(len(encoding.encode(s)) for s in summaries)
        logger.info(
            f"Case {case.provisional_case_name} (Row {row_index}): "
            f"Initial summaries count: {len(summaries)}, total tokens: {total_tokens}"
        )

        if case.ocr_texts and all(len(encoding.encode(t)) < 250 for t in case.ocr_texts if t):
            logger.warning(
                f"Case {case.provisional_case_name} (Row {row_index}): "
                "All documents have less than 50 tokens of OCR text, classifying as Unclear"
            )
            return CaseTypeResult(
                provisional_case_name=case.provisional_case_name,
                classification=CaseClassifications(
                    use_of_force="Unclear",
                    misconduct="Unclear",
                    officer_involved_shooting="Unclear",
                ),
                note="All documents have insufficient text content (< 50 tokens)",
            )

        important = filter_important_summaries(
            llm,
            summaries,
            filter_prompt,
            log=logger,
            case_name=case.provisional_case_name,
            row_index=row_index,
            add_missing_fallback_indices=True,
            cap_combined=False,
        )
        concatenated = "\n\n===== DOCUMENT BREAK =====\n\n".join(important)

        token_count = len(encoding.encode(concatenated))
        logger.info(
            f"Case {case.provisional_case_name} (Row {row_index}): "
            f"After filtering, concatenated token count: {token_count}"
        )

        result = classify_case(llm, concatenated)
        return CaseTypeResult(
            provisional_case_name=case.provisional_case_name,
            classification=result,
            note=str(result),
        )
    except Exception as e:
        logger.error(f"Error processing case {case.provisional_case_name} (Row {row_index}): {e}")
        return CaseTypeResult(
            provisional_case_name=case.provisional_case_name,
            classification=None,
            note=f"Error: {e}",
        )


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 50,
) -> RunResult:
    """Classify cases from a jsonl of `CaseRecord`s into use-of-force / misconduct / OIS labels.

    Each input line is a `CaseRecord`; each output line is a `CaseTypeResult`.
    """
    from prap_core.io import read_jsonl

    cases = [CaseRecord.model_validate(rec) for rec in read_jsonl(input_path)]
    if llm is None:
        llm = LLM()

    filter_prompt = _load_prompt("filter")
    results: list[CaseTypeResult | None] = [None] * len(cases)

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
