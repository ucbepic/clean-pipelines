"""Officer-name validation pipeline.

Two-stage LLM chain (extract → validate). Each unique cleaned name is
classified at most once; the result is mapped back over the full input
list of officer-name strings.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from string import Template

from prap_core.io import read_jsonl, write_jsonl
from prap_core.llm import LLM, LLMError
from tqdm import tqdm

from .cleaning import clean_officer_name, is_obviously_invalid
from .schemas import (
    NameClassification,
    NameExtractionResult,
    NameRecord,
    NameValidationResult,
    RunResult,
)

logger = logging.getLogger("prap.split_officer_names")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_split_officer_names.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


def _extract(llm: LLM, name_string: str) -> NameExtractionResult:
    prompt = Template(_load_prompt("extract")).safe_substitute(name_string=name_string)
    return llm.complete(prompt, response_format=NameExtractionResult)


def _validate(llm: LLM, name_string: str, extraction: NameExtractionResult) -> NameValidationResult:
    prompt = Template(_load_prompt("validate")).safe_substitute(
        name_string=name_string,
        is_valid_name=str(extraction.is_valid_name),
        first_name=extraction.extracted_parts.first_name,
        last_name=extraction.extracted_parts.last_name,
        middle_name=extraction.extracted_parts.middle_name,
        suffix=extraction.extracted_parts.suffix,
    )
    return llm.complete(prompt, response_format=NameValidationResult)


def classify_name(llm: LLM, name_string: str) -> dict:
    """Classify a single officer-name string. Returns the per-name dict."""
    empty = {
        "valid_name": 0,
        "first_name": "",
        "last_name": "",
        "middle_name": "",
        "suffix": "",
    }
    if not name_string or not name_string.strip():
        return empty

    cleaned = clean_officer_name(name_string)
    if is_obviously_invalid(cleaned):
        logger.debug(f"Name '{name_string}' -> '{cleaned}' rejected by rules-based filter")
        return empty

    try:
        extraction = _extract(llm, cleaned)
        parts = extraction.extracted_parts
        out = {
            "valid_name": 0,
            "first_name": parts.first_name,
            "last_name": parts.last_name,
            "middle_name": parts.middle_name,
            "suffix": parts.suffix,
        }
        if not extraction.is_valid_name:
            return out
        validation = _validate(llm, cleaned, extraction)
        if validation.final_decision:
            out["valid_name"] = 1
        return out
    except LLMError as e:
        logger.error(f"Error classifying name '{name_string}': {e}")
        return empty


def _classify_unique_name(llm: LLM, name: str) -> tuple[str, dict]:
    time.sleep(0.1)  # rate limiting
    return name, classify_name(llm, name)


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 20,
) -> RunResult:
    """Validate officer names from a jsonl of `NameRecord`s.

    Unique cleaned-name strings are classified once and the result is mapped
    back across all records that share that cleaned name.
    """
    records = [NameRecord.model_validate(r) for r in read_jsonl(input_path)]
    if llm is None:
        llm = LLM()

    cleaned_per_record = [clean_officer_name(r.officer_name) for r in records]
    unique_names = sorted({n for n in cleaned_per_record if n})

    logger.info(f"Loaded {len(records)} records with {len(unique_names)} unique cleaned names")

    classifications: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(_classify_unique_name, llm, n): n for n in unique_names}
        for fut in tqdm(as_completed(futures), total=len(unique_names), desc="Classifying"):
            name, result = fut.result()
            classifications[name] = result

    empty = {
        "valid_name": 0,
        "first_name": "",
        "last_name": "",
        "middle_name": "",
        "suffix": "",
    }
    out_records: list[NameClassification] = []
    for record, cleaned in zip(records, cleaned_per_record, strict=True):
        c = classifications.get(cleaned, empty)
        out_records.append(
            NameClassification(
                officer_name=record.officer_name,
                cleaned_name=cleaned,
                valid_name=c["valid_name"],
                first_name=c["first_name"],
                last_name=c["last_name"],
                middle_name=c["middle_name"],
                suffix=c["suffix"],
                case_id=record.case_id,
            )
        )

    write_jsonl(output_path, [r.model_dump() for r in out_records])
    n_valid = sum(1 for r in out_records if r.valid_name == 1)
    logger.info(
        f"Wrote {len(out_records)} records to {output_path}; "
        f"{n_valid} marked valid ({n_valid / max(len(out_records), 1):.1%})"
    )
    return RunResult(
        n_records=len(out_records),
        n_unique_names=len(unique_names),
        n_valid=n_valid,
        output_path=str(output_path),
    )
