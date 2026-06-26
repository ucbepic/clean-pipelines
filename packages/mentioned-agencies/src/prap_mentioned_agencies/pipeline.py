"""Mentioned-agencies extraction pipeline.

Three stages: per-page extraction -> case-level validation -> fuzzy dedup.
Backoff/retry is handled by `prap_core.llm`; concurrency uses ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from string import Template

from fuzzywuzzy import fuzz
from prap_core.io import read_jsonl, write_jsonl
from prap_core.llm import LLM
from tqdm import tqdm

from .schemas import CaseBundle, CasePage, MentionedAgenciesResult, RunResult

logger = logging.getLogger("prap.mentioned_agencies")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_mentioned_agencies.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    return cleaned


def _extract_page_agencies(llm: LLM, prompt_template: str, page: CasePage) -> list[str]:
    page_context = f"File: {page.file_name}, Page: {page.page_number}"
    prompt = Template(prompt_template).safe_substitute(
        page_text=page.text, page_context=page_context
    )
    response = llm.complete(prompt).text
    cleaned = _strip_code_fences(response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"  Page {page.page_number}: Failed to parse JSON - {e}")
        return []
    if not isinstance(parsed, list):
        logger.warning(f"  Page {page.page_number}: Unexpected response format")
        return []
    return [str(a) for a in parsed]


def _validate_agencies(
    llm: LLM,
    prompt_template: str,
    extracted_agencies: list[str],
    context_sample: str,
) -> dict:
    agencies_json = json.dumps(extracted_agencies, indent=2)
    prompt = Template(prompt_template).safe_substitute(
        agencies_json=agencies_json, context_sample=context_sample[:3000]
    )
    response = llm.complete(prompt).text
    cleaned = _strip_code_fences(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse validation response as JSON: {e}")
        return {
            "validated_agencies": extracted_agencies,
            "removed_entries": [],
            "corrections_made": [],
            "confidence": "low",
            "validation_notes": "JSON parsing failed, returning original list",
        }


def deduplicate_agencies_fuzzy(agencies: list[str], threshold: int = 85) -> list[str]:
    """Deduplicate agency names via fuzzy string matching, preferring longer names."""
    if not agencies:
        return []
    unique = list(set(agencies))
    unique.sort(key=len, reverse=True)
    deduplicated: list[str] = []
    for agency in unique:
        is_dup = False
        for existing in deduplicated:
            if fuzz.ratio(agency.lower(), existing.lower()) >= threshold:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(agency)
    return sorted(deduplicated)


def flatten_case_pages(bundle_dict: dict) -> CaseBundle:
    """Project a raw `agency_case_file_bundle` dict into a `CaseBundle`.

    Skips pages whose OCR text is empty after strip. Caller supplies dicts with
    the cpost schema: top-level `HIDDEN_provisional_case_name` plus a
    `case_files: [{file_name, ocr_doc_text_per_page: {page_texts: [...]}}]` list.
    """
    name = bundle_dict.get("HIDDEN_provisional_case_name") or bundle_dict.get(
        "provisional_case_name", "unknown"
    )
    pages: list[CasePage] = []
    for file_obj in bundle_dict.get("case_files", []):
        file_name = file_obj.get("file_name", "unknown")
        ocr_data = file_obj.get("ocr_doc_text_per_page", {})
        for page_obj in ocr_data.get("page_texts", []):
            text = (page_obj.get("text") or "").strip()
            if not text:
                continue
            pages.append(
                CasePage(
                    file_name=file_name,
                    page_number=int(page_obj.get("page_number", 0)),
                    text=text,
                )
            )
    return CaseBundle(provisional_case_name=name, pages=pages)


def process_case(
    llm: LLM,
    case: CaseBundle,
    extract_prompt: str,
    validate_prompt: str,
    *,
    dedup_threshold: int = 85,
) -> MentionedAgenciesResult:
    try:
        logger.info(f"Processing case: {case.provisional_case_name}")
        if not case.pages:
            logger.warning(f"No pages with text for case: {case.provisional_case_name}")
            return MentionedAgenciesResult(
                provisional_case_name=case.provisional_case_name,
                mentioned_agencies=[],
            )

        # Stage 1: per-page extraction.
        all_extracted: list[str] = []
        for page in case.pages:
            try:
                all_extracted.extend(_extract_page_agencies(llm, extract_prompt, page))
            except Exception as e:
                logger.warning(f"  Page {page.page_number}: extraction error - {e}")
                continue

        if not all_extracted:
            return MentionedAgenciesResult(
                provisional_case_name=case.provisional_case_name,
                mentioned_agencies=[],
                n_pages_processed=len(case.pages),
            )

        # Stage 2: case-level validation. Context sample = first 5 pages, 500 chars each.
        context_sample = "\n\n".join(p.text[:500] for p in case.pages[:5])
        validation = _validate_agencies(llm, validate_prompt, all_extracted, context_sample)
        validated = validation.get("validated_agencies", all_extracted)

        # Stage 3: fuzzy dedup.
        deduped = deduplicate_agencies_fuzzy(validated, threshold=dedup_threshold)

        return MentionedAgenciesResult(
            provisional_case_name=case.provisional_case_name,
            mentioned_agencies=deduped,
            n_pages_processed=len(case.pages),
            n_raw_extractions=len(all_extracted),
            n_after_validation=len(validated),
            validation_confidence=validation.get("confidence"),
        )
    except Exception as e:
        logger.error(f"Error processing case {case.provisional_case_name}: {e}")
        logger.error(traceback.format_exc())
        return MentionedAgenciesResult(
            provisional_case_name=case.provisional_case_name,
            mentioned_agencies=[],
            error=str(e),
        )


def _load_cases(input_path: str | Path) -> list[CaseBundle]:
    """Accept either a jsonl of pre-flattened CaseBundle records OR a directory
    of `agency_case_file_bundle-*.json` files (cpost format)."""
    p = Path(input_path)
    if p.is_dir():
        bundles: list[CaseBundle] = []
        for jf in sorted(p.glob("agency_case_file_bundle-*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    bundles.append(flatten_case_pages(json.load(f)))
            except Exception as e:
                logger.error(f"Error reading {jf}: {e}")
        return bundles
    # Jsonl: each line is either a flattened CaseBundle or a raw bundle dict.
    cases: list[CaseBundle] = []
    for rec in read_jsonl(p):
        if "pages" in rec:
            cases.append(CaseBundle.model_validate(rec))
        else:
            cases.append(flatten_case_pages(rec))
    return cases


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 16,
    dedup_threshold: int = 85,
) -> RunResult:
    """Extract mentioned law-enforcement agencies for each case.

    `input_path` is either a directory of `agency_case_file_bundle-*.json`
    files (cpost format) or a jsonl of CaseBundle records. Output is jsonl
    of `MentionedAgenciesResult`.
    """
    cases = _load_cases(input_path)
    if llm is None:
        llm = LLM()

    extract_prompt = _load_prompt("extract_per_page")
    validate_prompt = _load_prompt("validate_agencies")

    results: list[MentionedAgenciesResult | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {
            ex.submit(
                process_case,
                llm,
                case,
                extract_prompt,
                validate_prompt,
                dedup_threshold=dedup_threshold,
            ): i
            for i, case in enumerate(cases)
        }
        for fut in tqdm(as_completed(futures), total=len(cases), desc="Processing cases"):
            results[futures[fut]] = fut.result()

    write_jsonl(output_path, [r.model_dump() for r in results if r is not None])
    logger.info(f"Wrote {len(cases)} results to {output_path}")
    return RunResult(n_cases=len(cases), output_path=str(output_path))
