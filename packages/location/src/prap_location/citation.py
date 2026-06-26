"""Targeted-sample citation pass for `prap-location`.

Reads an existing `prap-location run` output (jsonl of LocationResult) +
the per-file documents table, filters cases to a configurable keyword
list (e.g. San Diego CDPs/unincorporated communities), and for each
filtered case runs a two-stage per-page LLM citation analysis followed
by an aggregate re-validation prompt. Writes a CSV suitable for manual
human validation.

The 4 LLM prompts (`citation_page_primary`, `citation_page_validator`,
`citation_aggregate_reasoning`, `citation_aggregate_parse`) live as
`*.txt` files alongside the extraction prompts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
from prap_core.llm import LLM
from tqdm import tqdm

logger = logging.getLogger("prap.location.citation")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_location.prompts").joinpath(f"{name}.txt").read_text(encoding="utf-8")
    )


def load_targeted_filter(name_or_path: str | Path) -> list[str]:
    """Load a list of lowercased substring patterns from a text file.

    `name_or_path` is either a bare filter name (e.g. `sd_cdp`, resolved
    against the package's `targeted_filters/` resource dir) or a path to
    a file on disk. Blank lines and lines starting with `#` are ignored.
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = (
            resources.files("prap_location.targeted_filters")
            .joinpath(f"{name_or_path}.txt")
            .read_text(encoding="utf-8")
        )
    return [
        line.strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _matches_filter(value: str | None, patterns: Iterable[str]) -> bool:
    if not value:
        return False
    v = value.lower()
    return any(p in v for p in patterns)


# ---- per-page citation analysis ----


@dataclass(frozen=True)
class PageTask:
    case_name: str
    gdrive_url: str
    page_num: int
    page_text: str
    initial_analysis: str | None


@dataclass
class PageResult:
    case_name: str
    gdrive_url: str
    page_num: int
    match: bool
    reasoning: str
    quote: str


def _parse_validator_response(response: str) -> dict[str, Any]:
    data = json.loads(response)
    return {
        "match": str(data.get("final_decision", "")).upper() == "YES",
        "reasoning": data.get("validator_reasoning", "Validator did not provide reasoning"),
        "quote": data.get("verified_quote", "Validator did not provide quote"),
    }


def _analyze_page(llm: LLM, task: PageTask) -> PageResult:
    primary_prompt = Template(_load_prompt("citation_page_primary")).safe_substitute(
        initial_analysis_summary=task.initial_analysis or "No initial analysis available",
        page_text=task.page_text,
    )
    try:
        primary = llm.complete(primary_prompt).text
        validator_prompt = Template(_load_prompt("citation_page_validator")).safe_substitute(
            page_text=task.page_text, primary_response=primary
        )
        validator = llm.complete(validator_prompt).text
        parsed = _parse_validator_response(validator)
        return PageResult(
            case_name=task.case_name,
            gdrive_url=task.gdrive_url,
            page_num=task.page_num,
            match=parsed["match"],
            reasoning=parsed["reasoning"],
            quote=parsed["quote"],
        )
    except Exception as e:
        logger.warning(f"page analysis failed for {task.case_name}/{task.page_num}: {e}")
        return PageResult(
            case_name=task.case_name,
            gdrive_url=task.gdrive_url,
            page_num=task.page_num,
            match=False,
            reasoning=f"Analysis error: {e}",
            quote="",
        )


def _parse_ocr_pages(ocr_text_json: Any) -> list[dict[str, Any]]:
    if ocr_text_json is None or (isinstance(ocr_text_json, float) and pd.isna(ocr_text_json)):
        return []
    if not isinstance(ocr_text_json, str) or not ocr_text_json.strip():
        return []
    try:
        pages = json.loads(ocr_text_json)
    except json.JSONDecodeError:
        return []
    if isinstance(pages, dict) and "messages" in pages:
        pages = pages["messages"]
    if not isinstance(pages, list):
        return []
    out: list[dict[str, Any]] = []
    for p in pages:
        if not isinstance(p, dict):
            continue
        page_num = p.get("page_number") or p.get("page")
        page_text = p.get("text") or p.get("page_content") or ""
        if page_num is not None and page_text:
            out.append({"page_number": int(page_num), "text": str(page_text)})
    return out


def _collect_page_tasks(filtered: pd.DataFrame, documents: pd.DataFrame) -> list[PageTask]:
    tasks: list[PageTask] = []
    docs_by_case = dict(tuple(documents.groupby("provisional_case_name")))
    for _, row in filtered.iterrows():
        case_name = row["provisional_case_name"]
        if case_name not in docs_by_case:
            continue
        initial = row.get("initial_analysis") or ""
        for _, doc in docs_by_case[case_name].iterrows():
            gdrive_url = doc.get("gdrive_url", "")
            pages = _parse_ocr_pages(doc.get("ocr_text"))
            for p in pages:
                tasks.append(
                    PageTask(
                        case_name=case_name,
                        gdrive_url=str(gdrive_url),
                        page_num=p["page_number"],
                        page_text=p["text"],
                        initial_analysis=initial.strip() or None,
                    )
                )
    return tasks


def _group_page_results(
    results: list[PageResult],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        case_bucket = grouped.setdefault(r.case_name, {})
        doc_bucket = case_bucket.setdefault(
            r.gdrive_url,
            {"gdrive_url": r.gdrive_url, "location_pages": [], "page_details": {}},
        )
        doc_bucket["page_details"][r.page_num] = {
            "reasoning": r.reasoning,
            "quote": r.quote,
            "match": r.match,
        }
        if r.match:
            doc_bucket["location_pages"].append(r.page_num)
    for case in grouped.values():
        for doc in case.values():
            doc["location_pages"] = sorted(set(doc["location_pages"]))
    return grouped


def _doc_summary(doc: dict[str, Any]) -> str:
    parts = [f"Document: {doc.get('gdrive_url', 'Unknown')}", "=" * 50]
    pages = doc.get("location_pages") or []
    if pages:
        parts.append(f"\nLOCATION CITATIONS - Pages: {', '.join(map(str, pages))}")
        for page in pages:
            details = doc["page_details"].get(page, {})
            parts.append(f"   Page {page}:")
            parts.append(f'     Quote: "{details.get("quote", "")}"')
            parts.append(f"     Reasoning: {details.get('reasoning', '')}")
    else:
        parts.append("\nNo location citation pages found.")
    return "\n".join(parts)


# ---- aggregate re-validation ----


def _aggregate_validate(
    llm: LLM, extracted_location: str, initial_reasoning: str, case_docs: list[dict[str, Any]]
) -> str:
    parts: list[str] = []
    total = 0
    for doc in case_docs:
        pages = doc.get("location_pages") or []
        total += len(pages)
        url = doc.get("gdrive_url", "Unknown")
        if pages:
            parts.append(f"Document {url}:")
            parts.append(f"  - Found {len(pages)} matching pages: {pages}")
            for page in pages:
                d = doc["page_details"].get(page, {})
                parts.append(f"    Page {page}:")
                parts.append(f'      Quote: "{d.get("quote", "")}"')
                parts.append(f"      Reasoning: {d.get('reasoning', '')}")
        else:
            parts.append(f"Document {url}: No matching pages found")
    summary = (
        f"Total matching pages found: {total}\n\n" + "\n".join(parts)
        if parts
        else "No documents were analyzed or no citation results available."
    )
    try:
        reasoning_prompt = Template(_load_prompt("citation_aggregate_reasoning")).safe_substitute(
            extracted_location=extracted_location,
            initial_reasoning=initial_reasoning,
            citation_summary=summary,
        )
        reasoning = llm.complete(reasoning_prompt).text
        parse_prompt = Template(_load_prompt("citation_aggregate_parse")).safe_substitute(
            reasoning_response=reasoning
        )
        decision = llm.complete(parse_prompt).text.strip().lower()
        return {"true": "True", "false": "False", "unclear": "Unclear"}.get(decision, "Unclear")
    except Exception as e:
        logger.error(f"aggregate validation failed: {e}")
        return "Unclear"


# ---- public entrypoint ----


def run_targeted_sample(
    *,
    run_jsonl: Path,
    documents_table: Path,
    output_csv: Path,
    filter_name_or_path: str | Path,
    llm: LLM | None = None,
    n_threads: int = 8,
    revalidate: bool = True,
) -> int:
    """Build the targeted-sample CSV. Returns the number of rows written."""
    from prap_core.io import read_jsonl

    if llm is None:
        llm = LLM()

    patterns = load_targeted_filter(filter_name_or_path)
    logger.info(f"Loaded {len(patterns)} filter patterns from {filter_name_or_path}")

    results = list(read_jsonl(run_jsonl))
    run_df = pd.DataFrame(results)
    if "extracted_location" not in run_df.columns:
        raise ValueError(f"run jsonl missing 'extracted_location' column: {run_jsonl}")

    filtered = run_df[
        run_df["extracted_location"].apply(lambda v: _matches_filter(v, patterns))
    ].copy()
    logger.info(f"Filtered {len(filtered)} / {len(run_df)} cases match the targeted filter")
    if filtered.empty:
        filtered.to_csv(output_csv, index=False)
        return 0

    documents = (
        pd.read_parquet(documents_table)
        if documents_table.suffix == ".parquet"
        else pd.read_csv(documents_table)
    )

    page_tasks = _collect_page_tasks(filtered, documents)
    logger.info(f"Collected {len(page_tasks)} page analysis tasks")

    page_results: list[PageResult] = []
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(_analyze_page, llm, t): t for t in page_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Citation pages"):
            page_results.append(fut.result())

    grouped = _group_page_results(page_results)

    citations_col: list[str] = []
    citations_summary_col: list[str] = []
    gdrive_urls_col: list[str] = []
    filenames_col: list[str] = []
    revalidated_col: list[str | None] = []

    docs_by_case = dict(tuple(documents.groupby("provisional_case_name")))

    for _, row in filtered.iterrows():
        case_name = row["provisional_case_name"]
        case_docs_list: list[dict[str, Any]] = []
        for _url, doc_result in grouped.get(case_name, {}).items():
            doc_result = dict(doc_result)
            doc_result["summary"] = _doc_summary(doc_result)
            case_docs_list.append(doc_result)

        citations_col.append(json.dumps(case_docs_list, indent=2))

        if case_docs_list:
            header = [f"CASE: {case_name}", "=" * 60]
            for d in case_docs_list:
                header.append(d["summary"])
                header.append("-" * 60)
            citations_summary_col.append("\n".join(header))
        else:
            citations_summary_col.append(f"CASE: {case_name}\nNo matching pages found.")

        case_docs_rows = docs_by_case.get(case_name)
        if case_docs_rows is not None:
            gdrive_urls_col.append(
                "; ".join(case_docs_rows.get("gdrive_url", pd.Series()).astype(str).tolist())
            )
            filenames_col.append(
                "; ".join(case_docs_rows.get("filename", pd.Series()).astype(str).tolist())
                if "filename" in case_docs_rows.columns
                else ""
            )
        else:
            gdrive_urls_col.append("")
            filenames_col.append("")

        if revalidate and row.get("extracted_location"):
            revalidated_col.append(
                _aggregate_validate(
                    llm,
                    str(row["extracted_location"]),
                    str(row.get("initial_analysis") or ""),
                    case_docs_list,
                )
            )
        else:
            revalidated_col.append(None)

    out = filtered.copy()
    out["gdrive_urls"] = gdrive_urls_col
    out["filenames"] = filenames_col
    out["citations"] = citations_col
    out["citations_summary"] = citations_summary_col
    out["citation_revalidation"] = revalidated_col
    out["correct"] = ""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(out)} rows to {output_csv}")
    return len(out)
