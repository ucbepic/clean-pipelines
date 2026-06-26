"""Page-stream segmentation pipeline.

Classify-then-TOC chain over pre-OCR'd page text. Implementation notes:
- Input is pre-OCR'd page text via the `DocText` pydantic schema; for OCR
  go through `prap_core.ocr` separately.
- LLM calls go through `prap_core.llm.LLM` (sync) + ThreadPoolExecutor
  across documents. Page classification stays sequential within a document
  because each page needs prior-page context + history.
- Relies on `prap_core.llm` defaults (temperature=0.0).
- Prompts are Jinja2 templates ({% if %} / {% for %} conditionals).
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path

from jinja2 import Template
from prap_core.io import read_jsonl, write_jsonl
from prap_core.llm import LLM
from tqdm import tqdm

from .schemas import (
    DocText,
    DocumentTOC,
    PageClassification,
    PageText,
    RunResult,
    TOCEntry,
)

logger = logging.getLogger("prap.page_stream_segmentation")

DOMAIN_SYSTEM_MSG = (
    "You are knowledgeable about investigative and administrative procedures in "
    "law enforcement, including misconduct investigations, use-of-force reviews, "
    "and officer-involved shooting investigations. You have extensive experience "
    "reading and organizing documents related to these cases."
)
GENERIC_SYSTEM_MSG = "You are a helpful document classification assistant."
DOCTYPE_FALLBACK_SYSTEM_MSG = "You are a helpful metadata extraction assistant"
TOC_SYSTEM_MSG = (
    "You are a police records archivist with experience organizing and summarizing "
    "documents related to law enforcement cases, including misconduct investigations, "
    "use-of-force incidents, and officer-involved shootings."
)


def _load_prompt_template(name: str) -> Template:
    text = (
        resources.files("prap_page_stream_segmentation.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )
    return Template(text)


def _build_hybrid_history(
    segments: list[dict],
    current_segment: dict | None,
    page_history: list[tuple[int, str]],
    recent_window: int,
) -> str | None:
    if not page_history:
        return None
    parts: list[str] = []
    if segments:
        collapsed = []
        for seg in segments:
            if seg["start"] == seg["end"]:
                collapsed.append(f"Page {seg['start']}: {seg['doc_type']}")
            else:
                collapsed.append(f"Pages {seg['start']}-{seg['end']}: {seg['doc_type']}")
        parts.append("Previous documents in this file:\n" + "\n".join(collapsed))
    recent_pages = page_history[-recent_window:]
    recent_start = recent_pages[0][0]
    if current_segment and current_segment["start"] < recent_start:
        parts.append(
            f"Current document (started on page {current_segment['start']}): "
            f"{current_segment['doc_type']}"
        )
    recent_lines = [f"Page {p}:\n{m}" for p, m in recent_pages]
    parts.append("Recent page classifications:\n" + "\n".join(recent_lines))
    return "\n\n".join(parts)


def _structured_doc_info(llm: LLM, meta: str, doctype_template: Template) -> tuple[str, bool]:
    """Pull (document_type, continuation) out of the classifier's metadata block.

    Fast path: if a "DOCUMENT TYPE:" line is present, parse it directly. Slow
    path: ask the LLM to extract the doctype from the metadata.
    """
    for line in meta.splitlines():
        if "DOCUMENT TYPE:" in line.upper():
            doctype = line.split(":", 1)[1].split("-")[0].strip()
            continuation = "CONTINUATION" in line.upper()
            return doctype, continuation
    prompt = doctype_template.render(meta=meta)
    response = llm.complete(prompt, system=DOCTYPE_FALLBACK_SYSTEM_MSG).text
    if "CONTINUATION" in response.upper():
        return response.upper().replace("CONTINUATION", "").strip(), True
    return response.strip(), False


def _classify_page(
    llm: LLM,
    sha1: str,
    page: PageText,
    history: str | None,
    previous_page_context: str | None,
    use_domain: bool,
    classify_template: Template,
    doctype_template: Template,
) -> PageClassification:
    rendered = classify_template.render(
        page_text=page.text,
        history=history,
        previous_page_context=previous_page_context,
        use_domain=use_domain,
    )
    system = DOMAIN_SYSTEM_MSG if use_domain else GENERIC_SYSTEM_MSG
    response = llm.complete(rendered, system=system).text

    lines = response.strip().splitlines()
    reasoning_start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(r"^REASONING.{0,3}", line.strip(), re.IGNORECASE)
        ),
        len(lines),
    )
    meta = "\n".join(lines[:reasoning_start]).strip()
    reasoning = "\n".join(lines[reasoning_start:]).strip()
    doctype, continuation = _structured_doc_info(llm, meta, doctype_template)

    return PageClassification(
        sha1=sha1,
        page_number=page.page_number,
        document_type=doctype,
        continuation=continuation,
        meta=meta,
        reasoning=reasoning,
    )


def classify_pages(
    llm: LLM,
    doc: DocText,
    *,
    use_domain: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    recent_window: int = 15,
    classify_template: Template | None = None,
    doctype_template: Template | None = None,
) -> list[PageClassification]:
    """Sequentially classify every page of one source document."""
    classify_template = classify_template or _load_prompt_template("classify_page")
    doctype_template = doctype_template or _load_prompt_template("extract_doctype")

    pages = doc.pages
    # If every page has <150 chars, treat the document as empty.
    if pages and all(len(p.text.strip()) < 150 for p in pages):
        pages = []

    classifications: list[PageClassification] = []
    segments: list[dict] = []
    current_segment: dict | None = None
    page_history: list[tuple[int, str]] = []
    previous_page_context: str | None = None

    for page in pages:
        history = (
            _build_hybrid_history(segments, current_segment, page_history, recent_window)
            if use_history
            else None
        )
        pc = _classify_page(
            llm,
            doc.sha1,
            page,
            history,
            previous_page_context if use_context else None,
            use_domain,
            classify_template,
            doctype_template,
        )
        classifications.append(pc)

        if pc.continuation and current_segment is not None:
            current_segment["end"] = page.page_number
        else:
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {
                "start": page.page_number,
                "end": page.page_number,
                "doc_type": pc.document_type,
                "meta": pc.meta,
            }
        page_history.append((page.page_number, pc.meta))
        previous_page_context = "\n".join(page.text.splitlines()[-5:])

    return classifications


def _pages_to_toc_entry(
    llm: LLM, pages: list[PageClassification], toc_template: Template
) -> TOCEntry:
    sha1 = pages[0].sha1
    start_page = pages[0].page_number
    prompt = toc_template.render(sha1=sha1, start_page=start_page, page_classifications=pages)
    response = llm.complete(prompt, system=TOC_SYSTEM_MSG).text
    clean = response.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"toc JSON parse failed for {sha1} p{start_page}; using empty fields")
        data = {"Headline": "", "Date": "", "People": {}, "Agency or agencies": []}
    return TOCEntry(
        sha1=sha1,
        start_page=start_page,
        headline=data.get("Headline", "") or "",
        date=data.get("Date", "") or "",
        people=data.get("People", {}) or {},
        agencies=data.get("Agency or agencies", []) or [],
        page_classifications=pages,
    )


def build_toc(
    llm: LLM,
    doc: DocText,
    *,
    use_domain: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    recent_window: int = 15,
) -> DocumentTOC:
    """Classify every page of `doc`, then group runs into TOC entries."""
    classify_template = _load_prompt_template("classify_page")
    doctype_template = _load_prompt_template("extract_doctype")
    toc_template = _load_prompt_template("toc_item")

    try:
        classifications = classify_pages(
            llm,
            doc,
            use_domain=use_domain,
            use_history=use_history,
            use_context=use_context,
            recent_window=recent_window,
            classify_template=classify_template,
            doctype_template=doctype_template,
        )
        entries: list[TOCEntry] = []
        current: list[PageClassification] = []
        for i, pc in enumerate(classifications):
            new_document = not (i > 0 and pc.continuation)
            if new_document:
                if current:
                    entries.append(_pages_to_toc_entry(llm, current, toc_template))
                current = [pc]
            else:
                current.append(pc)
        if current:
            entries.append(_pages_to_toc_entry(llm, current, toc_template))
        return DocumentTOC(sha1=doc.sha1, entries=entries)
    except Exception as e:
        logger.error(f"Error processing {doc.sha1}: {e}")
        logger.error(traceback.format_exc())
        return DocumentTOC(sha1=doc.sha1, entries=[], error=str(e))


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLM | None = None,
    n_threads: int = 8,
    use_domain: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    recent_window: int = 15,
) -> RunResult:
    """Segment every document in `input_path` (jsonl of DocText) into TOC entries.

    Output is jsonl of `DocumentTOC` (one line per source PDF).
    """
    docs = [DocText.model_validate(rec) for rec in read_jsonl(input_path)]
    if llm is None:
        llm = LLM()

    results: list[DocumentTOC | None] = [None] * len(docs)
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {
            ex.submit(
                build_toc,
                llm,
                doc,
                use_domain=use_domain,
                use_history=use_history,
                use_context=use_context,
                recent_window=recent_window,
            ): i
            for i, doc in enumerate(docs)
        }
        for fut in tqdm(as_completed(futures), total=len(docs), desc="Documents"):
            results[futures[fut]] = fut.result()

    write_jsonl(output_path, [r.model_dump() for r in results if r is not None])
    logger.info(f"Wrote {len(docs)} document TOCs to {output_path}")
    return RunResult(n_documents=len(docs), output_path=str(output_path))
