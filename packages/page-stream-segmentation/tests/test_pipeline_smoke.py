"""Smoke tests — no real LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from prap_page_stream_segmentation.evaluation import (
    predictions_from_tocs,
    score,
    score_by_group,
)
from prap_page_stream_segmentation.pipeline import (
    _build_hybrid_history,
    _load_prompt_template,
    _structured_doc_info,
    build_toc,
    run,
)
from prap_page_stream_segmentation.schemas import (
    DocText,
    DocumentTOC,
    PageClassification,
    PageText,
    TOCEntry,
)


def test_build_hybrid_history_empty():
    assert _build_hybrid_history([], None, [], 15) is None


def test_build_hybrid_history_collapses_segments():
    segments = [
        {"start": 1, "end": 1, "doc_type": "MEMO", "meta": ""},
        {"start": 2, "end": 5, "doc_type": "REPORT", "meta": ""},
    ]
    history = _build_hybrid_history(segments, None, [(6, "meta-6")], 15)
    assert "Page 1: MEMO" in history
    assert "Pages 2-5: REPORT" in history
    assert "Page 6:" in history


def test_structured_doc_info_fast_path():
    """When meta contains 'DOCUMENT TYPE:', we should not call the LLM."""
    template = _load_prompt_template("extract_doctype")

    class _Boom:
        def complete(self, *a, **kw):
            raise AssertionError("LLM should not be called on fast path")

    doctype, cont = _structured_doc_info(_Boom(), "- DOCUMENT TYPE: INVESTIGATIVE REPORT", template)
    assert doctype == "INVESTIGATIVE REPORT"
    assert cont is False

    doctype, cont = _structured_doc_info(
        _Boom(), "- DOCUMENT TYPE: MEMORANDUM - CONTINUATION", template
    )
    assert doctype == "MEMORANDUM"
    assert cont is True


def test_predictions_from_tocs():
    toc = DocumentTOC(
        sha1="abc",
        entries=[
            TOCEntry(
                sha1="abc",
                start_page=1,
                headline="h",
                date="2024-01-01",
                page_classifications=[
                    PageClassification(
                        sha1="abc",
                        page_number=1,
                        document_type="MEMO",
                        continuation=False,
                        meta="",
                        reasoning="",
                    ),
                    PageClassification(
                        sha1="abc",
                        page_number=2,
                        document_type="MEMO",
                        continuation=True,
                        meta="",
                        reasoning="",
                    ),
                ],
            ),
            TOCEntry(
                sha1="abc",
                start_page=3,
                headline="h2",
                date="2024-01-02",
                page_classifications=[
                    PageClassification(
                        sha1="abc",
                        page_number=3,
                        document_type="REPORT",
                        continuation=False,
                        meta="",
                        reasoning="",
                    ),
                ],
            ),
        ],
    )
    df = predictions_from_tocs([toc])
    assert list(df["predicted"]) == [1, 0, 1]
    assert list(df["page_number"]) == [1, 2, 3]


def test_score_overall_and_by_stratum():
    df = pd.DataFrame(
        {
            "sha1": ["a", "a", "b", "b"],
            "page_number": [1, 2, 1, 2],
            "label": [1, 0, 1, 1],
            "predicted": [1, 0, 1, 0],
            "stratum": ["x", "x", "y", "y"],
        }
    )
    prf = score(df)
    assert prf.true_positive == 2
    assert prf.false_negative == 1
    assert prf.true_negative == 1
    assert prf.precision == 1.0

    grouped = score_by_group(df, "stratum")
    assert set(grouped["group"]) == {"overall", "x", "y"}


class _StubLLM:
    """Returns canned classifier output, then canned TOC JSON."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self._SimpleNamespace = SimpleNamespace
        self.calls = 0

    def complete(self, prompt: str, **_kwargs):
        self.calls += 1
        if "canonical index entry" in prompt or "Headline" in prompt:
            payload = {
                "Headline": "Test doc",
                "Date": "2024-01-01",
                "People": {"Alice": "Author"},
                "Agency or agencies": ["Test Agency"],
            }
            return self._SimpleNamespace(text=json.dumps(payload))
        # classify_page response
        body = "- Document type: INVESTIGATIVE REPORT\n\nREASONING:\n- header says so"
        return self._SimpleNamespace(text=body)


def test_build_toc_with_stub_llm():
    doc = DocText(
        sha1="abc",
        pages=[
            PageText(page_number=1, text="X" * 200 + " investigative report header"),
            PageText(page_number=2, text="Y" * 200 + " more content"),
        ],
    )
    result = build_toc(_StubLLM(), doc)
    assert result.sha1 == "abc"
    assert result.error is None
    assert len(result.entries) >= 1
    assert result.entries[0].headline == "Test doc"


def test_run_end_to_end_with_stub_llm(tmp_path: Path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    doc = DocText(
        sha1="abc",
        pages=[PageText(page_number=1, text="X" * 200 + " investigative report")],
    )
    with open(input_path, "w") as f:
        f.write(doc.model_dump_json() + "\n")

    result = run(input_path, output_path, llm=_StubLLM(), n_threads=1)
    assert result.n_documents == 1
    line = output_path.read_text().strip().splitlines()[0]
    out = json.loads(line)
    assert out["sha1"] == "abc"
    assert out["error"] is None
