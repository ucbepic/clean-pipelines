"""Smoke tests — no real LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

from prap_mentioned_agencies.pipeline import (
    _strip_code_fences,
    deduplicate_agencies_fuzzy,
    flatten_case_pages,
    run,
)
from prap_mentioned_agencies.schemas import CaseBundle, CasePage


def test_strip_code_fences_plain():
    assert _strip_code_fences('["a"]') == '["a"]'


def test_strip_code_fences_json_block():
    text = '```json\n["a", "b"]\n```'
    assert _strip_code_fences(text) == '["a", "b"]'


def test_dedup_prefers_longer_name():
    out = deduplicate_agencies_fuzzy(
        ["Fresno PD", "Fresno Police Department", "California Highway Patrol"]
    )
    # "Fresno PD" gets collapsed into "Fresno Police Department" at threshold 85.
    assert "Fresno Police Department" in out
    assert "California Highway Patrol" in out


def test_flatten_case_pages_drops_empty_text():
    raw = {
        "HIDDEN_provisional_case_name": "case-1",
        "case_files": [
            {
                "file_name": "f.pdf",
                "ocr_doc_text_per_page": {
                    "page_texts": [
                        {"page_number": 1, "text": "hello"},
                        {"page_number": 2, "text": "   "},
                        {"page_number": 3, "text": ""},
                    ]
                },
            }
        ],
    }
    bundle = flatten_case_pages(raw)
    assert bundle.provisional_case_name == "case-1"
    assert len(bundle.pages) == 1
    assert bundle.pages[0].page_number == 1


class _StubLLM:
    """Minimal LLM stand-in: returns canned per-page extraction, then a canned
    validation JSON. Order matters because pages are dispatched concurrently —
    we keep the stub deterministic by returning the same value every call."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, **_kwargs):
        from types import SimpleNamespace

        self.calls += 1
        if "validating" in prompt or "validated_agencies" in prompt:
            payload = {
                "validated_agencies": ["Fresno Police Department"],
                "removed_entries": [],
                "corrections_made": [],
                "confidence": "high",
                "validation_notes": "ok",
            }
            return SimpleNamespace(text=json.dumps(payload))
        return SimpleNamespace(text='["Fresno PD"]')


def test_run_end_to_end_with_stub_llm(tmp_path: Path):
    input_path = tmp_path / "in.jsonl"
    bundle = CaseBundle(
        provisional_case_name="case-1",
        pages=[
            CasePage(file_name="f.pdf", page_number=1, text="Officer from Fresno PD"),
            CasePage(file_name="f.pdf", page_number=2, text="More text"),
        ],
    )
    with open(input_path, "w") as f:
        f.write(bundle.model_dump_json() + "\n")

    output_path = tmp_path / "out.jsonl"
    result = run(input_path, output_path, llm=_StubLLM(), n_threads=2)

    assert result.n_cases == 1
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["provisional_case_name"] == "case-1"
    assert rec["mentioned_agencies"] == ["Fresno Police Department"]
    assert rec["n_pages_processed"] == 2
    assert rec["validation_confidence"] == "high"
