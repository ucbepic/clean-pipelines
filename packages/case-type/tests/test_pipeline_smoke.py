"""End-to-end smoke test with an injected fake LLM. No network calls."""

from prap_case_type import run
from prap_case_type.schemas import CaseRecord
from prap_core.config import Settings
from prap_core.io import read_jsonl, write_jsonl
from prap_core.llm import LLM


def _fake_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_hidden_params": {"response_cost": 0.0},
    }


def test_run_master_plus_three_booleans(tmp_path):
    """One case with 2 summaries: filter short-circuits, then master + 3 booleans = 4 calls."""
    seq = ["master analysis", "true", "false", "unclear"]
    state = {"i": 0}

    def fake(**kwargs):
        idx = state["i"]
        state["i"] += 1
        return _fake_response(seq[idx])

    llm = LLM(Settings(_env_file=None, cache_enabled=False), completion_fn=fake)

    input_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "out.jsonl"
    case = CaseRecord(
        provisional_case_name="case-a",
        summaries=["summary one", "summary two"],
        ocr_texts=["long ocr text " * 200],  # > 250 tokens so the short-circuit is skipped
    )
    write_jsonl(input_path, [case.model_dump()])

    result = run(input_path, output_path, llm=llm, n_threads=1)
    assert result.n_cases == 1
    out = list(read_jsonl(output_path))
    assert out[0]["provisional_case_name"] == "case-a"
    assert out[0]["classification"] == {
        "use_of_force": "True",
        "misconduct": "False",
        "officer_involved_shooting": "Unclear",
    }
    assert state["i"] == 4


def test_short_ocr_shortcuts_to_unclear(tmp_path):
    """When every OCR doc is < 250 tokens, the pipeline bypasses the LLM entirely."""

    def fake(**kwargs):
        raise AssertionError("LLM should not be called on the short-OCR path")

    llm = LLM(Settings(_env_file=None, cache_enabled=False), completion_fn=fake)

    input_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "out.jsonl"
    case = CaseRecord(
        provisional_case_name="case-b",
        summaries=["x"],
        ocr_texts=["short", "also short"],
    )
    write_jsonl(input_path, [case.model_dump()])

    run(input_path, output_path, llm=llm, n_threads=1)
    out = list(read_jsonl(output_path))
    assert out[0]["classification"] == {
        "use_of_force": "Unclear",
        "misconduct": "Unclear",
        "officer_involved_shooting": "Unclear",
    }
