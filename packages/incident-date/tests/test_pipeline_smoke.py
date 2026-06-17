"""End-to-end smoke test with an injected fake LLM. No network calls."""

from prap_core.config import Settings
from prap_core.io import read_jsonl, write_jsonl
from prap_core.llm import LLM
from prap_incident_date import run
from prap_incident_date.schemas import CaseRecord


def _fake_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_hidden_params": {"response_cost": 0.0},
    }


def test_run_three_stage_chain(tmp_path):
    state = {"i": 0}

    # Sequence the fake responses to mirror what the pipeline asks for per case:
    # (1) extract  (2) verify  (3) ISO-8601
    def fake(**kwargs):
        state["i"] += 1
        idx = state["i"]
        if idx % 3 == 1:
            return _fake_response("INCIDENT TYPE: Use of Force\nPRIMARY INCIDENT DATE: 07/18/2024")
        if idx % 3 == 2:
            return _fake_response(
                "VERIFICATION RESULT: CONFIRMED\nPRIMARY INCIDENT DATE: 07/18/2024"
            )
        return _fake_response("2024-07-18")

    llm = LLM(Settings(_env_file=None, cache_enabled=False), completion_fn=fake)

    input_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "out.jsonl"
    case = CaseRecord(
        provisional_case_name="case-a",
        # 2 summaries means filter_important_summaries short-circuits (total <= final_count=2)
        summaries=["foo bar incident on 7/18/2024", "follow-up details"],
    )
    write_jsonl(input_path, [case.model_dump()])

    result = run(input_path, output_path, llm=llm, n_threads=1)
    assert result.n_cases == 1
    out = list(read_jsonl(output_path))
    assert len(out) == 1
    assert out[0]["provisional_case_name"] == "case-a"
    assert out[0]["extracted_date"] == ["2024-07-18"]


def test_ocr_path_skips_filter(tmp_path):
    state = {"i": 0}

    def fake(**kwargs):
        state["i"] += 1
        if state["i"] == 1:
            return _fake_response("PRIMARY INCIDENT DATE: 01/02/2021")
        if state["i"] == 2:
            return _fake_response("VERIFICATION RESULT: CONFIRMED")
        return _fake_response("2021-01-02")

    llm = LLM(Settings(_env_file=None, cache_enabled=False), completion_fn=fake)

    input_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "out.jsonl"
    case = CaseRecord(
        provisional_case_name="case-b",
        summaries=[],
        ocr_pages=["page 1 text", "page 2 text", "page 3 text"],
    )
    write_jsonl(input_path, [case.model_dump()])

    result = run(input_path, output_path, llm=llm, n_threads=1)
    assert result.n_cases == 1
    # Only 3 calls: the OCR path bypasses the chunk filter entirely.
    assert state["i"] == 3
    out = list(read_jsonl(output_path))
    assert out[0]["extracted_date"] == ["2021-01-02"]
