"""End-to-end smoke tests with an injected fake LLM."""

import json

from prap_core.config import Settings
from prap_core.io import read_jsonl, write_jsonl
from prap_core.llm import LLM
from prap_split_officer_names import run
from prap_split_officer_names.cleaning import clean_officer_name, is_obviously_invalid


def _resp(payload: dict) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_hidden_params": {"response_cost": 0.0},
    }


def test_rank_stripping_verbatim():
    assert clean_officer_name("Police Officer II Fernando Cuevas") == "Fernando Cuevas"
    assert clean_officer_name("Sgt. John Smith") == "John Smith"
    assert clean_officer_name("K-9 Handler Anthony Haidet") == "Anthony Haidet"
    assert clean_officer_name("John Smith #3446") == "John Smith"
    assert clean_officer_name("[Name Redacted]") == ""


def test_obvious_invalid_filter():
    assert is_obviously_invalid("Officers")
    assert is_obviously_invalid("A. Marks")
    assert is_obviously_invalid("[Redacted]")
    assert not is_obviously_invalid("John Smith")


def test_run_two_stage_chain(tmp_path):
    # 2 records share the same cleaned name → only one classification call
    records = [
        {"officer_name": "Police Officer John Smith"},
        {"officer_name": "Sgt. John Smith"},
        {"officer_name": "Officers"},  # rejected by is_obviously_invalid
    ]
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_jsonl(input_path, records)

    state = {"i": 0}

    def fake(**kwargs):
        state["i"] += 1
        # Stage 1 = extract, stage 2 = validate (alternating per unique name)
        if state["i"] % 2 == 1:
            return _resp(
                {
                    "is_valid_name": True,
                    "extracted_parts": {
                        "first_name": "John",
                        "last_name": "Smith",
                        "middle_name": "",
                        "suffix": "",
                    },
                }
            )
        return _resp({"final_decision": True})

    llm = LLM(Settings(_env_file=None, cache_enabled=False), completion_fn=fake)
    result = run(input_path, output_path, llm=llm, n_threads=1)

    assert result.n_records == 3
    assert result.n_unique_names == 1  # "John Smith" (the third is "" after cleaning)
    assert result.n_valid == 2  # two records share the validated name

    out = list(read_jsonl(output_path))
    assert out[0]["valid_name"] == 1
    assert out[0]["first_name"] == "John"
    assert out[1]["valid_name"] == 1
    assert out[2]["valid_name"] == 0  # rules-based reject


def test_extraction_invalid_skips_validation(tmp_path):
    """If stage-1 says invalid, stage-2 must not be called."""
    write_jsonl(tmp_path / "in.jsonl", [{"officer_name": "John Smith"}])

    state = {"i": 0}

    def fake(**kwargs):
        state["i"] += 1
        if state["i"] == 1:
            return _resp(
                {
                    "is_valid_name": False,
                    "extracted_parts": {
                        "first_name": "",
                        "last_name": "Smith",
                        "middle_name": "",
                        "suffix": "",
                    },
                }
            )
        raise AssertionError("validation should not run when stage-1 rejects")

    llm = LLM(Settings(_env_file=None, cache_enabled=False), completion_fn=fake)
    result = run(tmp_path / "in.jsonl", tmp_path / "out.jsonl", llm=llm, n_threads=1)
    assert result.n_valid == 0
    assert state["i"] == 1
