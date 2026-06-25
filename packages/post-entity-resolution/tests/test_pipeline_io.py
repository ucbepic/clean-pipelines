"""Offline tests for the jsonl `prepare` / `run` pipeline functions.

No network/model: `run` takes an injected matcher. Exercises the prap_core.io
jsonl round-trip (CSV -> mentions jsonl -> results jsonl). Deps: pandas, pydantic,
prap-core.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from prap_core.io import read_jsonl
from prap_post_entity_resolution.resolve.pipeline import MentionResult, _result_record, run
from prap_post_entity_resolution.resolve.prepare import prepare
from prap_post_entity_resolution.schemas import OfficerMention

_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_prepare_csv_to_jsonl_mentions(tmp_path):
    out = tmp_path / "mentions.jsonl"
    n = prepare(_FIXTURES / "resolve_sample.csv", out, default_state="CA")

    records = list(read_jsonl(out))
    assert n == len(records) == 12
    # every line round-trips back into an OfficerMention
    mentions = [OfficerMention.model_validate(r) for r in records]
    assert all(m.state == "CA" for m in mentions)
    assert mentions[0].mention_incident_date.year == 2017


class _FakeMatcher:
    def resolve_batch(self, mentions):
        return [
            MentionResult(mention=m, status="auto_matched", match={"post_person_nbr": "x"})
            for m in mentions
        ]


def test_run_jsonl_to_jsonl_results(tmp_path):
    mentions_jsonl = tmp_path / "mentions.jsonl"
    prepare(_FIXTURES / "resolve_sample.csv", mentions_jsonl, default_state="CA")

    out = tmp_path / "results.jsonl"
    result = run(mentions_jsonl, out, matcher=_FakeMatcher())

    rows = list(read_jsonl(out))
    assert result.n_mentions == len(rows) == 12
    assert all(r["status"] == "auto_matched" for r in rows)
    assert rows[0]["post_match"]["post_person_nbr"] == "x"


def test_result_record_is_json_serializable_with_timestamps():
    """Live NPI candidates carry pandas Timestamps (post_start_date/post_end_date)
    from `df.to_dict("records")`; the result record must still json-serialize."""

    class _M:
        mention_uid = "u1"
        mention_first_name = "Jane"
        mention_middle_name = ""
        mention_last_name = "Doe"
        mention_suffix = ""
        mention_agency = "Oakland Police Department"
        mention_incident_date = "2017-01-01"
        state = "CA"
        mentioned_agencies = ""

    ts_match = {"post_person_nbr": "123", "post_start_date": pd.Timestamp("2010-05-01")}
    cand = {"post_person_nbr": "123", "post_end_date": pd.Timestamp("2020-01-01")}
    rec = _result_record(
        MentionResult(mention=_M(), status="auto_matched", match=ts_match, candidates=[cand])
    )

    # must not raise — and Timestamps become JSON-native (strings)
    dumped = json.loads(json.dumps(rec))
    assert isinstance(dumped["post_match"]["post_start_date"], str)
    assert isinstance(dumped["candidates"][0]["post_end_date"], str)
