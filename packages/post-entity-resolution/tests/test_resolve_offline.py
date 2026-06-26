"""Offline characterization tests for the resolve pipeline.

No network, no model, no LLM: the client / scorer / validator are all injected,
so these exercise the PostMatcher decision logic against the real package layout
(`prap_post_entity_resolution.*`) without the heavy deps. Deps: pandas, pydantic.
"""

from __future__ import annotations

import datetime as dt

import pytest
from prap_post_entity_resolution.resolve.pipeline import PostMatcher
from prap_post_entity_resolution.resolve.validation import validate_agency_match
from prap_post_entity_resolution.schemas import AgencyType, OfficerMention


def _mention(first="DOMINIC", last="DEGUILIO", agency="Napa Police Department", state="CA"):
    return OfficerMention(
        mention_uid="uid-1",
        mention_agency_type=AgencyType.POLICE,
        mention_incident_date=dt.date(2017, 1, 1),
        mention_first_name=first,
        mention_last_name=last,
        mention_agency=agency,
        state=state,
        mentioned_agencies=agency,
    )


class _FakeClient:
    """Returns one exact-name POST candidate employed across the incident year."""

    def __init__(self, candidates, same_name_persons=1):
        self._candidates = candidates
        self._same = same_name_persons

    def get_officers_by_name(self, first_name, last_name, state=None):
        return [{"post_person_nbr": f"p{i}"} for i in range(self._same)]

    def get_candidates_for_mention(
        self, first_name, last_name, incident_year, agency_type="POLICE", state=None
    ):
        return self._candidates

    def get_county_for_agency(self, agency_name):
        return None


def _exact_candidate(first="DOMINIC", last="DEGUILIO", agency="Napa Police Department"):
    return {
        "post_person_nbr": "b54-t85",
        "post_first_name": first,
        "post_middle_name": None,
        "post_last_name": last,
        "post_suffix": None,
        "post_agency_name": agency,
        "post_agency_type": "POLICE",
        "post_start_date": "2010-01-01T00:00:00",
        "post_end_date": "2020-01-01T00:00:00",
        "post_separation_reason": None,
        "state": "california",
        "county": "",
    }


def test_exact_name_high_score_valid_agency_auto_matches():
    matcher = PostMatcher(
        client=_FakeClient([_exact_candidate()]),
        scorer=lambda df: [0.9] * len(df),
        validator=lambda *a: (True, ""),
        common_last_names=set(),
    )
    result = matcher.resolve_one(_mention())
    assert result.status == "auto_matched"
    assert result.match["post_person_nbr"] == "b54-t85"


def test_common_last_name_routes_to_review():
    matcher = PostMatcher(
        client=_FakeClient([_exact_candidate()]),
        scorer=lambda df: [0.9] * len(df),
        validator=lambda *a: (True, ""),
        common_last_names={"DEGUILIO"},
    )
    result = matcher.resolve_one(_mention())
    assert result.status == "review"
    assert "Common last name" in result.reason


def test_invalid_agency_routes_to_review():
    matcher = PostMatcher(
        client=_FakeClient([_exact_candidate()]),
        scorer=lambda df: [0.9] * len(df),
        validator=lambda *a: (False, "Agency cannot be validated"),
        common_last_names=set(),
    )
    result = matcher.resolve_one(_mention())
    assert result.status == "review"


@pytest.mark.parametrize("llm_reply,expected", [("MATCH", True), ("NO_MATCH", False)])
def test_validate_agency_match_parses_llm_verdict(llm_reply, expected):
    valid, _reason = validate_agency_match(
        "Napa Police Department",
        "",
        "Napa Police Department",
        llm_fn=lambda prompt: llm_reply,
    )
    assert valid is expected
