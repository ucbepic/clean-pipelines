"""Offline test for the `eval` step: link-level P/R/F1 of auto-matches vs GT.

Uses prap_core.eval.prf over (officer_uid, post_person_nbr) link sets.
"""

from __future__ import annotations

from prap_core.io import write_jsonl
from prap_post_entity_resolution.resolve.evaluation import evaluate


def _result(uid, status, person=""):
    return {
        "input_officer": {"officer_uid": uid},
        "status": status,
        "post_match": {"post_person_nbr": person} if person else {},
    }


def test_evaluate_link_level_prf(tmp_path):
    results = tmp_path / "results.jsonl"
    write_jsonl(
        results,
        [
            _result("a", "auto_matched", "P1"),  # correct
            _result("b", "auto_matched", "P2"),  # wrong person (gold says P3)
            _result("c", "review"),  # missed (gold has P4)
        ],
    )
    gt = tmp_path / "gt.csv"
    gt.write_text("officer_uid,post_person_nbr\na,P1\nb,P3\nc,P4\n")

    prf = evaluate(results, gt)
    assert prf.true_positive == 1
    assert prf.precision == 0.5  # 1 correct of 2 predicted matches
    assert abs(prf.recall - 1 / 3) < 1e-9  # 1 correct of 3 gold matches
