"""The `eval` step: link-level precision/recall/F1 of auto-matches vs ground truth.

Ground truth is a table of `officer_uid -> post_person_nbr` (the correct POST person
for each mention; blank means "no correct match"). We score the set of predicted
auto-match links against the gold links with `prap_core.eval.prf`.
"""

from __future__ import annotations


def evaluate(results_path, ground_truth_path):
    """Score a resolve results jsonl against a GT csv. Returns a `prap_core.eval.PRF`."""
    import pandas as pd
    from prap_core.eval import prf
    from prap_core.io import read_jsonl

    gt = pd.read_csv(ground_truth_path).fillna("")
    gold = {
        (str(uid), str(person))
        for uid, person in zip(gt["officer_uid"], gt["post_person_nbr"], strict=False)
        if str(person).strip()
    }

    predicted = set()
    for rec in read_jsonl(results_path):
        if rec.get("status") != "auto_matched":
            continue
        uid = (rec.get("input_officer") or {}).get("officer_uid", "")
        person = (rec.get("post_match") or {}).get("post_person_nbr", "")
        if str(person).strip():
            predicted.add((str(uid), str(person)))

    return prf(predicted, gold)
