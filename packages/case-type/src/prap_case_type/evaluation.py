"""Scoring for prap-case-type.

Computes per-field precision / recall / F1 (use_of_force, misconduct,
officer_involved_shooting) plus a micro-averaged overall score.

`Unclear` values — in both predictions and ground truth — are treated as
`False`, matching how downstream consumers interpret the tristate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from prap_core.eval import BinaryPRF, binary_prf

logger = logging.getLogger("prap.case_type.evaluation")


FIELDS: list[tuple[str, str]] = [
    ("use_of_force", "UOF_case_type"),
    ("misconduct", "Misconduct_case_type"),
    ("officer_involved_shooting", "OIS_case_type"),
]


def _norm(value: object) -> bool:
    """Normalize tristate (True / False / Unclear / NaN / missing) to bool.

    `Unclear` and any missing / unrecognized value normalize to False. The
    ground-truth column uses upper-case strings (`TRUE` / `FALSE` /
    `UNCLEAR`); the pipeline output uses title-case (`True` / `False` /
    `Unclear`). Both shapes are accepted.
    """
    if value is None:
        return False
    s = str(value).strip().upper()
    if s in ("TRUE", "T", "1", "YES", "Y"):
        return True
    return False


def load_predictions(results_path: str | Path) -> pd.DataFrame:
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cls = r.get("classification") or {}
            rows.append(
                {
                    "provisional_case_name": r["provisional_case_name"],
                    "use_of_force": cls.get("use_of_force"),
                    "misconduct": cls.get("misconduct"),
                    "officer_involved_shooting": cls.get("officer_involved_shooting"),
                }
            )
    return pd.DataFrame(rows)


def score(merged: pd.DataFrame) -> dict[str, BinaryPRF]:
    """Per-field + overall (micro-averaged) metrics."""
    per_field: dict[str, BinaryPRF] = {}
    all_pred: list[bool] = []
    all_gold: list[bool] = []
    for pred_col, gold_col in FIELDS:
        pred = [_norm(v) for v in merged[pred_col]]
        gold = [_norm(v) for v in merged[gold_col]]
        per_field[pred_col] = binary_prf(pred, gold)
        all_pred.extend(pred)
        all_gold.extend(gold)
    per_field["overall"] = binary_prf(all_pred, all_gold)
    return per_field


def write_metrics(out_dir: str | Path, model_name: str, metrics: dict[str, BinaryPRF]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [{"model": model_name, "field": field, **asdict(m)} for field, m in metrics.items()]
    csv_path = out_dir / f"metrics__{model_name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Wrote metrics CSV to {csv_path}")

    txt_path = out_dir / f"metrics__{model_name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"model: {model_name}\n")
        f.write("Unclear treated as False (predictions and ground truth)\n\n")
        f.write(f"{'field':30s} {'n':>5} {'P':>7} {'R':>7} {'F1':>7} {'acc':>7}\n")
        for field, m in metrics.items():
            f.write(
                f"{field:30s} {m.n:>5d} "
                f"{m.precision:>7.4f} {m.recall:>7.4f} {m.f1:>7.4f} {m.accuracy:>7.4f}\n"
            )
    logger.info(f"Wrote metrics summary to {txt_path}")


def log_summary(metrics: dict[str, BinaryPRF]) -> None:
    logger.info("===== CASE-TYPE PERFORMANCE METRICS (Unclear=False) =====")
    for field, m in metrics.items():
        logger.info(
            f"{field:30s} n={m.n:4d} TP={m.true_positive:3d} FP={m.false_positive:3d} "
            f"FN={m.false_negative:3d} TN={m.true_negative:3d} "
            f"P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}"
        )
