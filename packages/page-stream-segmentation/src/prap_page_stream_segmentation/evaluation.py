"""Page-level binary eval.

Each page in the ground truth is labeled 1 if it is the start of a new
document (a "boundary"), 0 if it is a continuation. The pipeline's
prediction for a page is 1 if the page is the start-page of a TOC entry,
else 0. Reduces to per-page binary PRF — `prap_core.eval.binary_prf` works
directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from prap_core.eval import BinaryPRF, binary_prf

from .schemas import DocumentTOC

logger = logging.getLogger("prap.page_stream_segmentation.eval")


def predictions_from_tocs(tocs: list[DocumentTOC]) -> pd.DataFrame:
    """Flatten a list of DocumentTOC into (sha1, page_number, predicted, predicted_doctype)."""
    rows: list[dict] = []
    for toc in tocs:
        for entry in toc.entries:
            start_page = entry.start_page
            for pc in entry.page_classifications:
                rows.append(
                    {
                        "sha1": toc.sha1,
                        "page_number": pc.page_number,
                        "predicted": 1 if pc.page_number == start_page else 0,
                        "predicted_doctype": pc.document_type,
                    }
                )
    return pd.DataFrame(rows)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load a labeled-sample xlsx/csv/parquet with columns (sha1, page_number, label, ...)."""
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def score(merged: pd.DataFrame) -> BinaryPRF:
    """PRF over the predicted vs. label columns in a joined GT+predictions df."""
    return binary_prf(merged["predicted"].astype(bool), merged["label"].astype(bool))


def score_by_group(merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-group PRF; returns one row per group plus an 'overall' row."""
    rows: list[dict] = []
    overall = score(merged)
    rows.append({"group": "overall", **_prf_row(merged, overall)})
    if group_col in merged.columns:
        for g, sub in merged.groupby(group_col, dropna=False):
            prf = binary_prf(sub["predicted"].astype(bool), sub["label"].astype(bool))
            rows.append({"group": str(g), **_prf_row(sub, prf)})
    return pd.DataFrame(rows)


def _prf_row(df: pd.DataFrame, prf: BinaryPRF) -> dict:
    return {
        "n_files": df["sha1"].nunique(),
        "n_pages": len(df),
        "tp": prf.true_positive,
        "fp": prf.false_positive,
        "fn": prf.false_negative,
        "tn": prf.true_negative,
        "precision": round(prf.precision, 4),
        "recall": round(prf.recall, 4),
        "f1": round(prf.f1, 4),
        "accuracy": round(prf.accuracy, 4),
    }
