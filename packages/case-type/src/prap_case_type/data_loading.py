"""Convert parquet/csv ground truth into the jsonl input format consumed by `pipeline.run()`."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("prap.case_type.data_loading")


FILE_NAMES_TO_CLEAR_DATES = [
    "1718081113360-ofq",
    "1718081310723-lue",
    "1718081312742-hzc",
    "1718081323305-tvk",
    "1718081438666-fto",
    "1718121589659-jhw",
]

PROBLEMATIC_CASES = [
    "1717696024591-kzk, 1717695709058-gau",
    "1717696024591-kzk",
    "1717695709058-gau",
]


def _normalize_case_names(df: pd.DataFrame, col: str = "provisional_case_name") -> pd.DataFrame:
    df.loc[:, col] = (
        df[col]
        .str.lower()
        .str.strip()
        .fillna("")
        .str.replace(r"\n", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    return df


def _clear_date_fields_for_specific_cases(
    groundtruth_df: pd.DataFrame, file_names_to_clear: list[str]
) -> pd.DataFrame:
    mask = groundtruth_df["provisional_case_name"].isin(file_names_to_clear)
    if mask.any():
        for col in [
            "Start_year",
            "Start_month",
            "Start_day",
            "Misconduct_case_type",
            "UOF_case_type",
            "OIS_case_type",
        ]:
            if col in groundtruth_df.columns:
                groundtruth_df[col] = groundtruth_df[col].astype(object)
                groundtruth_df.loc[mask, col] = ""
    return groundtruth_df


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_ground_truth(gt_path: str | Path) -> pd.DataFrame:
    gt_path = Path(gt_path)
    gt = _read(gt_path)
    gt = gt[~gt.provisional_case_name.isin(PROBLEMATIC_CASES)]
    gt = _normalize_case_names(gt)
    gt = gt[gt.provisional_case_name.fillna("") != ""]
    for col in ["Misconduct_case_type", "UOF_case_type", "OIS_case_type"]:
        if col in gt.columns:
            gt.loc[:, col] = gt[col].fillna("UNCLEAR")
    gt = _clear_date_fields_for_specific_cases(gt, FILE_NAMES_TO_CLEAR_DATES)
    return gt


def load_documents(doc_path: str | Path) -> pd.DataFrame:
    doc_path = Path(doc_path)
    doc = _read(doc_path)
    doc = _normalize_case_names(doc)
    doc = doc[~doc.provisional_case_name.isin(PROBLEMATIC_CASES)]
    return doc[doc.provisional_case_name.fillna("") != ""]


def build_case_records(groundtruth_df: pd.DataFrame, document_df: pd.DataFrame) -> list[dict]:
    """Produce the jsonl-shaped CaseRecord list for `pipeline.run()`."""
    valid_cases = set(groundtruth_df["provisional_case_name"].values)
    filtered = document_df[document_df["provisional_case_name"].isin(valid_cases)]
    if "file_belong" in filtered.columns:
        filtered = filtered[filtered.file_belong == "TRUE"]

    records: list[dict] = []
    processed: set[str] = set()
    for case_name, group in filtered.groupby("provisional_case_name"):
        summaries = (
            group["first_look_summary"].dropna().tolist()
            if "first_look_summary" in group.columns
            else []
        )
        ocr_texts = group["ocr_text"].dropna().tolist() if "ocr_text" in group.columns else []
        if not summaries:
            summaries = ["No summary"]
        records.append(
            {
                "provisional_case_name": case_name,
                "summaries": summaries,
                "ocr_texts": ocr_texts,
            }
        )
        processed.add(case_name)

    for case_name in valid_cases - processed:
        records.append(
            {"provisional_case_name": case_name, "summaries": ["No summary"], "ocr_texts": []}
        )
    return records
