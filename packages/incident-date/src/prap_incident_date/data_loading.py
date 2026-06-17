"""Convert parquet ground truth into the jsonl input format consumed by `pipeline.run()`."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("prap.incident_date.data_loading")


SPECIAL_CASE_PREFIXES = [
    "1728506173798-unm",
    "1728506188915-ikx",
    "1728506198876-yxe",
    "1728506226039-het",
    "1728506250055-mpa",
    "1728506280750-mgr",
    "1725639855798-hhv",
    "1725640019247-qdl",
]

FILE_NAMES_TO_CLEAR_DATES = [
    "1718081113360-ofq",
    "1718081310723-lue",
    "1718081312742-hzc",
    "1718081323305-tvk",
    "1718081438666-fto",
    "1718121589659-jhw",
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
                # The original CSV path had string dtypes; parquet has typed
                # columns (float64 for Start_*). Cast to object before assigning
                # empty strings.
                groundtruth_df[col] = groundtruth_df[col].astype(object)
                groundtruth_df.loc[mask, col] = ""
    return groundtruth_df


def clean_date_fields(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Start_year",
        "Start_month",
        "Start_day",
        "End_year",
        "End_month",
        "End_day",
        "Misconduct_date_ranges",
    ]
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.replace(r"na", "", regex=True)
                .str.replace(r"\s+", "", regex=True)
            )
            if col != "Misconduct_date_ranges":
                df[col] = df[col].replace("", np.nan)
                if col.endswith(("_year", "_month", "_day")):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ground_truth(gt_path: str | Path) -> pd.DataFrame:
    """Load the per-case ground truth (CSV or Parquet) and normalize."""
    gt_path = Path(gt_path)
    gt = pd.read_parquet(gt_path) if gt_path.suffix == ".parquet" else pd.read_csv(gt_path)
    gt = _normalize_case_names(gt)
    gt = _clear_date_fields_for_specific_cases(gt, FILE_NAMES_TO_CLEAR_DATES)
    gt.loc[:, "Misconduct_date_ranges"] = (
        gt["Misconduct_date_ranges"]
        .astype(str)
        .str.lower()
        .str.strip()
        .fillna("")
        .str.replace(r"na", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    return gt


def load_documents(doc_path: str | Path) -> pd.DataFrame:
    doc_path = Path(doc_path)
    doc = pd.read_parquet(doc_path) if doc_path.suffix == ".parquet" else pd.read_csv(doc_path)
    return _normalize_case_names(doc)


def _filter_documents(document_df: pd.DataFrame, groundtruth_df: pd.DataFrame) -> pd.DataFrame:
    valid_cases = set(groundtruth_df["provisional_case_name"].values)
    by_case = document_df[document_df["provisional_case_name"].isin(valid_cases)]
    if "file_belong" in by_case.columns:
        by_case["file_belong"] = by_case["file_belong"].astype(str).str.lower() == "true"
        return by_case[by_case["file_belong"] == True]  # noqa: E712
    logger.warning(
        "file_belong column not found in document_df. Using all documents with valid case names."
    )
    return by_case


def _extract_ocr_pages_for_special_case(case_name: str, group: pd.DataFrame) -> list[str]:
    ocr_contents: list[str] = []
    for _, row in group.iterrows():
        if not (
            pd.notna(row.get("page_start"))
            and pd.notna(row.get("page_end"))
            and pd.notna(row.get("ocr_text"))
        ):
            continue
        try:
            start, end = int(row["page_start"]), int(row["page_end"])
            ocr_data = row["ocr_text"]
            if isinstance(ocr_data, str):
                try:
                    ocr_data = json.loads(ocr_data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Case {case_name}: Could not parse OCR text as JSON. Using as raw text."
                    )
                    ocr_contents.append(ocr_data)
                    continue
            messages = []
            if isinstance(ocr_data, dict) and "messages" in ocr_data:
                messages = ocr_data["messages"]
            elif isinstance(ocr_data, list):
                messages = ocr_data
            extracted = [
                m["page_content"]
                for m in messages
                if isinstance(m, dict)
                and "page_number" in m
                and "page_content" in m
                and start <= m["page_number"] <= end
            ]
            if extracted:
                ocr_contents.append("\n\n===== PAGE BREAK =====\n\n".join(extracted))
            else:
                fallback = [
                    m["page_content"]
                    for m in messages
                    if isinstance(m, dict) and "page_content" in m
                ]
                if fallback:
                    ocr_contents.append("\n\n===== PAGE BREAK =====\n\n".join(fallback))
        except Exception as e:
            logger.error(f"Error extracting OCR pages for case {case_name}: {e}")
            logger.error(traceback.format_exc())
    return ocr_contents


def build_case_records(groundtruth_df: pd.DataFrame, document_df: pd.DataFrame) -> list[dict]:
    """Produce the jsonl-shaped CaseRecord list for `pipeline.run()`."""
    filtered = _filter_documents(document_df, groundtruth_df)
    records: list[dict] = []
    for case_name, group in filtered.groupby("provisional_case_name"):
        is_special = any(case_name.startswith(p) for p in SPECIAL_CASE_PREFIXES)
        if (
            is_special
            and "page_start" in group.columns
            and "page_end" in group.columns
            and "ocr_text" in group.columns
        ):
            ocr_contents = _extract_ocr_pages_for_special_case(case_name, group)
            if ocr_contents:
                records.append(
                    {"provisional_case_name": case_name, "summaries": [], "ocr_pages": ocr_contents}
                )
                continue

        raw = group["first_look_summary"].tolist() if "first_look_summary" in group.columns else []
        summaries: list[str] = []
        for s in raw:
            if s is None or (isinstance(s, float) and pd.isna(s)):
                summaries.append("No summary")
            elif not isinstance(s, str):
                summaries.append(str(s))
            else:
                summaries.append(s)
        if not summaries:
            summaries = ["No summary"]
        records.append(
            {"provisional_case_name": case_name, "summaries": summaries, "ocr_pages": None}
        )
    return records
