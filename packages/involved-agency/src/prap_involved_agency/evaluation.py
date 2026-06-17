"""Evaluation: match extracted agencies against ground truth and score.

Matching uses an LLM-based agency-name comparator; scoring uses
`prap_core.eval.binary_prf` for record-level binary precision / recall / F1.

A record here is a (case, gt_agency) pair: gold=True if the GT row's
`correct == '1'`, predicted=True if some extraction in the same case
matches the GT name via the LLM comparator. Extractions present in a case
but not in the GT contribute a (gold=False, predicted=True) row.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import pandas as pd
from jinja2 import Template
from prap_core.eval import binary_prf

from .schemas import AgencyNameMatch

if TYPE_CHECKING:
    from prap_core.llm import LLM

logger = logging.getLogger("prap.involved_agency.evaluation")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_involved_agency.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


def compare_agency_names_llm(llm: "LLM", name1: str, name2: str) -> bool:
    """LLM-based judgment of whether two agency names refer to the same entity."""
    prompt = Template(_load_prompt("compare_agency_names")).render(
        name1=name1, name2=name2
    )

    try:
        result = llm.complete(prompt, response_format=AgencyNameMatch)
        logger.debug(
            f"Agency match: '{name1}' vs '{name2}' = {result.is_match} ({result.confidence})"
        )
        return result.is_match
    except Exception as e:
        logger.error(f"Error comparing agency names: {e}")
        # Fallback to exact string match
        return name1.strip().lower() == name2.strip().lower()


def _compare_wrapper(args: Tuple["LLM", str, str, int]) -> Tuple[int, bool]:
    llm, name1, name2, task_id = args
    return task_id, compare_agency_names_llm(llm, name1, name2)


def match_extractions_to_groundtruth(
    llm: "LLM",
    groundtruth_df: pd.DataFrame,
    extractions_df: pd.DataFrame,
    max_workers: int = 20,
) -> pd.DataFrame:
    """Match extracted agencies to groundtruth using LLM name matching.

    Returns a DataFrame with one row per (case, gt_agency) plus rows for
    extractions not in groundtruth (false positives). Columns include
    `extracted` (bool), `gt_correct` (str/None), and `result_type`.
    """
    logger.info("=" * 80)
    logger.info("MATCHING EXTRACTIONS TO GROUNDTRUTH")
    logger.info("=" * 80)

    gt_by_case = groundtruth_df.groupby("case_name")
    ext_by_case = extractions_df.groupby("case_name")
    all_cases = set(groundtruth_df["case_name"].unique())

    # Stage 1: queue every (gt_row, ext_row) pair for comparison
    comparison_tasks = []
    comparison_index: Dict[int, Dict] = {}

    for case_name in all_cases:
        if case_name not in gt_by_case.groups:
            continue

        gt_agencies = gt_by_case.get_group(case_name)
        ext_agencies = (
            ext_by_case.get_group(case_name)
            if case_name in ext_by_case.groups
            else pd.DataFrame()
        )

        for gt_idx, gt_row in gt_agencies.iterrows():
            for ext_idx, ext_row in ext_agencies.iterrows():
                task_id = len(comparison_tasks)
                comparison_tasks.append(
                    (llm, gt_row["agency_name"], ext_row["agency_name"], task_id)
                )
                comparison_index[task_id] = {
                    "case_name": case_name,
                    "gt_idx": gt_idx,
                    "ext_idx": ext_idx,
                }

    logger.info(f"Queued {len(comparison_tasks)} agency name comparisons")

    comparison_results: Dict[int, bool] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_compare_wrapper, task) for task in comparison_tasks
        ]
        for fut in as_completed(futures):
            task_id, is_match = fut.result()
            comparison_results[task_id] = is_match

    logger.info(f"Completed all {len(comparison_tasks)} comparisons")

    # Stage 2: build per-(case, gt) and per-extraction rows
    results = []
    for case_name in all_cases:
        if case_name not in gt_by_case.groups:
            continue

        gt_agencies = gt_by_case.get_group(case_name)
        ext_agencies = (
            ext_by_case.get_group(case_name)
            if case_name in ext_by_case.groups
            else pd.DataFrame()
        )

        for gt_idx, gt_row in gt_agencies.iterrows():
            gt_name = gt_row["agency_name"]
            gt_type = gt_row["agency_type"]
            gt_correct = str(gt_row["correct"])

            match_found = False
            matched_ext_name = None
            matched_ext_type = None

            for ext_idx, ext_row in ext_agencies.iterrows():
                task_id = None
                for tid, info in comparison_index.items():
                    if (
                        info["case_name"] == case_name
                        and info["gt_idx"] == gt_idx
                        and info["ext_idx"] == ext_idx
                    ):
                        task_id = tid
                        break
                names_match = (
                    comparison_results[task_id] if task_id is not None else False
                )
                if names_match:
                    match_found = True
                    matched_ext_name = ext_row["agency_name"]
                    matched_ext_type = ext_row["agency_type"]
                    break

            if gt_correct == "1":
                result_type = "true_positive" if match_found else "false_negative"
            elif gt_correct == "0":
                result_type = "false_positive" if match_found else "true_negative"
            else:
                result_type = "uncertain"

            results.append(
                {
                    "case_name": case_name,
                    "gt_agency_name": gt_name,
                    "gt_agency_type": gt_type,
                    "gt_correct": gt_correct,
                    "extracted": match_found,
                    "ext_agency_name": matched_ext_name if match_found else None,
                    "ext_agency_type": matched_ext_type if match_found else None,
                    "result_type": result_type,
                }
            )

        # Extractions not in GT → false_positive rows
        for ext_idx, ext_row in ext_agencies.iterrows():
            found_in_gt = False
            for gt_idx, _ in gt_agencies.iterrows():
                task_id = None
                for tid, info in comparison_index.items():
                    if (
                        info["case_name"] == case_name
                        and info["gt_idx"] == gt_idx
                        and info["ext_idx"] == ext_idx
                    ):
                        task_id = tid
                        break
                if task_id is not None and comparison_results[task_id]:
                    found_in_gt = True
                    break

            if not found_in_gt:
                results.append(
                    {
                        "case_name": case_name,
                        "gt_agency_name": None,
                        "gt_agency_type": None,
                        "gt_correct": None,
                        "extracted": True,
                        "ext_agency_name": ext_row["agency_name"],
                        "ext_agency_type": ext_row["agency_type"],
                        "result_type": "false_positive",
                    }
                )

    return pd.DataFrame(results)


def score(results_df: pd.DataFrame):
    """Return `prap_core.eval.BinaryPRF` over the matched results.

    Each row contributes one (gold, predicted) pair:
      - GT row with correct=='1' → gold=True; predicted=`extracted`.
      - GT row with correct=='0' → gold=False; predicted=`extracted`.
      - Extraction-only row (FP) → gold=False; predicted=True.
      - GT rows with correct in {'', NaN} are dropped as `uncertain`.
    """
    df = results_df[results_df["result_type"] != "uncertain"].copy()

    gold = []
    predicted = []
    for _, row in df.iterrows():
        rt = row["result_type"]
        if rt == "true_positive":
            gold.append(True)
            predicted.append(True)
        elif rt == "false_negative":
            gold.append(True)
            predicted.append(False)
        elif rt == "false_positive":
            gold.append(False)
            predicted.append(True)
        elif rt == "true_negative":
            gold.append(False)
            predicted.append(False)

    return binary_prf(predicted=predicted, gold=gold)


def write_metrics(out_dir: str | Path, model_name: str, prf) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"metrics__{model_name}.csv"
    pd.DataFrame(
        [
            {
                "model": model_name,
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
                "accuracy": prf.accuracy,
                "n": prf.n,
                "true_positive": prf.true_positive,
                "false_positive": prf.false_positive,
                "true_negative": prf.true_negative,
                "false_negative": prf.false_negative,
            }
        ]
    ).to_csv(csv_path, index=False)
    logger.info(f"Wrote metrics CSV to {csv_path}")

    txt_path = out_dir / f"metrics__{model_name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"n:         {prf.n}\n")
        f.write(f"precision: {prf.precision:.4f}\n")
        f.write(f"recall:    {prf.recall:.4f}\n")
        f.write(f"f1:        {prf.f1:.4f}\n")
        f.write(f"accuracy:  {prf.accuracy:.4f}\n")
        f.write(f"tp: {prf.true_positive}  fp: {prf.false_positive}  ")
        f.write(f"tn: {prf.true_negative}  fn: {prf.false_negative}\n")
    logger.info(f"Wrote metrics summary to {txt_path}")


def load_groundtruth(
    gt_path: str | Path, fp_path: str | Path | None = None
) -> pd.DataFrame:
    """Load + filter groundtruth CSV (and optional false-positives-labeled CSV)."""
    gt_df = pd.read_csv(gt_path)
    gt_df = gt_df[gt_df["correct"].astype(str).str.contains(r"1")]
    if "corrected_name" in gt_df.columns:
        gt_df.loc[:, "corrected_name"] = gt_df.corrected_name.fillna("")
        gt_df = gt_df.drop_duplicates(
            subset=["agency_name", "corrected_name", "agency_type", "case_name"]
        )

    if fp_path and Path(fp_path).exists():
        fp_df = pd.read_csv(fp_path)
        fp_df = fp_df[fp_df["correct"].astype(str).str.contains(r"1")]
        fp_cols = ["case_name", "agency_name", "agency_type", "correct"]
        fp_df = fp_df[fp_cols]
        gt_df = pd.concat([gt_df, fp_df], ignore_index=True)
        gt_df = gt_df.drop_duplicates(
            subset=["case_name", "agency_name", "agency_type"], keep="first"
        )

    return gt_df


def filter_extractions(
    extractions_df: pd.DataFrame, gt_cases: set
) -> pd.DataFrame:
    """Apply standard filters: in-GT-cases, no fire dept, citations > 0, dedup."""
    df = extractions_df[extractions_df["case_name"].isin(gt_cases)]
    df = df[~df["agency_name"].astype(str).str.contains("fire", case=False, na=False)]
    if "num_citations" in df.columns:
        df = df[df["num_citations"] > 0]
    df = df.drop_duplicates(
        subset=["case_name", "agency_name", "agency_type"], keep="first"
    )
    return df
