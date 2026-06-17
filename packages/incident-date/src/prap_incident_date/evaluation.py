"""Match evaluation, metrics, and result-writing."""

import json
import logging
import os

import pandas as pd

from .helpers import get_ground_truth_dates

logger = logging.getLogger("prap.incident_date.evaluation")


CASE_TYPES: list[str] = ["Misconduct_case_type", "UOF_case_type", "OIS_case_type"]


def add_ground_truth_dates(merged_df: pd.DataFrame) -> pd.DataFrame:
    merged_df["all_ground_truth_dates"] = merged_df.apply(get_ground_truth_dates, axis=1)
    merged_df["all_ground_truth_dates_str"] = merged_df["all_ground_truth_dates"].apply(
        lambda dates: ", ".join(dates) if dates else None
    )
    return merged_df


def check_date_match(row: pd.Series) -> bool:
    if not row["all_ground_truth_dates"] or row["extracted_date"] is None:
        if (
            (pd.isna(row["Start_year"]) or row["Start_year"] == "")
            and (pd.isna(row["Start_month"]) or row["Start_month"] == "")
            and (pd.isna(row["Start_day"]) or row["Start_day"] == "")
            and (
                row["extracted_date"] is None
                or row["extracted_date"] == ""
                or (isinstance(row["extracted_date"], list) and len(row["extracted_date"]) == 0)
            )
        ):
            return True
        return False

    extracted = (
        row["extracted_date"]
        if isinstance(row["extracted_date"], list)
        else [row["extracted_date"]]
    )
    extracted = [str(d) for d in extracted if pd.notna(d)]
    gt = [str(d) for d in row["all_ground_truth_dates"] if pd.notna(d)]
    return any(e in gt for e in extracted)


def evaluate_matches(merged_df: pd.DataFrame) -> pd.DataFrame:
    merged_df["match"] = merged_df.apply(check_date_match, axis=1)
    return merged_df


def _row_classify(row: pd.Series) -> tuple:
    has_gt = (
        isinstance(row["all_ground_truth_dates"], list) and len(row["all_ground_truth_dates"]) > 0
    )
    has_ext = row["extracted_date"] is not None
    if has_ext and isinstance(row["extracted_date"], list):
        has_ext = len(row["extracted_date"]) > 0
    is_match = bool(row["match"])
    if has_gt:
        if is_match:
            return "true_positive", has_gt, has_ext
        if has_ext:
            return "false_negative_and_false_positive", has_gt, has_ext
        return "false_negative", has_gt, has_ext
    if has_ext:
        return "false_positive", has_gt, has_ext
    return "true_negative", has_gt, has_ext


def _is_case_type_true(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().upper() == "TRUE" or str(value).strip() in ("true", "True")


def log_performance_metrics(merged_df: pd.DataFrame, out_dir: str | None = None) -> pd.DataFrame:
    total = len(merged_df)
    tp = fp = tn = fn = 0
    mismatches = []
    merged_df["classification"] = None

    case_type_metrics = {
        ct: {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "total": 0,
        }
        for ct in CASE_TYPES
        if ct in merged_df.columns
    }

    for idx, row in merged_df.iterrows():
        cls, has_gt, has_ext = _row_classify(row)
        if cls == "true_positive":
            tp += 1
        elif cls == "false_negative_and_false_positive":
            fn += 1
            fp += 1
            mismatches.append(_mismatch_row(row, cls))
        elif cls == "false_negative":
            fn += 1
            mismatches.append(_mismatch_row(row, cls))
        elif cls == "false_positive":
            fp += 1
            mismatches.append(_mismatch_row(row, cls))
        else:
            tn += 1

        merged_df.at[idx, "classification"] = cls

        for ct, metrics in case_type_metrics.items():
            if not _is_case_type_true(row.get(ct)):
                continue
            metrics["total"] += 1
            if cls == "true_positive":
                metrics["true_positives"] += 1
            elif cls == "true_negative":
                metrics["true_negatives"] += 1
            elif cls == "false_positive":
                metrics["false_positives"] += 1
            elif cls in ("false_negative", "false_negative_and_false_positive"):
                metrics["false_negatives"] += 1
                if has_ext:
                    metrics["false_positives"] += 1

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info("===== OVERALL PERFORMANCE METRICS =====")
    logger.info(f"Total: {total}  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    logger.info(f"Accuracy: {accuracy:.4f}  Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}  F1: {f1:.4f}")

    merged_df["metric_true_positives"] = tp
    merged_df["metric_false_positives"] = fp
    merged_df["metric_true_negatives"] = tn
    merged_df["metric_false_negatives"] = fn
    merged_df["metric_accuracy"] = accuracy
    merged_df["metric_precision"] = precision
    merged_df["metric_recall"] = recall
    merged_df["metric_f1_score"] = f1

    for ct, metrics in case_type_metrics.items():
        if metrics["total"] == 0:
            continue
        ct_total = metrics["total"]
        ct_tp = metrics["true_positives"]
        ct_fp = metrics["false_positives"]
        ct_tn = metrics["true_negatives"]
        ct_fn = metrics["false_negatives"]
        ct_acc = (ct_tp + ct_tn) / ct_total
        ct_p = ct_tp / (ct_tp + ct_fp) if (ct_tp + ct_fp) > 0 else 0
        ct_r = ct_tp / (ct_tp + ct_fn) if (ct_tp + ct_fn) > 0 else 0
        ct_f1 = 2 * ct_p * ct_r / (ct_p + ct_r) if (ct_p + ct_r) > 0 else 0
        prefix = ct.replace("_case_type", "")
        merged_df[f"{prefix}_metric_accuracy"] = ct_acc
        merged_df[f"{prefix}_metric_precision"] = ct_p
        merged_df[f"{prefix}_metric_recall"] = ct_r
        merged_df[f"{prefix}_metric_f1_score"] = ct_f1

    merged_df["is_true_positive"] = merged_df["classification"] == "true_positive"
    merged_df["is_false_positive"] = merged_df["classification"].isin(
        ["false_positive", "false_negative_and_false_positive"]
    )
    merged_df["is_true_negative"] = merged_df["classification"] == "true_negative"
    merged_df["is_false_negative"] = merged_df["classification"].isin(
        ["false_negative", "false_negative_and_false_positive"]
    )

    if mismatches and out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "date_mismatches.json")
            with open(path, "w") as f:
                json.dump(mismatches, f, indent=2, default=str)
            logger.info(f"Saved {len(mismatches)} mismatches to {path}")
        except Exception as e:
            logger.warning(f"Could not save mismatches: {e}")

    return merged_df


def _mismatch_row(row: pd.Series, error_type: str) -> dict:
    return {
        "case_name": row["provisional_case_name"],
        "ground_truth_dates": row["all_ground_truth_dates"],
        "extracted_date": row["extracted_date"],
        "nl_date": row["nl_date"],
        "error_type": error_type,
    }


def write_metrics(out_dir: str, model_name: str, results_df: pd.DataFrame) -> None:
    precision = float(results_df["metric_precision"].iloc[0]) if not results_df.empty else 0.0
    recall = float(results_df["metric_recall"].iloc[0]) if not results_df.empty else 0.0
    f1 = float(results_df["metric_f1_score"].iloc[0]) if not results_df.empty else 0.0

    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/metrics__{model_name}.csv"
    pd.DataFrame(
        [{"model": model_name, "precision": precision, "recall": recall, "f1": f1}]
    ).to_csv(csv_path, index=False)
    logger.info(f"Wrote metrics CSV to {csv_path}")

    txt_path = f"{out_dir}/metrics__{model_name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall:    {recall:.4f}\n")
        f.write(f"f1:        {f1:.4f}\n")
    logger.info(f"Wrote metrics summary to {txt_path}")
