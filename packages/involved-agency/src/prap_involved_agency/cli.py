"""Thin Typer CLI: prap-involved-agency {prepare, run, eval}."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from .pipeline import run as pipeline_run

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def prepare(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        help="Directory of `agency_case_file_bundle-*.json` case bundles.",
    ),
    output: Path = typer.Option(..., "--output", help="Path to write the case-records jsonl."),
    ground_truth: Path | None = typer.Option(
        None,
        "--ground-truth",
        help="Optional GT CSV: only include cases whose `case_name` appears there.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Convert a directory of case-file bundles into the jsonl input for `run`."""
    from prap_core.io import write_jsonl

    _setup_logging(log_level)

    bundle_files = sorted(input_dir.glob("agency_case_file_bundle-*.json"))
    typer.echo(f"Found {len(bundle_files)} case bundles in {input_dir}")

    gt_cases: set[str] | None = None
    if ground_truth is not None:
        gt_df = pd.read_csv(ground_truth)
        gt_cases = set(gt_df["case_name"].unique())
        typer.echo(f"Filtering to {len(gt_cases)} cases from {ground_truth}")

    records = []
    for bf in bundle_files:
        case_name = bf.stem.replace("agency_case_file_bundle-", "")
        if gt_cases is not None and case_name not in gt_cases:
            continue
        with open(bf) as f:
            case_data = json.load(f)
        case_data["case_name"] = case_name
        case_data.setdefault("provisional_case_name", case_name)
        records.append(case_data)

    n = write_jsonl(output, records)
    typer.echo(f"Wrote {n} case records to {output}")


@app.command()
def run(
    input: Path = typer.Option(..., "--input", help="Path to input jsonl of case records."),
    output: Path = typer.Option(..., "--output", help="Path to output CSV."),
    n_threads: int = typer.Option(15, "--n-threads"),
    resume: bool = typer.Option(False, "--resume", help="Skip cases already in output CSV."),
    save_every: int = typer.Option(10, "--save-every", help="Checkpoint every N cases."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Extract agencies from a jsonl of case records, write per-(case,agency,role) rows to CSV."""
    _setup_logging(log_level)
    result = pipeline_run(
        input,
        output,
        n_threads=n_threads,
        resume=resume,
        save_every=save_every,
    )
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def eval(
    results: Path = typer.Option(..., "--results", help="Pipeline output CSV."),
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Groundtruth CSV."),
    ground_truth_fp: Path | None = typer.Option(
        None,
        "--ground-truth-fp",
        help="Optional groundtruth_false_positives_labeled CSV.",
    ),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for metrics + mismatch files."),
    model_name: str = typer.Option("model", "--model-name"),
    n_threads: int = typer.Option(20, "--n-threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Score extraction CSV against groundtruth_labeled.csv."""
    from prap_core.llm import LLM

    from .evaluation import (
        filter_extractions,
        load_groundtruth,
        match_extractions_to_groundtruth,
        score,
        write_metrics,
    )

    _setup_logging(log_level)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_df = load_groundtruth(ground_truth, fp_path=ground_truth_fp)
    extractions_df = pd.read_csv(results)
    extractions_df = filter_extractions(extractions_df, set(gt_df["case_name"].unique()))

    llm = LLM()
    matched = match_extractions_to_groundtruth(
        llm, gt_df, extractions_df, max_workers=n_threads
    )

    detail_path = out_dir / f"agency_eval_results__{model_name}.csv"
    matched.to_csv(detail_path, index=False)
    typer.echo(f"Wrote detailed results to {detail_path}")

    # Per-case mismatch CSV (FP + FN rows)
    mismatches = matched[
        matched["result_type"].isin(["false_positive", "false_negative"])
    ]
    mismatch_path = out_dir / f"agency_eval_mismatches__{model_name}.csv"
    mismatches.to_csv(mismatch_path, index=False)
    typer.echo(f"Wrote mismatches to {mismatch_path}")

    prf = score(matched)
    write_metrics(out_dir, model_name, prf)
    typer.echo(
        json.dumps(
            {
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
                "accuracy": prf.accuracy,
                "n": prf.n,
            },
            indent=2,
        )
    )
