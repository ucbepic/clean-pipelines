"""Thin Typer CLI. Parses args, builds Settings, calls `pipeline.run()`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from .data_loading import build_case_records, clean_date_fields, load_documents, load_ground_truth
from .evaluation import add_ground_truth_dates, evaluate_matches, log_performance_metrics
from .pipeline import run as pipeline_run

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def run(
    input: Path = typer.Option(..., "--input", help="Path to input jsonl of CaseRecords."),
    output: Path = typer.Option(..., "--output", help="Path to output jsonl."),
    n_threads: int = typer.Option(20, "--n-threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Extract incident dates from a jsonl of case records."""
    _setup_logging(log_level)
    result = pipeline_run(input, output, n_threads=n_threads)
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def eval(
    results: Path = typer.Option(..., "--results", help="Pipeline output jsonl."),
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Per-case GT (parquet or csv)."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for metrics + mismatch files."),
    model_name: str = typer.Option("model", "--model-name", help="Tag used in metric filenames."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Score the run output against the ground-truth tables and write metrics."""
    from .evaluation import write_metrics

    _setup_logging(log_level)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(ground_truth)
    gt = clean_date_fields(gt)

    results_df = pd.read_json(results, lines=True)
    merged = results_df.merge(gt, on="provisional_case_name", how="left")
    if merged.empty:
        raise typer.BadParameter(
            "merge produced 0 rows; check that provisional_case_name matches between results and GT"
        )

    merged = add_ground_truth_dates(merged)
    merged = evaluate_matches(merged)
    merged = log_performance_metrics(merged, out_dir=str(out_dir))

    detail_path = out_dir / f"date_extraction_results__{model_name}.csv"
    merged.to_csv(detail_path, index=False)
    typer.echo(f"Wrote detailed results to {detail_path}")
    write_metrics(str(out_dir), model_name, merged)


@app.command()
def prepare(
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Per-case GT (parquet or csv)."),
    documents: Path = typer.Option(..., "--documents", help="Per-file documents (parquet or csv)."),
    output: Path = typer.Option(..., "--output", help="Path to write the case-records jsonl."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Convert ground-truth + documents tables into the jsonl input."""
    _setup_logging(log_level)
    from prap_core.io import write_jsonl

    gt = load_ground_truth(ground_truth)
    doc = load_documents(documents)
    records = build_case_records(gt, doc)
    n = write_jsonl(output, records)
    typer.echo(f"Wrote {n} case records to {output}")
