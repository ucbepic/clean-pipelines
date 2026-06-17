"""Thin Typer CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .data_loading import build_case_records, load_documents, load_ground_truth
from .evaluation import load_predictions, log_summary, score, write_metrics
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
    n_threads: int = typer.Option(50, "--n-threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Classify cases from a jsonl of CaseRecords."""
    _setup_logging(log_level)
    result = pipeline_run(input, output, n_threads=n_threads)
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def eval(
    results: Path = typer.Option(..., "--results", help="Pipeline output jsonl."),
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Per-case GT (parquet or csv)."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for metrics files."),
    model_name: str = typer.Option("model", "--model-name", help="Tag used in metric filenames."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Score the run output against the ground-truth tristate columns."""
    _setup_logging(log_level)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(ground_truth)
    preds = load_predictions(results)
    merged = preds.merge(
        gt[["provisional_case_name", "UOF_case_type", "Misconduct_case_type", "OIS_case_type"]],
        on="provisional_case_name",
        how="inner",
    )
    if merged.empty:
        raise typer.BadParameter(
            "merge produced 0 rows; check that provisional_case_name matches between results and GT"
        )

    metrics = score(merged)
    log_summary(metrics)

    detail_path = out_dir / f"case_type_results__{model_name}.csv"
    merged.to_csv(detail_path, index=False)
    typer.echo(f"Wrote detailed results to {detail_path}")
    write_metrics(out_dir, model_name, metrics)


@app.command()
def prepare(
    ground_truth: Path = typer.Option(..., "--ground-truth"),
    documents: Path = typer.Option(..., "--documents"),
    output: Path = typer.Option(..., "--output"),
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
