"""Thin Typer CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from prap_core.io import read_jsonl

from .evaluation import (
    load_ground_truth,
    predictions_from_tocs,
    score_by_group,
)
from .pipeline import run as pipeline_run
from .schemas import DocumentTOC

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def run(
    input: Path = typer.Option(..., "--input", help="jsonl of DocText (sha1 + pre-OCR'd pages)."),
    output: Path = typer.Option(..., "--output", help="jsonl of DocumentTOC."),
    n_threads: int = typer.Option(8, "--n-threads"),
    no_domain: bool = typer.Option(
        False, "--no-domain", help="Disable the SB-1421 domain preamble."
    ),
    no_history: bool = typer.Option(
        False, "--no-history", help="Disable rolling page-history context."
    ),
    no_context: bool = typer.Option(
        False, "--no-context", help="Disable previous-page tail context."
    ),
    recent_window: int = typer.Option(15, "--recent-window"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Segment + index every document in `input`."""
    _setup_logging(log_level)
    result = pipeline_run(
        input,
        output,
        n_threads=n_threads,
        use_domain=not no_domain,
        use_history=not no_history,
        use_context=not no_context,
        recent_window=recent_window,
    )
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def eval(
    results: Path = typer.Option(..., "--results", help="Pipeline output jsonl of DocumentTOC."),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", help="Per-page GT (xlsx/csv/parquet) with sha1, page_number, label."
    ),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for metrics files."),
    model_name: str = typer.Option("model", "--model-name"),
    stratum_col: str = typer.Option(
        "stratum", "--stratum-col", help="Optional GT column for per-group breakdown."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Score per-page binary boundary detection against a labeled sample."""
    _setup_logging(log_level)
    out_dir.mkdir(parents=True, exist_ok=True)

    tocs = [DocumentTOC.model_validate(r) for r in read_jsonl(results)]
    pred = predictions_from_tocs(tocs)
    gt = load_ground_truth(ground_truth)

    merged = gt.merge(pred, on=["sha1", "page_number"], how="inner")
    if merged.empty:
        raise typer.BadParameter(
            "merge produced 0 rows; check sha1/page_number alignment between GT and results"
        )

    detail_path = out_dir / f"page_classifications__{model_name}.csv"
    merged.to_csv(detail_path, index=False)
    typer.echo(f"Wrote per-page comparison to {detail_path}")

    metrics_df = score_by_group(merged, stratum_col)
    metrics_path = out_dir / f"metrics__{model_name}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    typer.echo(f"Wrote metrics to {metrics_path}")
    typer.echo(metrics_df.to_string(index=False))
