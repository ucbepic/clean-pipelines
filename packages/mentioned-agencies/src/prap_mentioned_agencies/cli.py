"""Thin Typer CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .pipeline import run as pipeline_run

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def run(
    input: Path = typer.Option(
        ...,
        "--input",
        help=(
            "Either a directory of agency_case_file_bundle-*.json files or a jsonl of CaseBundles."
        ),
    ),
    output: Path = typer.Option(..., "--output", help="Path to output jsonl."),
    n_threads: int = typer.Option(16, "--n-threads"),
    dedup_threshold: int = typer.Option(85, "--dedup-threshold"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Extract mentioned law-enforcement agencies per case."""
    _setup_logging(log_level)
    result = pipeline_run(input, output, n_threads=n_threads, dedup_threshold=dedup_threshold)
    typer.echo(json.dumps(result.model_dump(), indent=2))
