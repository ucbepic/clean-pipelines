"""Thin Typer CLI."""

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
def run(
    input: Path = typer.Option(..., "--input", help="Path to input jsonl of NameRecords."),
    output: Path = typer.Option(..., "--output", help="Path to output jsonl."),
    n_threads: int = typer.Option(20, "--n-threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Validate officer names from a jsonl of NameRecords."""
    _setup_logging(log_level)
    result = pipeline_run(input, output, n_threads=n_threads)
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def prepare(
    input_csv: Path = typer.Option(
        ..., "--input-csv", help="CSV with `officer_name` (and optional `roles`, `case_id`)."
    ),
    output: Path = typer.Option(..., "--output"),
    drop_mentioned_roles: bool = typer.Option(True, "--drop-mentioned/--keep-mentioned"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Convert an involved-officer CSV into the NameRecord jsonl input."""
    _setup_logging(log_level)
    from prap_core.io import write_jsonl

    df = pd.read_csv(input_csv)
    if drop_mentioned_roles and "roles" in df.columns:
        df = df[~df.roles.str.lower().str.contains("mentioned", na=False)]

    # Accept either `officer_name` or `officer_name_string`.
    name_col = (
        "officer_name"
        if "officer_name" in df.columns
        else "officer_name_string"
        if "officer_name_string" in df.columns
        else None
    )
    if name_col is None:
        raise typer.BadParameter(
            "input csv must have an `officer_name` or `officer_name_string` column"
        )

    case_col = (
        "case_id" if "case_id" in df.columns else "case_name" if "case_name" in df.columns else None
    )

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "officer_name": str(row[name_col]) if pd.notna(row[name_col]) else "",
                "case_id": str(row[case_col])
                if case_col is not None and pd.notna(row.get(case_col))
                else None,
            }
        )
    n = write_jsonl(output, records)
    typer.echo(f"Wrote {n} name records to {output}")
