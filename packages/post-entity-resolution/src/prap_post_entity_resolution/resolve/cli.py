"""Typer CLI for the resolve entity-resolution pipeline.

Standard PRAP verbs (mirrors the other pipelines); invoke as
`prap-post-entity-resolution <verb> ...`:

    prepare --input mentions.csv  --output mentions.jsonl --default-state CA
    run     --input mentions.jsonl --output results.jsonl --api-url $NPI_API_URL
    eval    --results results.jsonl --ground-truth gt.csv

`prepare` turns a mentions CSV into the jsonl input; `run` resolves it against the NPI
API (XGBoost scoring + agency validation via prap_core); `eval` scores auto-matches
against ground truth. The NPI API base URL comes from --api-url or the NPI_API_URL env.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def prepare(
    input: Path = typer.Option(..., "--input", help="Mentions CSV."),
    output: Path = typer.Option(..., "--output", help="Output jsonl of OfficerMention records."),
    default_state: str = typer.Option(
        None, "--default-state", help="State for rows that lack one."
    ),
    sample_n: int = typer.Option(None, "--sample-n", help="Optionally subsample N rows."),
    sample_seed: int = typer.Option(None, "--sample-seed"),
) -> None:
    """Convert a mentions CSV into the jsonl pipeline input."""
    from .prepare import prepare as prepare_fn

    n = prepare_fn(
        input, output, default_state=default_state, sample_n=sample_n, sample_seed=sample_seed
    )
    typer.echo(f"Wrote {n} mention records to {output}")


@app.command()
def run(
    input: Path = typer.Option(..., "--input", help="Mentions jsonl (from `prepare`)."),
    output: Path = typer.Option(..., "--output", help="Output results jsonl."),
    api_url: str = typer.Option(None, "--api-url", help="NPI API base URL (else NPI_API_URL env)."),
    threshold: float = typer.Option(0.5, "--threshold"),
    require_state: bool = typer.Option(True, "--require-state/--no-require-state"),
) -> None:
    """Resolve a jsonl of mentions against the NPI API into a jsonl of results."""
    from .pipeline import run as run_fn

    result = run_fn(
        input, output, api_url=api_url, require_state=require_state, threshold=threshold
    )
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def eval(
    results: Path = typer.Option(..., "--results", help="Results jsonl (from `run`)."),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", help="GT csv with columns: officer_uid, post_person_nbr."
    ),
) -> None:
    """Score auto-matches against ground truth (link-level precision/recall/F1)."""
    from .evaluation import evaluate

    prf = evaluate(results, ground_truth)
    typer.echo(json.dumps(asdict(prf), indent=2))


if __name__ == "__main__":
    app()
