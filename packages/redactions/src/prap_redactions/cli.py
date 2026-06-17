"""Thin Typer CLI for prap-redactions.

DISABLED: this pipeline depends on Azure Content Safety (graphic-imagery
classifier) and Azure Blob Storage (PDF download), neither of which is
wired up in the current release. Source is preserved in-tree pending an
open-source classifier replacement; the CLI exits non-zero until then.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input: Path = typer.Option(None, "--input"),
    output: Path = typer.Option(None, "--output"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """DISABLED — depends on Azure Content Safety + Blob Storage."""
    typer.echo(
        "prap-redactions is disabled in this release.\n"
        "It depends on Azure Content Safety (graphic-imagery classifier) "
        "and Azure Blob Storage (PDF source); an open-source replacement "
        "for the classifier is planned. See packages/redactions/README.md."
    )
    raise typer.Exit(code=2)
