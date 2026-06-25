"""CLI wiring smoke test: `prepare` end-to-end via Typer's CliRunner (offline)."""

from __future__ import annotations

from pathlib import Path

from prap_core.io import read_jsonl
from prap_post_entity_resolution.resolve.cli import app
from typer.testing import CliRunner

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "resolve_sample.csv"


def test_cli_prepare(tmp_path):
    out = tmp_path / "mentions.jsonl"
    result = CliRunner().invoke(
        app,
        ["prepare", "--input", str(_FIXTURE), "--output", str(out), "--default-state", "CA"],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 12 mention records" in result.output
    assert len(list(read_jsonl(out))) == 12
