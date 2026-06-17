"""Thin Typer CLI for the clustering pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command("extract-features")
def extract_features_cmd(
    input: Path = typer.Option(..., "--input", help="Input CSV with OCR text + metadata."),
    output: Path = typer.Option(None, "--output", help="Output CSV (default: derived from input)."),
    force: bool = typer.Option(False, "--force", help="Re-extract even if already extracted."),
    limit: int = typer.Option(None, "--limit", help="Limit to first N documents (testing)."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run regex + LLM feature extraction over an input CSV."""
    _setup_logging(log_level)
    # Delegate by shimming argv — the underlying main() uses argparse.
    from .feature_extraction import extract_features as _ef

    argv_backup = sys.argv
    new_argv = [argv_backup[0], str(input)]
    if output is not None:
        new_argv += ["-o", str(output)]
    if force:
        new_argv += ["--force"]
    if limit is not None:
        new_argv += ["--limit", str(limit)]
    sys.argv = new_argv
    try:
        _ef.main()
    finally:
        sys.argv = argv_backup


@app.command("cluster-handengineered")
def cluster_handengineered_cmd(
    input: Path = typer.Option(..., "--input", help="Input CSV with extracted features."),
    output_dir: Path = typer.Option(
        Path("data/output/ablations"),
        "--output-dir",
        help="Directory to save results.",
    ),
    ablation: str = typer.Option(
        "baseline_v2",
        "--ablation",
        help="Ablation config name (from handengineered YAML).",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the hand-engineered hybrid clustering pipeline."""
    _setup_logging(log_level)
    from .clustering import hybrid_handengineered as he

    asyncio.run(he.main(str(input), ablation, str(output_dir)))


@app.command("cluster-ml")
def cluster_ml_cmd(
    input: Path = typer.Option(..., "--input", help="Input CSV with extracted features."),
    output_dir: Path = typer.Option(
        Path("data/output/ablations_ml"),
        "--output-dir",
        help="Directory to save results.",
    ),
    ablation: str = typer.Option(
        ..., "--ablation", help="ML ablation config name (e.g. rf_both, dt_cascade)."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the ML-based hybrid clustering pipeline."""
    _setup_logging(log_level)
    from .clustering import hybrid_ml as ml

    asyncio.run(ml.main(str(input), ablation, str(output_dir)))


@app.command("eval")
def eval_cmd(
    results: Path = typer.Option(..., "--results", help="Single results CSV to score."),
    ground_truth: str = typer.Option(
        "provisional_case_name",
        "--ground-truth",
        help="Ground-truth column name in the results CSV.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Score a single results CSV against an embedded ground-truth column."""
    _setup_logging(log_level)
    from . import evaluation as ev

    report = ev.score(results, ground_truth_col=ground_truth)
    typer.echo(report.model_dump_json(indent=2))


@app.command("eval-batch")
def eval_batch_cmd(
    results_dir: Path = typer.Option(
        ..., "--results-dir", help="Directory of per-agency ablation results (V2 / handengineered)."
    ),
    ml_results_dir: Path = typer.Option(
        None, "--ml-results-dir", help="Directory of per-agency ML ablation results (optional)."
    ),
    reports_dir: Path = typer.Option(
        ..., "--reports-dir", help="Directory to write evaluation reports."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the multi-agency ablation evaluation (writes per-agency + cross-agency reports)."""
    _setup_logging(log_level)
    os.environ["PRAP_CLUSTERING_ABLATIONS_V2_DIR"] = str(results_dir)
    if ml_results_dir is not None:
        os.environ["PRAP_CLUSTERING_ABLATIONS_ML_DIR"] = str(ml_results_dir)
    os.environ["PRAP_CLUSTERING_REPORTS_DIR"] = str(reports_dir)

    from . import evaluation as ev

    ev.main()


@app.command("embeddings-features")
def embeddings_features_cmd(
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Generate sentence-transformers embeddings for each AGENCIES entry (multi-agency)."""
    _setup_logging(log_level)
    from .embeddings_pipeline.feature_extraction import generate_embeddings as ge

    ge.main()


@app.command("embeddings-cluster")
def embeddings_cluster_cmd(
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run embeddings-based clustering over pre-generated embedding CSVs (multi-agency)."""
    _setup_logging(log_level)
    from .embeddings_pipeline.cluster import cluster_embeddings as ce

    ce.main()


@app.command("metadata-features")
def metadata_features_cmd(
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Extract regex-only features from filenames/filepaths (single-agency runner)."""
    _setup_logging(log_level)
    from .metadata_pipeline.feature_extraction import extract_filename_features as ef

    ef.main()


@app.command("metadata-cluster")
def metadata_cluster_cmd(
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Use the deterministic clustering variant."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the metadata-only clustering (single-agency runner)."""
    _setup_logging(log_level)
    if deterministic:
        from .metadata_pipeline.clustering import cluster_deterministic as cd

        cd.main()
    else:
        from .metadata_pipeline.clustering import cluster as mc

        mc.run_clustering()


if __name__ == "__main__":
    app()
