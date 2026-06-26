"""Thin Typer CLI for prap-location."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from .pipeline import SPECIAL_CASE_PREFIXES
from .pipeline import run as pipeline_run

app = typer.Typer(add_completion=False, no_args_is_help=True)

logger = logging.getLogger("prap.location")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def run(
    input: Path = typer.Option(..., "--input", help="Path to input jsonl of CaseRecords."),
    output: Path = typer.Option(..., "--output", help="Path to output jsonl."),
    n_threads: int = typer.Option(8, "--n-threads"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Extract incident cities from a jsonl of CaseRecords."""
    _setup_logging(log_level)
    result = pipeline_run(input, output, n_threads=n_threads)
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command("targeted-sample")
def targeted_sample(
    run_jsonl: Path = typer.Option(..., "--run", help="Existing prap-location run output (jsonl)."),
    documents: Path = typer.Option(
        ..., "--documents", help="Per-file table (parquet/csv) with gdrive_url + ocr_text."
    ),
    output: Path = typer.Option(..., "--output", help="Output CSV for manual validation."),
    filter_: str = typer.Option(
        "sd_cdp",
        "--filter",
        help="Targeted-filter name (e.g. 'sd_cdp') or path to a .txt of substrings.",
    ),
    n_threads: int = typer.Option(8, "--n-threads"),
    no_revalidate: bool = typer.Option(
        False, "--no-revalidate", help="Skip the aggregate citation re-validation prompt pair."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Build a targeted-sample CSV (citations + per-case quotes) for manual validation."""
    _setup_logging(log_level)
    from .citation import run_targeted_sample

    n = run_targeted_sample(
        run_jsonl=run_jsonl,
        documents_table=documents,
        output_csv=output,
        filter_name_or_path=filter_,
        n_threads=n_threads,
        revalidate=not no_revalidate,
    )
    typer.echo(f"Wrote {n} rows to {output}")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _normalize_case_names(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "provisional_case_name"] = (
        df.provisional_case_name.str.lower()
        .str.strip()
        .fillna("")
        .str.replace(r"\n", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    return df


PROBLEMATIC_CASES = [
    "1717696024591-kzk, 1717695709058-gau",
    "1717696024591-kzk",
    "1717695709058-gau",
]


@app.command()
def prepare(
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", help="Per-case GT (parquet or csv) with location_city column."
    ),
    documents: Path = typer.Option(
        ..., "--documents", help="Per-file table (parquet or csv) with first_look_summary."
    ),
    output: Path = typer.Option(..., "--output", help="Path to output jsonl."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Convert the per-case GT + per-file documents tables into the jsonl input."""
    _setup_logging(log_level)

    gt = _read(ground_truth)
    gt = _normalize_case_names(gt)
    gt = gt[gt.provisional_case_name.fillna("") != ""]

    doc = _read(documents)
    doc = _normalize_case_names(doc)
    doc = doc[~doc.provisional_case_name.isin(PROBLEMATIC_CASES)]
    doc = doc[doc.provisional_case_name.fillna("") != ""]

    valid_cases = set(gt.provisional_case_name.unique().tolist())
    doc = doc[doc.provisional_case_name.isin(valid_cases)]

    logger.info(f"Groundtruth shape {gt.shape}")
    logger.info(f"Doc shape {doc.shape}")

    records: list[dict] = []
    for case_name, group in doc.groupby("provisional_case_name"):
        is_special_case = any(str(case_name).startswith(prefix) for prefix in SPECIAL_CASE_PREFIXES)

        if (
            is_special_case
            and "page_start" in group.columns
            and "page_end" in group.columns
            and "ocr_text" in group.columns
        ):
            ocr_contents: list[str] = []
            for _, row in group.iterrows():
                if (
                    pd.notna(row["page_start"])
                    and pd.notna(row["page_end"])
                    and pd.notna(row["ocr_text"])
                ):
                    ocr_data = row["ocr_text"]
                    start_page = int(row["page_start"])
                    end_page = int(row["page_end"])
                    if isinstance(ocr_data, str):
                        try:
                            ocr_data = json.loads(ocr_data)
                        except json.JSONDecodeError:
                            ocr_contents.append(row["ocr_text"])
                            continue
                    messages = []
                    if isinstance(ocr_data, dict) and "messages" in ocr_data:
                        messages = ocr_data["messages"]
                    elif isinstance(ocr_data, list):
                        messages = ocr_data
                    extracted_pages = [
                        m["page_content"]
                        for m in messages
                        if isinstance(m, dict)
                        and "page_number" in m
                        and "page_content" in m
                        and start_page <= m["page_number"] <= end_page
                    ]
                    if extracted_pages:
                        ocr_contents.append("\n\n===== PAGE BREAK =====\n\n".join(extracted_pages))
                    else:
                        all_pages = [
                            m["page_content"]
                            for m in messages
                            if isinstance(m, dict) and "page_content" in m
                        ]
                        if all_pages:
                            ocr_contents.append("\n\n===== PAGE BREAK =====\n\n".join(all_pages))
            if ocr_contents:
                records.append(
                    {
                        "provisional_case_name": case_name,
                        "summaries_or_ocr_texts": ocr_contents,
                        "is_special_case": True,
                    }
                )
                continue

        summaries: list[str] = []
        if "first_look_summary" in group.columns:
            for s in group["first_look_summary"].tolist():
                if s is None or pd.isna(s):
                    summaries.append("No summary")
                elif not isinstance(s, str):
                    summaries.append(str(s))
                else:
                    summaries.append(s)
        if not summaries:
            summaries = ["No summary"]
        records.append(
            {
                "provisional_case_name": case_name,
                "summaries_or_ocr_texts": summaries,
                "is_special_case": False,
            }
        )

    from prap_core.io import write_jsonl

    n = write_jsonl(output, records)
    typer.echo(f"Wrote {n} case records to {output}")
