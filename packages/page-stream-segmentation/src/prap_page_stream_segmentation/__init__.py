"""PRAP page-stream segmentation pipeline."""

from .pipeline import build_toc, classify_pages, run
from .schemas import DocText, PageClassification, RunResult, TOCEntry

__all__ = [
    "DocText",
    "PageClassification",
    "RunResult",
    "TOCEntry",
    "build_toc",
    "classify_pages",
    "run",
]
