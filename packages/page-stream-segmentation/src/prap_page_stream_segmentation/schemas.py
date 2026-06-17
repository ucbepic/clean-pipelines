from __future__ import annotations

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int
    text: str


class DocText(BaseModel):
    """One input record per source document (a concatenated PDF's OCR text)."""

    sha1: str
    pages: list[PageText] = Field(default_factory=list)


class PageClassification(BaseModel):
    """Per-page output of the classifier."""

    sha1: str
    page_number: int
    document_type: str
    continuation: bool
    meta: str
    reasoning: str


class TOCEntry(BaseModel):
    """One contiguous document inside a source PDF."""

    sha1: str
    start_page: int
    headline: str
    date: str
    people: dict[str, str] = Field(default_factory=dict)
    agencies: list[str] = Field(default_factory=list)
    page_classifications: list[PageClassification] = Field(default_factory=list)


class DocumentTOC(BaseModel):
    """Top-level output: one source PDF -> ordered list of TOC entries."""

    sha1: str
    entries: list[TOCEntry] = Field(default_factory=list)
    error: str | None = None


class RunResult(BaseModel):
    n_documents: int
    output_path: str
