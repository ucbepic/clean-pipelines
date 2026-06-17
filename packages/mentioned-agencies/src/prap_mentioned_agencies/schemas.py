from __future__ import annotations

from pydantic import BaseModel, Field


class CasePage(BaseModel):
    """One OCR page within a case file."""

    file_name: str
    page_number: int
    text: str


class CaseBundle(BaseModel):
    """One input record per police case.

    Mirrors the `agency_case_file_bundle-*.json` format used by cpost: a case
    has many files, each file has many OCR pages. The `pages` field is the
    flattened-and-filtered view consumed by `pipeline.run`. Empty-text pages
    should be filtered out by the caller (see `pipeline.flatten_case_pages`).
    """

    provisional_case_name: str
    pages: list[CasePage] = Field(default_factory=list)


class MentionedAgenciesResult(BaseModel):
    """One output record per case."""

    provisional_case_name: str
    mentioned_agencies: list[str] = Field(default_factory=list)
    n_pages_processed: int = 0
    n_raw_extractions: int = 0
    n_after_validation: int = 0
    validation_confidence: str | None = None
    error: str | None = None


class RunResult(BaseModel):
    """Summary returned by `pipeline.run(...)`."""

    n_cases: int
    output_path: str
