from pydantic import BaseModel, Field


class CaseRecord(BaseModel):
    """One input record per police case."""

    provisional_case_name: str
    summaries: list[str] = Field(
        default_factory=list,
        description=(
            "Per-document summaries for this case. If `ocr_pages` is provided "
            "those are used instead and `summaries` is ignored."
        ),
    )
    ocr_pages: list[str] | None = Field(
        default=None,
        description="Per-document OCR text blocks. Used for the special-cases code path.",
    )


class IncidentDateResult(BaseModel):
    """One output record per police case."""

    provisional_case_name: str
    extracted_date: list[str] | None = None
    nl_date: str = ""


class RunResult(BaseModel):
    """Summary returned by `pipeline.run(...)`."""

    n_cases: int
    output_path: str
