from pydantic import BaseModel, Field


class ExtractedParts(BaseModel):
    first_name: str = ""
    last_name: str = ""
    middle_name: str = ""
    suffix: str = ""


class NameExtractionResult(BaseModel):
    """Stage-1 LLM response: extracted name parts + initial validity decision."""

    is_valid_name: bool = Field(..., description="True if the string is a valid human name")
    extracted_parts: ExtractedParts


class NameValidationResult(BaseModel):
    """Stage-2 LLM response: final decision after verifying the extraction."""

    final_decision: bool


class NameRecord(BaseModel):
    """One input record per officer-name string."""

    officer_name: str
    case_id: str | None = None  # passthrough for downstream joins; optional


class NameClassification(BaseModel):
    """One output record per officer-name string."""

    officer_name: str
    cleaned_name: str
    valid_name: int = 0  # 0 / 1, written to CSV
    first_name: str = ""
    last_name: str = ""
    middle_name: str = ""
    suffix: str = ""
    case_id: str | None = None


class RunResult(BaseModel):
    n_records: int
    n_unique_names: int
    n_valid: int
    output_path: str
