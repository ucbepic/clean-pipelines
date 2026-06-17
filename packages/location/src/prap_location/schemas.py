from pydantic import BaseModel


class CaseRecord(BaseModel):
    """One input record per police case.

    `summaries_or_ocr_texts` is either the per-document first_look summaries
    (regular case) or per-document OCR text sections (special case).
    """

    provisional_case_name: str
    summaries_or_ocr_texts: list[str] = []
    is_special_case: bool = False


class ValidationResult(BaseModel):
    validation_decision: str = ""
    final_decision: str = ""
    validator_reasoning: str = ""
    verified_quote: str = ""
    confidence: str = ""
    city_completeness: str = ""
    specificity_assessment: str = ""
    additional_details: str = ""


class LocationResult(BaseModel):
    """One output record per police case."""

    provisional_case_name: str
    extracted_location: str | None = None
    initial_analysis: str = ""
    validation_result: ValidationResult | None = None
    pipeline_stage_completed: str = ""
    is_special_case: bool = False
    note: str = ""


class RunResult(BaseModel):
    n_cases: int
    output_path: str
