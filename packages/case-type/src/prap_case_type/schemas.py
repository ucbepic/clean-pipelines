from pydantic import BaseModel


class CaseRecord(BaseModel):
    """One input record per police case."""

    provisional_case_name: str
    summaries: list[str] = []
    ocr_texts: list[str] = []


class CaseClassifications(BaseModel):
    use_of_force: str | None = None
    misconduct: str | None = None
    officer_involved_shooting: str | None = None


class CaseTypeResult(BaseModel):
    """One output record per police case."""

    provisional_case_name: str
    classification: CaseClassifications | None = None
    note: str = ""


class RunResult(BaseModel):
    n_cases: int
    output_path: str
