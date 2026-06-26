"""Pydantic models for LLM structured outputs + the pipeline RunResult."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PersonExtraction(BaseModel):
    """Model for a single extracted person"""

    name: str = Field(description="Full proper name (first and last)")
    person_type: str = Field(
        description="Type: victim|witness|domestic_violence_victim|sexual_assault_victim|child"
    )
    reasoning: str = Field(description="Specific textual evidence for classification")
    confidence: str = Field(description="Confidence level: high|medium|low")
    context: str = Field(description="Direct quote from document (10-20 words)")


class NameExtractionResponse(BaseModel):
    """Response model for name extraction"""

    extracted_persons: list[PersonExtraction] = Field(
        default_factory=list, description="List of extracted persons"
    )


class PersonVerification(BaseModel):
    """Model for verifying if a person should be included"""

    name: str = Field(description="Full name of the person")
    should_be_included: bool = Field(description="Whether person should be included")
    reasoning: str = Field(description="Explanation for inclusion/exclusion decision")
    evidence_from_text: str | None = Field(
        default=None, description="Direct quote supporting the decision"
    )
    exclusion_category: str | None = Field(
        default=None,
        description=(
            "Category if excluded: law_enforcement|defendant|legal_professional|"
            "court_staff|government_official|other"
        ),
    )


class VerificationResponse(BaseModel):
    """Response model for person verification"""

    verified_persons: list[PersonVerification] = Field(description="List of person verifications")


class FormattedPerson(BaseModel):
    """Model for a formatted person in final output"""

    name: str = Field(description="Full name of person")
    person_type: list[str] = Field(description="Array of all applicable types")
    confidence: str = Field(description="Confidence level: high|medium|low")
    reasoning: str = Field(description="Concise explanation of inclusion")


class FormattedOutput(BaseModel):
    """Final formatted output model"""

    extracted_persons: list[FormattedPerson] = Field(
        default_factory=list, description="List of verified persons to include"
    )


class AddressFinding(BaseModel):
    """Model for address finding result"""

    name: str = Field(description="Person's name exactly as provided")
    address_found: bool = Field(description="Whether address was found")
    address: str | None = Field(default=None, description="Complete address if found")
    confidence: str | None = Field(default=None, description="Confidence level if found")
    reasoning: str = Field(description="Explanation of findings")
    context: str | None = Field(
        default=None, description="Direct quote containing address if found"
    )


class AddressResponse(BaseModel):
    """Response model for address findings"""

    addresses: list[AddressFinding] = Field(description="Address findings for each person")


class RunResult(BaseModel):
    n_cases_attempted: int
    n_cases_processed: int
    n_cases_errored: int
    n_files_with_redactions: int
    output_path: str
