"""Pydantic schemas for involved-agency extraction."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Agency(BaseModel):
    """Individual agency with evidence and role description."""

    agency_name: str = Field(..., description="Full official name of the agency")
    evidence: List[str] = Field(
        ...,
        description="Direct quotes or references from text supporting this agency's involvement",
    )
    role_description: str = Field(
        ..., description="Clear description of the agency's role (what they did)"
    )
    has_dual_role: bool = Field(
        default=False,
        description="True if agency serves both investigating and responding roles",
    )
    dual_role_note: Optional[str] = Field(
        default=None, description="Explanation of dual role if applicable"
    )


class AgencyExtraction(BaseModel):
    """Result of extracting investigating and responding agencies from document summaries."""

    incident_type: str = Field(
        ..., description="Use of Force / Misconduct / Both / Unclear"
    )

    responding_agencies: List[Agency] = Field(
        default=[],
        description="Agencies whose officers were involved in or present at the incident",
    )

    investigating_agencies: List[Agency] = Field(
        default=[],
        description="Agencies that conducted investigations into the incident",
    )

    confidence_level: str = Field(
        ..., description="HIGH / MEDIUM / LOW - overall confidence in extraction"
    )

    extraction_reasoning: str = Field(
        ...,
        description="Explanation of how agencies were identified and categorized",
    )

    ambiguous_agencies: List[str] = Field(
        default=[],
        description="Agency names mentioned but role could not be determined",
    )

    additional_notes: Optional[str] = Field(
        default=None,
        description="Any important observations or caveats about the extraction",
    )


class AgencyVerificationResult(BaseModel):
    """Result of verifying an agency extraction against source documents."""

    verification_status: str = Field(
        ...,
        description="CONFIRMED / CORRECTED / REJECTED - status of the verification",
    )

    incident_type: str = Field(
        ..., description="Use of Force / Misconduct / Both / Unclear"
    )

    responding_agencies: List[Agency] = Field(
        default=[],
        description="Verified agencies whose officers were involved in or present at the incident",
    )

    investigating_agencies: List[Agency] = Field(
        default=[],
        description="Verified agencies that conducted investigations",
    )

    confidence_level: str = Field(
        ..., description="HIGH / MEDIUM / LOW - confidence after verification"
    )

    verification_reasoning: str = Field(
        ...,
        description="Detailed explanation of verification including what was confirmed, corrected, or rejected",
    )

    changes_made: List[str] = Field(
        default=[],
        description="List of specific changes made during verification (if any)",
    )

    agencies_added: List[str] = Field(
        default=[],
        description="Agency names added during verification that were missing in initial extraction",
    )

    agencies_removed: List[str] = Field(
        default=[],
        description="Agency names removed during verification due to insufficient evidence",
    )


class SingleAgencyVerification(BaseModel):
    verification_status: str = Field(
        ..., description="CONFIRMED / CORRECTED / REJECTED"
    )
    corrected_agency_type: Optional[str] = Field(
        ...,
        description="INVESTIGATING / RESPONDING / BOTH / null (your independent assessment)",
    )
    verified_agency_name: str = Field(..., description="Full official agency name")
    verified_evidence: List[str] = Field(..., description="Verified evidence quotes")
    verified_role_description: str = Field(
        ..., description="Verified role description"
    )
    has_dual_role: bool = Field(..., description="True if dual role (type = BOTH)")
    dual_role_note: Optional[str] = Field(
        default=None, description="Dual role explanation"
    )
    confidence_level: str = Field(..., description="HIGH / MEDIUM / LOW")
    verification_reasoning: str = Field(
        ...,
        description="Detailed explanation of verification including type determination",
    )
    changes_made: List[str] = Field(
        default=[], description="Changes made if corrected"
    )
    recommendation: str = Field(..., description="INCLUDE or EXCLUDE")
    type_mismatch_explanation: Optional[str] = Field(
        default=None, description="Explanation if type was corrected"
    )


# ============================================================================
# CITATION SCHEMAS (from citations.py)
# ============================================================================


class PrimaryCitationAnalysis(BaseModel):
    """Result of primary analysis for agency citation on a page."""

    has_citation: bool = Field(
        ..., description="True if page contains agency citation with action verbs"
    )
    reasoning: str = Field(..., description="Explanation of whether citation is present")
    quote: str = Field(..., description="Exact quote from page supporting the decision")
    confidence: str = Field(..., description="HIGH / MEDIUM / LOW")


class ValidatorCitationResult(BaseModel):
    """Result of validator verification for agency citation."""

    final_decision: bool = Field(
        ..., description="True if citation is valid and high-quality"
    )
    validator_reasoning: str = Field(
        ..., description="Detailed explanation of validation decision"
    )
    verified_quote: str = Field(
        ..., description="Exact verified quote from page or explanation if invalid"
    )
    evidence_strength: str = Field(..., description="EXPLICIT / CIRCUMSTANTIAL / WEAK")


# ============================================================================
# EVALUATION SCHEMAS (from evaluation.py)
# ============================================================================


class AgencyNameMatch(BaseModel):
    """Result of LLM-based agency name comparison."""

    is_match: bool = Field(
        ..., description="True if agency names refer to the same entity"
    )
    reasoning: str = Field(..., description="Explanation of matching decision")
    confidence: str = Field(..., description="HIGH / MEDIUM / LOW")


# ============================================================================
# PIPELINE RESULT
# ============================================================================


class RunResult(BaseModel):
    n_cases: int
    n_agencies_extracted: int
    output_path: str
