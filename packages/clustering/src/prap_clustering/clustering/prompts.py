"""
LLM prompts for document clustering comparison.

This module contains the Tier 3 summary-comparison prompt (loaded from
``prap_clustering/prompts/summary_comparison.txt``) and the validation
helper that parses LLM responses.
"""

from pathlib import Path

from prap_core.prompts import PromptDir
from pydantic import BaseModel, Field

_PROMPT_DIR = PromptDir(Path(__file__).resolve().parent.parent / "prompts")

SUMMARY_COMPARISON_PROMPT = _PROMPT_DIR.load("summary_comparison")


class SummaryComparisonResult(BaseModel):
    """Response model for summary comparison."""

    similarity: float = Field(
        description="Similarity score: 1.0 for same incident, 0.5 for uncertain, 0.0 for different"
    )
    reasoning: str = Field(description="Brief explanation of the score")


def validate_summary_comparison_response(response: str) -> float | None:
    """
    Validate and extract similarity score from LLM response.

    Args:
        response: Raw LLM response (should be JSON)

    Returns:
        Similarity score (0.0, 0.5, or 1.0) or None if invalid
    """
    try:
        cleaned = response.strip()
        result = SummaryComparisonResult.model_validate_json(cleaned)
        if result.similarity in [0.0, 0.5, 1.0]:
            return result.similarity
        return None
    except Exception:
        try:
            if "1.0" in response or '"similarity": 1' in response:
                return 1.0
            elif "0.5" in response or ".5" in response or '"similarity": 0.5' in response:
                return 0.5
            elif "0.0" in response or '"similarity": 0' in response:
                return 0.0
        except Exception:
            pass
        return None
