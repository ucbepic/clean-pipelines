"""
Generic extraction pipeline for feature extraction.

Simplified from incident_date_extraction/extract/src/extract.py
Accepts document summary and feature-specific prompts, returns extracted values.
"""

import logging
from typing import Any

from jinja2 import Template

from prap_clustering._llm import get_llm

logger = logging.getLogger(__name__)


def extract_feature(summary: str, prompts: dict[str, str]) -> str:
    """
    Extract specific feature value from document summary.

    Args:
        summary: Bulletpoint summary from summarize_pipeline
        prompts: Dictionary of prompt templates from feature-specific prompt module
                 Must have 'extraction' key with: extract, verification templates

    Returns:
        Extracted feature as string (format depends on feature type)
    """
    logger.info("Starting feature extraction from summary")
    logger.debug(f"Summary length: {len(summary)} chars")

    # Step 1: Initial extraction
    template = Template(prompts["extraction"]["extract"])
    prompt = template.render(source_text=summary)

    logger.info("Running initial extraction")
    initial_extraction = get_llm().complete(prompt).text
    logger.debug(f"Initial extraction: {initial_extraction[:200]}...")

    # Step 2: Verification
    template = Template(prompts["extraction"]["verification"])
    prompt = template.render(
        initial_dates=initial_extraction,  # Generic name for now, works for all features
        source_text=summary,
    )

    logger.info("Running extraction verification")
    verified_extraction = get_llm().complete(prompt).text
    logger.debug(f"Verified extraction: {verified_extraction[:200]}...")

    logger.info("Feature extraction complete")
    return verified_extraction


def convert_to_standard_format(extraction_result: str, prompts: dict[str, str]) -> Any | None:
    """
    Convert extracted feature to standard format.

    Args:
        extraction_result: Raw extraction result from extract_feature()
        prompts: Dictionary of prompt templates
                 Must have 'extraction' key with 'format_conversion' template

    Returns:
        Standardized value(s) or None if no valid value found
        Type depends on feature (dates: list of strings, names: list, etc.)
    """
    # Some features may not need format conversion
    if prompts["extraction"].get("format_conversion") is None:
        logger.info("No format conversion template provided, returning raw extraction")
        return extraction_result

    logger.info("Converting extraction to standard format")

    template = Template(prompts["extraction"]["format_conversion"])
    prompt = template.render(source_text=extraction_result)

    result = get_llm().complete(prompt).text.strip()
    logger.debug(f"Format conversion result: {result}")

    # Parse result based on format
    # For dates: returns comma-separated ISO dates or "None"
    # For names: might return JSON list or comma-separated
    # For IDs: might return standardized ID format

    if result.lower() == "none":
        logger.info("No valid value found after format conversion")
        return None

    logger.info(f"Format conversion complete: {result}")
    return result


def extract_and_convert(summary: str, prompts: dict[str, str]) -> Any | None:
    """
    Main entry point: Extract feature from summary and convert to standard format.

    This is a two-step process:
    1. Extract feature value(s) from summary using LLM
    2. Convert to standard format if needed

    Args:
        summary: Bulletpoint summary from summarize_pipeline
        prompts: Dictionary of prompt templates from feature-specific prompt module
                 Must have 'extraction' key with: extract, verification, format_conversion

    Returns:
        Extracted and formatted feature value(s), or None if not found

    Example:
        >>> from prompts.dates import PROMPTS as DATE_PROMPTS
        >>> dates = extract_and_convert(summary, DATE_PROMPTS)
        >>> # Returns: ['2024-01-26', '2024-01-28'] or None
    """
    logger.info("=" * 60)
    logger.info("EXTRACTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Extract feature
    extraction_result = extract_feature(summary, prompts)

    if not extraction_result or not extraction_result.strip():
        logger.warning("No extraction result returned")
        return None

    # Step 2: Convert to standard format (if template provided)
    formatted_result = convert_to_standard_format(extraction_result, prompts)

    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Result: {formatted_result}")
    logger.info("=" * 60)

    return formatted_result


# Convenience function for common case: summarize + extract
def summarize_and_extract(ocr_text: str, prompts: dict[str, str]) -> Any | None:
    """
    Convenience function: Summarize document and extract feature in one call.

    Args:
        ocr_text: Full OCR text of document
        prompts: Dictionary of prompt templates from feature-specific prompt module
                 Must have both 'summarization' and 'extraction' keys

    Returns:
        Extracted and formatted feature value(s), or None if not found

    Example:
        >>> from prompts.dates import PROMPTS as DATE_PROMPTS
        >>> dates = summarize_and_extract(ocr_text, DATE_PROMPTS)
    """
    from summarize_pipeline import summarize_document

    logger.info("Starting combined summarize + extract pipeline")

    # Step 1: Summarize
    summary = summarize_document(ocr_text, prompts)

    if not summary:
        logger.warning("No summary generated")
        return None

    # Step 2: Extract
    result = extract_and_convert(summary, prompts)

    return result


# For backwards compatibility and testing
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test with dates prompts
    from prompts.dates import PROMPTS as DATE_PROMPTS

    sample_summary = """
    January 26, 2014, 6:56 a.m. - Incident Date: Officer Butera responded to call
    January 26, 2014, 7:37 a.m. - Incident Date: Officer discharged firearm
    January 28, 2014 - Investigation Milestone: Investigation initiated
    """

    dates = extract_and_convert(sample_summary, DATE_PROMPTS)
    print("Extracted dates:", dates)
