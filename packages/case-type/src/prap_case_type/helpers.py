"""Task-local utilities for case-type classification."""

import logging

logger = logging.getLogger("prap.case_type.helpers")


def natural_language_to_tristate_enum(
    natural_language: str,
    true_value: str = "true",
    false_value: str = "false",
    unknown_value: str = "unclear",
) -> str | None:
    """Map LLM tristate response to 'True' / 'False' / 'Unclear' (None on parse failure)."""
    s = natural_language.strip().lower()
    if s == true_value:
        return "True"
    if s == false_value:
        return "False"
    if s == unknown_value:
        return "Unclear"
    logger.error(f"Unexpected response to tristate classification question: {natural_language}")
    return None


def convert_groundtruth_value(value: str) -> str:
    """Map ground-truth 'TRUE' / 'FALSE' / anything-else to 'True' / 'False' / 'Unclear'."""
    if value == "TRUE":
        return "True"
    if value == "FALSE":
        return "False"
    return "Unclear"
