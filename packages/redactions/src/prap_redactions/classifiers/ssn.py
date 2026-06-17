"""SSN classifier."""

import logging
import re
from typing import Dict, List

logger = logging.getLogger("prap.redactions.ssn")


def classify_pages_with_ssn(ocr_text_pages: List[Dict]) -> List[int]:
    """
    Identify pages containing social security numbers (SSNs)

    This function uses high-precision pattern matching to detect SSN references
    in OCR text.

    Args:
        ocr_text_pages: List of page dictionaries with 'text' and 'page_number' keys

    Returns:
        List of page numbers (integers) that contain SSN references
    """
    pages_with_ssn = set()
    ssn_pattern = r'[0-9]{3}-[0-9]{2}-[0-9]{4}'
    ssn_regex = re.compile(ssn_pattern)
    for page in ocr_text_pages:
        if 'text' not in page or not page['text']:
            continue
        # sometimes the ssn pattern matches the evidence number from a photograph,
        # usually in those cases the evidence number is the only extracted text,
        if len(page['text']) < 15:
            continue
        page_text = page['text']
        page_number = page.get('page_number', 0)
        if ssn_regex.search(page_text):
            pages_with_ssn.add(page_number)
            logger.debug(f"Found SSN on page {page_number}")

    # Convert set to sorted list
    result = sorted(list(pages_with_ssn))

    if result:
        logger.info(f"Found SSN references on {len(result)} pages: {result}")
    else:
        logger.info("No SSN references found")

    return result


def classify_file_for_ssn(ocr_text_pages: List[Dict], sha1: str) -> Dict:
    """
    Classify a file for SSN presence and return structured results.

    This is a convenience wrapper that provides results in a format
    consistent with other classifiers in the pipeline.

    Args:
        ocr_text_pages: List of page dictionaries with OCR text
        sha1: SHA1 hash of the file

    Returns:
        Dict with:
            - sha1: The file SHA1
            - pages_with_ssn: List of page numbers containing SSN
            - success: Boolean indicating if processing was successful
            - error: Error message if processing failed (None otherwise)
    """
    result = {
        "sha1": sha1,
        "pages_with_ssn": [],
        "success": False,
        "error": None,
    }

    try:
        pages_with_ssn = classify_pages_with_ssn(ocr_text_pages)
        result["pages_with_ssn"] = pages_with_ssn
        result["success"] = True

        logger.info(
            f"Successfully classified {sha1}: "
            f"{len(pages_with_ssn)} pages with SSN references"
        )

    except Exception as e:
        logger.error(f"Error classifying {sha1} for SSN: {e}", exc_info=True)
        result["error"] = str(e)

    return result
