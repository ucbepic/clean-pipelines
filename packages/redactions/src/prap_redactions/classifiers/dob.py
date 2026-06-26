"""DOB classifier."""

import logging
import re

logger = logging.getLogger("prap.redactions.dob")


def classify_pages_with_dob(ocr_text_pages: list[dict]) -> list[int]:
    """
    Identify pages containing dates of birth based on common DOB identifiers.

    This function uses high-precision pattern matching to detect DOB references
    in OCR text. It looks for explicit DOB labels followed by date patterns.

    Args:
        ocr_text_pages: List of page dictionaries with 'text' and 'page_number' keys

    Returns:
        List of page numbers (integers) that contain DOB references
    """
    pages_with_dob = set()

    # DOB label patterns (case-insensitive)
    # These are explicit indicators that a date of birth follows
    # Use word boundaries (\b) to prevent matching inside longer words/identifiers
    dob_labels = [
        # Standard variations with word boundaries
        r"\bdate\s+of\s+birth\s*:?",
        r"\bbirth\s+date\s*:?",
        r"\bbirthdate\s*:?",
        r"\bd\.?\s*o\.?\s*b\.?\s*:?",  # DOB, D.O.B., D O B
        r"\bdob\b\s*:?",  # DOB with word boundary to prevent matching "DOB1", "DOBSON", etc.
        # Form-style labels
        r"\bdate\s+of\s+birth\s*\(?.*?\)?\s*:?",  # "Date of Birth (MM/DD/YYYY):"
        r"\bbirth\s*date\s*\(?.*?\)?\s*:?",
        # Narrative variations
        r"\bborn\s+on\s*:?",
        r"\bborn\b\s*:?",
        r"\bdate\s+born\s*:?",
        # Multi-word with optional punctuation
        r"\bpatient\s+(?:date\s+of\s+)?birth\s*:?",
        r"\bsubject\s+(?:date\s+of\s+)?birth\s*:?",
        r"\bvictim\s+(?:date\s+of\s+)?birth\s*:?",
        r"\bwitness\s+(?:date\s+of\s+)?birth\s*:?",
        # Abbreviated forms common in forms
        r"\bdt\.?\s+of\s+birth\s*:?",
        r"\bbirth\s+dt\.?\s*:?",
    ]

    # Date pattern components
    # Month names (abbreviated and full)
    months_full = (
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    )
    months_abbr = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    months_any = f"(?:{months_full}|{months_abbr})"

    # Day patterns (1-31, with optional leading zero and optional st/nd/rd/th suffix)
    day = r"(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?"

    # Year patterns (1900-2099 to avoid false positives)
    year_full = r"(?:19|20)\d{2}"

    # Common separators
    sep = r"[\s\-/\.,]+"
    opt_sep = r"[\s\-/\.,]*"

    # Date format patterns
    date_patterns = [
        # Numeric formats with 4-digit year
        rf"\d{{1,2}}{sep}\d{{1,2}}{sep}{year_full}",  # MM/DD/YYYY, M-D-YYYY, etc.
        rf"{year_full}{sep}\d{{1,2}}{sep}\d{{1,2}}",  # YYYY-MM-DD (ISO format)
        # Numeric formats with 2-digit year
        rf"\d{{1,2}}{sep}\d{{1,2}}{sep}\d{{2}}",  # MM/DD/YY, M-D-YY, etc.
        # Month name formats (full and abbreviated)
        rf"{months_any}{sep}{day}{opt_sep},{opt_sep}{year_full}",  # January 15, 1985
        rf"{day}{sep}{months_any}{opt_sep},{opt_sep}{year_full}",  # 15 January, 1985
        rf"{months_any}{sep}{day}{sep}{year_full}",  # January 15 1985
        rf"{day}{sep}{months_any}{sep}{year_full}",  # 15 January 1985
        # Month/Year only (less common but sometimes used)
        rf"{months_any}{opt_sep},{opt_sep}{year_full}",  # January, 1985
        rf"{months_any}{sep}{year_full}",  # January 1985
    ]

    # Compile patterns for efficiency
    compiled_label_patterns = [re.compile(label, re.IGNORECASE) for label in dob_labels]

    compiled_date_patterns = [re.compile(date, re.IGNORECASE) for date in date_patterns]

    # Process each page
    for page in ocr_text_pages:
        if "text" not in page or not page["text"]:
            continue

        page_text = page["text"]
        page_number = page.get("page_number", 0)

        # Check for DOB label patterns
        for label_pattern in compiled_label_patterns:
            matches = label_pattern.finditer(page_text)

            for match in matches:
                # Get text after the label (next 25 characters - reduced from 50)
                # This is where we expect to find the actual date
                # DOB values typically appear immediately after the label
                start_pos = match.end()
                context_window = page_text[start_pos : start_pos + 25]

                # Skip if the context window is too sparse (likely incomplete/truncated label)
                # Require at least some non-whitespace content after the label
                stripped_context = context_window.strip()
                if len(stripped_context) < 3:  # Need at least 3 chars for a valid date
                    continue

                # Check if a date pattern follows the label
                date_found = False
                for date_pattern in compiled_date_patterns:
                    date_match = date_pattern.search(context_window)
                    if date_match:
                        # Additional validation: date should start close to the beginning of context
                        # This prevents matching dates that just happen to be nearby
                        date_start_pos = date_match.start()

                        # Allow up to 5 characters of whitespace/punctuation before the date
                        # This handles formats like "DOB: 01/15/1985" or "DOB\n01/15/1985"
                        if date_start_pos <= 5:
                            # Additional check: the text BEFORE the date shouldn't have multiple
                            # newlines
                            # This ensures we're matching a date on the same/next line, not far away
                            text_before_date = context_window[:date_start_pos]
                            if text_before_date.count("\n") > 1:
                                continue  # Date is too far from label

                            pages_with_dob.add(page_number)
                            logger.debug(
                                f"Found DOB on page {page_number}: "
                                f"Label='{match.group()}', Date='{date_match.group()}', "
                                f"Context='{context_window[:30]}...'"
                            )
                            date_found = True
                            break

                # If we found a match on this page, no need to check more patterns
                if date_found:
                    break

            # If we found a match on this page, no need to check more label patterns
            if page_number in pages_with_dob:
                break

    # Convert set to sorted list
    result = sorted(list(pages_with_dob))

    if result:
        logger.info(f"Found DOB references on {len(result)} pages: {result}")
    else:
        logger.info("No DOB references found")

    return result


def classify_file_for_dob(ocr_text_pages: list[dict], sha1: str) -> dict:
    """
    Classify a file for DOB presence and return structured results.

    This is a convenience wrapper that provides results in a format
    consistent with other classifiers in the pipeline.

    Args:
        ocr_text_pages: List of page dictionaries with OCR text
        sha1: SHA1 hash of the file

    Returns:
        Dict with:
            - sha1: The file SHA1
            - pages_with_dob: List of page numbers containing DOB
            - success: Boolean indicating if processing was successful
            - error: Error message if processing failed (None otherwise)
    """
    result = {
        "sha1": sha1,
        "pages_with_dob": [],
        "success": False,
        "error": None,
    }

    try:
        pages_with_dob = classify_pages_with_dob(ocr_text_pages)
        result["pages_with_dob"] = pages_with_dob
        result["success"] = True

        logger.info(
            f"Successfully classified {sha1}: {len(pages_with_dob)} pages with DOB references"
        )

    except Exception as e:
        logger.error(f"Error classifying {sha1} for DOB: {e}", exc_info=True)
        result["error"] = str(e)

    return result
