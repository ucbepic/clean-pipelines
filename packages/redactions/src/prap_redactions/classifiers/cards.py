"""Credit/debit card classifier."""

import logging
import re

logger = logging.getLogger("prap.redactions.cards")


CARD_TYPES = {
    "Visa": {
        "prefix_pattern": r"^4",
        "lengths": [16],
    },
    "Mastercard": {
        "prefix_pattern": r"^(5[1-5]|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)",
        "lengths": [16],
    },
    "American Express": {
        "prefix_pattern": r"^(34|37)",
        "lengths": [15],
    },
    "Discover": {
        "prefix_pattern": r"^(6011|65|64[4-9])",
        "lengths": [16],
    },
    "Diners Club": {
        "prefix_pattern": r"^(36|38)",
        "lengths": [14],
    },
    "JCB": {
        "prefix_pattern": r"^(352[8-9]|35[3-8][0-9])",
        "lengths": [16],
    },
}


def luhn_validate(card_number: str) -> bool:
    """
    Validate a card number using the Luhn algorithm (mod-10 checksum).

    The Luhn algorithm:
    1. Starting from the right, double every second digit
    2. If doubling results in a 2-digit number, sum those digits
    3. Add all digits together
    4. If sum % 10 == 0, the number is valid

    Args:
        card_number: String of digits (no separators)

    Returns:
        True if valid, False otherwise
    """
    if not card_number.isdigit():
        return False

    # Reverse the number for easier processing (work from right to left)
    digits = [int(d) for d in card_number][::-1]

    # Double every second digit (index 1, 3, 5, ...)
    for i in range(1, len(digits), 2):
        digits[i] *= 2
        # If result is 2 digits, sum them (e.g., 16 -> 1+6 = 7)
        if digits[i] > 9:
            digits[i] = digits[i] // 10 + digits[i] % 10

    # Sum all digits
    total = sum(digits)

    # Valid if divisible by 10
    return total % 10 == 0


def get_card_type(card_number: str) -> str | None:
    """
    Determine the card type based on prefix and length.

    Args:
        card_number: String of digits (no separators)

    Returns:
        Card type name if valid, None otherwise
    """
    if not card_number.isdigit():
        return None

    card_length = len(card_number)

    for card_type, rules in CARD_TYPES.items():
        # Check if length matches
        if card_length not in rules["lengths"]:
            continue

        # Check if prefix matches
        if re.match(rules["prefix_pattern"], card_number):
            return card_type

    return None


def is_valid_card_number(card_number: str) -> tuple[bool, str | None]:
    """
    Validate a card number using prefix check and Luhn algorithm.

    Args:
        card_number: String of digits (no separators)

    Returns:
        Tuple of (is_valid, card_type)
    """
    # Check card type prefix and length
    card_type = get_card_type(card_number)
    if not card_type:
        return False, None

    # Validate with Luhn algorithm
    if not luhn_validate(card_number):
        return False, None

    return True, card_type


def find_card_numbers_in_text(text: str) -> list[tuple[str, str]]:
    """
    Find valid credit card numbers in text using high-precision pattern matching.

    Only matches numbers with CONSISTENT separators (all spaces, all dashes,
    all dots, or no separators). No mixed separators allowed.

    Args:
        text: Text to search

    Returns:
        List of tuples: (card_number, card_type)
    """
    found_cards = []

    # Define patterns for different formats
    # Each pattern enforces CONSISTENT separators only

    patterns = [
        # 16-digit cards (Visa, Mastercard, Discover, JCB)
        # Format: XXXX XXXX XXXX XXXX (spaces only)
        r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b",
        # Format: XXXX-XXXX-XXXX-XXXX (dashes only)
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
        # Format: XXXX.XXXX.XXXX.XXXX (dots only)
        r"\b\d{4}\.\d{4}\.\d{4}\.\d{4}\b",
        # Format: XXXXXXXXXXXXXXXX (no separators)
        r"\b\d{16}\b",
        # 15-digit cards (American Express)
        # Format: XXXX XXXXXX XXXXX (spaces, 4-6-5 grouping)
        r"\b\d{4}\s\d{6}\s\d{5}\b",
        # Format: XXXX-XXXXXX-XXXXX (dashes, 4-6-5 grouping)
        r"\b\d{4}-\d{6}-\d{5}\b",
        # Format: XXXX.XXXXXX.XXXXX (dots, 4-6-5 grouping)
        r"\b\d{4}\.\d{6}\.\d{5}\b",
        # Format: XXXXXXXXXXXXXXX (no separators)
        r"\b\d{15}\b",
        # 14-digit cards (Diners Club)
        # Format: XXXX XXXX XXXX XX (spaces, 4-4-4-2 grouping)
        r"\b\d{4}\s\d{4}\s\d{4}\s\d{2}\b",
        # Format: XXXX-XXXX-XXXX-XX (dashes, 4-4-4-2 grouping)
        r"\b\d{4}-\d{4}-\d{4}-\d{2}\b",
        # Format: XXXX.XXXX.XXXX.XX (dots, 4-4-4-2 grouping)
        r"\b\d{4}\.\d{4}\.\d{4}\.\d{2}\b",
        # Format: XXXXXX XXXXXX XX (spaces, 6-6-2 grouping, alternate Diners format)
        r"\b\d{6}\s\d{6}\s\d{2}\b",
        # Format: XXXXXX-XXXXXX-XX (dashes, 6-6-2 grouping)
        r"\b\d{6}-\d{6}-\d{2}\b",
        # Format: XXXXXX.XXXXXX.XX (dots, 6-6-2 grouping)
        r"\b\d{6}\.\d{6}\.\d{2}\b",
        # Format: XXXXXXXXXXXXXX (no separators)
        r"\b\d{14}\b",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text)

        for match in matches:
            candidate = match.group()

            # Normalize: remove all separators
            normalized = re.sub(r"[\s\-\.]", "", candidate)

            # Validate using prefix + Luhn
            is_valid, card_type = is_valid_card_number(normalized)

            if is_valid:
                found_cards.append((normalized, card_type))
                logger.debug(f"Found valid {card_type} card: {normalized[:6]}...{normalized[-4:]}")

    return found_cards


def classify_pages_with_cards(ocr_text_pages: list[dict]) -> list[int]:
    """
    Identify pages containing valid credit/debit card numbers.

    Uses high-precision validation:
    1. Pattern matching with consistent separators only
    2. Card type prefix validation
    3. Luhn algorithm checksum validation

    Args:
        ocr_text_pages: List of page dictionaries with 'text' and 'page_number' keys

    Returns:
        List of page numbers (integers) that contain valid card numbers
    """
    pages_with_cards = set()

    for page in ocr_text_pages:
        if "text" not in page or not page["text"]:
            continue

        page_text = page["text"]
        page_number = page.get("page_number", 0)

        # Find card numbers in this page
        found_cards = find_card_numbers_in_text(page_text)

        if found_cards:
            pages_with_cards.add(page_number)
            card_summary = ", ".join(
                f"{card_type} ({number[:6]}...{number[-4:]})" for number, card_type in found_cards
            )
            logger.info(f"Page {page_number}: Found {len(found_cards)} card(s) - {card_summary}")

    # Convert set to sorted list
    result = sorted(list(pages_with_cards))

    if result:
        logger.info(f"Found credit/debit cards on {len(result)} pages: {result}")
    else:
        logger.info("No credit/debit cards found")

    return result


def classify_file_for_cards(ocr_text_pages: list[dict], sha1: str) -> dict:
    """
    Classify a file for credit/debit card presence and return structured results.

    This is a convenience wrapper that provides results in a format
    consistent with other classifiers in the pipeline.

    Args:
        ocr_text_pages: List of page dictionaries with OCR text
        sha1: SHA1 hash of the file

    Returns:
        Dict with:
            - sha1: The file SHA1
            - pages_with_cards: List of page numbers containing valid cards
            - success: Boolean indicating if processing was successful
            - error: Error message if processing failed (None otherwise)
    """
    result = {
        "sha1": sha1,
        "pages_with_cards": [],
        "success": False,
        "error": None,
    }

    try:
        pages_with_cards = classify_pages_with_cards(ocr_text_pages)
        result["pages_with_cards"] = pages_with_cards
        result["success"] = True

        logger.info(
            f"Successfully classified {sha1}: {len(pages_with_cards)} pages with card numbers"
        )

    except Exception as e:
        logger.error(f"Error classifying {sha1} for cards: {e}", exc_info=True)
        result["error"] = str(e)

    return result
