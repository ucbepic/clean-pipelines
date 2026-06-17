"""Utility helpers for the redactions pipeline."""

import re


def has_penal_code_mentions(ocr_text_pages):
    """
    Check if any page in the OCR text contains penal code patterns or descriptions.
    """
    penal_code_variations = [
        "368\\(b\\) - elder abuse",
        "368\\(b\\) elder abuse",
        "368 - elder abuse",
        "273\\.5\\(a\\) - corporal injury cohabitant/spouse",
        "273\\.5 - corporal injury cohabitant/spouse",
        "corporal injury cohabitant/spouse",
        "243\\(e\\)\\(1\\) - battery on a spouse",
        "243\\(e\\) - battery on a spouse",
        "243 - battery on a spouse",
        "battery on a spouse",
        "209\\(b\\) - kidnap to commit sex crime",
        "kidnap to commit sex crime",
        "209 - kidnap to commit sex crime",
        "220\\(a\\) - sexual assault",
        "220 - sexual assault",
        "236\\.1 - human sex trafficking",
        "human sex trafficking",
        "243\\.4\\(a\\) - sexual battery",
        "243\\.4\\(c\\) - sexual battery",
        "243\\.4\\(d\\) - sexual battery",
        "243\\.4\\(e\\) - sexual battery",
        "243\\.4 - sexual battery",
        "261 - rape",
        "rape",
        "264\\.1 - gang rape",
        "gang rape",
        "266 - prostitution child",
        "prostitution child",
        "647 - prostitution",
        "prostitution",
        "269 - child sex abuse",
        "child sex abuse",
        "285 - incest",
        "incest",
        "286 - sodomy",
        "sodomy",
        "287 - oral copulation with minors/forced",
        "oral copulation with minors/forced",
        "288 - lewd acts with children",
        "lewd acts with children",
        "289 - sexual penetration",
        "sexual penetration",
        "288\\.5 - repeated sexual abuse of a child",
        "repeated sexual abuse of a child",
        "311 - possession of child pornography",
        "possession of child pornography",
        "314 - indecent exposure",
        "indecent exposure",
        "644 - attempted rape",
        "attempted rape",
        "368 elder abuse",
        "273\\.5\\(a\\) corporal injury cohabitant/spouse",
        "273\\.5 corporal injury cohabitant/spouse",
        "243\\(e\\)\\(1\\) battery on a spouse",
        "243\\(e\\) battery on a spouse",
        "243 battery on a spouse",
        "209\\(b\\) kidnap to commit sex crime",
        "209 kidnap to commit sex crime",
        "220\\(a\\) sexual assault",
        "220 sexual assault",
        "236\\.1 human sex trafficking",
        "243\\.4\\(a\\) sexual battery",
        "243\\.4\\(c\\) sexual battery",
        "243\\.4\\(d\\) sexual battery",
        "243\\.4\\(e\\) sexual battery",
        "243\\.4 sexual battery",
        "261 rape",
        "264\\.1 gang rape",
        "266 prostitution child",
        "647 prostitution",
        "269 child sex abuse",
        "285 incest",
        "286 sodomy",
        "287 oral copulation with minors/forced",
        "288 lewd acts with children",
        "289 sexual penetration",
        "288\\.5 repeated sexual abuse of a child",
        "311 possession of child pornography",
        "314 indecent exposure",
        "644 attempted rape",
    ]

    pattern = "|".join(penal_code_variations)
    regex = re.compile(pattern, re.IGNORECASE)

    # Check each page for matches
    for page in ocr_text_pages:
        if "text" in page and page["text"]:
            if regex.search(page["text"]):
                return True

    # No matches found in any page
    return False


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.

    Args:
        text: The text to count words in

    Returns:
        Number of words in the text
    """
    if not text or not isinstance(text, str):
        return 0

    # Split on whitespace and filter out empty strings
    words = text.split()
    return len(words)


def has_low_word_count(text: str, threshold: int = 20) -> bool:
    """
    Check if a text has a low word count (likely an image page).

    Pages with low word counts are more likely to be primarily images
    rather than text documents, making them candidates for graphic
    imagery classification.

    Args:
        text: The text to check
        threshold: Maximum word count to be considered "low" (default: 20)

    Returns:
        True if word count is <= threshold, False otherwise
    """
    return count_words(text) <= threshold
