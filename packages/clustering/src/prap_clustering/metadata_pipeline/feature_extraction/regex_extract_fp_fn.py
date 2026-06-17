import re

from pydantic import BaseModel, Field


class IncidentDate(BaseModel):
    incident_date: list[str] = Field(description="Incident date parsed from img")


class CaseNumbers(BaseModel):
    case_numbers: list[str] = Field(description="Case ids parsed from fp or fn")


class Names(BaseModel):
    names: list[str] = Field(description="Names ids parsed from fp or fn")


def extract_date_from_metadata(text: str) -> list[str]:
    """
    Extract all dates from text with extremely flexible pattern matching.
    Prioritizes recall over precision to catch all possible date formats.
    """
    if not text:
        return []

    date_patterns = [
        # Written month formats (e.g., "August 8, 2022", "Jan 15, 2024", "December 1st, 2023")
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
        # YYYY-MM-DD (standard ISO format)
        r"\b(\d{4}-\d{1,2}-\d{1,2})\b",
        # MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY (US format)
        r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{4})\b",
        # MM/DD/YY, MM-DD-YY, MM.DD.YY (short year)
        r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2})\b",
        # YYYY/MM/DD, YYYY.MM.DD (alternate ISO)
        r"\b(\d{4}[\/\.]\d{1,2}[\/\.]\d{1,2})\b",
        # YYYYMMDD (compact format without separators) - relaxed for filepaths
        r"(\d{4}[01]\d[0-3]\d)",
        # Relaxed matching for anything that looks like a date
        # This will catch dates within filenames and without word boundaries
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})",
        # Format like 04.27.21 that might be in filenames
        r"(\d{2}\.\d{2}\.\d{2})",
        # MMDDYY format (e.g., 030524)
        r"([01]\d[0-3]\d\d{2})",
    ]

    all_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            date = match[0] if isinstance(match, tuple) else match

            if "(" in date:
                date = date.split("(")[0]
            if "__" in date:
                date = date.split("__")[0]

            date = date.strip()
            if date and date not in all_dates:
                all_dates.append(date)

    return all_dates


def extract_ids_from_metadata(text: str) -> list[str]:
    """
    Extract all possible case IDs from text with extremely flexible pattern matching.
    Prioritizes recall over precision to catch all possible ID formats.
    """
    if not text:
        return []

    id_patterns = [
        # IAD with numbers directly after (IAD552)
        r"((?:IAD|IA|OIA|OAI)[-_]?\d+)",
        # Standard complex format (H-OIA-097-20-A)
        r"\b([A-Z]-[A-Z]{2,4}-\d{1,3}-\d{1,2}-[A-Z])\b",
        # Various case ID formats with year and number
        r"((?:IAD|IA|OIA|OAI)[-_]?\d{2,4}[-_]?\d{1,4})",
        # Case number with hash, more relaxed
        r"Case#([A-Za-z0-9-]+)",
        # Any uppercase letter followed by hyphen and numbers
        r"([A-Z]-\d+)",
        # Any format like N-BPH-284-18-A
        r"([A-Z]-[A-Z]{2,4}-\d{1,3}-\d{1,2}-[A-Z])",
        # Additional pattern for case numbers that appear between underscores
        r"_((?:IAD|IA|OIA|OAI)\d+)_",
        # Look for case numbers surrounded by non-alphanumeric characters
        r"(?:^|[^a-zA-Z0-9])((?:IAD|IA|OIA|OAI)\d+)(?:$|[^a-zA-Z0-9])",
        # Numeric case IDs (e.g., "22-1460", "2018-4567", "18-123", "01-20621")
        # Match 2-4 digit year followed by hyphen and 2-5 digit case number
        # Use lookahead/lookbehind to handle underscores, slashes, etc. without consuming chars
        r"(?:^|(?<=[^0-9]))(\d{2,4}-\d{2,5})(?=[^0-9]|$)",
        # OIS (Officer Involved Shooting) patterns
        r"\b(OIS[-_]?[A-Za-z0-9-]+)\b",
    ]

    all_ids = []
    for pattern in id_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            id_val = match[0] if isinstance(match, tuple) else match

            clean_id = re.sub(r"\s+", "", id_val)
            if "(" in clean_id:
                clean_id = clean_id.split("(")[0]
            if "__" in clean_id:
                clean_id = clean_id.split("__")[0]

            clean_id = clean_id.strip()

            if clean_id.startswith("_"):
                clean_id = clean_id[1:]
            if clean_id.endswith("_"):
                clean_id = clean_id[:-1]

            if clean_id and clean_id not in all_ids:
                all_ids.append(clean_id)

    prefixes = ["IAD", "IA", "OIA", "OAI"]
    for prefix in prefixes:
        prefix_pattern = rf"({prefix}\d+)"
        matches = re.findall(prefix_pattern, text, re.IGNORECASE)
        for match in matches:
            if match and match not in all_ids:
                all_ids.append(match)

    return all_ids


def extract_names_from_metadata(text: str) -> list[str]:
    """Extract all names from text with regex, focusing on human names only"""
    if not text:
        return []

    patterns = [
        # Names with titles: Officer John Smith, Det. Jane Doe
        r"\b(?:Officer|Det\.|Detective|Sgt\.|Sergeant|Lt\.|Lieutenant|Cpl\.|Corporal|Chief|Sheriff|Deputy)\s+([A-Z][a-z]+\s+(?:[A-Z][a-z]*\s+)?[A-Z][a-z]+)\b",
        # Last name, First initial format (like "Hourigan, K")
        r"\b([A-Z][a-z]{2,}),\s+([A-Z])(?:\b|\s|\.|\s-\s)",
        # Names with suffixes like Jr., Sr., III
        r"\b([A-Z][a-z]{2,}\s+(?:[A-Z][a-z]{2,}\s+)?[A-Z][a-z]{2,}\s+(?:Jr\.|Sr\.|I{1,3}|IV))\b",
    ]

    all_names = []

    lastname_initial_pattern = r"\b([A-Z][a-z]{2,}),\s+([A-Z])(?:\b|\s|\.|\s-\s)"
    for match in re.finditer(lastname_initial_pattern, text):
        lastname = match.group(1)
        initial = match.group(2)
        all_names.append(f"{lastname}, {initial}")

    for pattern in patterns:
        if pattern != lastname_initial_pattern:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                all_names.extend([match.strip() for match in matches])

    filtered_names = []
    for name in all_names:
        name.lower()

        if len(name) < 4:
            continue

        filtered_names.append(name)

    seen = set()
    unique_names = []
    for name in filtered_names:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique_names.append(name)

    return unique_names
