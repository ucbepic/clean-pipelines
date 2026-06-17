import logging
import re

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IncidentDate(BaseModel):
    incident_date: list[str] = Field(description="Incident date parsed from img")


class CaseNumbers(BaseModel):
    case_numbers: list[str] = Field(description="Case ids parsed from fp or fn")


class Names(BaseModel):
    names: list[str] = Field(description="Names ids parsed from fp or fn")


# ============================================================================
# EXCLUSION PATTERNS - Filter out generic/ambiguous features
# ============================================================================

EXCLUDE_CASE_ID_PATTERNS = [
    r'^F\d{1,3}$',           # F313, F990, F101 (form numbers)
    r'^TF\d+[A-Z]?$',        # TF967, TF752, TF967B, TF1050A (traffic forms)
    r'^FMT\d+$',             # FMT5 (form codes)
    r'^\d{1,6}$',            # 109, 1208, 190416 (too short, likely not unique)
    r'^[A-Z]{1,2}\d{1,2}$',  # V1, V2, W3 (too generic)
    r'^[A-Z]{2,6}$',         # OISLL, COM, RAVA (abbreviations without numbers)
    r'^FILE\d+$',            # FILE20150803011837 (file system timestamps)
    r'^IADFORM\d*$',         # IADFORM13, IADFORM11 (IA form templates)
    r'^INCIDENT\d*$',        # INCIDENT1 (generic incident placeholders)
    r'^REPORT\d*$',          # REPORT (generic report IDs)
    r'^FORM\d*$',            # FORM (generic form references)
]

EXCLUDE_NAME_PATTERNS = [
    r'^b\s*\d+\s*b',                    # b 6 b, b 5 b (redacted)
    r'^\d+\s*\d+\s*b\s*\d+\s*[a-z]',    # 832 7 b 6 b, 832 7 b 6 a (redacted)
    r'subject\s*b\s*\d+\s*b',           # subject b 6 b
    r'witness\s+\d+',                    # witness 1, witness 3
    r'unnamed\s+(suspect|person|individual)',
    r'name\s*redacted',
    r'^victim$',
    r'^suspect$',
    r'^confidential\s+informant',        # confidential informant ci
    r'\bci\b',                           # CI (confidential informant)
    r'^[a-z]+$',                         # Single word names (too ambiguous alone)
]


def should_exclude_case_id(case_id: str) -> bool:
    """Check if a case ID should be excluded based on blocklist patterns."""
    for pattern in EXCLUDE_CASE_ID_PATTERNS:
        if re.match(pattern, case_id, re.IGNORECASE):
            return True
    return False


def should_exclude_name(name: str) -> bool:
    """Check if a name should be excluded based on blocklist patterns."""
    name_lower = name.lower()
    for pattern in EXCLUDE_NAME_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return True
    return False


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
        # MMDDYYYY format (e.g., 01212021 = Jan 21, 2021) - common in San Jose PD folder names
        r"([01]\d[0-3]\d\d{4})",
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
    Extract case IDs from text, excluding known generic/ambiguous patterns.
    """
    if not text:
        return []

    id_patterns = [
        # IAD/IA/OIA with numbers (specific incident IDs)
        r"((?:IAD|IA|OIA|OAI)[-_]?\d{2,4}[-_]?\d{1,4})",  # Require year+number format
        # Case number with hash
        r"Case#([A-Za-z0-9-]+)",
        # Complex case ID formats (H-OIA-097-20-A)
        r"\b([A-Z]-[A-Z]{2,4}-\d{1,3}-\d{1,2}-[A-Z])\b",
        # 3-part numeric case IDs (e.g., "21-021-0129", "21-151-0821") - San Jose PD format
        r"(?:^|(?<=[^0-9]))(\d{2,4}-\d{3,5}-\d{3,5})(?=[^0-9]|$)",
        # 2-part numeric case IDs with year (e.g., "2018-4567", "22-1460")
        # Require at least 2 digit year and 3 digit case number for uniqueness
        r"(?:^|(?<=[^0-9]))(\d{2,4}-\d{3,5})(?=[^0-9]|$)",
        # Long alphanumeric case IDs (at least 6 digits after letters)
        r"(?:^|(?<=[^A-Za-z0-9]))([A-Z]{2,4}\d{6,})(?=[^A-Za-z0-9]|$)",
        # OIS (Officer Involved Shooting) with specific numbers
        r"\b(OIS[-_]?\d{4,})\b",  # Require at least 4 digits
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

            clean_id = clean_id.strip().strip("_")

            # Apply exclusion filters
            if clean_id and not should_exclude_case_id(clean_id):
                if clean_id not in all_ids:
                    all_ids.append(clean_id)

    # Deduplicate: remove IDs that are substrings of other IDs
    # Keep longer, more specific IDs
    filtered_ids = []
    for id1 in all_ids:
        is_substring = False
        for id2 in all_ids:
            if id1 != id2 and id1 in id2:
                # id1 is a substring of id2, skip it
                is_substring = True
                break
        if not is_substring:
            filtered_ids.append(id1)

    return filtered_ids


def extract_names_from_metadata(text: str) -> list[str]:
    """Extract all names from text with regex, focusing on human names only"""
    if not text:
        return []

    patterns = [
        # Names with titles: Officer John Smith, Det. Jane Doe
        # IMPORTANT: Use negative lookahead to reject organizational fragments like "Chief of Police"
        r"\b(?:Officer|Det\.|Detective|Sgt\.|Sergeant|Lt\.|Lieutenant|Cpl\.|Corporal|Chief|Sheriff|Deputy)\s+(?!of\s|or\s)([A-Z][a-z]+\s+(?:[A-Z][a-z]*\s+)?[A-Z][a-z]+)\b",
        # Last name, First initial format (like "Hourigan, K")
        r"\b([A-Z][a-z]{2,}),\s+([A-Z])(?:\b|\s|\.|\s-\s)",
        # Names with suffixes like Jr., Sr., III
        r"\b([A-Z][a-z]{2,}\s+(?:[A-Z][a-z]{2,}\s+)?[A-Z][a-z]{2,}\s+(?:Jr\.|Sr\.|I{1,3}|IV))\b",
    ]

    all_names = []

    # Extract last name, initial format with exclusion filtering
    lastname_initial_pattern = r"\b([A-Z][a-z]{2,}),\s+([A-Z])(?:\b|\s|\.|\s-\s)"
    for match in re.finditer(lastname_initial_pattern, text):
        lastname = match.group(1)
        initial = match.group(2)
        full_name = f"{lastname}, {initial}"

        # Apply exclusion filter
        if not should_exclude_name(full_name):
            all_names.append(full_name)

    # Extract other patterns with exclusion filtering
    for pattern in patterns:
        if pattern != lastname_initial_pattern:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    name = match.strip() if isinstance(match, str) else match[0].strip()

                    # Apply length and exclusion filters
                    if len(name) >= 4 and not should_exclude_name(name):
                        # Additional filter: reject organizational fragments
                        # These are fragments like "of Police", "of Police Commissioners", etc.
                        name_lower = name.lower()
                        org_fragments = [
                            "of police", "or police", "of the police",
                            "police department", "police commission", "police board",
                            "city of", "county of", "state of",
                            "department of", "board of", "bureau of"
                        ]
                        if not any(fragment in name_lower for fragment in org_fragments):
                            all_names.append(name)

    # Deduplicate
    seen = set()
    unique_names = []
    for name in all_names:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique_names.append(name)

    return unique_names
