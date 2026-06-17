"""
Normalization and feature parsing utilities for hybrid clustering pipeline.

This module provides functions to parse and normalize features extracted from
documents (filepath, filename, LLM) to enable consistent comparison.
"""

import ast
import json
import re
from datetime import datetime
from typing import Any

import pandas as pd


def parse_feature_list(feature_value: Any) -> list[str]:
    """
    Parse feature value from CSV into list of strings.

    Handles multiple input formats:
    - String representations: "['val1', 'val2']" → ['val1', 'val2']
    - JSON strings: '["val1", "val2"]' → ['val1', 'val2']
    - Comma-separated: "val1, val2" → ['val1', 'val2']
    - None/empty: None, "None", "[]" → []
    - Already parsed lists: ['val1', 'val2'] → ['val1', 'val2']

    Args:
        feature_value: Value from CSV column (can be various types)

    Returns:
        List of string values, empty list if None/empty
    """
    if pd.isna(feature_value) or feature_value in [None, 'None', '[]', '']:
        return []

    # Already a list
    if isinstance(feature_value, list):
        return [str(v).strip() for v in feature_value if v]

    # String representation of list: "['val1', 'val2']"
    if isinstance(feature_value, str):
        feature_value = feature_value.strip()

        # Try ast.literal_eval for Python list strings
        if feature_value.startswith('['):
            try:
                parsed = ast.literal_eval(feature_value)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if v]
            except:
                pass

        # Try JSON parsing
        try:
            parsed = json.loads(feature_value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
            elif isinstance(parsed, dict):
                # Handle structured outputs like {"incident_date": "2024-01-15"}
                # Extract all non-null values
                values = [str(v) for v in parsed.values() if v is not None]
                return values
        except:
            pass

        # Try comma-separated
        if ',' in feature_value:
            return [v.strip() for v in feature_value.split(',') if v.strip()]

        # Single value
        if feature_value:
            return [feature_value.strip()]

    return []


def normalize_case_ids(case_ids: list[str]) -> set[str]:
    """
    Normalize case IDs to consistent format.

    Rules:
    - Uppercase all letters
    - Remove all spaces, dashes, and punctuation (keep only letters and numbers)
    - Strip leading zeros from numeric parts (e.g., "IA2018-017" → "IA201817")
    - Deduplicate

    Examples:
        ["IA2018-017", "ia2018-17"] → {"IA201817"}
        ["OIS-001", "ois-1"] → {"OIS1"}
        ["RD #99-092462", "RD#99-92462"] → {"RD9992462"}
        ["F-0754", "F 0754"] → {"F754"}

    Args:
        case_ids: List of case ID strings

    Returns:
        Set of normalized case IDs
    """
    normalized = set()

    for case_id in case_ids:
        if not case_id:
            continue

        # Uppercase and strip
        case_id = str(case_id).upper().strip()

        # Remove all non-alphanumeric characters (spaces, dashes, #, ., etc.)
        case_id = re.sub(r'[^A-Z0-9]', '', case_id)

        # Remove leading zeros from numeric parts
        # Pattern: Find sequences of digits and remove leading zeros
        def remove_leading_zeros(match):
            num = match.group(0)
            # Keep at least one digit (don't turn "000" into "")
            return str(int(num)) if num.isdigit() else num

        case_id = re.sub(r'\d+', remove_leading_zeros, case_id)

        if case_id:
            normalized.add(case_id)

    return normalized


def normalize_dates(dates: list[str]) -> set[str]:
    """
    Normalize dates to YYYY-MM-DD format.

    Rules:
    - Parse various formats (MM/DD/YYYY, YYYYMMDD, "January 15, 2024", etc.)
    - Convert to ISO format YYYY-MM-DD
    - Handle 2-digit years (20 → 2020, 95 → 1995)
    - Filter out timestamps (HH:MM:SS patterns)
    - Deduplicate

    Examples:
        ["01/15/2024", "2024-01-15", "January 15, 2024"] → {"2024-01-15"}
        ["20240115"] → {"2024-01-15"}
        ["021320"] → {"2020-02-13"}
        ["15.08.03"] → {"2015-08-03"}

    Args:
        dates: List of date strings in various formats

    Returns:
        Set of dates in YYYY-MM-DD format
    """
    normalized = set()

    for date_str in dates:
        if not date_str:
            continue

        date_str = str(date_str).strip()

        # Filter out timestamps (HH:MM:SS patterns)
        # Patterns like "18.14.24", "11-52-06", "23:07:34"
        timestamp_pattern = r'^\d{1,2}[:\-\.]\d{1,2}[:\-\.]\d{1,2}$'
        if re.match(timestamp_pattern, date_str):
            continue

        # Filter out values that look like times (HH:MM format)
        time_pattern = r'^\d{1,2}:\d{2}$'
        if re.match(time_pattern, date_str):
            continue

        # Try multiple date formats (order matters - most specific first)
        formats = [
            # ISO formats with full year
            "%Y-%m-%d",           # 2024-01-15
            "%Y/%m/%d",           # 2024/01/15
            "%Y.%m.%d",           # 2024.01.15

            # US formats with full year
            "%m/%d/%Y",           # 01/15/2024
            "%m-%d-%Y",           # 01-15-2024
            "%m.%d.%Y",           # 01.15.2024

            # 2-digit year formats (YY-MM-DD style)
            "%y-%m-%d",           # 20-08-23
            "%y/%m/%d",           # 20/08/23
            "%y.%m.%d",           # 15.08.03 (YY.MM.DD)

            # US formats with 2-digit year
            "%m/%d/%y",           # 01/15/24
            "%m-%d-%y",           # 01-15-24, 10-13-21
            "%m.%d.%y",           # 01.15.24, 8.5.21

            # Compact 8-digit formats
            "%Y%m%d",             # 20240115

            # Month name formats
            "%B %d, %Y",          # January 15, 2024
            "%b %d, %Y",          # Jan 15, 2024
        ]

        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

        # Handle compact 6-digit format MMDDYY (e.g., "021320" → Feb 13, 2020)
        if not parsed_date and re.match(r'^\d{6}$', date_str):
            try:
                parsed_date = datetime.strptime(date_str, "%m%d%y")
            except ValueError:
                # Try YYMMDD format as fallback
                try:
                    parsed_date = datetime.strptime(date_str, "%y%m%d")
                except ValueError:
                    pass

        # Handle compact 8-digit format without separators (e.g., "20240115")
        if not parsed_date and re.match(r'^\d{8}$', date_str):
            try:
                parsed_date = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                # Try MMDDYYYY format as fallback
                try:
                    parsed_date = datetime.strptime(date_str, "%m%d%Y")
                except ValueError:
                    pass

        # Handle 2-digit years (convert 00-29 to 2000-2029, 30-99 to 1930-1999)
        if parsed_date and parsed_date.year < 100:
            if parsed_date.year < 30:
                parsed_date = parsed_date.replace(year=parsed_date.year + 2000)
            else:
                parsed_date = parsed_date.replace(year=parsed_date.year + 1900)

        # Validate the parsed date is reasonable (1900-2100)
        if parsed_date:
            if 1900 <= parsed_date.year <= 2100:
                normalized.add(parsed_date.strftime("%Y-%m-%d"))

    return normalized


def normalize_names(names: list[str]) -> set[str]:
    """
    Normalize person names to consistent format.

    Rules:
    - Lowercase (for case-insensitive matching)
    - Strip titles (Officer, Det., Sgt., etc.)
    - Remove all punctuation (commas, periods, hyphens, etc.)
    - Remove extra whitespace
    - Deduplicate

    Examples:
        ["Officer John Smith", "john smith", "JOHN SMITH"] → {"john smith"}
        ["Doe, J.", "Doe J", "Doe,J"] → {"doe j"}
        ["Mary-Ann", "Mary Ann"] → {"mary ann"}

    Args:
        names: List of person name strings

    Returns:
        Set of normalized names
    """
    normalized = set()

    # Titles to remove
    titles = [
        r'\bOfficer\b', r'\bOff\.\b', r'\bOfc\.\b',
        r'\bDetective\b', r'\bDet\.\b',
        r'\bSergeant\b', r'\bSgt\.\b',
        r'\bLieutenant\b', r'\bLt\.\b',
        r'\bCorporal\b', r'\bCpl\.\b',
        r'\bChief\b', r'\bSheriff\b', r'\bDeputy\b',
        r'\bMr\.\b', r'\bMrs\.\b', r'\bMs\.\b', r'\bDr\.\b'
    ]

    for name in names:
        if not name:
            continue

        name = str(name).strip()

        # Remove titles
        for title in titles:
            name = re.sub(title, '', name, flags=re.IGNORECASE)

        # Lowercase
        name = name.lower()

        # Remove all punctuation (commas, periods, hyphens, apostrophes, etc.)
        # Replace with space to preserve word boundaries
        name = re.sub(r'[^\w\s]', ' ', name)

        # Normalize whitespace (collapse multiple spaces into one)
        name = re.sub(r'\s+', ' ', name).strip()

        if name and len(name) >= 2:  # Filter out single letters
            normalized.add(name)

    return normalized


def combine_features(fp_values: Any, fn_values: Any) -> list[str]:
    """
    Combine features from filepath and filename sources.

    Args:
        fp_values: Feature values from filepath (any format)
        fn_values: Feature values from filename (any format)

    Returns:
        Combined list of unique values (order preserved)
    """
    fp_list = parse_feature_list(fp_values)
    fn_list = parse_feature_list(fn_values)

    # Combine and deduplicate while preserving order
    combined = []
    seen = set()

    for val in fp_list + fn_list:
        if val and val not in seen:
            combined.append(val)
            seen.add(val)

    return combined


# ============================================================================
# STRUCTURED JSON PARSING (for *_llm_structured columns)
# ============================================================================


def parse_structured_dates(structured_value: Any) -> list[str]:
    """
    Parse structured date JSON: {"incident_date": "2024-01-26"}

    Args:
        structured_value: JSON string or parsed dict from structured column

    Returns:
        List containing incident date if found, empty list otherwise
    """
    if pd.isna(structured_value) or not structured_value:
        return []

    try:
        # Parse JSON if it's a string
        if isinstance(structured_value, str):
            data = json.loads(structured_value)
        else:
            data = structured_value

        # Extract incident_date
        if isinstance(data, dict) and 'incident_date' in data:
            incident_date = data['incident_date']
            if incident_date and incident_date is not None:
                return [str(incident_date)]

    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return []


def parse_structured_case_ids(structured_value: Any) -> list[str]:
    """
    Parse structured case ID JSON: [{"id": "IA2018-0167"}, {"id": "2018-0167"}]

    Args:
        structured_value: JSON string or parsed list from structured column

    Returns:
        List of case ID strings
    """
    if pd.isna(structured_value) or not structured_value:
        return []

    try:
        # Parse JSON if it's a string
        if isinstance(structured_value, str):
            data = json.loads(structured_value)
        else:
            data = structured_value

        # Extract IDs from array of objects
        if isinstance(data, list):
            ids = []
            for item in data:
                if isinstance(item, dict) and 'id' in item:
                    case_id = item['id']
                    if case_id:
                        ids.append(str(case_id))
            return ids

    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return []


def parse_structured_subject_names(structured_value: Any) -> list[str]:
    """
    Parse structured subject name JSON: [{"name": "Kevin Bushnell", "subject_type": "suspect"}]

    Args:
        structured_value: JSON string or parsed list from structured column

    Returns:
        List of subject name strings (without subject_type)
    """
    if pd.isna(structured_value) or not structured_value:
        return []

    try:
        # Parse JSON if it's a string
        if isinstance(structured_value, str):
            data = json.loads(structured_value)
        else:
            data = structured_value

        # Extract names from array of objects
        if isinstance(data, list):
            names = []
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    name = item['name']
                    if name:
                        names.append(str(name))
            return names

    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return []


def parse_structured_officer_names(structured_value: Any) -> list[str]:
    """
    Parse structured officer name JSON: [{"name": "Officer Butera", "context": "responded to scene"}]

    Args:
        structured_value: JSON string or parsed list from structured column

    Returns:
        List of officer name strings (without context)
    """
    if pd.isna(structured_value) or not structured_value:
        return []

    try:
        # Parse JSON if it's a string
        if isinstance(structured_value, str):
            data = json.loads(structured_value)
        else:
            data = structured_value

        # Extract names from array of objects
        if isinstance(data, list):
            names = []
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    name = item['name']
                    if name:
                        names.append(str(name))
            return names

    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return []
