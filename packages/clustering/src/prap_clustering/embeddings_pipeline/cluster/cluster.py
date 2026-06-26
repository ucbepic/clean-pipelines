import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import networkx as nx
import numpy as np
import pandas as pd
from jinja2 import Template
from pydantic import BaseModel, Field

from prap_clustering._llm import get_llm

AGENCIES = [
    ### hold out
    ## run 1
    {
        "name": "Bakersfield Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Bakersfield Police Department--2025-04-09_22-08-24 - autofolio_1.2.0_output--Bakersfield Police Department--2025-04-09_22-08-24_ocr_col_dropped.csv",
    },
    {
        "name": "Santa Monica Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Monica Police Department--2024-12-21_02-18-20 - autofolio_1.1.0_output--Santa Monica Police Department--2024-12-21_02-18-20_ocr_col_dropped.csv",
    },
    {
        "name": "Richmond Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Richmond Police Department--2025-04-09_20-59-59 - autofolio_1.2.0_output--Richmond Police Department--2025-04-09_20-59-59_ocr_col_dropped.csv",
    },
    {
        "name": "Los Angeles District Attorney",
        "csv_path": "../data/input/autofolio_1.1.0_output--Los Angeles District Attorney--2024-11-27_05-47-23 - autofolio_1.1.0_output--Los Angeles District Attorney--2024-11-27_05-47-23_ocr_col_dropped.csv",
    },
    {
        "name": "California Department of Justice",
        "csv_path": "../data/input/autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52 - autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52_ocr_col_dropped.csv",
    },
    {
        "name": "Office of Inspector General for Prisons",
        "csv_path": "../data/input/autofolio_1.2.0_output--Office of Inspector General for Prisons--2025-04-27_21-48-50 - autofolio_1.2.0_output--Office of Inspector General for Prisons--2025-04-27_21-48-50_ocr_col_dropped.csv",
    },
    {
        "name": "Santa Ana Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Ana Police Department--2025-02-13_01-55-05 - autofolio_1.1.0_output--Santa Ana Police Department--2025-02-13_01-55-05_ocr_col_dropped.csv",
    },
    {
        "name": "San Francisco Police Commission",
        "csv_path": "../data/input/autofolio_1.2.0_output--San Francisco Police Commission--2025-04-09_21-20-14 - autofolio_1.2.0_output--San Francisco Police Commission--2025-04-09_21-20-14_ocr_col_dropped.csv",
    },
    {
        "name": "Kern County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Kern County Sheriff--2024-07-15_23-59-08 - autofolio_1.1.0_output--Kern County Sheriff--2024-07-15_23-59-08_ocr_col_dropped.csv",
    },
    {
        "name": "Santa Clara County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Clara County Sheriff--2024-12-13_15-09-21 - autofolio_1.1.0_output--Santa Clara County Sheriff--2024-12-13_15-09-21_ocr_col_dropped.csv",
    },
    {
        "name": "Fresno County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42 - autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42_ocr_col_dropped.csv",
    },
    {
        "name": "Sacramento County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Sacramento County Sheriff--2025-03-04_10-11-24 - autofolio_1.1.0_output--Sacramento County Sheriff--2025-03-04_10-11-24_ocr_col_dropped.csv",
    },
    {
        "name": "San Francisco County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Francisco County Sheriff--2024-07-20_01-57-38 - autofolio_1.1.0_output--San Francisco County Sheriff--2024-07-20_01-57-38_ocr_col_dropped.csv",
    },
    {
        "name": "California Department of Corrections and Rehabilitation",
        "csv_path": "../data/input/OLDautofolio_1.2.0_output--California Department of Corrections and Rehabilitation--2025-03-24_23-57-31 - autofolio_1.2.0_output--California Department of Corrections and Rehabilitation--2025-03-24_23-57-31_ocr_col_dropped.csv",
    },
    # run 2
    {
        "name": "Folsom Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Folsom Police Department--2025-02-27_23-14-01 - autofolio_1.1.0_output--Folsom Police Department--2025-02-27_23-14-01_ocr_col_dropped.csv",
    },
    {
        "name": "UC Davis Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--UC Davis Police Department--2024-07-20_08-53-00 - autofolio_1.1.0_output--UC Davis Police Department--2024-07-20_08-53-00_ocr_col_dropped.csv",
    },
    {
        "name": "Seal Beach Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Seal Beach Police Department--2025-02-13_08-16-22 - autofolio_1.1.0_output--Seal Beach Police Department--2025-02-13_08-16-22_ocr_col_dropped.csv",
    },
    {
        "name": "Contra Costa County District Attorney",
        "csv_path": "../data/input/autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04 - autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04_ocr_col_dropped.csv",
    },
    {
        "name": "Contra Costa County Sheriff",
        "csv_path": "../data/input/autofolio_1.1.0_output--Contra Costa County Sheriff--2024-12-21_01-47-17 - autofolio_1.1.0_output--Contra Costa County Sheriff--2024-12-21_01-47-17_ocr_col_dropped.csv",
    },
    {
        "name": "Shasta County District Attorney",
        "csv_path": "../data/input/autofolio_1.1.0_output--Shasta County District Attorney--2024-12-18_05-02-13 - autofolio_1.1.0_output--Shasta County District Attorney--2024-12-18_05-02-13_ocr_col_dropped.csv",
    },
    {
        "name": "Riverside County Department of Public Social Services",
        "csv_path": "../data/input/autofolio_1.1.0_output--Riverside County Department of Public Social Services--2024-07-05_20-47-44 - autofolio_1.1.0_output--Riverside County Department of Public Social Services--2024-07-05_20-47-44_ocr_col_dropped.csv",
    },
    {
        "name": "Cal State East Bay University Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Cal State East Bay University Police Department--2025-04-09_20-14-32 - autofolio_1.2.0_output--Cal State East Bay University Police Department--2025-04-09_20-14-32_ocr_col_dropped.csv",
    },
    {
        "name": "San Joaquin County Medical Examiner",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44 - autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44_ocr_col_dropped.csv",
    },
    {
        "name": "Pasadena Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Pasadena Police Department--2025-01-24_06-19-29 - autofolio_1.1.0_output--Pasadena Police Department--2025-01-24_06-19-29_ocr_col_dropped.csv",
    },
    {
        "name": "Irvine Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Irvine Police Department--2024-07-09_01-45-57 - autofolio_1.1.0_output--Irvine Police Department--2024-07-09_01-45-57_ocr_col_dropped.csv",
    },
    {
        "name": "San Diego County Medical Examiner",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Diego County Medical Examiner--2025-01-07_23-53-09 - autofolio_1.1.0_output--San Diego County Medical Examiner--2025-01-07_23-53-09_ocr_col_dropped.csv",
    },
    {
        "name": "San Leandro Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--San Leandro Police Department--2024-08-15_22-50-51 - autofolio_1.1.0_output--San Leandro Police Department--2024-08-15_22-50-51_ocr_col_dropped.csv",
    },
    {
        "name": "Santa Clara Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Santa Clara Police Department--2024-06-01_21-38-12 - autofolio_1.1.0_output--Santa Clara Police Department--2024-06-01_21-38-12_ocr_col_dropped.csv",
    },
    {
        "name": "Hayward Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37 - autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37_ocr_col_dropped.csv",
    },
    {
        "name": "Vallejo Police Department",
        "csv_path": "../data/input/autofolio_1.2.0_output--Vallejo Police Department--2025-03-25_00-20-52 - autofolio_1.2.0_output--Vallejo Police Department--2025-03-25_00-20-52_ocr_col_dropped.csv",
    },
    {
        "name": "Chula Vista Police Department",
        "csv_path": "../data/input/autofolio_1.1.0_output--Chula Vista Police Department--2025-02-20_07-43-22 - autofolio_1.1.0_output--Chula Vista Police Department--2025-02-20_07-43-22_ocr_col_dropped.csv",
    },
    ## hold out
]

OUTPUT_PATH = "../data/output/"
HISTORICAL_CSV_PATH = None
DEBUG = True

logging.basicConfig(
    format="%(process)d\t%(asctime)s\t%(levelname)s\t| %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("idonea." + __name__).setLevel(logging.INFO)

logger = logging.getLogger("idonea." + __name__)


def get_embeddings(names):
    if not names:
        return np.array([])
    return np.array(get_llm().embed(names))


@dataclass
class DocumentNode(TypedDict):
    id: str
    date: str | None
    name: str | None
    case_numbers: str | None
    summary: str | None


class SimilarityScore(BaseModel):
    """Response model for document similarity scoring."""

    similarity: float = Field(
        description="Similarity score between documents: 1.0 for same incident, 0.0 for different, 0.5 for uncertain"
    )


def validate_response(response: str) -> float | None:
    """Validate and convert LLM response to proper numeric value."""
    try:
        # Clean the response
        cleaned_response = response.strip().lower()

        if cleaned_response in [".5", "0.5", "0.50"]:
            return 0.5

        if cleaned_response in ["0", "0.0", "1", "1.0"]:
            return float(cleaned_response)

        try:
            score_response = SimilarityScore.model_validate_json(cleaned_response)
            if score_response.similarity in [0.0, 0.5, 1.0]:
                return score_response.similarity
        except:
            pass

        return None
    except:
        return None


def pairwise_comparison_of_paths(filepath_1: str, filepath_2: str, max_retries: int = 3) -> float:
    """
    Compare two document summaries and determine their similarity score.
    """
    template = Template(
        """<task>Your task is to compare two filepaths and determine if they should be assigned to the same incident cluster.</task>

<clustering_rules>
1. Strong indicators that files belong to the same incident (Score: 1.0):
- Same individual names AND same case number
- Same incident date AND same individual names
- Same case number AND matching agency

2. Moderate indicators suggesting possible relation (Score: 0.5):
- Same case number but different agencies (may indicate multi-agency involvement)
- Same individual names and close dates (within 1-2 days) but different case numbers
- Same incident location and date but slight variations in names

3. Files should NOT be clustered (Score: 0) when:
- Different individual names AND different case numbers
- Different agencies AND different case numbers
- Similar case numbers but clearly different years
- Different incident dates with no other matching identifiers
- Generic file paths without specific identifiers
</clustering_rules>

<example_input_1>
Bakersfield Police Department/BakersfieldPoliceDeptEnriqueMosqueda10132018/KCETBakersfieldPoliceDepartmentEnriqueMosqueda10132018/IA2018-17Photos and Audio
Bakersfield Police Department/BakersfieldPoliceDeptEnriqueMosqueda10132018/KCETBakersfieldPoliceDepartmentEnriqueMosqueda10132018/IA2018-17Photos and Audio
</example_input_1>

<example_output_1>
1
</example_output_1>

<example_input_2>
Bakersfield Police Department/BakersfieldPoliceDeptEnriqueMosqueda10132018/IA2018-17
Bakersfield Police Department/BakersfieldPoliceDeptEnriqueMosqueda10132018/KCETBakersfieldPoliceDepartmentEnriqueMosqueda10132018/Backlight files - California Reporting Project Case Files/Split files/IA2018-017.pdf
</example_input_2>

<example_output_2>
1
</example_output_2>

<example_input_3>
Bakersfield Police Department/BakersfieldPoliceDeptEnriqueMosqueda10132018/KCETBakersfieldPoliceDepartmentEnriqueMosqueda10132018/Backlight files - California Reporting Project Case Files/Split files/IA2018-017.pdf
Bakersfield Police Department/BakersfieldPoliceDeptEnriqueMosqueda10132018/IA2018-17/Backlight files - California Reporting Project Case Files/Split files/IA2018-017.pdf
</example_input_3>

<example_output_3>
1
</example_output_3>

<example_input_4>
Bakersfield Police Department/BakersfieldPoliceDeptJohnConner/IA2010-17
Bakersfield District Attorney's Office/BakersfieldPoliceDeptEnriqueMosqueda10132018/IA2018-17
</example_input_4>

<example_output_4>
0
</example_output_4>

<example_input_5>
Chicago Police Department/ChicagoPoliceDeptTimRobbins10132018/ChicagodPoliceDepartmentTimRobbins10/Split files/IA2005-017.pdf
Arizona Police Department/IA2005-017.pdf
</example_input_5>

<example_output_5>
0
</example_output_5>

<priority_order>
When evaluating filepath similarity, consider these elements in order of importance:
1. Individual names (officers, civilians, witnesses)
2. Case numbers/incident numbers
3. Incident dates
4. Agency names
5. Location information
</priority_order>

<special_cases>
- Multiple agency involvement: Files from different agencies may belong to the same incident if they share case numbers or involved individuals
- Date formats: Consider that dates may appear in different formats (MMDDYYYY, MM-DD-YYYY, etc.)
- Case number variations: Account for leading zeros or different formatting (e.g., IA2018-17 vs IA2018-017)
</special_cases>

Valid responses include:
1
0.5  (Possibly related incidents)
0    (Different incidents)

Below are the summaries to compare:

Filepath 1:
--------------------
{{ filepath_1 }}
--------------------

Filepath 2:
--------------------
{{ filepath_2 }}
--------------------
"""
    )

    prompt = template.render(filepath_1=filepath_1, filepath_2=filepath_2)

    for attempt in range(max_retries):
        try:
            # Model is whatever PRAP_LLM_MODEL resolves to.
            response = get_llm().complete(prompt).text
            validated_response = validate_response(response)

            if validated_response is not None:
                return validated_response

        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")

        if attempt < max_retries - 1:
            continue
    return 0.0


def get_directory_overlap(path1: str, path2: str) -> tuple[int, int]:
    """
    Calculate directory overlap between two paths.
    Returns tuple of (overlap_count, total_dirs)
    """
    dirs1 = Path(path1).parts
    dirs2 = Path(path2).parts

    # Count matching directories from the start
    overlap = 0
    for d1, d2 in zip(dirs1, dirs2, strict=False):
        if d1 == d2:
            overlap += 1
        else:
            break

    return (overlap, max(len(dirs1), len(dirs2)))


def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity score between two names using embeddings."""
    embeddings = get_embeddings([name1, name2])
    if len(embeddings) != 2:
        return 0.0

    # Calculate cosine similarity on raw embeddings
    norm1 = np.linalg.norm(embeddings[0])
    norm2 = np.linalg.norm(embeddings[1])

    # Check for zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2)
    return (similarity + 1) / 2


def calculate_path_similarity(path1: str, path2: str) -> float:
    """
    Calculate similarity between two paths using enhanced rules.
    """
    # Rule 0: If paths are exactly equal, they're a match
    if path1 == path2:
        return 1.0

    # Extract all incident numbers (now returns sets)
    incidents1 = extract_incident_numbers(path1)
    incidents2 = extract_incident_numbers(path2)

    # Check directory structure
    overlap_count, total_dirs = get_directory_overlap(path1, path2)

    # Rule 1: If both paths have incident numbers and there's any overlap
    if incidents1 and incidents2 and incidents1 & incidents2:  # Use & instead of .intersection()
        return 1.0

    # Rule 2: If paths have >2 subdirectories
    if total_dirs > 2:
        # Rule 2a: If only first 2 dirs match before diverging
        if overlap_count <= 2 and overlap_count < total_dirs:
            return 0.0

        # Rule 2b: If >2 dirs match but no incident number match
        if overlap_count > 2:
            # Fallback to pairwise comparison
            return pairwise_comparison_of_paths(path1, path2)

    # Default to pairwise comparison for other cases
    return pairwise_comparison_of_paths(path1, path2)


# Fix 3: Update calculate_id_similarity function
def calculate_id_similarity(a: str, b: str) -> float:
    """Calculate similarity score between two inputs."""
    if not a or not b:
        return 0.0

    # Extract all incident numbers (now returns sets)
    incidents1 = extract_incident_numbers(a)
    incidents2 = extract_incident_numbers(b)

    # Rule 1: If both have incident numbers and there's any overlap
    if incidents1 and incidents2 and incidents1 & incidents2:  # Use & instead of .intersection()
        return 1.0
    else:
        return 0.0


# Fix 4: Update get_incident_id function
def get_incident_id(doc1: dict, doc2: dict) -> tuple:
    """Extract all incident IDs from document names and paths using comprehensive pattern matching."""

    # Extract IDs from document names (now returns sets)
    incident_ids1 = extract_incident_numbers(doc1["gdrive_name"])
    incident_ids2 = extract_incident_numbers(doc2["gdrive_name"])

    # Also extract from paths for better coverage (now returns sets)
    path_ids1 = extract_incident_numbers(doc1["gdrive_path"])
    path_ids2 = extract_incident_numbers(doc2["gdrive_path"])

    # Combine IDs from both sources (union of sets)
    all_ids1 = list(incident_ids1 | path_ids1)  # Convert back to list for comma joining
    all_ids2 = list(incident_ids2 | path_ids2)  # Convert back to list for comma joining

    doc1["incident_id"] = ", ".join(all_ids1) if all_ids1 else None
    doc2["incident_id"] = ", ".join(all_ids2) if all_ids2 else None

    return doc1, doc2


def get_file_extension(document_name: str) -> str:
    """Extract file extension from document name."""
    if not document_name:
        return ""
    name_lower = document_name.lower()

    if name_lower.endswith(".pdf"):
        return ".pdf"

    if "." in name_lower:
        return "." + name_lower.split(".")[-1]

    return ""


def extract_incident_numbers(text: str) -> set:
    """
    Extract all possible case IDs from text with extremely flexible pattern matching.
    Prioritizes recall over precision to catch all possible ID formats.
    Returns a SET instead of list to avoid intersection errors.
    """
    if not text:
        return set()

    id_patterns = [
        # IAD with numbers directly after (IAD552)
        r"((?:IAD|IA|OIA|OAI)[-_]?\d+)",
        # Standard complex format (H-OIA-097-20-A)
        r"\b([A-Z]-[A-Z]{2,4}-\d{1,3}-\d{1,2}-[A-Z])\b",
        # Various case ID formats with year and number
        r"((?:IAD|IA|OIA|OAI)[-_]?\d{2,4}[-_]?\d{1,4})",
        # Case number with hash, more relaxed
        r"Case#([A-Za-z0-9-]+)",
        # Any format like N-BPH-284-18-A
        r"([A-Z]-[A-Z]{2,4}-\d{1,3}-\d{1,2}-[A-Z])",
        # Additional pattern for case numbers that appear between underscores
        r"_((?:IAD|IA|OIA|OAI)\d+)_",
        # Look for case numbers surrounded by non-alphanumeric characters
        r"(?:^|[^a-zA-Z0-9])((?:IAD|IA|OIA|OAI)\d+)(?:$|[^a-zA-Z0-9])",
        # DDDD-DDDD
        r"\b(\d{4}-\d{4})\b",
        r"\b(\d{2,4}-\d{2,4})\b",
    ]

    all_ids = set()  # Changed to set
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

            if clean_id:
                all_ids.add(clean_id)  # Changed to add()

    prefixes = ["IAD", "IA", "OIA", "OAI"]
    for prefix in prefixes:
        prefix_pattern = rf"({prefix}\d+)"
        matches = re.findall(prefix_pattern, text, re.IGNORECASE)
        for match in matches:
            if match:
                all_ids.add(match)  # Changed to add()

    return all_ids  # Returns set


def clean_names(doc1, doc2):
    """Clean name fields for both documents by removing unnamed values."""

    def _clean_single_name(name: str) -> str:
        if not name:
            return None

        names = [n.strip() for n in name.split(",")]

        # Filter out any variations of "unnamed"
        cleaned_names = [n for n in names if "unnamed" not in n.lower()]

        if not cleaned_names:
            return None

        return ", ".join(cleaned_names)

    doc1["name"] = _clean_single_name(doc1.get("name"))
    doc2["name"] = _clean_single_name(doc2.get("name"))

    return doc1, doc2


def is_valid_date(date_str):
    """Check if a string is a valid date in YYYY-MM-DD format."""
    if not date_str or not isinstance(date_str, str):
        return False

    # Check if it matches YYYY-MM-DD pattern
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return False

    # Try to parse it
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def extract_date(text: str) -> list[str]:
    """
    Extract all dates from text with extremely flexible pattern matching.
    Prioritizes recall over precision to catch all possible date formats.
    """
    if not text:
        return []

    date_patterns = [
        # YYYY-MM-DD (standard ISO format)
        r"\b(\d{4}-\d{1,2}-\d{1,2})\b",
        # YYYY-MM (standard ISO format)
        r"\b(\d{4}-\d{1,2})\b",
        # MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY (US format)
        r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{4})\b",
        # MM/DD/YY, MM-DD-YY, MM.DD.YY (short year)
        r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2})\b",
        # YYYY/MM/DD, YYYY.MM.DD (alternate ISO)
        r"\b(\d{4}[\/\.]\d{1,2}[\/\.]\d{1,2})\b",
        # YYYYMMDD (compact format without separators)
        r"\b(\d{4}[01]\d[0-3]\d)\b",
        # Relaxed matching for anything that looks like a date
        # This will catch dates within filenames and without word boundaries
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})",
        # Format like 04.27.21 that might be in filenames
        r"(\d{2}\.\d{2}\.\d{2})",
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


def standardize_date(date_str: str) -> datetime | None:
    """
    Attempt to standardize various date formats into a datetime object.
    Returns None if the date can't be parsed.
    """
    formats = [
        "%Y-%m-%d",  # 2023-01-15
        "%m/%d/%Y",  # 01/15/2023
        "%m-%d-%Y",  # 01-15-2023
        "%m.%d.%Y",  # 01.15.2023
        "%Y/%m/%d",  # 2023/01/15
        "%Y.%m.%d",  # 2023.01.15
        "%Y%m%d",  # 20230115
        "%m/%d/%y",  # 01/15/23
        "%m-%d-%y",  # 01-15-23
        "%m.%d.%y",  # 01.15.23
        "%d.%m.%y",  # 15.01.23
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def get_dates_from_filepath(doc1_text: str, doc2_text: str) -> float:
    """
    Compare dates extracted from text strings and return similarity score.
    """
    doc1_dates = extract_date(doc1_text)
    doc2_dates = extract_date(doc2_text)

    doc1_dt = [dt for dt in (standardize_date(d) for d in doc1_dates) if dt]
    doc2_dt = [dt for dt in (standardize_date(d) for d in doc2_dates) if dt]
    return doc1_dt, doc2_dt


def parse_date_safe(date_str):
    """Safely parse a date string, returning None if invalid."""
    if is_valid_date(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d")
    return None


def calculate_days_difference(date1_str, date2_str):
    """Calculate days difference between two date strings, handling invalid formats."""
    date1 = parse_date_safe(date1_str)
    date2 = parse_date_safe(date2_str)

    if date1 and date2:
        return abs((date1 - date2).days)
    return None


def calculate_edge_weight(doc1: dict, doc2: dict) -> float:
    """Calculate similarity score between two documents based on all available signals."""

    doc1, doc2 = get_incident_id(doc1, doc2)
    doc1, doc2 = clean_names(doc1, doc2)

    doc1_ext = get_file_extension(doc1["document_name"])
    doc2_ext = get_file_extension(doc2["document_name"])

    if doc1_ext and doc2_ext and doc1_ext != ".pdf" and doc2_ext != ".pdf":
        sim_score_fn = calculate_id_similarity(doc1["document_name"], doc2["document_name"])

        if sim_score_fn >= 0.5:
            return 1.0
        else:
            return 0.0

    if doc1_ext and doc2_ext and doc1_ext == ".pdf" and doc2_ext != ".pdf":
        sim_score_fn = calculate_id_similarity(doc1["gdrive_path"], doc2["gdrive_path"])

        if sim_score_fn >= 0.5:
            return 1.0
        else:
            return 0.0

    # Check if documents have names and valid dates
    if doc1.get("name") and doc2.get("name") and doc1.get("date") and doc2.get("date"):
        doc1_names = set(doc1["name"].split(", "))
        doc2_names = set(doc2["name"].split(", "))

        days_diff = calculate_days_difference(doc1["date"], doc2["date"])

        # Only proceed if we have valid dates
        if days_diff is not None and doc1_names & doc2_names and days_diff < 90:
            return 1.0

    if doc1_ext and doc2_ext and doc1_ext == ".pdf" and doc2_ext == ".pdf":
        # block on incident ids
        if doc1.get("incident_id") and doc2.get("incident_id"):
            # Convert incident_ids to sets if they aren't already
            doc1_ids = (
                set([doc1["incident_id"]])
                if isinstance(doc1["incident_id"], (str, int))
                else set(doc1["incident_id"])
            )
            doc2_ids = (
                set([doc2["incident_id"]])
                if isinstance(doc2["incident_id"], (str, int))
                else set(doc2["incident_id"])
            )

            if doc1_ids & doc2_ids:
                return 1.0
            else:
                return 0.0

        # Block on case numbers and check dates
        if (
            doc1.get("case_numbers")
            and doc2.get("case_numbers")
            and doc1.get("date")
            and doc2.get("date")
        ):
            doc1_nums = set(doc1["case_numbers"].split(", "))
            doc2_nums = set(doc2["case_numbers"].split(", "))

            days_diff = calculate_days_difference(doc1["date"], doc2["date"])

            if days_diff is not None and doc1_nums & doc2_nums and days_diff < 90:
                return 1.0
            else:
                return 0.0

        weights = []

        # Date similarity (if both have valid dates)
        days_diff = calculate_days_difference(doc1.get("date"), doc2.get("date"))
        if days_diff is not None:
            # Strong signal if within 30 days, moderate up to 90 days
            if days_diff <= 30:
                date_score = 1.0
            elif days_diff <= 60:
                date_score = 0.7
            elif days_diff <= 90:
                date_score = 0.5
            else:
                date_score = 0.0
            weights.append(date_score)

        # Name similarity (if both have names)
        if doc1.get("name") and doc2.get("name"):
            doc1_case_names = set(doc1["name"].split(", "))
            doc2_case_names = set(doc2["name"].split(", "))

            if doc1_case_names & doc2_case_names:
                weights.append(1.0)
            else:
                weights.append(0.0)

        if not weights:
            return 0.0

        # Return weighted average, with extra weight given to summary similarities
        weighted_sum = sum(w * multiplier for w, multiplier in zip(weights, [1, 1], strict=False))

        return weighted_sum / sum([1, 1][: len(weights)])

    return 0.0


def process_pair(pair, calculate_edge_weight_func):
    """Process a single pair of documents and return edge data if weight is sufficient."""
    doc1, doc2 = pair
    try:
        weight = calculate_edge_weight_func(doc1, doc2)
        if weight and weight > 0.5:  # Only return edges for reasonably strong connections
            return (doc1["id"], doc2["id"], weight)
    except Exception as e:
        print(
            f"Error processing pair {doc1.get('document_name', 'unknown')} and {doc2.get('document_name', 'unknown')}: {str(e)}"
        )
    return None


def preprocess_documents(data: list[dict]) -> list[dict]:
    """Preprocess all documents to extract incident IDs and dates for blocking."""
    print("Preprocessing documents for incident IDs and dates...")

    for doc in data:
        # Extract incident IDs from both filename and path (now returns sets)
        filename_ids = extract_incident_numbers(doc.get("gdrive_name", ""))
        path_ids = extract_incident_numbers(doc.get("gdrive_path", ""))

        # Combine and deduplicate incident IDs (union of sets)
        all_incident_ids = list(filename_ids | path_ids)  # Convert to list

        doc["extracted_incident_ids"] = all_incident_ids

        # Also update the incident_id field to match existing format
        if all_incident_ids:
            doc["incident_id"] = ", ".join(all_incident_ids)
        else:
            doc["incident_id"] = doc.get("incident_id")  # Keep existing value if any

        doc["has_incident_ids"] = len(all_incident_ids) > 0

        # Extract dates from both filename and path
        filename_dates = extract_date(doc.get("gdrive_name", ""))
        path_dates = extract_date(doc.get("gdrive_path", ""))

        # Combine and standardize dates
        all_dates = list(set(filename_dates + path_dates))
        standardized_dates = []
        for date_str in all_dates:
            std_date = standardize_date(date_str)
            if std_date:
                standardized_dates.append(std_date)

        doc["extracted_dates"] = standardized_dates
        doc["has_extracted_dates"] = len(standardized_dates) > 0

        # Also check for existing date field
        doc["has_date_field"] = bool(doc.get("date"))
        doc["has_any_date"] = doc["has_extracted_dates"] or doc["has_date_field"]

    # Print preprocessing statistics
    total_docs = len(data)
    with_incident_ids = sum(1 for doc in data if doc["has_incident_ids"])
    with_case_numbers = sum(1 for doc in data if doc.get("case_numbers"))
    with_incident_field = sum(1 for doc in data if doc.get("incident_id"))
    with_extracted_dates = sum(1 for doc in data if doc["has_extracted_dates"])
    with_date_field = sum(1 for doc in data if doc["has_date_field"])

    print("Preprocessing complete:")
    print(f"  Total documents: {total_docs}")
    print(
        f"  With extracted incident IDs: {with_incident_ids} ({with_incident_ids / total_docs * 100:.1f}%)"
    )
    print(
        f"  With case_numbers field: {with_case_numbers} ({with_case_numbers / total_docs * 100:.1f}%)"
    )
    print(
        f"  With incident_id field: {with_incident_field} ({with_incident_field / total_docs * 100:.1f}%)"
    )
    print(
        f"  With extracted dates: {with_extracted_dates} ({with_extracted_dates / total_docs * 100:.1f}%)"
    )
    print(f"  With date field: {with_date_field} ({with_date_field / total_docs * 100:.1f}%)")

    return data


def should_compare_documents(doc1: dict, doc2: dict) -> bool:
    """
    Determine if two documents should be compared based on blocking rules.
    Returns True if documents should be compared, False if they should be blocked.
    """

    # Block 1: incident_id field presence mismatch
    if bool(doc1.get("incident_id")) != bool(doc2.get("incident_id")):
        return False

    # Block 2: case_numbers field presence mismatch
    if bool(doc1.get("case_numbers")) != bool(doc2.get("case_numbers")):
        return False

    # Block 3: Extracted incident IDs presence mismatch
    if doc1["has_incident_ids"] != doc2["has_incident_ids"]:
        return False

    # Block 4: If both have extracted incident IDs, check for overlap
    if doc1["has_incident_ids"] and doc2["has_incident_ids"]:
        ids1_set = set(doc1["extracted_incident_ids"])
        ids2_set = set(doc2["extracted_incident_ids"])

        # If they have incident IDs but no overlap, block
        if not (ids1_set & ids2_set):
            return False

    # Block 5: Date-based blocking
    # Get all available dates for each document
    doc1_dates = []
    doc2_dates = []

    # Add extracted dates
    if doc1["has_extracted_dates"]:
        doc1_dates.extend(doc1["extracted_dates"])
    if doc2["has_extracted_dates"]:
        doc2_dates.extend(doc2["extracted_dates"])

    # If both have dates, check if any are within reasonable range
    if doc1_dates and doc2_dates:
        min_days_diff = float("inf")
        for d1 in doc1_dates:
            for d2 in doc2_dates:
                days_diff = abs((d1 - d2).days)
                min_days_diff = min(min_days_diff, days_diff)

        # Block if all dates are more than 180 days apart
        if min_days_diff > 180:
            return False

    return True


async def process_pairs_generator(doc_pairs, max_workers=50, batch_size=10000):
    """Generator that yields edge results in batches to avoid memory accumulation."""
    total_batches = (len(doc_pairs) + batch_size - 1) // batch_size
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_pair, calculate_edge_weight_func=calculate_edge_weight)

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(doc_pairs))
            batch_pairs = doc_pairs[batch_start:batch_end]

            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(executor, process_func, pair) for pair in batch_pairs]

            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Yield results immediately instead of accumulating
                batch_edges = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"Warning: Task failed: {result}")
                    elif result is not None:
                        batch_edges.append(result)

                # Progress reporting
                elapsed = time.time() - start_time
                processed_pairs = (batch_idx + 1) * batch_size
                rate = processed_pairs / elapsed if elapsed > 0 else 0
                remaining_pairs = len(doc_pairs) - processed_pairs
                eta_seconds = remaining_pairs / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60

                print(
                    f"Batch {batch_idx + 1}/{total_batches} complete. "
                    f"Found {len(batch_edges)} edges this batch. "
                    f"Rate: {rate:.1f} pairs/sec. "
                    f"ETA: {eta_minutes:.1f} minutes"
                )

                # Yield the batch results
                if batch_edges:
                    yield batch_edges

                # Force garbage collection every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    import gc

                    gc.collect()
                    print(f"  Memory cleanup at batch {batch_idx + 1}")

            except Exception as e:
                print(f"Batch {batch_idx + 1} processing error: {e}")
                continue


async def cluster_results(
    data: list[dict], historical_data: dict = None, max_workers: int = None
) -> list[dict]:
    """Memory-efficient clustering using generator pattern to avoid accumulating results."""

    # Preprocess documents to extract incident IDs and dates
    data = preprocess_documents(data)

    G = nx.Graph()
    for doc in data:
        G.add_node(doc["id"], **doc)

    # Create provisional case mapping if historical data is available
    provisional_cases = {}
    if historical_data is not None and all(
        k in historical_data for k in ["gdrive_name", "provisional_case_name"]
    ):
        for name, prov_name in zip(
            historical_data["gdrive_name"], historical_data["provisional_case_name"], strict=False
        ):
            if not pd.isna(prov_name) and prov_name not in ["", None]:
                provisional_cases[name] = prov_name

    # Generate document pairs with enhanced blocking
    doc_pairs = []
    blocked_counts = {"total_blocked": 0}

    print("Generating document pairs with blocking rules...")

    for i, doc1 in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing document {i}/{len(data)}")

        doc1_name = doc1.get("document_name", "")
        doc1_case = provisional_cases.get(doc1_name)

        for doc2 in data[i + 1 :]:
            doc2_name = doc2.get("document_name", "")
            doc2_case = provisional_cases.get(doc2_name)

            # Skip pairs that are already known to be in the same provisional case
            if doc1_case and doc2_case and doc1_case == doc2_case:
                G.add_edge(doc1["id"], doc2["id"], weight=0.5)
                continue

            # Apply blocking rules
            if not should_compare_documents(doc1, doc2):
                blocked_counts["total_blocked"] += 1
                continue

            doc_pairs.append((doc1, doc2))

    print("\nBlocking results:")
    print(f"  Total possible pairs: {len(data) * (len(data) - 1) // 2}")
    print(f"  Pairs after blocking: {len(doc_pairs)}")
    print(f"  Pairs blocked: {blocked_counts['total_blocked']}")
    print(
        f"  Blocking efficiency: {blocked_counts['total_blocked'] / (len(data) * (len(data) - 1) // 2) * 100:.1f}%"
    )

    # Process pairs using generator to avoid memory accumulation
    max_workers = 50
    batch_size = 10000  # Smaller batches

    print(
        f"Processing {len(doc_pairs)} document pairs with {max_workers} workers in batches of {batch_size}"
    )
    print("Using memory-efficient generator pattern...")

    total_edges_added = 0

    # Process pairs in batches and add edges immediately
    async for batch_edges in process_pairs_generator(doc_pairs, max_workers, batch_size):
        # Add edges to graph immediately to avoid accumulating in memory
        for edge_data in batch_edges:
            doc1_id, doc2_id, weight = edge_data
            G.add_edge(doc1_id, doc2_id, weight=weight)
            total_edges_added += 1

        # Optional: Save intermediate progress every 1000 batches
        if total_edges_added % 1000 == 0:
            print(f"  Added {total_edges_added} edges to graph so far")

    print(f"Added {total_edges_added} total edges to graph")

    # Find connected components (clusters)
    print("Computing connected components...")
    clusters = list(nx.connected_components(G))
    node_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = i

    print(f"Initial clustering from similarity: {len(clusters)} clusters")

    # Apply incident ID grouping with consistent extraction
    print("Applying incident ID grouping...")
    incident_groups = {}
    for doc in data:
        path_incidents = extract_incident_numbers(doc["gdrive_path"])
        doc_name_incidents = extract_incident_numbers(doc["document_name"])

        all_incidents = path_incidents.union(doc_name_incidents)

        if all_incidents:
            for incident_id in all_incidents:
                if incident_id not in incident_groups:
                    incident_groups[incident_id] = []
                incident_groups[incident_id].append(doc["id"])

    print(f"Found {len(incident_groups)} incident groups")

    # Update clusters based on incident groups
    next_cluster_id = max(node_to_cluster.values()) + 1 if node_to_cluster else 0
    merged_groups = 0

    for incident_id, incident_docs in incident_groups.items():
        if len(incident_docs) > 1:
            current_clusters = {node_to_cluster.get(doc_id, -1) for doc_id in incident_docs}

            if len(current_clusters) > 1:
                for doc_id in incident_docs:
                    node_to_cluster[doc_id] = next_cluster_id
                merged_groups += 1
                next_cluster_id += 1

    print(f"Merged {merged_groups} incident groups")

    # Apply provisional case name refinement
    if provisional_cases:
        print("Applying provisional case refinement...")
        prov_case_groups = defaultdict(list)
        for doc in data:
            doc_name = doc.get("document_name", "")
            if doc_name in provisional_cases:
                prov_case = provisional_cases[doc_name]
                if prov_case:
                    prov_case_groups[prov_case].append(doc["id"])

        merged_prov_cases = 0
        for prov_case, doc_ids in prov_case_groups.items():
            if len(doc_ids) > 1:
                clusters_involved = {node_to_cluster.get(doc_id, -1) for doc_id in doc_ids}

                if len(clusters_involved) > 1:
                    for doc_id in doc_ids:
                        node_to_cluster[doc_id] = next_cluster_id
                    merged_prov_cases += 1
                    next_cluster_id += 1

        print(f"Merged {merged_prov_cases} provisional case groups")

    # Build final result
    print("Building final results...")
    result = []
    for doc in data:
        doc_copy = doc.copy()
        doc_copy["Parent Clusters"] = [node_to_cluster.get(doc["id"], -1)]

        doc_name = doc.get("document_name", "")
        if doc_name in provisional_cases:
            doc_copy["provisional_case_name"] = provisional_cases[doc_name]

        result.append(doc_copy)

    final_clusters = len(set(node_to_cluster.values()))
    print(f"Clustering complete: {final_clusters} clusters formed")
    return result


def load_historical_data(historical_csv_path: str | None = None) -> dict | None:
    """Load historical/provisional case data if available."""
    if not historical_csv_path or not Path(historical_csv_path).exists():
        return None

    print(f"Loading historical data from: {historical_csv_path}")
    df_historical = pd.read_csv(historical_csv_path)

    return {
        "gdrive_name": df_historical.get("gdrive_name", []).tolist(),
        "provisional_case_name": df_historical.get("provisional_case_name", []).tolist(),
    }


def load_csv_data(csv_path: str) -> list[dict[str, Any]]:
    """Load CSV data and convert to list of dictionaries for clustering."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} documents")
    print(f"Columns: {list(df.columns)}")

    # Convert DataFrame to list of dictionaries
    documents = []
    for idx, row in df.iterrows():
        doc = {
            "id": str(idx),  # Use row index as unique ID
            "document_name": row.get("gdrive_name", ""),
            "gdrive_path": row.get("gdrive_path", ""),
            "gdrive_name": row.get("gdrive_name", ""),
            "date": row.get("incident_date", None),
            "name": row.get("subject_name", None),
            "case_numbers": row.get("case_numbers", None),
        }

        # Handle NaN values by converting to None
        for key, value in doc.items():
            if pd.isna(value):
                doc[key] = None

        documents.append(doc)

    return documents


def save_clustering_results(results: list[dict], output_path: str):
    """Save clustering results to CSV file."""
    # Convert results back to DataFrame
    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print clustering summary
    cluster_counts = (
        df_results["Parent Clusters"]
        .apply(lambda x: x[0] if isinstance(x, list) else x)
        .value_counts()
    )
    print("\nClustering Summary:")
    print(f"Total documents: {len(df_results)}")
    print(f"Total clusters: {len(cluster_counts)}")
    print(f"Largest cluster size: {cluster_counts.max()}")
    print(f"Average cluster size: {cluster_counts.mean():.2f}")

    # Show clusters with more than 1 document
    multi_doc_clusters = cluster_counts[cluster_counts > 1]
    if len(multi_doc_clusters) > 0:
        print("\nClusters with multiple documents:")
        for cluster_id, count in multi_doc_clusters.head(10).items():
            print(f"  Cluster {cluster_id}: {count} documents")


async def main():
    """Main async function to run document clustering with hardcoded configuration."""
    try:
        print("=== DOCUMENT CLUSTERING SCRIPT (ASYNC) ===")
        print(f"Input file: {CSV_PATH}")
        print(f"Output file: {OUTPUT_PATH}")
        print(f"Historical data: {HISTORICAL_CSV_PATH}")
        print(f"Debug mode: {DEBUG}")
        print("=" * 40)

        # Load the main dataset
        print("Loading main dataset...")
        documents = load_csv_data(CSV_PATH)
        print(f"Loaded {len(documents)} documents")

        if DEBUG:
            print("\nSample document:")
            print(json.dumps(documents[0], indent=2, default=str))

        # Load historical data if provided
        print("Loading historical data...")
        historical_data = load_historical_data(HISTORICAL_CSV_PATH)

        if historical_data:
            print(f"Historical data loaded with {len(historical_data['gdrive_name'])} entries")
        else:
            print("No historical data loaded")

        # Run clustering asynchronously
        print("\nStarting async clustering process...")
        start_time = asyncio.get_event_loop().time()

        clustered_results = await cluster_results(documents, historical_data)

        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        print(f"Clustering completed in {processing_time:.2f} seconds")

        # Save results
        print("Saving results...")
        save_clustering_results(clustered_results, OUTPUT_PATH)

        # Print summary statistics
        cluster_ids = [doc.get("Parent Clusters", [-1])[0] for doc in clustered_results]
        unique_clusters = len(set(cluster_ids))
        unclustered = sum(1 for cid in cluster_ids if cid == -1)

        print("\n=== CLUSTERING SUMMARY ===")
        print(f"Total documents processed: {len(clustered_results)}")
        print(f"Clusters formed: {unique_clusters}")
        print(f"Unclustered documents: {unclustered}")
        print(f"Average cluster size: {len(clustered_results) / max(unique_clusters, 1):.1f}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Results saved to: {OUTPUT_PATH}")
        print("=" * 40)

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {str(e)}")
        print(f"Make sure {CSV_PATH} exists in the current directory")
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def run_clustering():
    """Wrapper function to run the async main function."""
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClustering interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
