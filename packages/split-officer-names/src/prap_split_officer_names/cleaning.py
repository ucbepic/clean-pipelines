"""Rules-based name cleaning + obvious-invalidity checks.

Provides rank lists and the strip_rank_prefix / clean_name_string /
is_obviously_invalid / clean_officer_name / drop_incomplete_names helpers.
"""

from __future__ import annotations

import re

import pandas as pd

MULTI_WORD_RANKS = [
    "Police Officer III+1",
    "Police Officer III",
    "Police Officer II",
    "Police Officer I",
    "Police Officer",
    "Fish and Wildlife Officer",
    "BART Officer",
    "K-9 Handler",
    "K-9 Officer",
    "K9 Handler",
    "K9 Officer",
    "Deputy U.S. Marshal",
    "U.S. Marshal",
    "S. Marshal",
    "RPD Officer",
    "RPD Sergeant",
    "Corrections Officer",
    "Correctional Officer",
    "Sr. DSO",
    "Sr. Officer",
    "Sr. Deputy",
    "Sergeant I",
    "DSO",
    "DO",
    "Police Department K-9 Officer",
    "Police Department Officers",
    "Desert Hot Springs Police Officer",
    "South Gate Police Officer",
    "Bluff Police Sergeant",
    "Chapman University Public Safety Officer",
    "Sr. Deputy",
    "Security Guard",
    "Police Officer",
]

SINGLE_WORD_RANKS = [
    "Officer", "Sergeant", "Detective", "Corporal", "Deputy", "LAPD", "Traffic",
    "Unnamed", "Senior", "Lieutenant", "Correctional", "Director", "PSO", "K9",
    "Special", "CHP", "Security", "Recruit", "CO", "DSOs", "SDSO", "SOIs",
    "Investigator", "DPO", "USBP", "Reserve", "Custody", "JIO", "DJCO", "Sheriff",
    "Does", "unnamed", "Additional", "Per", "Sacramento", "Responding", "CD", "FTO",
    "Agent", "First", "Chief", "Field", "Deputies", "Parole", "Four", "Hemet",
    "SACMCTF", "Assistant", "Team", "CA", "Other", "Supervisor", "Tactical", "Captain",
    "The", "Placer", "Acting", "PCO", "Detention", "RSO", "District", "SDPO", "Orange",
    "MCSO", "SPD", "Explorer", "Master", "E3", "JTO", "Unidentified", "SWAT",
    "California", "Inspector", "SOT", "Three", "President", "Sgt", "Unit", "FBI",
    "Probation", "other", "Warden", "Elk", "Patrol", "Unspecified", "DOES", "MTA",
    "Ranger", "PCSO", "CSI", "OIC", "State", "Unknown", "COIV", "RCPD", "DPC", "five",
    "Supervising", "Claremont", "Two", "Group", "Main", "Cal", "Game", "Right",
    "Antioch", "SAPD", "Officers", "Canine", "METRO", "MPD", "Anaheim", "Trainee",
    "San", "Stockton", "Initial", "DA", "PO", "the", "Jailer", "Uniformed", "SEB",
    "Cover", "Fire", "Detentions", "National", "Red", "Park", "Multiple", "Branch",
    '"Redacted"', "Sgt.", "Lt.", "Cpl.", "Det.", "Ofc.", "S/O", "P/O", "D/S", "FNU",
    "LNU", "II", "III", "IV", "El", "PC", "Constable", "Dep.", "Marshal", "Leader",
    "Specialist", "Counselor", "Witness",
]  # fmt: skip

_MULTI_WORD_NORMALIZED = [r.lower().strip() for r in MULTI_WORD_RANKS]
_SINGLE_WORD_NORMALIZED = {r.lower().strip() for r in SINGLE_WORD_RANKS}


def strip_rank_prefix(name_string: str) -> str:
    """Remove rank/title prefixes from an officer-name string."""
    if pd.isna(name_string) or not name_string:
        return ""
    name = name_string.strip()
    name_lower = name.lower()
    for rank_lower in _MULTI_WORD_NORMALIZED:
        if name_lower.startswith(rank_lower + " "):
            remainder = name[len(rank_lower) :].strip()
            return remainder if remainder else ""
    first_word = name.split(" ", 1)[0]
    if first_word.lower().strip() in _SINGLE_WORD_NORMALIZED:
        return name.split(" ", 1)[1] if " " in name else ""
    return name


def clean_name_string(name_string: str) -> str:
    """Strip badge numbers, bracketed/parenthetical noise, FNU/LNU, whitespace."""
    if pd.isna(name_string) or not name_string:
        return ""
    name = name_string.strip()
    name = re.sub(r"#\d+", "", name)
    name = re.sub(r"\b\d+\.?\d*\b", "", name)
    name = re.sub(r"\bFNU\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bLNU\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[,\.]\s*$", "", name).strip()
    return name


def is_obviously_invalid(name_string: str) -> bool:
    """Rules-based quick rejection so the LLM is not called on garbage strings."""
    if not name_string or not name_string.strip():
        return True
    name = name_string.strip().lower()
    name_orig = name_string.strip()

    if " " not in name_orig and "," not in name_orig:
        return True
    if re.match(r"^[#\d\.\s]+$", name_orig):
        return True
    if re.match(r"^k-?9\s+", name, re.IGNORECASE):
        return True

    placeholder_keywords = ["redacted", "unknown", "not provided", "unidentified", "unnamed"]
    if any(kw in name for kw in placeholder_keywords):
        return True

    generic_keywords = [
        "officers",
        "detectives",
        "deputies",
        "personnel",
        "agents",
        "does 1",
        "does through",
        "referred to as",
        "division officers",
        "team members",
        "response team",
        "pd detectives",
        "pd officers",
        "staff members",
    ]
    if any(kw in name for kw in generic_keywords):
        return True

    invalid_patterns = [
        r"^\w\.\s+\w+$",
        r"^[A-Z]\.\s",
        r",\s*[A-Z]\.$",
        r"^on\s+\w+$",
        r"^\d+\s+\w+$",
        r"involved\s+in",
        r"^exact\s+",
        r"\'s\s+team$",
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, name_orig, re.IGNORECASE):
            return True
    return False


def clean_officer_name(name_string: str) -> str:
    """Full pipeline: strip rank, clean noise, strip rank again."""
    if pd.isna(name_string) or not name_string:
        return ""
    name = strip_rank_prefix(name_string)
    name = clean_name_string(name)
    name = strip_rank_prefix(name)
    return name


def drop_incomplete_names(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with a multi-word or comma-containing cleaned_name."""
    df = df.copy()
    df.loc[:, "cleaned_name"] = (
        df.cleaned_name.fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    )
    mask = (df.cleaned_name != "") & (
        df.cleaned_name.str.contains(" ", na=False) | df.cleaned_name.str.contains(",", na=False)
    )
    return df[mask]
