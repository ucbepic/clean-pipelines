"""Domain schemas for the entity-resolution pipeline.

Ported from the npi-api `shared/models.py`. Only the types the resolve pipeline
actually uses live here (the API-server request/response models stayed with the API).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel


class AgencyType(str, Enum):
    POLICE = "POLICE"
    CORRECTIONS = "CORRECTIONS"


class PostEmploymentRecord(BaseModel):
    """A POST employment record (one stint) as returned by the NPI API."""

    post_person_nbr: str
    post_first_name: str
    post_middle_name: str | None = None
    post_last_name: str
    post_suffix: str | None = None
    post_agency_name: str
    post_agency_type: AgencyType = AgencyType.POLICE
    post_start_date: datetime | None = None
    post_end_date: datetime | None = None
    post_separation_reason: str | None = None
    state: str | None = None
    county: str | None = None


class OfficerMention(BaseModel):
    """An officer mention extracted from an incident report (pipeline input)."""

    mention_uid: str
    mention_agency_type: AgencyType = AgencyType.POLICE
    mention_incident_date: date
    mention_first_name: str | None = None
    mention_middle_name: str | None = None
    mention_suffix: str | None = None
    mention_last_name: str
    mention_rank: str | None = None
    mention_agency: str | None = None
    state: str | None = None
    mentioned_agencies: str | None = ""


class RunResult(BaseModel):
    """Summary returned by the resolve `run(...)` step."""

    n_mentions: int
    output_path: str
