"""
Feature extraction prompts.

Each feature module exports a PROMPTS dictionary with:
- summarization: 7 prompts for creating bulletpoint summaries
- extraction: 3 prompts for extracting specific values
- citations: 2 prompts for finding supporting evidence
"""

from .case_ids import PROMPTS as CASE_IDS_PROMPTS
from .dates import PROMPTS as DATES_PROMPTS
from .officer_names import PROMPTS as OFFICER_NAMES_PROMPTS
from .subject_names import PROMPTS as SUBJECT_NAMES_PROMPTS

__all__ = ['DATES_PROMPTS', 'OFFICER_NAMES_PROMPTS', 'SUBJECT_NAMES_PROMPTS', 'CASE_IDS_PROMPTS']
