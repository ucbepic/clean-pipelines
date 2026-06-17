"""
Dynamic frequency-based filtering for features that appear in too many documents.

This module identifies and filters out non-discriminative features:
- Dates appearing in 300+ documents (metadata/processing dates)
- Names appearing in 300+ documents (organizational fragments)
- Case IDs appearing in 50+ documents (form templates)

These features cause massive false positive merges and must be filtered out.
"""

from collections import Counter

import pandas as pd

from .helpers import (
    normalize_case_ids,
    normalize_dates,
    normalize_names,
    parse_feature_list,
    parse_structured_case_ids,
)


class FrequencyFilter:
    """
    Precompute feature frequencies and filter out non-discriminative features.

    This filter is built from the full dataset before clustering begins.

    Conservative approach: Case ID filtering is optional (set to None to disable).
    """

    def __init__(
        self,
        date_threshold: int = 300,
        name_threshold: int = 300,
        case_id_threshold: int = None  # None = disabled (conservative)
    ):
        """
        Initialize frequency filter with thresholds.

        Args:
            date_threshold: Dates appearing in this many docs are filtered (default: 300)
            name_threshold: Names appearing in this many docs are filtered (default: 300)
            case_id_threshold: Case IDs appearing in this many docs are filtered (default: None = disabled)
        """
        self.date_threshold = date_threshold
        self.name_threshold = name_threshold
        self.case_id_threshold = case_id_threshold

        # These get populated by build_from_dataframe()
        self.blocked_dates: set[str] = set()
        self.blocked_names: set[str] = set()
        self.blocked_case_ids: set[str] = set()

        # Statistics for reporting
        self.date_frequencies: dict[str, int] = {}
        self.name_frequencies: dict[str, int] = {}
        self.case_id_frequencies: dict[str, int] = {}

    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Build frequency filter from a dataframe.

        Analyzes all Tier 1 features (filepath + filename) and identifies
        non-discriminative features to block.

        Args:
            df: DataFrame with extracted_dates_fp/fn, extracted_names_fp/fn,
                extracted_case_ids_fp/fn columns
        """
        print(f"\nBuilding frequency filter from {len(df)} documents...")
        if self.case_id_threshold is not None:
            print(f"Thresholds: dates={self.date_threshold}, names={self.name_threshold}, case_ids={self.case_id_threshold}")
        else:
            print(f"Thresholds: dates={self.date_threshold}, names={self.name_threshold}, case_ids=disabled")

        # Count date frequencies
        date_counter = Counter()
        for _, row in df.iterrows():
            fp_dates = normalize_dates(parse_feature_list(row.get("extracted_dates_fp")))
            fn_dates = normalize_dates(parse_feature_list(row.get("extracted_dates_fn")))
            date_counter.update(fp_dates | fn_dates)

        # Count name frequencies
        name_counter = Counter()
        for _, row in df.iterrows():
            fp_names = normalize_names(parse_feature_list(row.get("extracted_names_fp")))
            fn_names = normalize_names(parse_feature_list(row.get("extracted_names_fn")))
            name_counter.update(fp_names | fn_names)

        # Count case ID frequencies (only if enabled)
        # Include BOTH Tier 1 (filepath/filename) AND Tier 2 (LLM-extracted) case IDs
        case_id_counter = Counter()
        if self.case_id_threshold is not None:
            for _, row in df.iterrows():
                # Tier 1: filepath + filename
                fp_ids = normalize_case_ids(parse_feature_list(row.get("extracted_case_ids_fp")))
                fn_ids = normalize_case_ids(parse_feature_list(row.get("extracted_case_ids_fn")))
                # Tier 2: LLM-extracted structured case IDs
                llm_ids = normalize_case_ids(parse_structured_case_ids(row.get("extracted_case_ids_llm_structured")))
                case_id_counter.update(fp_ids | fn_ids | llm_ids)

        # Store frequencies
        self.date_frequencies = dict(date_counter)
        self.name_frequencies = dict(name_counter)
        self.case_id_frequencies = dict(case_id_counter)

        # Identify features to block
        self.blocked_dates = {
            date for date, count in date_counter.items()
            if count >= self.date_threshold
        }
        self.blocked_names = {
            name for name, count in name_counter.items()
            if count >= self.name_threshold
        }
        if self.case_id_threshold is not None:
            self.blocked_case_ids = {
                cid for cid, count in case_id_counter.items()
                if count >= self.case_id_threshold
            }
        else:
            self.blocked_case_ids = set()  # Empty set when disabled

        # Report
        print("\nFrequency filter built:")
        print(f"  Blocking {len(self.blocked_dates)} high-frequency dates (>={self.date_threshold} docs)")
        if self.blocked_dates:
            for date in sorted(self.blocked_dates):
                count = self.date_frequencies[date]
                pct = count / len(df) * 100
                print(f"    - {date}: {count} docs ({pct:.1f}%)")

        print(f"  Blocking {len(self.blocked_names)} high-frequency names (>={self.name_threshold} docs)")
        if self.blocked_names:
            for name in sorted(self.blocked_names):
                count = self.name_frequencies[name]
                pct = count / len(df) * 100
                print(f"    - '{name}': {count} docs ({pct:.1f}%)")

        if self.case_id_threshold is not None:
            print(f"  Blocking {len(self.blocked_case_ids)} high-frequency case IDs (>={self.case_id_threshold} docs)")
            if self.blocked_case_ids:
                for cid in sorted(self.blocked_case_ids):
                    count = self.case_id_frequencies[cid]
                    pct = count / len(df) * 100
                    print(f"    - {cid}: {count} docs ({pct:.1f}%)")
        else:
            print("  Case ID filtering: DISABLED (conservative approach)")

    def filter_dates(self, dates: set[str]) -> set[str]:
        """Remove high-frequency dates from a set."""
        return dates - self.blocked_dates

    def filter_names(self, names: set[str]) -> set[str]:
        """Remove high-frequency names from a set."""
        return names - self.blocked_names

    def filter_case_ids(self, case_ids: set[str]) -> set[str]:
        """Remove high-frequency case IDs from a set."""
        return case_ids - self.blocked_case_ids
