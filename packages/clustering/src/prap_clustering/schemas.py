"""Top-level pydantic schemas re-exported from the clustering subpackages.

This is a convenience surface — the canonical definitions still live in
the module that uses them.
"""

from pydantic import BaseModel, Field

from .clustering.prompts import SummaryComparisonResult
from .clustering.regex_extract_fp_fn import CaseNumbers, IncidentDate, Names


class ScoreReport(BaseModel):
    """Single result-set evaluation against ground truth."""

    results_csv: str = Field(description="Path to the evaluated results CSV.")
    ground_truth_column: str = Field(
        default="provisional_case_name",
        description="Column in the CSV that carries the ground-truth cluster label.",
    )
    cluster_column: str = Field(
        description="Detected cluster column from the test results (e.g. 'Parent Clusters')."
    )
    num_groundtruth_clusters: int
    num_test_clusters: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    clusters_with_splits: int = 0
    clusters_with_merges: int = 0


__all__ = [
    "CaseNumbers",
    "IncidentDate",
    "Names",
    "ScoreReport",
    "SummaryComparisonResult",
]
