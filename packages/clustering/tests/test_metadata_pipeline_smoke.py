"""Smoke tests for the metadata_pipeline sub-package."""

from __future__ import annotations


def test_metadata_pipeline_imports():
    from prap_clustering.metadata_pipeline.clustering import (  # noqa: F401
        cluster,
        cluster_deterministic,
        cluster_graph,
    )
    from prap_clustering.metadata_pipeline.feature_extraction import (  # noqa: F401
        extract_filename_features,
        regex_extract_fp_fn,
    )


def test_metadata_regex_extraction_runs():
    from prap_clustering.metadata_pipeline.feature_extraction.regex_extract_fp_fn import (
        extract_date_from_metadata,
        extract_ids_from_metadata,
        extract_names_from_metadata,
    )

    dates = extract_date_from_metadata("Agency/2023-01-15_report.pdf")
    ids = extract_ids_from_metadata("IA2018-0167_report.pdf")
    names = extract_names_from_metadata("Officer John Smith report.pdf")
    assert dates is not None
    assert ids is not None
    assert names is not None


def test_metadata_cluster_main_callables():
    from prap_clustering.metadata_pipeline.clustering.cluster import run_clustering
    from prap_clustering.metadata_pipeline.clustering.cluster_deterministic import main as det_main
    from prap_clustering.metadata_pipeline.feature_extraction.extract_filename_features import (
        main as features_main,
    )

    assert callable(run_clustering)
    assert callable(det_main)
    assert callable(features_main)
