"""Smoke tests for the embeddings_pipeline sub-package."""

from __future__ import annotations


def test_embeddings_pipeline_imports():
    from prap_clustering.embeddings_pipeline.cluster import (  # noqa: F401
        cluster,
        cluster_embeddings,
    )

    # NB: ``eda`` is a top-level script that reads a CSV at import time and is
    # not safe to import in tests; we only import the library-safe modules.
    from prap_clustering.embeddings_pipeline.feature_extraction import (  # noqa: F401
        generate_embeddings,
    )


def test_embeddings_pipeline_main_callables():
    from prap_clustering.embeddings_pipeline.cluster.cluster_embeddings import main as cluster_main
    from prap_clustering.embeddings_pipeline.feature_extraction.generate_embeddings import (
        main as features_main,
    )

    assert callable(features_main)
    assert callable(cluster_main)


def test_cluster_embeddings_pure_helpers():
    """parse_embedding_safe + compute_cosine_similarity_matrix run on synthetic data."""
    import numpy as np
    from prap_clustering.embeddings_pipeline.cluster.cluster_embeddings import (
        compute_cosine_similarity_matrix,
        parse_embedding_safe,
    )

    arr = parse_embedding_safe("[0.1, 0.2, 0.3]")
    assert arr is not None
    assert len(arr) == 3

    embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    sim = compute_cosine_similarity_matrix(embeddings)
    assert sim.shape == (3, 3)
