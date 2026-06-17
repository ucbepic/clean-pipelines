"""Smoke tests for prap-clustering — no real LLM calls.

Verifies that all public modules import cleanly and that a few pure helpers
work on synthetic data. The clustering pipeline itself requires async LLM
calls + a real CSV, so we do not exercise the full pipeline here.
"""

from __future__ import annotations

import pandas as pd


def test_imports():
    import prap_clustering  # noqa: F401
    import prap_clustering.embeddings  # noqa: F401
    import prap_clustering.evaluation  # noqa: F401
    import prap_clustering.metrics  # noqa: F401
    import prap_clustering.schemas  # noqa: F401
    from prap_clustering.clustering import (  # noqa: F401
        directory_depth_analyzer,
        frequency_filter,
        helpers,
        hybrid_handengineered,
        hybrid_ml,
        regex_extract_fp_fn,
        singleton_merge_validator,
    )
    from prap_clustering.clustering.prompts import (  # noqa: F401
        SUMMARY_COMPARISON_PROMPT,
        SummaryComparisonResult,
        validate_summary_comparison_response,
    )


def test_summary_comparison_prompt_loaded():
    from prap_clustering.clustering.prompts import SUMMARY_COMPARISON_PROMPT

    assert "{{ summary_1 }}" in SUMMARY_COMPARISON_PROMPT
    assert "{{ summary_2 }}" in SUMMARY_COMPARISON_PROMPT


def test_validate_summary_comparison_response():
    from prap_clustering.clustering.prompts import validate_summary_comparison_response

    assert validate_summary_comparison_response('{"similarity": 1.0, "reasoning": "x"}') == 1.0
    assert validate_summary_comparison_response('{"similarity": 0.5, "reasoning": "x"}') == 0.5
    assert validate_summary_comparison_response('{"similarity": 0.0, "reasoning": "x"}') == 0.0


def test_helpers_normalize():
    from prap_clustering.clustering.helpers import (
        normalize_case_ids,
        parse_feature_list,
    )

    # parse_feature_list handles JSON-list strings + list-repr strings.
    assert parse_feature_list("['a', 'b']") == ["a", "b"]
    assert parse_feature_list('["a", "b"]') == ["a", "b"]

    # normalize_case_ids returns a set of normalized strings.
    out = normalize_case_ids(["IA2018-0167", "ia2018-0167"])
    assert isinstance(out, set)
    assert len(out) >= 1


def test_load_ablation_config():
    from prap_clustering.clustering.hybrid_handengineered import load_ablation_config

    cfg = load_ablation_config("baseline_v2")
    assert cfg.name == "baseline_v2"


def test_embeddings_module_surface():
    # Just verify the symbols exist without loading the actual model.
    from prap_clustering.embeddings import (
        EMBEDDING_MODEL_NAME,
        cosine_similarity,
        embed_texts,
        embed_texts_async,
    )

    assert EMBEDDING_MODEL_NAME.startswith("sentence-transformers/")
    assert callable(embed_texts)
    assert callable(embed_texts_async)
    assert callable(cosine_similarity)


def test_regex_extraction_runs():
    from prap_clustering.clustering.regex_extract_fp_fn import (
        extract_date_from_metadata,
        extract_ids_from_metadata,
    )

    # Just check the functions don't crash on representative input.
    dates = extract_date_from_metadata("Agency/2023-01-15_report.pdf")
    ids = extract_ids_from_metadata("IA2018-0167_report.pdf")
    assert dates is not None
    assert ids is not None


def test_metrics_evaluate_clusters_on_synthetic():
    from prap_clustering.metrics import evaluate_clusters

    # evaluate_clusters has a specific signature; just verify it's callable.
    assert callable(evaluate_clusters)
    _ = pd  # silence the unused-import warning if dataframes are removed

