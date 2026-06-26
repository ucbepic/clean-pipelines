"""
ML-based ablation clustering: joint and cascade models.

Two test designs:

  Test 1 — Joint model (use_learned_model=True):
    A single classifier trained on tier1_only, tier2_only, or both atomic features.
    6 configs: 2 model types (DT, RF) × 3 feature subsets.
    Answers: are Tier 1 and Tier 2 signals complementary for a joint classifier?

  Test 2 — Cascade model (use_cascade_model=True):
    Two separately-trained models deployed sequentially:
      - Tier 1 model runs first; if it predicts no match, Tier 2 model gets a chance.
    3 configs: DT cascade, RF cascade, LightGBM cascade.
    Reuses the tier1_only and tier2_only models from Test 1 (no extra training needed).
    Answers: does the cascade architecture generalise with learned rules?
"""

import itertools
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml
from tqdm import tqdm

from .hybrid_handengineered import (
    CASE_ID_FREQUENCY_THRESHOLD_PERCENT,
    DATE_FREQUENCY_THRESHOLD_PERCENT,
    ENABLE_DIRECTORY_FALLBACK,
    ENABLE_VALIDATION,
    NAME_FREQUENCY_THRESHOLD_PERCENT,
    TIER1_MIN_SHARED_DIRS,
    VALIDATION_THRESHOLD,
    FrequencyFilter,
    MatchResult,
    extract_document_features,
    extract_pair_features_for_model,
    load_trained_model,
    merge_singletons_by_directory,
    rerun_regex_extraction,
    should_compare_tier1,
    validate_and_split_clusters,
)

logger = logging.getLogger(__name__)


def _extract_pair_features_for_ml_wrapper(pair_tuple):
    """
    Wrapper for ML inference: calls extract_pair_features_for_model then adds
    ML-specific features that are in the trained model's feature set.

    Structural features: depth3plus, exact_dir_match.
    Date availability flags: tier1_dates_available, tier2_dates_available
      (True iff both docs had extractable dates — matches training feature).
    Tier availability flags: tier1_has_any_data, tier2_has_any_data
      (True iff both docs in the pair had any extractable data for that tier).
    Date diff sentinel: -1 (missing) replaced with 9999 so the value falls
      outside all proximity thresholds, consistent with training.

    Must be at module level for ProcessPoolExecutor pickling.
    """
    doc1, doc2 = pair_tuple
    features = extract_pair_features_for_model(doc1, doc2)

    # Structural features
    features["depth3plus"] = int(features.get("shared_dir_depth", 0) >= 3)
    dir_parts1 = tuple(Path(doc1.filepath).parts[:-1]) if doc1.filepath else ()
    dir_parts2 = tuple(Path(doc2.filepath).parts[:-1]) if doc2.filepath else ()
    features["exact_dir_match"] = int(len(dir_parts1) > 3 and dir_parts1 == dir_parts2)

    # Date availability flags (must be computed before sentinel replacement)
    features["tier1_dates_available"] = int(features.get("tier1_min_date_diff_days", -1) >= 0)
    features["tier2_dates_available"] = int(features.get("tier2_min_date_diff_days", -1) >= 0)

    # Tier availability flags — mirrors learn_matching_rules.py extract_pair_features.
    # True only when BOTH docs in the pair had extractable data for that tier.
    features["tier1_has_any_data"] = int(
        bool(doc1.tier1_case_ids or doc1.tier1_dates or doc1.tier1_names)
        and bool(doc2.tier1_case_ids or doc2.tier1_dates or doc2.tier1_names)
    )
    features["tier2_has_any_data"] = int(
        bool(
            doc1.tier2_case_ids
            or doc1.tier2_dates
            or doc1.tier2_subject_names
            or doc1.tier2_officer_names
        )
        and bool(
            doc2.tier2_case_ids
            or doc2.tier2_dates
            or doc2.tier2_subject_names
            or doc2.tier2_officer_names
        )
    )

    # Replace -1 sentinel with 9999 so missing dates fall outside all thresholds
    for col in ("tier1_min_date_diff_days", "tier2_min_date_diff_days"):
        if features.get(col, -1) == -1:
            features[col] = 9999

    return features


# ============================================================================
# ML ABLATION CONFIG
# ============================================================================


@dataclass
class MLAblationConfig:
    """Configuration for an ML ablation run."""

    name: str
    description: str
    date_proximity_days: int
    require_dates_present: bool
    enabled_tiers: list[int]
    enabled_rules: dict[str, bool]

    # Test 1: single joint classifier
    use_learned_model: bool = False
    model_path: str | None = None
    feature_subset: str | None = None  # 'tier1_only' | 'tier2_only' | 'both'

    # Test 2: sequential cascade (tier1 first; if no match → tier2)
    use_cascade_model: bool = False
    cascade_tier1_model_path: str | None = None
    cascade_tier2_model_path: str | None = None

    # Feature-level disables (unused by ML inference but kept for config parity)
    disable_case_ids: bool = False
    disable_dates: bool = False
    disable_subject_names: bool = False
    disable_officer_names: bool = False


def load_ml_ablation_config(config_name: str) -> MLAblationConfig:
    """Load ML ablation configuration from ablation_configs_ml.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ablation_configs_ml.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"ML config file not found: {config_path}")

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    available = []
    for ablation in yaml_data.get("ablations", []):
        available.append(ablation["name"])
        if ablation["name"] == config_name:
            return MLAblationConfig(
                name=ablation["name"],
                description=ablation["description"],
                date_proximity_days=ablation["date_proximity_days"],
                require_dates_present=ablation["require_dates_present"],
                enabled_tiers=ablation["enabled_tiers"],
                enabled_rules=ablation["enabled_rules"],
                use_learned_model=ablation.get("use_learned_model", False),
                model_path=ablation.get("model_path", None),
                feature_subset=ablation.get("feature_subset", None),
                use_cascade_model=ablation.get("use_cascade_model", False),
                cascade_tier1_model_path=ablation.get("cascade_tier1_model_path", None),
                cascade_tier2_model_path=ablation.get("cascade_tier2_model_path", None),
                disable_case_ids=ablation.get("disable_case_ids", False),
                disable_dates=ablation.get("disable_dates", False),
                disable_subject_names=ablation.get("disable_subject_names", False),
                disable_officer_names=ablation.get("disable_officer_names", False),
            )

    raise ValueError(f"ML ablation config '{config_name}' not found. Available: {available}")


# ============================================================================
# ML CLUSTERING PIPELINE
# ============================================================================


def _run_batch_joint(
    batch_feature_dicts: list[dict],
    batch_doc_ids: list[tuple[str, str]],
    model,
    feature_names: list[str],
    use_nan: bool,
    threshold: float = 0.5,
) -> list[MatchResult]:
    """Run a single joint model over one batch of pairs."""
    # Structural and date features are already handled by _extract_pair_features_for_ml_wrapper:
    # depth3plus, exact_dir_match, dates_available flags, tier_has_any_data flags,
    # and -1 → 9999 sentinel replacement.
    X = pd.DataFrame(batch_feature_dicts, columns=feature_names).fillna(0).astype(float)
    if use_nan:
        # LightGBM path: convert the 9999 sentinel back to NaN for native missing handling
        for col in [c for c in X.columns if "min_date_diff_days" in c]:
            X[col] = X[col].replace(9999.0, float("nan"))

    if threshold == 0.5:
        preds = model.predict(X)
    else:
        preds = (model.predict_proba(X)[:, 1] > threshold).astype(int)

    matches = []
    for idx, pred in enumerate(preds):
        if pred == 1:
            matches.append(
                MatchResult(
                    matched=True,
                    tier=1,
                    weight=1.0,
                    reason="joint_model match",
                    doc_ids=batch_doc_ids[idx],
                    match_type="joint_model",
                    shared_features={},
                )
            )
    return matches


def _run_batch_cascade(
    batch_feature_dicts: list[dict],
    batch_doc_ids: list[tuple[str, str]],
    tier1_model,
    tier1_feature_names: list[str],
    tier1_use_nan: bool,
    tier2_model,
    tier2_feature_names: list[str],
    tier2_use_nan: bool,
) -> list[MatchResult]:
    """
    Run tier1 model; for unmatched pairs, run tier2 model.

    Match type records which tier made the final positive prediction so
    downstream analysis can break down tier1 vs tier2 contribution.
    """
    # depth3plus and exact_dir_match are added by _extract_pair_features_for_ml_wrapper

    # --- Tier 1 pass ---
    X1 = pd.DataFrame(batch_feature_dicts, columns=tier1_feature_names).astype(float)
    if tier1_use_nan:
        for col in [c for c in X1.columns if "min_date_diff_days" in c]:
            X1[col] = X1[col].replace(-1.0, float("nan"))
    t1_preds = list(tier1_model.predict(X1))

    # --- Tier 2 pass for tier1-unmatched pairs ---
    unmatched_idx = [i for i, p in enumerate(t1_preds) if p == 0]
    if unmatched_idx:
        unmatched_fds = [batch_feature_dicts[i] for i in unmatched_idx]
        X2 = pd.DataFrame(unmatched_fds, columns=tier2_feature_names).astype(float)
        if tier2_use_nan:
            for col in [c for c in X2.columns if "min_date_diff_days" in c]:
                X2[col] = X2[col].replace(-1.0, float("nan"))
        t2_preds = tier2_model.predict(X2)
        for j, orig_idx in enumerate(unmatched_idx):
            if t2_preds[j] == 1:
                t1_preds[orig_idx] = 2  # sentinel: matched by tier2

    matches = []
    for idx, pred in enumerate(t1_preds):
        if pred > 0:
            match_type = "cascade_tier2_match" if pred == 2 else "cascade_tier1_match"
            matches.append(
                MatchResult(
                    matched=True,
                    tier=1 if pred == 1 else 2,
                    weight=1.0,
                    reason=match_type,
                    doc_ids=batch_doc_ids[idx],
                    match_type=match_type,
                    shared_features={},
                )
            )
    return matches


async def cluster_documents_ml(data: list[dict], config: MLAblationConfig) -> dict:
    """
    ML clustering pipeline.

    Handles both Test 1 (joint model) and Test 2 (cascade) inference.
    No Tier 3 LLM comparisons — ML models use only Tier 1 and Tier 2 features.
    """
    if not config.use_learned_model and not config.use_cascade_model:
        raise ValueError(
            "MLAblationConfig must have use_learned_model=True or use_cascade_model=True"
        )

    mode = "cascade" if config.use_cascade_model else "joint"
    logger.info("=" * 80)
    logger.info(f"ML CLUSTERING — {config.name} ({mode})")
    logger.info("=" * 80)
    logger.info(f"  {config.description}")
    logger.info(f"  Documents: {len(data):,}")

    # ========== STEP 0: Frequency filter ==========
    df = pd.DataFrame(data)
    corpus_size = len(data)
    freq_filter = FrequencyFilter(
        date_threshold=int(corpus_size * DATE_FREQUENCY_THRESHOLD_PERCENT),
        name_threshold=int(corpus_size * NAME_FREQUENCY_THRESHOLD_PERCENT),
        case_id_threshold=int(corpus_size * CASE_ID_FREQUENCY_THRESHOLD_PERCENT),
    )
    freq_filter.build_from_dataframe(df)

    # ========== STEP 1: Extract features ==========
    features_map = {}
    for doc in data:
        row = pd.Series(doc)
        features = extract_document_features(row, freq_filter=freq_filter)
        features_map[features.doc_id] = features
    logger.info(f"Extracted features for {len(features_map):,} documents")

    # ========== STEP 2: Load model(s) ==========
    if config.use_cascade_model:
        logger.info("Loading cascade models...")
        t1_bundle = load_trained_model(config.cascade_tier1_model_path)
        t2_bundle = load_trained_model(config.cascade_tier2_model_path)

        tier1_model = t1_bundle["model"]
        tier1_feature_names = list(t1_bundle["feature_cols"])
        tier1_use_nan = t1_bundle.get("use_nan", False)

        tier2_model = t2_bundle["model"]
        tier2_feature_names = list(t2_bundle["feature_cols"])
        tier2_use_nan = t2_bundle.get("use_nan", False)

        logger.info(f"  Tier 1: {type(tier1_model).__name__}, {len(tier1_feature_names)} features")
        logger.info(f"  Tier 2: {type(tier2_model).__name__}, {len(tier2_feature_names)} features")

    else:
        logger.info("Loading joint model...")
        bundle = load_trained_model(config.model_path)
        joint_model = bundle["model"]
        joint_feature_names = list(bundle["feature_cols"])
        joint_use_nan = bundle.get("use_nan", False)
        joint_threshold = bundle.get("threshold", 0.5)
        logger.info(
            f"  Model: {type(joint_model).__name__}, {len(joint_feature_names)} features ({config.feature_subset}), threshold={joint_threshold}"
        )

    # ========== STEP 3: Batched pair inference ==========
    doc_ids = list(features_map.keys())
    total_pairs = len(doc_ids) * (len(doc_ids) - 1) // 2
    logger.info(f"\nProcessing {total_pairs:,} pairs (batch size 1M)...")

    BATCH_SIZE = 1_000_000
    all_edges = []
    tier1_blocked = 0
    pairs_processed = 0

    batch_pairs = []
    batch_doc_ids = []

    pbar = tqdm(
        total=total_pairs,
        desc=config.name,
        unit="pairs",
        unit_scale=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    phase_start = time.time()

    def _flush_batch():
        if not batch_pairs:
            return

        num_workers = min(10, len(batch_pairs))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            feature_dicts = list(executor.map(_extract_pair_features_for_ml_wrapper, batch_pairs))

        if config.use_cascade_model:
            matches = _run_batch_cascade(
                feature_dicts,
                batch_doc_ids,
                tier1_model,
                tier1_feature_names,
                tier1_use_nan,
                tier2_model,
                tier2_feature_names,
                tier2_use_nan,
            )
        else:
            matches = _run_batch_joint(
                feature_dicts,
                batch_doc_ids,
                joint_model,
                joint_feature_names,
                joint_use_nan,
                threshold=joint_threshold,
            )

        all_edges.extend(matches)
        pbar.set_postfix(
            {
                "blocked": f"{tier1_blocked:,}",
                "matches": f"{len(all_edges):,}",
            }
        )
        batch_pairs.clear()
        batch_doc_ids.clear()

    for id1, id2 in itertools.combinations(doc_ids, 2):
        doc1 = features_map[id1]
        doc2 = features_map[id2]

        pairs_processed += 1
        pbar.update(1)

        if not should_compare_tier1(doc1, doc2, min_shared_dirs=TIER1_MIN_SHARED_DIRS):
            tier1_blocked += 1
            continue

        batch_pairs.append((doc1, doc2))
        batch_doc_ids.append((id1, id2))

        if len(batch_pairs) >= BATCH_SIZE:
            _flush_batch()

    _flush_batch()  # process any remaining pairs
    pbar.close()

    elapsed = time.time() - phase_start
    rate = total_pairs / elapsed if elapsed > 0 else 0
    logger.info(f"\nInference complete in {elapsed:.1f}s ({rate:,.0f} pairs/s)")
    logger.info(f"  Blocked by dir filter: {tier1_blocked:,}")
    logger.info(f"  Matches: {len(all_edges):,}")

    if config.use_cascade_model:
        t1_matches = sum(1 for e in all_edges if e.match_type == "cascade_tier1_match")
        t2_matches = sum(1 for e in all_edges if e.match_type == "cascade_tier2_match")
        logger.info(f"  Cascade breakdown: Tier1={t1_matches:,}, Tier2={t2_matches:,}")

    # ========== STEP 4: Build graph ==========
    G = nx.Graph()
    for doc_id in features_map:
        G.add_node(doc_id)
    for edge in all_edges:
        id1, id2 = edge.doc_ids
        G.add_edge(id1, id2, weight=edge.weight, reason=edge.reason)

    # ========== STEP 5: Connected components + validation ==========
    candidate_clusters = list(nx.connected_components(G))
    logger.info(f"\nFound {len(candidate_clusters):,} candidate clusters")

    if ENABLE_VALIDATION:
        clusters = validate_and_split_clusters(
            candidate_clusters,
            G,
            threshold=VALIDATION_THRESHOLD,
            debug=False,
            features_map=features_map,
        )
        logger.info(
            f"After validation: {len(clusters):,} clusters ({len(clusters) - len(candidate_clusters):+,} net splits)"
        )
    else:
        clusters = candidate_clusters

    # ========== STEP 6: Build output ==========
    node_to_cluster = {}
    for cluster_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            node_to_cluster[node] = cluster_id

    results = []
    for doc in data:
        doc_id = str(doc.get("gdrive_id", doc.get("id")))
        doc_copy = doc.copy()
        doc_copy["Parent Clusters"] = [node_to_cluster.get(doc_id, -1)]
        results.append(doc_copy)

    cluster_sizes = [len(c) for c in clusters]

    t1_matches = sum(
        1 for e in all_edges if "tier1" in e.match_type or e.match_type == "joint_model"
    )
    t2_matches = sum(1 for e in all_edges if "tier2" in e.match_type)

    return {
        "results": results,
        "statistics": {
            "total_documents": len(data),
            "total_pairs": total_pairs,
            "total_clusters": len(clusters),
            "singleton_clusters": sum(1 for s in cluster_sizes if s == 1),
            "multi_doc_clusters": sum(1 for s in cluster_sizes if s > 1),
            "edges_added": len(all_edges),
            "tier1_blocked": tier1_blocked,
            "tier1_matches": t1_matches,
            "tier2_matches": t2_matches,
            "tier3_matches": 0,
            "tier3_comparisons": 0,
        },
        "edges": all_edges,
        "features_map": features_map,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main(
    csv_path: str,
    ablation_config_name: str,
    output_dir: str = "../../data/output/ablations_ml",
):
    """
    Main entry point for ML ablation clustering.

    Same signature as cluster_ablations_handengineered.main() so callers can
    invoke either interchangeably via a MODE switch.
    """
    logger.info("=" * 80)
    logger.info("ML ABLATION CLUSTERING PIPELINE")
    logger.info("=" * 80)

    config = load_ml_ablation_config(ablation_config_name)
    logger.info(f"Config: {config.name}")
    logger.info(f"  {config.description}")
    logger.info("")

    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} documents")
    df = df[~(df.provisional_case_name.fillna("?").str.contains(r"n\/a|\?"))]
    logger.info(f"{len(df):,} documents after filtering unassigned rows")

    df = rerun_regex_extraction(df)
    data = df.to_dict("records")

    output = await cluster_documents_ml(data, config)

    # Post-processing: directory-based singleton merging
    if ENABLE_DIRECTORY_FALLBACK:
        output["results"], dir_stats = merge_singletons_by_directory(
            output["results"], output["features_map"]
        )
        output["statistics"]["directory_clustering"] = dir_stats
    else:
        cluster_counts = {}
        for doc in output["results"]:
            cid = doc["Parent Clusters"][0]
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        singleton_count = sum(1 for count in cluster_counts.values() if count == 1)
        output["statistics"]["directory_clustering"] = {
            "enabled": False,
            "singletons_before": singleton_count,
            "singletons_merged": 0,
            "new_clusters_from_dirs": 0,
            "singletons_after": singleton_count,
        }

    # Save results
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"clustering_results_ablation_{config.name}.csv"

    pd.DataFrame(output["results"]).to_csv(output_path, index=False)
    logger.info(f"Saved results to: {output_path}")

    edge_path = str(output_path).replace(".csv", "_edge_metadata.pkl")
    with open(edge_path, "wb") as f:
        pickle.dump({"edges": output["edges"], "features_map": output["features_map"]}, f)

    logger.info("=" * 80)
    logger.info("ML CLUSTERING COMPLETE")
    logger.info("=" * 80)

    return output
