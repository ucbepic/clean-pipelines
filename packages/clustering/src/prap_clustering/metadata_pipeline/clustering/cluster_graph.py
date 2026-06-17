"""
Metadata-Only Clustering Pipeline

This script clusters documents using ONLY filepath and filename regex-extracted features.
No LLM calls, no embeddings - pure metadata baseline.

USAGE:
------
1. Set FEATURE_SOURCE to one of: "filepath_only", "filename_only", "combined"
2. Set ENABLE_VALIDATION to True or False to toggle validation refinement
3. Run: python cluster_metadata.py

CONFIGURATION OPTIONS:
----------------------
FEATURE_SOURCE:
  - "filepath_only": Uses only features from directory path (_fp columns)
  - "filename_only": Uses only features from filename (_fn columns)
  - "combined": Uses union of both _fp and _fn features

ENABLE_VALIDATION:
  - True: Apply non-transitive validation (50% threshold)
    * Each node must have edges to ≥50% of other nodes in cluster
    * Invalid clusters are split into valid sub-clusters
    * This is the REFINED approach

  - False: Use NetworkX connected components as-is (NAIVE baseline)
    * Accepts transitive chains (A-B-C where A and C not connected)
    * Faster, but may produce spurious clusters
    * Use this to compare against the refined approach

EXAMPLE WORKFLOW:
-----------------
Run 6 variations to compare:
1. filepath_only + validation=False → naive filepath baseline
2. filepath_only + validation=True → refined filepath
3. filename_only + validation=False → naive filename baseline
4. filename_only + validation=True → refined filename
5. combined + validation=False → naive combined baseline
6. combined + validation=True → refined combined (best expected)
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import networkx as nx
import pandas as pd

from .cluster import (
    calculate_edge_weight_metadata,
    validate_and_split_clusters,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# INPUT/OUTPUT
INPUT_CSV = "../../data/output/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36_metadata.csv"

OUTPUT_PATHS = {
    ("filepath_only", False): "../../data/output/test_clustering_results_graph_filepath_only_no_validation.csv",
    ("filepath_only", True): "../../data/output/test_clustering_results_graph_filepath_only_with_validation.csv",
    ("filename_only", False): "../../data/output/test_clustering_results_graph_filename_only_no_validation.csv",
    ("filename_only", True): "../../data/output/test_clustering_results_graph_filename_only_with_validation.csv",
    ("combined", False): "../../data/output/test_clustering_results_graph_combined_no_validation.csv",
    ("combined", True): "../../data/output/test_clustering_results_graph_combined_with_validation.csv",
}

# FEATURE SOURCE VARIATIONS - Set to True to run that feature source
# Default: All True (runs all 3 feature sources)
RUN_FILEPATH_ONLY = True
RUN_FILENAME_ONLY = True
RUN_COMBINED = True

# VALIDATION VARIATIONS - Set to True to run with that validation setting
# Default: All True (runs both validation=True and validation=False for each feature source)
RUN_VALIDATION_TRUE = True   # Apply 50% threshold validation (refined)
RUN_VALIDATION_FALSE = True  # Use NetworkX connected components as-is (naive)

VALIDATION_THRESHOLD = 0.5  # Each node must connect to 50% of cluster

DEBUG = True
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_csv_data(csv_path: str) -> list[dict]:
    """Load CSV with metadata features."""
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    if 'ocr_text_per_page' in df.columns:
        df = df.drop(columns=["ocr_text_per_page"])

    # Add ID column if not present
    if 'id' not in df.columns:
        df['id'] = df.index

    # Convert to list of dicts
    documents = df.to_dict('records')

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def process_pair(pair: tuple[dict, dict], source_mode: str) -> tuple[int, int, float] | None:
    """Process a single document pair and return edge data if weight > 0."""
    doc1, doc2 = pair

    try:
        weight = calculate_edge_weight_metadata(doc1, doc2, source_mode)

        # Only create edge if weight is positive
        if weight > 0:
            return (doc1['id'], doc2['id'], weight)
    except Exception as e:
        logger.error(f"Error processing pair ({doc1.get('id')}, {doc2.get('id')}): {e}")

    return None


def process_pairs_in_batches(pairs: list[tuple], max_workers: int, batch_size: int, source_mode: str):
    """
    Process document pairs in batches using ProcessPoolExecutor.
    Yields results batch by batch to avoid memory accumulation.
    """
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    total_edges = 0
    start_time = time.time()

    # Use number of CPU cores if max_workers not specified
    if max_workers is None:
        max_workers = os.cpu_count()

    logger.info(f"  Using {max_workers} worker processes")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_pair, source_mode=source_mode)

        for batch_idx in range(0, len(pairs), batch_size):
            batch_start = time.time()
            batch_num = batch_idx // batch_size + 1
            batch_pairs = pairs[batch_idx:batch_idx + batch_size]

            # Submit all pairs in this batch
            logger.info(f"[Batch {batch_num}/{total_batches}] Processing {len(batch_pairs):,} pairs...")

            # Use map to process in parallel
            results = list(executor.map(process_func, batch_pairs, chunksize=100))

            # Filter out None and exceptions
            valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
            total_edges += len(valid_results)

            batch_elapsed = time.time() - batch_start
            total_elapsed = time.time() - start_time

            # Calculate progress and ETA
            progress_pct = (batch_num / total_batches) * 100
            avg_batch_time = total_elapsed / batch_num
            remaining_batches = total_batches - batch_num
            eta_seconds = remaining_batches * avg_batch_time
            eta_minutes = eta_seconds / 60

            logger.info(f"[Batch {batch_num}/{total_batches}] Complete: {len(valid_results):,} edges found in {batch_elapsed:.1f}s")
            logger.info(f"[Progress] {progress_pct:.1f}% | Edges: {total_edges:,} | Elapsed: {total_elapsed/60:.1f}m | ETA: {eta_minutes:.1f}m")
            logger.info(f"{'='*80}")

            yield valid_results


def cluster_documents(data: list[dict], source_mode: str, enable_validation: bool) -> dict[int, int]:
    """
    Cluster documents using metadata features only.

    Args:
        data: List of document dictionaries
        source_mode: Feature source ("filepath_only", "filename_only", "combined")
        enable_validation: Whether to apply non-transitive validation

    Returns:
        node_to_cluster mapping
    """
    logger.info(f"=== GRAPH-BASED CLUSTERING: {source_mode} (validation={enable_validation}) ===")
    phase_start = time.time()

    # Create graph
    logger.info("PHASE 1: Building graph...")
    graph_start = time.time()
    G = nx.Graph()
    for doc in data:
        G.add_node(doc['id'])
    logger.info(f"  ✓ Created graph with {len(data):,} nodes in {time.time() - graph_start:.2f}s")

    # Generate document pairs
    logger.info("PHASE 2: Generating document pairs...")
    pairs_start = time.time()
    doc_pairs = [(data[i], data[j]) for i in range(len(data)) for j in range(i+1, len(data))]
    pairs_time = time.time() - pairs_start
    logger.info(f"  ✓ Generated {len(doc_pairs):,} document pairs in {pairs_time:.2f}s")
    logger.info(f"  ℹ This is O(n²) = {len(data)}² / 2 = {len(doc_pairs):,} comparisons")

    # Process pairs in parallel
    logger.info("PHASE 3: Computing pairwise similarities...")
    logger.info("  Configuration: batch_size=10,000")
    logger.info("  Starting parallel processing...")
    logger.info(f"{'='*80}")

    max_workers = None  # Use all CPU cores
    batch_size = 10000
    process_start = time.time()

    total_edges_added = 0

    for batch_edges in process_pairs_in_batches(doc_pairs, max_workers, batch_size, source_mode):
        for edge_data in batch_edges:
            doc1_id, doc2_id, weight = edge_data
            G.add_edge(doc1_id, doc2_id, weight=weight)
            total_edges_added += 1

    process_time = time.time() - process_start
    logger.info(f"\n  ✓ Phase 3 complete: Added {total_edges_added:,} edges in {process_time/60:.1f}m")
    logger.info(f"  ℹ Edge density: {total_edges_added/len(doc_pairs)*100:.2f}% of pairs matched")

    # Find connected components
    logger.info("PHASE 4: Computing connected components...")
    cc_start = time.time()
    candidate_clusters = list(nx.connected_components(G))
    cc_time = time.time() - cc_start
    logger.info(f"  ✓ Found {len(candidate_clusters):,} candidate clusters in {cc_time:.2f}s")

    # Cluster size distribution
    cluster_sizes = [len(c) for c in candidate_clusters]
    singletons = sum(1 for size in cluster_sizes if size == 1)
    logger.info(f"  ℹ Singleton clusters: {singletons:,} ({singletons/len(candidate_clusters)*100:.1f}%)")
    logger.info(f"  ℹ Multi-doc clusters: {len(candidate_clusters) - singletons:,}")
    if cluster_sizes:
        logger.info(f"  ℹ Largest cluster: {max(cluster_sizes):,} documents")

    # Apply non-transitive validation if enabled
    if enable_validation:
        logger.info("\nPHASE 5: Non-transitive validation...")
        logger.info(f"  Threshold: {VALIDATION_THRESHOLD * 100}% (each node must connect to ≥{VALIDATION_THRESHOLD * 100}% of cluster)")
        validation_start = time.time()

        validated_clusters = validate_and_split_clusters(
            candidate_clusters,
            G,
            threshold=VALIDATION_THRESHOLD,
            debug=DEBUG
        )
        validation_time = time.time() - validation_start
        final_clusters = validated_clusters

        logger.info(f"  ✓ Validation complete in {validation_time:.2f}s")
        logger.info(f"  ℹ Split {len(candidate_clusters) - len(validated_clusters)} clusters")
    else:
        logger.info("\nPHASE 5: Validation DISABLED - using connected components as-is")
        final_clusters = candidate_clusters

    logger.info(f"\n  ✓ Final cluster count: {len(final_clusters):,}")

    # Build node-to-cluster mapping
    logger.info("Building cluster mapping...")
    node_to_cluster = {}
    for cluster_id, cluster_nodes in enumerate(final_clusters):
        for node in cluster_nodes:
            node_to_cluster[node] = cluster_id

    total_time = time.time() - phase_start
    logger.info(f"\n{'='*80}")
    logger.info(f"CLUSTERING COMPLETE: Total time {total_time/60:.1f}m")
    logger.info(f"{'='*80}")

    return node_to_cluster


def build_final_results(data: list[dict], node_to_cluster: dict[int, int]) -> pd.DataFrame:
    """Build final results DataFrame with cluster assignments."""
    df = pd.DataFrame(data)

    # Add Parent Clusters column
    df['Parent Clusters'] = df['id'].map(lambda idx: [node_to_cluster.get(idx, -1)])

    return df


def run_single_variation(data: list[dict], feature_source: str, enable_validation: bool, output_path: str):
    """Run clustering for a single variation (feature source + validation setting)."""
    logger.info("\n" + "="*80)
    logger.info(f"STARTING VARIATION: {feature_source} (validation={enable_validation})")
    logger.info(f"Output: {output_path}")
    logger.info("="*80)

    # Cluster
    node_to_cluster = cluster_documents(data, feature_source, enable_validation)

    # Build results
    logger.info("Building final results...")
    results_df = build_final_results(data, node_to_cluster)

    # Save
    logger.info(f"Saving results to {output_path}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("CLUSTERING SUMMARY")
    logger.info(f"Feature source: {feature_source}")
    logger.info(f"Validation: {'ENABLED' if enable_validation else 'DISABLED'}")
    if enable_validation:
        logger.info(f"Validation threshold: {VALIDATION_THRESHOLD * 100}%")
    logger.info(f"Total documents: {len(results_df)}")
    logger.info(f"Clusters formed: {len(set(node_to_cluster.values()))}")
    logger.info(f"Singleton clusters: {sum(1 for cid in node_to_cluster.values() if list(node_to_cluster.values()).count(cid) == 1)}")
    logger.info(f"Largest cluster size: {max(list(node_to_cluster.values()).count(cid) for cid in set(node_to_cluster.values()))}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("="*80 + "\n")


def main():
    """Main function for graph-based clustering."""
    # Determine which variations to run
    feature_sources = []
    if RUN_FILEPATH_ONLY:
        feature_sources.append("filepath_only")
    if RUN_FILENAME_ONLY:
        feature_sources.append("filename_only")
    if RUN_COMBINED:
        feature_sources.append("combined")

    validation_settings = []
    if RUN_VALIDATION_FALSE:
        validation_settings.append(False)
    if RUN_VALIDATION_TRUE:
        validation_settings.append(True)

    if not feature_sources:
        logger.error("No feature sources enabled! Set at least one RUN_*_ONLY or RUN_COMBINED flag to True.")
        return

    if not validation_settings:
        logger.error("No validation settings enabled! Set at least one RUN_VALIDATION_* flag to True.")
        return

    # Generate all combinations
    variations = [(fs, vs) for fs in feature_sources for vs in validation_settings]

    logger.info("="*80)
    logger.info("GRAPH-BASED CLUSTERING - MULTI-VARIATION RUN")
    logger.info(f"Input: {INPUT_CSV}")
    logger.info(f"Feature sources: {', '.join(feature_sources)}")
    logger.info(f"Validation settings: {', '.join(str(v) for v in validation_settings)}")
    logger.info(f"Total variations: {len(variations)}")
    logger.info("="*80)

    # Load data once
    data = load_csv_data(INPUT_CSV)

    if DEBUG and data:
        logger.info("\nSample document features:")
        sample = data[0]
        logger.info(f"  extracted_dates_fp: {sample.get('extracted_dates_fp')}")
        logger.info(f"  extracted_dates_fn: {sample.get('extracted_dates_fn')}")
        logger.info(f"  extracted_case_ids_fp: {sample.get('extracted_case_ids_fp')}")
        logger.info(f"  extracted_case_ids_fn: {sample.get('extracted_case_ids_fn')}")
        logger.info(f"  extracted_names_fp: {sample.get('extracted_names_fp')}")
        logger.info(f"  extracted_names_fn: {sample.get('extracted_names_fn')}")

    # Run each enabled variation
    for idx, (feature_source, enable_validation) in enumerate(variations, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# VARIATION {idx}/{len(variations)}: {feature_source} + validation={enable_validation}")
        logger.info(f"{'#'*80}")

        output_path = OUTPUT_PATHS[(feature_source, enable_validation)]
        run_single_variation(data, feature_source, enable_validation, output_path)

    # Final summary
    logger.info("\n" + "#"*80)
    logger.info("ALL VARIATIONS COMPLETE")
    logger.info(f"Ran {len(variations)} variations")
    logger.info("#"*80)


