"""
Deterministic Feature-Based Grouping (Option 1 - Simplest Baseline)

This script uses direct grouping with NO graph, NO transitivity validation, NO pairwise comparisons.
Documents are grouped deterministically based on shared normalized features.

ALGORITHM:
----------
1. Group all documents that share AT LEAST 1 normalized case ID
2. For unclustered documents, group by AT LEAST 1 shared date AND AT LEAST 1 shared name
3. Remaining documents become singleton clusters

PROS:
-----
- Deterministic and reproducible
- Highly interpretable: "These docs share case ID X"
- Natural transitivity (A-B-C chain via shared features)
- Fast - O(n) instead of O(n²) pairwise comparisons
- True "naive baseline" for comparison

CONS:
-----
- Accepts all transitivity without validation
- May create large clusters via chains of weak connections
- Simpler than graph-based approaches

USAGE:
------
Set FEATURE_SOURCE to one of: "filepath_only", "filename_only", "combined"
Run: python cluster_deterministic.py

This creates the SIMPLEST possible baseline for comparison against graph-based approaches.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Import helper functions from cluster.py
from .cluster import (
    extract_incident_numbers,
    get_feature_value_metadata,
    normalize_name,
    standardize_date,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# INPUT/OUTPUT
INPUT_CSV = "../../data/output/autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36 - autofolio_1.2.0_output--Oakland Police Department--2025-04-14_21-57-36_metadata.csv"  # Test sample: 50 cases, 6751 docs

OUTPUT_PATHS = {
    "filepath_only": "../../data/output/test_clustering_results_deterministic_filepath_only.csv",
    "filename_only": "../../data/output/test_clustering_results_deterministic_filename_only.csv",
    "combined": "../../data/output/test_clustering_results_deterministic_combined.csv",
}

# VARIATION CONTROL - Set to True to run that variation
# Default: All True (runs all 3 variations)
RUN_FILEPATH_ONLY = True
RUN_FILENAME_ONLY = True
RUN_COMBINED = True

DEBUG = True
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find data structure for efficient grouping."""

    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, elem):
        """Find root of element with path compression."""
        if self.parent[elem] != elem:
            self.parent[elem] = self.find(self.parent[elem])
        return self.parent[elem]

    def union(self, elem1, elem2):
        """Union two elements' sets."""
        root1 = self.find(elem1)
        root2 = self.find(elem2)

        if root1 != root2:
            # Union by rank
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def get_groups(self) -> dict[any, list[any]]:
        """Get all groups as dict: root -> list of members."""
        groups = defaultdict(list)
        for elem in self.parent:
            root = self.find(elem)
            groups[root].append(elem)
        return groups


def get_normalized_case_ids(doc: dict, source_mode: str) -> set[str]:
    """Extract and normalize all case IDs for a document."""
    row = pd.Series(doc)
    case_ids_raw = get_feature_value_metadata(row, "case_ids", source_mode)

    normalized = set()
    if case_ids_raw:
        for cid in case_ids_raw:
            # Use existing extract_incident_numbers for normalization
            normalized.update(extract_incident_numbers(cid))

    return normalized


def get_standardized_dates(doc: dict, source_mode: str) -> set[str]:
    """Extract and standardize all dates for a document."""
    row = pd.Series(doc)
    dates_raw = get_feature_value_metadata(row, "dates", source_mode)

    standardized = set()
    if dates_raw:
        for date_str in dates_raw:
            parsed = standardize_date(date_str)
            if parsed:
                # Convert to string format YYYY-MM-DD
                standardized.add(parsed.strftime("%Y-%m-%d"))

    return standardized


def get_normalized_names(doc: dict, source_mode: str) -> set[str]:
    """Extract and normalize all names for a document."""
    row = pd.Series(doc)
    names_raw = get_feature_value_metadata(row, "names", source_mode)

    normalized = set()
    if names_raw:
        for name in names_raw:
            # Lowercase and strip for exact matching
            normalized.add(normalize_name(name))

    return normalized


def cluster_by_case_ids(docs: list[dict], source_mode: str) -> dict[int, int]:
    """
    Group documents that share at least 1 normalized case ID.

    Uses Union-Find for efficient grouping.

    Returns:
        doc_id -> cluster_id mapping
    """
    logger.info("PHASE 1: Grouping by shared case IDs...")
    phase_start = time.time()

    # Build index: case_id -> list of doc IDs that have it
    logger.info("  Building inverted index (case_id → docs)...")
    index_start = time.time()
    case_id_to_docs = defaultdict(list)

    docs_with_case_ids = 0
    for doc in docs:
        case_ids = get_normalized_case_ids(doc, source_mode)
        if case_ids:
            docs_with_case_ids += 1
        for case_id in case_ids:
            case_id_to_docs[case_id].append(doc["id"])

    logger.info(
        f"  ✓ Indexed {len(case_id_to_docs):,} unique case IDs in {time.time() - index_start:.2f}s"
    )
    logger.info(
        f"  ℹ Documents with case IDs: {docs_with_case_ids:,} ({docs_with_case_ids / len(docs) * 100:.1f}%)"
    )

    # Union-Find: merge all docs that share at least 1 case ID
    logger.info("  Merging documents via Union-Find...")
    uf_start = time.time()
    uf = UnionFind([doc["id"] for doc in docs])

    merge_count = 0
    case_ids_with_links = 0
    for case_id, doc_ids in case_id_to_docs.items():
        if len(doc_ids) > 1:
            case_ids_with_links += 1
            # Union all docs with this case ID
            for i in range(1, len(doc_ids)):
                uf.union(doc_ids[0], doc_ids[i])
                merge_count += 1

            if DEBUG and case_ids_with_links <= 10:  # Log first 10
                logger.info(f"    Case ID '{case_id}': links {len(doc_ids)} documents")

    logger.info(f"  ✓ Performed {merge_count:,} union operations in {time.time() - uf_start:.2f}s")

    # Avoid division by zero if no case IDs found
    case_id_link_pct = (
        (case_ids_with_links / len(case_id_to_docs) * 100) if len(case_id_to_docs) > 0 else 0.0
    )
    logger.info(f"  ℹ Case IDs that linked docs: {case_ids_with_links:,} ({case_id_link_pct:.1f}%)")

    # Get groups
    logger.info("  Computing final groups...")
    groups_start = time.time()
    groups = uf.get_groups()
    multi_doc_groups = {root: members for root, members in groups.items() if len(members) > 1}
    logger.info(f"  ✓ Grouped in {time.time() - groups_start:.2f}s")

    total_in_groups = sum(len(members) for members in multi_doc_groups.values())
    logger.info(f"  ✓ Formed {len(multi_doc_groups):,} groups with 2+ documents")
    logger.info(
        f"  ✓ Total documents in groups: {total_in_groups:,} ({total_in_groups / len(docs) * 100:.1f}%)"
    )

    # Group size distribution
    group_sizes = [len(members) for members in multi_doc_groups.values()]
    if group_sizes:
        logger.info(f"  ℹ Largest group: {max(group_sizes):,} documents")
        logger.info(f"  ℹ Average group size: {sum(group_sizes) / len(group_sizes):.1f} documents")

    # Create initial mapping (only for grouped docs)
    node_to_cluster = {}
    for cluster_id, (_root, members) in enumerate(multi_doc_groups.items()):
        for doc_id in members:
            node_to_cluster[doc_id] = cluster_id

    logger.info(f"  ⏱ Phase 1 complete in {time.time() - phase_start:.2f}s")
    logger.info(f"{'=' * 80}")

    return node_to_cluster


def cluster_by_date_and_names(
    docs: list[dict], already_clustered: dict[int, int], source_mode: str
) -> dict[int, int]:
    """
    Group unclustered documents that share at least 1 date AND at least 1 name.

    Returns:
        Updated doc_id -> cluster_id mapping
    """
    logger.info("PHASE 2: Grouping unclustered docs by shared (date, name) pairs...")
    phase_start = time.time()

    # Filter to unclustered documents
    unclustered_docs = [doc for doc in docs if doc["id"] not in already_clustered]
    logger.info(
        f"  {len(unclustered_docs):,} documents remaining to cluster ({len(unclustered_docs) / len(docs) * 100:.1f}%)"
    )

    if not unclustered_docs:
        logger.info("  ℹ All documents already clustered, skipping phase 2")
        return already_clustered

    # Build indexes
    logger.info("  Building inverted indexes (date → docs, name → docs)...")
    index_start = time.time()
    date_to_docs = defaultdict(list)
    name_to_docs = defaultdict(list)
    doc_dates = {}
    doc_names = {}

    docs_with_dates = 0
    docs_with_names = 0
    for doc in unclustered_docs:
        dates = get_standardized_dates(doc, source_mode)
        names = get_normalized_names(doc, source_mode)

        if dates:
            docs_with_dates += 1
        if names:
            docs_with_names += 1

        doc_dates[doc["id"]] = dates
        doc_names[doc["id"]] = names

        for date in dates:
            date_to_docs[date].append(doc["id"])
        for name in names:
            name_to_docs[name].append(doc["id"])

    logger.info(
        f"  ✓ Indexed {len(date_to_docs):,} unique dates and {len(name_to_docs):,} unique names in {time.time() - index_start:.2f}s"
    )
    logger.info(
        f"  ℹ Docs with dates: {docs_with_dates:,} ({docs_with_dates / len(unclustered_docs) * 100:.1f}%)"
    )
    logger.info(
        f"  ℹ Docs with names: {docs_with_names:,} ({docs_with_names / len(unclustered_docs) * 100:.1f}%)"
    )

    # Union-Find for unclustered docs
    uf = UnionFind([doc["id"] for doc in unclustered_docs])

    # Generate candidate pairs (docs that share at least 1 date)
    logger.info("  Generating candidate pairs (docs sharing dates)...")
    cand_start = time.time()
    candidates = set()
    for date, doc_ids in date_to_docs.items():
        if len(doc_ids) > 1:
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    candidates.add((min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j])))

    logger.info(
        f"  ✓ Generated {len(candidates):,} candidate pairs in {time.time() - cand_start:.2f}s"
    )

    # Check each candidate pair for date AND name overlap
    logger.info("  Checking candidates for (date ∩ name) overlap...")
    check_start = time.time()
    merged_count = 0
    for idx, (doc_id1, doc_id2) in enumerate(candidates):
        dates1 = doc_dates[doc_id1]
        dates2 = doc_dates[doc_id2]
        names1 = doc_names[doc_id1]
        names2 = doc_names[doc_id2]

        # Check: at least 1 shared date AND at least 1 shared name
        if dates1 & dates2 and names1 & names2:
            uf.union(doc_id1, doc_id2)
            merged_count += 1

            if DEBUG and merged_count <= 10:  # Log first 10
                shared_dates = dates1 & dates2
                shared_names = names1 & names2
                logger.info(
                    f"    Merged docs {doc_id1}, {doc_id2}: date={list(shared_dates)[0]}, name={list(shared_names)[0]}"
                )

        # Progress logging for large datasets
        if (idx + 1) % 50000 == 0:
            progress = (idx + 1) / len(candidates) * 100
            logger.info(
                f"    Progress: {progress:.1f}% ({idx + 1:,}/{len(candidates):,}), {merged_count:,} merges so far"
            )

    logger.info(f"  ✓ Checked {len(candidates):,} pairs in {time.time() - check_start:.2f}s")
    logger.info(f"  ✓ Merged {merged_count:,} document pairs")

    # Get groups
    logger.info("  Computing final groups...")
    groups_start = time.time()
    groups = uf.get_groups()
    multi_doc_groups = {root: members for root, members in groups.items() if len(members) > 1}
    logger.info(f"  ✓ Grouped in {time.time() - groups_start:.2f}s")

    total_in_new_groups = sum(len(members) for members in multi_doc_groups.values())
    logger.info(f"  ✓ Formed {len(multi_doc_groups):,} new groups with 2+ documents")
    logger.info(
        f"  ✓ Total documents in new groups: {total_in_new_groups:,} ({total_in_new_groups / len(unclustered_docs) * 100:.1f}%)"
    )

    # Group size distribution
    group_sizes = [len(members) for members in multi_doc_groups.values()]
    if group_sizes:
        logger.info(f"  ℹ Largest new group: {max(group_sizes):,} documents")
        logger.info(
            f"  ℹ Average new group size: {sum(group_sizes) / len(group_sizes):.1f} documents"
        )

    # Create updated mapping
    node_to_cluster = already_clustered.copy()
    next_cluster_id = max(already_clustered.values()) + 1 if already_clustered else 0

    for _root, members in multi_doc_groups.items():
        for doc_id in members:
            node_to_cluster[doc_id] = next_cluster_id
        next_cluster_id += 1

    logger.info(f"  ⏱ Phase 2 complete in {time.time() - phase_start:.2f}s")
    logger.info(f"{'=' * 80}")

    return node_to_cluster


def assign_singletons(docs: list[dict], existing_mapping: dict[int, int]) -> dict[int, int]:
    """
    Assign singleton clusters to all remaining unclustered documents.

    Returns:
        Complete doc_id -> cluster_id mapping
    """
    logger.info("PHASE 3: Assigning singleton clusters to remaining documents...")
    phase_start = time.time()

    node_to_cluster = existing_mapping.copy()
    next_cluster_id = max(existing_mapping.values()) + 1 if existing_mapping else 0

    singleton_count = 0
    for doc in docs:
        if doc["id"] not in existing_mapping:
            node_to_cluster[doc["id"]] = next_cluster_id
            next_cluster_id += 1
            singleton_count += 1

    logger.info(
        f"  ✓ Assigned {singleton_count:,} singleton clusters ({singleton_count / len(docs) * 100:.1f}%)"
    )
    logger.info(f"  ⏱ Phase 3 complete in {time.time() - phase_start:.2f}s")
    logger.info(f"{'=' * 80}")

    return node_to_cluster


def cluster_documents_deterministic(data: list[dict], source_mode: str) -> dict[int, int]:
    """
    Deterministically cluster documents using shared normalized features.

    Algorithm:
    1. Group by at least 1 shared case ID (with transitivity)
    2. Group remaining by at least 1 shared date AND at least 1 shared name
    3. Assign singletons to the rest

    Returns:
        node_to_cluster mapping
    """
    logger.info(f"{'=' * 80}")
    logger.info(f"DETERMINISTIC CLUSTERING: {source_mode}")
    logger.info(f"Total documents: {len(data):,}")
    logger.info("Algorithm: Union-Find (O(n) complexity)")
    logger.info(f"{'=' * 80}\n")

    total_start = time.time()

    # Phase 1: Cluster by case IDs
    node_to_cluster = cluster_by_case_ids(data, source_mode)

    # Phase 2: Cluster unclustered docs by (date, names)
    node_to_cluster = cluster_by_date_and_names(data, node_to_cluster, source_mode)

    # Phase 3: Assign singletons
    node_to_cluster = assign_singletons(data, node_to_cluster)

    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 80}")
    logger.info("CLUSTERING COMPLETE")
    logger.info(f"Total time: {total_time:.2f}s ({total_time / 60:.1f}m)")
    logger.info(f"{'=' * 80}")

    return node_to_cluster


def run_single_variation(documents: list[dict], feature_source: str, output_path: str):
    """Run clustering for a single feature source variation."""
    logger.info("\n" + "=" * 80)
    logger.info(f"STARTING VARIATION: {feature_source}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)

    # Cluster
    node_to_cluster = cluster_documents_deterministic(documents, feature_source)

    # Build results
    logger.info("Building final results...")
    df = pd.DataFrame(documents)
    df["Parent Clusters"] = df["id"].map(lambda idx: [node_to_cluster.get(idx, -1)])

    # Save
    logger.info(f"Saving results to {output_path}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary statistics
    cluster_sizes = defaultdict(int)
    for cluster_id in node_to_cluster.values():
        cluster_sizes[cluster_id] += 1

    singleton_count = sum(1 for size in cluster_sizes.values() if size == 1)
    multi_doc_clusters = [size for size in cluster_sizes.values() if size > 1]

    logger.info("\n" + "=" * 80)
    logger.info("CLUSTERING SUMMARY")
    logger.info(f"Feature source: {feature_source}")
    logger.info("Algorithm: Deterministic (no graph, natural transitivity)")
    logger.info(f"Total documents: {len(df)}")
    logger.info(f"Total clusters: {len(set(node_to_cluster.values()))}")
    logger.info(f"Singleton clusters: {singleton_count}")
    logger.info(f"Multi-document clusters: {len(multi_doc_clusters)}")
    if multi_doc_clusters:
        logger.info(f"Largest cluster size: {max(multi_doc_clusters)}")
        logger.info(
            f"Average multi-doc cluster size: {sum(multi_doc_clusters) / len(multi_doc_clusters):.1f}"
        )
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80 + "\n")


def main():
    """Main function for deterministic clustering."""
    # Determine which variations to run
    variations_to_run = []
    if RUN_FILEPATH_ONLY:
        variations_to_run.append("filepath_only")
    if RUN_FILENAME_ONLY:
        variations_to_run.append("filename_only")
    if RUN_COMBINED:
        variations_to_run.append("combined")

    if not variations_to_run:
        logger.error("No variations enabled! Set at least one RUN_* flag to True.")
        return

    logger.info("=" * 80)
    logger.info("DETERMINISTIC CLUSTERING - MULTI-VARIATION RUN")
    logger.info(f"Input: {INPUT_CSV}")
    logger.info(f"Variations to run: {', '.join(variations_to_run)}")
    logger.info("=" * 80)

    # Load data once
    logger.info(f"\nLoading CSV from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if "ocr_text_per_page" in df.columns:
        df = df.drop(columns=["ocr_text_per_page"])

    # Add ID column if not present
    if "id" not in df.columns:
        df["id"] = df.index

    documents = df.to_dict("records")
    logger.info(f"Loaded {len(documents)} documents")

    if DEBUG and documents:
        logger.info("\nSample document features:")
        sample = documents[0]
        logger.info(f"  extracted_dates_fp: {sample.get('extracted_dates_fp')}")
        logger.info(f"  extracted_dates_fn: {sample.get('extracted_dates_fn')}")
        logger.info(f"  extracted_case_ids_fp: {sample.get('extracted_case_ids_fp')}")
        logger.info(f"  extracted_case_ids_fn: {sample.get('extracted_case_ids_fn')}")
        logger.info(f"  extracted_names_fp: {sample.get('extracted_names_fp')}")
        logger.info(f"  extracted_names_fn: {sample.get('extracted_names_fn')}")

    # Run each enabled variation
    for idx, feature_source in enumerate(variations_to_run, 1):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"# VARIATION {idx}/{len(variations_to_run)}: {feature_source}")
        logger.info(f"{'#' * 80}")

        output_path = OUTPUT_PATHS[feature_source]
        run_single_variation(documents, feature_source, output_path)

    # Final summary
    logger.info("\n" + "#" * 80)
    logger.info("ALL VARIATIONS COMPLETE")
    logger.info(f"Ran {len(variations_to_run)} variations: {', '.join(variations_to_run)}")
    logger.info("#" * 80)
