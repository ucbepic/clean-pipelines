"""
Core metrics for evaluating clustering results against ground truth.

Uses the "Best-Matching Cluster" strategy with macro-averaging:
- For each ground truth cluster, find the test cluster with the most overlap
- Calculate precision/recall/F1 for that ground truth cluster
- Macro-average across all ground truth clusters (each cluster weighted equally)

Also includes:
- B-Cubed metrics (document-weighted, penalizes errors in large clusters more)
- Adjusted Rand Index (pair-based, strong penalty for merges and splits)
"""

from collections import defaultdict

import pandas as pd

try:
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, ARI will not be computed")


def extract_cluster_id(cluster_value):
    """
    Extract cluster ID from various formats.

    Handles:
    - String representations of lists: "[0]"
    - Lists: [0]
    - Direct values: 0
    - None/NaN

    Returns:
        Cluster ID or None if invalid
    """
    if pd.isna(cluster_value):
        return None

    if isinstance(cluster_value, str):
        # Handle string representations of lists like "[0]"
        if cluster_value.startswith('[') and cluster_value.endswith(']'):
            try:
                inner = cluster_value.strip('[]').strip()
                if inner:
                    return int(inner)
                return None
            except:
                return cluster_value
        return cluster_value

    if isinstance(cluster_value, list) and len(cluster_value) > 0:
        return cluster_value[0]

    return cluster_value


def build_cluster_mapping(df: pd.DataFrame, cluster_column: str) -> dict[str, any]:
    """
    Build mapping from filename to cluster ID.

    Args:
        df: DataFrame with clustering results
        cluster_column: Name of column containing cluster assignments

    Returns:
        Dict mapping gdrive_name to cluster ID
    """
    df_copy = df.copy()
    df_copy['cluster_id'] = df_copy[cluster_column].apply(extract_cluster_id)

    # Remove rows with no cluster assignment
    df_copy = df_copy.dropna(subset=['cluster_id'])

    return dict(zip(df_copy['gdrive_name'], df_copy['cluster_id'], strict=False))


def evaluate_clusters(groundtruth_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Evaluate how well test clusters preserve the groundtruth clusters.

    Uses best-matching cluster strategy with macro-averaging:
    - For each groundtruth cluster, find the test cluster with most overlap
    - Calculate precision/recall/F1 for that pairing
    - Average metrics across all groundtruth clusters (equal weight per cluster)

    Args:
        groundtruth_df: DataFrame with columns ['gdrive_name', 'provisional_case_name']
        test_df: DataFrame with columns ['gdrive_name', 'Parent Clusters'] (or similar)

    Returns:
        Dict with overall metrics and per-cluster details
    """
    # Find cluster column in test_df
    cluster_column = None
    possible_cluster_columns = ['Parent Clusters', 'Clusters', 'cluster', 'parent_clusters']

    for col in possible_cluster_columns:
        if col in test_df.columns:
            cluster_column = col
            break

    if cluster_column is None:
        raise ValueError(f"Could not find cluster column. Available columns: {list(test_df.columns)}")

    print(f"Using cluster column: '{cluster_column}'")

    # Build mapping from filename to test cluster
    file_to_test_cluster = build_cluster_mapping(test_df, cluster_column)

    print(f"Found {len(file_to_test_cluster)} files with cluster assignments")

    # Evaluate each groundtruth cluster
    cluster_metrics = []

    # Group files by their groundtruth cluster
    for gt_cluster_name, group in groundtruth_df.groupby('provisional_case_name'):
        if pd.isna(gt_cluster_name):
            continue

        gt_cluster_name = str(gt_cluster_name).strip()
        gt_files = set(group['gdrive_name'].tolist())

        # Find all test clusters containing these files
        test_cluster_distribution = defaultdict(set)
        files_found_in_test = 0

        for file in gt_files:
            if file in file_to_test_cluster:
                test_cluster_distribution[file_to_test_cluster[file]].add(file)
                files_found_in_test += 1

        if files_found_in_test == 0:
            print(f"Warning: No files from groundtruth cluster '{gt_cluster_name}' found in test results")
            continue

        # Find the test cluster with the most files from this groundtruth cluster
        best_test_cluster = max(
            test_cluster_distribution.items(),
            key=lambda x: len(x[1]),
            default=(None, set())
        )
        best_cluster_id, best_cluster_files = best_test_cluster

        if best_cluster_id is not None:
            # How many files from the groundtruth cluster are in the best test cluster
            files_in_best_cluster = len(best_cluster_files)

            # How many total files are in this test cluster (including files from other groundtruth clusters)
            total_in_best_cluster = sum(
                1 for f, c in file_to_test_cluster.items()
                if c == best_cluster_id
            )

            # Metrics
            precision = files_in_best_cluster / total_in_best_cluster if total_in_best_cluster > 0 else 0
            recall = files_in_best_cluster / len(gt_files)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Files that should be in this cluster but were split into other clusters
            split_files = {
                test_cluster: files
                for test_cluster, files in test_cluster_distribution.items()
                if test_cluster != best_cluster_id
            }

            # Files from other groundtruth clusters that got merged into the best test cluster
            merged_files = [
                f for f, c in file_to_test_cluster.items()
                if c == best_cluster_id and f not in gt_files
            ]

            cluster_metrics.append({
                'groundtruth_cluster': gt_cluster_name,
                'best_matching_test_cluster': best_cluster_id,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_files_should_be_together': len(gt_files),
                'num_files_in_best_cluster': files_in_best_cluster,
                'total_files_in_best_cluster': total_in_best_cluster,
                'num_split_clusters': len(split_files),
                'num_incorrectly_merged_files': len(merged_files),
                'split_distribution': {
                    str(cluster): list(files)
                    for cluster, files in split_files.items()
                },
                'incorrectly_merged_files': merged_files
            })

    # Calculate overall metrics (macro-averaging)
    if cluster_metrics:
        avg_precision = sum(m['precision'] for m in cluster_metrics) / len(cluster_metrics)
        avg_recall = sum(m['recall'] for m in cluster_metrics) / len(cluster_metrics)
        avg_f1 = sum(m['f1'] for m in cluster_metrics) / len(cluster_metrics)

        # Count how many clusters had splits or merges
        clusters_with_splits = sum(1 for m in cluster_metrics if m['num_split_clusters'] > 0)
        clusters_with_merges = sum(1 for m in cluster_metrics if m['num_incorrectly_merged_files'] > 0)
    else:
        avg_precision = avg_recall = avg_f1 = 0
        clusters_with_splits = clusters_with_merges = 0

    # Compute B-Cubed metrics (document-weighted)
    bcubed_metrics = compute_bcubed_metrics(groundtruth_df, file_to_test_cluster)

    # Compute Adjusted Rand Index (pair-based)
    ari = compute_adjusted_rand_index(groundtruth_df, file_to_test_cluster)

    return {
        'overall': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'num_groundtruth_clusters_evaluated': len(cluster_metrics),
            'clusters_with_splits': clusters_with_splits,
            'clusters_with_merges': clusters_with_merges,
            # B-Cubed metrics (document-weighted)
            'bcubed_precision': bcubed_metrics['bcubed_precision'],
            'bcubed_recall': bcubed_metrics['bcubed_recall'],
            'bcubed_f1': bcubed_metrics['bcubed_f1'],
            # Adjusted Rand Index (pair-based)
            'ari': ari,
        },
        'cluster_details': cluster_metrics
    }


def compute_bcubed_metrics(groundtruth_df: pd.DataFrame, file_to_test_cluster: dict) -> dict:
    """
    Compute B-Cubed precision, recall, and F1.

    B-Cubed metrics naturally weight by document count - each document contributes equally,
    so errors in large clusters affect more documents' scores.

    For each document i:
    - B-Cubed Precision: proportion of items in i's predicted cluster that share i's GT cluster
    - B-Cubed Recall: proportion of items in i's GT cluster that are in i's predicted cluster

    Then average across all documents.

    Args:
        groundtruth_df: DataFrame with columns ['gdrive_name', 'provisional_case_name']
        file_to_test_cluster: Dict mapping gdrive_name to predicted cluster ID

    Returns:
        Dict with bcubed_precision, bcubed_recall, bcubed_f1
    """
    # Build mapping from file to groundtruth cluster
    file_to_gt_cluster = dict(zip(
        groundtruth_df['gdrive_name'],
        groundtruth_df['provisional_case_name'], strict=False
    ))

    # Build inverse mappings: cluster -> set of files
    gt_cluster_to_files = defaultdict(set)
    test_cluster_to_files = defaultdict(set)

    for file, gt_cluster in file_to_gt_cluster.items():
        if pd.notna(gt_cluster):
            gt_cluster_to_files[str(gt_cluster).strip()].add(file)

    for file, test_cluster in file_to_test_cluster.items():
        if test_cluster is not None:
            test_cluster_to_files[test_cluster].add(file)

    # Compute B-Cubed metrics per document
    bcubed_precisions = []
    bcubed_recalls = []

    # Only evaluate files that appear in both GT and test results
    evaluated_files = set(file_to_gt_cluster.keys()) & set(file_to_test_cluster.keys())

    for file in evaluated_files:
        gt_cluster = str(file_to_gt_cluster[file]).strip()
        test_cluster = file_to_test_cluster[file]

        if pd.isna(gt_cluster) or test_cluster is None:
            continue

        # Files that should be in the same cluster (GT perspective)
        gt_cluster_files = gt_cluster_to_files[gt_cluster]

        # Files that are in the same cluster (test perspective)
        test_cluster_files = test_cluster_to_files[test_cluster]

        # B-Cubed Precision: of files in my predicted cluster, how many share my GT cluster?
        if len(test_cluster_files) > 0:
            correct_in_test = len(test_cluster_files & gt_cluster_files)
            bcubed_precision = correct_in_test / len(test_cluster_files)
            bcubed_precisions.append(bcubed_precision)

        # B-Cubed Recall: of files in my GT cluster, how many are in my predicted cluster?
        if len(gt_cluster_files) > 0:
            correct_in_gt = len(test_cluster_files & gt_cluster_files)
            bcubed_recall = correct_in_gt / len(gt_cluster_files)
            bcubed_recalls.append(bcubed_recall)

    # Average across all documents
    if bcubed_precisions and bcubed_recalls:
        avg_bcubed_precision = sum(bcubed_precisions) / len(bcubed_precisions)
        avg_bcubed_recall = sum(bcubed_recalls) / len(bcubed_recalls)

        if (avg_bcubed_precision + avg_bcubed_recall) > 0:
            avg_bcubed_f1 = 2 * (avg_bcubed_precision * avg_bcubed_recall) / (avg_bcubed_precision + avg_bcubed_recall)
        else:
            avg_bcubed_f1 = 0.0
    else:
        avg_bcubed_precision = avg_bcubed_recall = avg_bcubed_f1 = 0.0

    return {
        'bcubed_precision': avg_bcubed_precision,
        'bcubed_recall': avg_bcubed_recall,
        'bcubed_f1': avg_bcubed_f1,
    }


def compute_adjusted_rand_index(groundtruth_df: pd.DataFrame, file_to_test_cluster: dict) -> float:
    """
    Compute Adjusted Rand Index using sklearn.

    ARI is a pair-based metric that penalizes both splits and merges harshly.
    It's adjusted for chance (random clustering gets ARI ≈ 0, perfect = 1).

    Args:
        groundtruth_df: DataFrame with columns ['gdrive_name', 'provisional_case_name']
        file_to_test_cluster: Dict mapping gdrive_name to predicted cluster ID

    Returns:
        Adjusted Rand Index score (float)
    """
    if not SKLEARN_AVAILABLE:
        return None

    # Build mapping from file to groundtruth cluster
    file_to_gt_cluster = dict(zip(
        groundtruth_df['gdrive_name'],
        groundtruth_df['provisional_case_name'], strict=False
    ))

    # Get files that appear in both GT and test
    common_files = sorted(set(file_to_gt_cluster.keys()) & set(file_to_test_cluster.keys()))

    if len(common_files) < 2:
        return 0.0

    # Build parallel lists of cluster assignments
    gt_labels = []
    test_labels = []

    for file in common_files:
        gt_cluster = file_to_gt_cluster[file]
        test_cluster = file_to_test_cluster[file]

        if pd.notna(gt_cluster) and test_cluster is not None:
            gt_labels.append(str(gt_cluster).strip())
            test_labels.append(str(test_cluster))

    if len(gt_labels) < 2:
        return 0.0

    # Compute ARI using sklearn
    ari = adjusted_rand_score(gt_labels, test_labels)
    return float(ari)


def get_summary_statistics(metrics: dict) -> dict:
    """
    Calculate summary statistics from evaluation metrics.

    Args:
        metrics: Output from evaluate_clusters()

    Returns:
        Dict with summary statistics
    """
    cluster_details = metrics['cluster_details']

    if not cluster_details:
        return {
            'perfect_clusters': 0,
            'high_quality_clusters': 0,
            'medium_quality_clusters': 0,
            'low_quality_clusters': 0,
        }

    # Count clusters by quality
    perfect = sum(1 for m in cluster_details if m['f1'] == 1.0)
    high = sum(1 for m in cluster_details if 0.9 <= m['f1'] < 1.0)
    medium = sum(1 for m in cluster_details if 0.7 <= m['f1'] < 0.9)
    low = sum(1 for m in cluster_details if m['f1'] < 0.7)

    # Get distribution of cluster sizes
    cluster_sizes = [m['num_files_should_be_together'] for m in cluster_details]

    return {
        'perfect_clusters': perfect,
        'high_quality_clusters': high,
        'medium_quality_clusters': medium,
        'low_quality_clusters': low,
        'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
        'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
    }
