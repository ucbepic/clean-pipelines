import ast
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_embedding_safe(emb_str):
    """Parse embedding string to numpy array, handling multiple formats."""
    if pd.isna(emb_str):
        return None
    if isinstance(emb_str, np.ndarray):
        return emb_str

    # Handle two formats:
    # 1. Python list: "[0.1, 0.2, 0.3]"
    # 2. Numpy array: "[-1.18838415e-01  4.82985936e-02 ...]" (no commas)
    emb_str = str(emb_str).strip()

    # Try Python list format first
    if ',' in emb_str:
        try:
            return np.array(ast.literal_eval(emb_str))
        except (ValueError, SyntaxError):
            pass

    # Try numpy array format (space-separated, no commas)
    try:
        emb_str = emb_str.strip('[]')
        values = emb_str.split()
        return np.array([float(v) for v in values])
    except:
        return None


def load_embeddings(csv_path: str, mode: str = "pure") -> pd.DataFrame:
    """
    Load embeddings from CSV and parse embedding strings to numpy arrays.

    Args:
        csv_path: Path to CSV with embedding column
        mode: "pure" (only files with embeddings) or "hybrid" (all files)

    Returns:
        DataFrame with parsed embeddings
    """
    logger.info(f"Loading embeddings from {csv_path} (mode={mode})")
    df = pd.read_csv(csv_path)

    # Parse embedding strings to numpy arrays
    df['embedding'] = df['embedding'].apply(parse_embedding_safe)

    # Add ID column if not present
    if 'id' not in df.columns:
        df['id'] = df.index.astype(int)

    if mode == "pure":
        # Filter to only documents with embeddings
        df = df[df['embedding'].notna()].copy()
        logger.info(f"Pure mode: {len(df)} documents with embeddings")
    else:  # hybrid
        # Keep ALL documents
        with_emb = df['embedding'].notna().sum()
        without_emb = df['embedding'].isna().sum()
        logger.info(f"Hybrid mode: {len(df)} total documents")
        logger.info(f"  - {with_emb} with embeddings ({with_emb/len(df)*100:.1f}%)")
        logger.info(f"  - {without_emb} without embeddings ({without_emb/len(df)*100:.1f}%)")

    return df


def compute_cosine_similarity_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for embeddings.

    Args:
        embeddings: List of numpy arrays (embeddings)

    Returns:
        Symmetric similarity matrix (n_docs × n_docs)
    """
    # Convert to 2D array (n_docs × embedding_dim)
    emb_matrix = np.vstack(embeddings)

    # L2 normalize each embedding vector
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    emb_normalized = emb_matrix / norms

    # Compute cosine similarity via dot product of normalized vectors
    similarity_matrix = emb_normalized @ emb_normalized.T

    return similarity_matrix


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(emb1, emb2) / (norm1 * norm2)


def calculate_hybrid_edge_weight(doc1: dict, doc2: dict, threshold: float,
                                  metadata_source_mode: str = "combined") -> float:
    """
    Calculate edge weight using hybrid strategy.

    Strategy:
    - If BOTH have embeddings: use cosine similarity >= threshold → 1.0 or 0.0
    - If AT LEAST ONE lacks embeddings: use metadata features → 1.0 or 0.0

    Args:
        doc1, doc2: Document dictionaries with 'embedding' field
        threshold: Cosine similarity threshold for embeddings
        metadata_source_mode: "filepath_only", "filename_only", or "combined"

    Returns:
        1.0 if should cluster, 0.0 otherwise
    """
    emb1 = doc1.get('embedding')
    emb2 = doc2.get('embedding')

    # Check if embeddings exist
    has_emb1 = emb1 is not None and isinstance(emb1, np.ndarray)
    has_emb2 = emb2 is not None and isinstance(emb2, np.ndarray)

    # Case 1: Both have embeddings - use cosine similarity
    if has_emb1 and has_emb2:
        similarity = compute_cosine_similarity(emb1, emb2)
        return 1.0 if similarity >= threshold else 0.0

    # Case 2: At least one lacks embeddings - use metadata features
    return calculate_edge_weight_metadata(doc1, doc2, metadata_source_mode)


def cluster_with_threshold(similarity_matrix: np.ndarray, threshold: float) -> dict[int, int]:
    """
    Cluster documents using similarity threshold and connected components.

    Args:
        similarity_matrix: Symmetric similarity matrix (n_docs × n_docs)
        threshold: Minimum similarity to create edge (e.g., 0.85)

    Returns:
        Dictionary mapping document index to cluster ID
    """
    n_docs = similarity_matrix.shape[0]

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(n_docs))

    # Add edges where similarity >= threshold
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(i, j)

    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Create mapping: doc_index -> cluster_id
    cluster_assignments = {}
    for cluster_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            cluster_assignments[node] = cluster_id

    return cluster_assignments


def cluster_with_threshold_hybrid(df: pd.DataFrame, threshold: float,
                                   metadata_source_mode: str = "combined") -> dict[int, int]:
    """
    Cluster documents using hybrid embeddings + metadata strategy.

    Args:
        df: DataFrame with all documents (including those without embeddings)
        threshold: Cosine similarity threshold for embeddings
        metadata_source_mode: Metadata feature source for fallback

    Returns:
        Dictionary mapping document ID to cluster ID
    """
    logger.info(f"  Clustering with hybrid strategy (threshold={threshold:.2f})...")

    # Convert to list of dicts
    documents = df.to_dict('records')

    # Build graph
    G = nx.Graph()
    for doc in documents:
        G.add_node(doc['id'])

    # Process all pairs
    total_pairs = len(documents) * (len(documents) - 1) // 2
    edges_added = 0

    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            weight = calculate_hybrid_edge_weight(
                documents[i], documents[j], threshold, metadata_source_mode
            )

            if weight > 0:
                G.add_edge(documents[i]['id'], documents[j]['id'], weight=weight)
                edges_added += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"    Processed {i+1}/{len(documents)} documents, {edges_added} edges so far...")

    logger.info(f"  Added {edges_added:,} edges out of {total_pairs:,} pairs")

    # Find connected components
    clusters = list(nx.connected_components(G))

    # Build cluster assignment mapping
    node_to_cluster = {}
    for cluster_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            node_to_cluster[node] = cluster_id

    logger.info(f"  Found {len(clusters):,} clusters")

    return node_to_cluster


def save_clustering_results(df: pd.DataFrame, cluster_assignments: dict[int, int],
                            threshold: float, output_dir: str, mode: str = "pure") -> str:
    """
    Save clustering results to CSV.

    Args:
        df: DataFrame with document data
        cluster_assignments: Mapping from doc ID to cluster ID
        threshold: Similarity threshold used
        output_dir: Directory to save output
        mode: "pure" or "hybrid"

    Returns:
        Path to output CSV
    """
    # Create copy to avoid modifying original
    df_output = df.copy()

    # Add cluster assignments (as list for compatibility with metadata pipeline)
    # Use 'id' column for hybrid mode, index for pure mode
    if mode == "hybrid":
        df_output['Parent Clusters'] = df_output['id'].map(
            lambda idx: [cluster_assignments.get(idx, -1)]
        )
    else:
        df_output['Parent Clusters'] = df_output.index.map(
            lambda idx: [cluster_assignments.get(idx, -1)]
        )

    # Save to CSV with mode in filename
    if mode == "hybrid":
        filename = f"test_clustering_results_embeddings_hybrid_{threshold:.2f}.csv"
    else:
        filename = f"test_clustering_results_embeddings_{threshold:.2f}.csv"

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    # Print statistics
    cluster_counts = pd.Series([c[0] for c in df_output['Parent Clusters']]).value_counts()
    logger.info(f"\nThreshold {threshold:.2f} ({mode.upper()}) Results:")
    logger.info(f"  Total documents: {len(df_output)}")
    logger.info(f"  Total clusters: {len(cluster_counts)}")
    logger.info(f"  Largest cluster: {cluster_counts.max()} documents")
    logger.info(f"  Average cluster size: {cluster_counts.mean():.2f}")
    logger.info(f"  Saved to: {output_path}")

    return str(output_path)


def process_threshold_pure(args):
    """
    Process a single threshold for PURE mode (for parallel execution).

    Args:
        args: Tuple of (df, similarity_matrix, threshold, output_dir)

    Returns:
        Output path
    """
    df, similarity_matrix, threshold, output_dir = args

    logger.info(f"Clustering with threshold {threshold:.2f} (PURE mode)...")
    cluster_assignments = cluster_with_threshold(similarity_matrix, threshold)
    output_path = save_clustering_results(df, cluster_assignments, threshold, output_dir, mode="pure")

    return output_path


def process_threshold_hybrid(args):
    """
    Process a single threshold for HYBRID mode (for parallel execution).

    Args:
        args: Tuple of (df, threshold, output_dir, metadata_source_mode)

    Returns:
        Output path
    """
    df, threshold, output_dir, metadata_source_mode = args

    logger.info(f"Clustering with threshold {threshold:.2f} (HYBRID mode)...")
    cluster_assignments = cluster_with_threshold_hybrid(df, threshold, metadata_source_mode)
    output_path = save_clustering_results(df, cluster_assignments, threshold, output_dir, mode="hybrid")

    return output_path


# ============================================================================
# CONFIGURATION
# ============================================================================
# Get absolute paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# One entry per agency — embeddings_csv must match generate_embeddings.py output filenames
# (input basename with _embeddings.csv appended, written to data/output/)
def _emb(filename: str) -> str:
    return str(PROJECT_DIR / "data/output" / filename)

AGENCIES = [
    # {'name': 'Bakersfield Police Department',     'embeddings_csv': _emb('autofolio_1.2.0_output--Bakersfield Police Department--2025-04-09_22-08-24 - autofolio_1.2.0_output--Bakersfield Police Department--2025-04-09_22-08-24_embeddings.csv')},
    {'name': 'Santa Monica Police Department',   'embeddings_csv': _emb('autofolio_1.1.0_output--Santa Monica Police Department--2024-12-21_02-18-20 - autofolio_1.1.0_output--Santa Monica Police Department--2024-12-21_02-18-20_embeddings.csv')},
    {'name': 'Richmond Police Department',       'embeddings_csv': _emb('autofolio_1.2.0_output--Richmond Police Department--2025-04-09_20-59-59 - autofolio_1.2.0_output--Richmond Police Department--2025-04-09_20-59-59_embeddings.csv')},
    {'name': 'Los Angeles District Attorney',    'embeddings_csv': _emb('autofolio_1.1.0_output--Los Angeles District Attorney--2024-11-27_05-47-23 - autofolio_1.1.0_output--Los Angeles District Attorney--2024-11-27_05-47-23_embeddings.csv')},
    {'name': 'California Department of Justice', 'embeddings_csv': _emb('autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52 - autofolio_1.2.0_output--California Department of Justice--2025-03-28_09-18-52_embeddings.csv')},
    {'name': 'Office of Inspector General for Prisons', 'embeddings_csv': _emb('autofolio_1.2.0_output--Office of Inspector General for Prisons--2025-04-27_21-48-50 - autofolio_1.2.0_output--Office of Inspector General for Prisons--2025-04-27_21-48-50_embeddings.csv')},
    {'name': 'Santa Ana Police Department',      'embeddings_csv': _emb('autofolio_1.1.0_output--Santa Ana Police Department--2025-02-13_01-55-05 - autofolio_1.1.0_output--Santa Ana Police Department--2025-02-13_01-55-05_embeddings.csv')},
    {'name': 'San Francisco Police Commission',  'embeddings_csv': _emb('autofolio_1.2.0_output--San Francisco Police Commission--2025-04-09_21-20-14 - autofolio_1.2.0_output--San Francisco Police Commission--2025-04-09_21-20-14_embeddings.csv')},
    {'name': 'Kern County Sheriff',              'embeddings_csv': _emb('autofolio_1.1.0_output--Kern County Sheriff--2024-07-15_23-59-08 - autofolio_1.1.0_output--Kern County Sheriff--2024-07-15_23-59-08_embeddings.csv')},
    {'name': 'Santa Clara County Sheriff',       'embeddings_csv': _emb('autofolio_1.1.0_output--Santa Clara County Sheriff--2024-12-13_15-09-21 - autofolio_1.1.0_output--Santa Clara County Sheriff--2024-12-13_15-09-21_embeddings.csv')},
    {'name': 'Fresno County Sheriff',            'embeddings_csv': _emb('autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42 - autofolio_1.1.0_output--Fresno County Sheriff--2024-10-05_00-33-42_embeddings.csv')},
    {'name': 'Sacramento County Sheriff',        'embeddings_csv': _emb('autofolio_1.1.0_output--Sacramento County Sheriff--2025-03-04_10-11-24 - autofolio_1.1.0_output--Sacramento County Sheriff--2025-03-04_10-11-24_embeddings.csv')},
    {'name': 'San Francisco County Sheriff',     'embeddings_csv': _emb('autofolio_1.1.0_output--San Francisco County Sheriff--2024-07-20_01-57-38 - autofolio_1.1.0_output--San Francisco County Sheriff--2024-07-20_01-57-38_embeddings.csv')},
    {'name': 'California Department of Corrections and Rehabilitation', 'embeddings_csv': _emb('OLDautofolio_1.2.0_output--California Department of Corrections and Rehabilitation--2025-03-24_23-57-31 - autofolio_1.2.0_output--California Department of Corrections and Rehabilitation--2025-03-24_23-57-31_embeddings.csv')},
    {'name': 'Folsom Police Department',         'embeddings_csv': _emb('autofolio_1.1.0_output--Folsom Police Department--2025-02-27_23-14-01 - autofolio_1.1.0_output--Folsom Police Department--2025-02-27_23-14-01_embeddings.csv')},
    {'name': 'UC Davis Police Department',       'embeddings_csv': _emb('autofolio_1.1.0_output--UC Davis Police Department--2024-07-20_08-53-00 - autofolio_1.1.0_output--UC Davis Police Department--2024-07-20_08-53-00_embeddings.csv')},
    {'name': 'Seal Beach Police Department',     'embeddings_csv': _emb('autofolio_1.1.0_output--Seal Beach Police Department--2025-02-13_08-16-22 - autofolio_1.1.0_output--Seal Beach Police Department--2025-02-13_08-16-22_embeddings.csv')},
    {'name': 'Contra Costa County District Attorney', 'embeddings_csv': _emb('autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04 - autofolio_1.1.0_output--Contra Costa County District Attorney--2024-11-06_07-38-04_embeddings.csv')},
    {'name': 'Contra Costa County Sheriff',      'embeddings_csv': _emb('autofolio_1.1.0_output--Contra Costa County Sheriff--2024-12-21_01-47-17 - autofolio_1.1.0_output--Contra Costa County Sheriff--2024-12-21_01-47-17_embeddings.csv')},
    {'name': 'Shasta County District Attorney',  'embeddings_csv': _emb('autofolio_1.1.0_output--Shasta County District Attorney--2024-12-18_05-02-13 - autofolio_1.1.0_output--Shasta County District Attorney--2024-12-18_05-02-13_embeddings.csv')},
    {'name': 'Riverside County Department of Public Social Services', 'embeddings_csv': _emb('autofolio_1.1.0_output--Riverside County Department of Public Social Services--2024-07-05_20-47-44 - autofolio_1.1.0_output--Riverside County Department of Public Social Services--2024-07-05_20-47-44_embeddings.csv')},
    {'name': 'Cal State East Bay University Police Department', 'embeddings_csv': _emb('autofolio_1.2.0_output--Cal State East Bay University Police Department--2025-04-09_20-14-32 - autofolio_1.2.0_output--Cal State East Bay University Police Department--2025-04-09_20-14-32_embeddings.csv')},
    {'name': 'San Joaquin County Medical Examiner', 'embeddings_csv': _emb('autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44 - autofolio_1.1.0_output--San Joaquin County Medical Examiner--2024-10-24_19-31-44_embeddings.csv')},
    {'name': 'Pasadena Police Department',       'embeddings_csv': _emb('autofolio_1.1.0_output--Pasadena Police Department--2025-01-24_06-19-29 - autofolio_1.1.0_output--Pasadena Police Department--2025-01-24_06-19-29_embeddings.csv')},
    {'name': 'Irvine Police Department',         'embeddings_csv': _emb('autofolio_1.1.0_output--Irvine Police Department--2024-07-09_01-45-57 - autofolio_1.1.0_output--Irvine Police Department--2024-07-09_01-45-57_embeddings.csv')},
    {'name': 'San Diego County Medical Examiner','embeddings_csv': _emb('autofolio_1.1.0_output--San Diego County Medical Examiner--2025-01-07_23-53-09 - autofolio_1.1.0_output--San Diego County Medical Examiner--2025-01-07_23-53-09_embeddings.csv')},
    {'name': 'San Leandro Police Department',    'embeddings_csv': _emb('autofolio_1.1.0_output--San Leandro Police Department--2024-08-15_22-50-51 - autofolio_1.1.0_output--San Leandro Police Department--2024-08-15_22-50-51_embeddings.csv')},
    {'name': 'Santa Clara Police Department',    'embeddings_csv': _emb('autofolio_1.1.0_output--Santa Clara Police Department--2024-06-01_21-38-12 - autofolio_1.1.0_output--Santa Clara Police Department--2024-06-01_21-38-12_embeddings.csv')},
    {'name': 'Hayward Police Department',        'embeddings_csv': _emb('autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37 - autofolio_1.2.0_output--Hayward Police Department--2025-04-24_18-36-37_embeddings.csv')},
    {'name': 'Vallejo Police Department',        'embeddings_csv': _emb('autofolio_1.2.0_output--Vallejo Police Department--2025-03-25_00-20-52 - autofolio_1.2.0_output--Vallejo Police Department--2025-03-25_00-20-52_embeddings.csv')},
    {'name': 'Chula Vista Police Department',    'embeddings_csv': _emb('autofolio_1.1.0_output--Chula Vista Police Department--2025-02-20_07-43-22 - autofolio_1.1.0_output--Chula Vista Police Department--2025-02-20_07-43-22_embeddings.csv')},


    # {'name': 'Orange County District Attorney',  'embeddings_csv': _emb('autofolio_1.2.0_output--Orange County District Attorney--2025-03-13_22-40-21 - autofolio_1.2.0_output--Orange County District Attorney--2025-03-13_22-40-21_embeddings.csv')},
    # {'name': 'Salinas Police Department',        'embeddings_csv': _emb('autofolio_1.1.0_output--Salinas Police Department--2024-12-05_13-18-13 - autofolio_1.1.0_output--Salinas Police Department--2024-12-05_13-18-13_embeddings.csv')},
    # {'name': 'CSU San Jose Police Department',   'embeddings_csv': _emb('autofolio_1.1.0_output--CSU San Jose Police Department--2024-05-31_13-46-58 - autofolio_1.1.0_output--CSU San Jose Police Department--2024-05-31_13-46-58_embeddings.csv')},
    # {'name': 'Brentwood Police Department',      'embeddings_csv': _emb('autofolio_1.1.0_output--Brentwood Police Department--2024-09-30_20-03-15 - autofolio_1.1.0_output--Brentwood Police Department--2024-09-30_20-03-15_embeddings.csv')},
    # {'name': 'San Ramon Police Department',      'embeddings_csv': _emb('OLDautofolio_1.1.0_output--San Ramon Police Department--2024-07-16_01-32-12 - autofolio_1.1.0_output--San Ramon Police Department--2024-07-16_01-32-12_embeddings.csv')},
    # {'name': 'Ontario Police Department',        'embeddings_csv': _emb('autofolio_1.2.0_output--Ontario Police Department--2025-04-24_16-59-27 - autofolio_1.2.0_output--Ontario Police Department--2025-04-24_16-59-27_embeddings.csv')},
    # {'name': 'Whittier Police Department',       'embeddings_csv': _emb('autofolio_1.1.0_output--Whittier Police Department--2024-08-05_15-28-07 - autofolio_1.1.0_output--Whittier Police Department--2024-08-05_15-28-07_embeddings.csv')},
]

OUTPUT_DIR = str(SCRIPT_DIR.parent.parent / "data" / "output" / "ablations_v2_dir_fallback")
THRESHOLDS = [0.75, 0.85]
METADATA_SOURCE_MODE = "combined"  # For hybrid mode fallback
# ============================================================================


def has_valid_embedding(row) -> bool:
    """Return True if this row has a real (non-null, non-zero) embedding."""
    emb = row.get('embedding')
    if emb is None or (isinstance(emb, float) and pd.isna(emb)):
        return False
    if isinstance(emb, np.ndarray) and np.allclose(emb, 0):
        return False
    return True


def process_threshold_for_agency(args):
    """
    Process a single threshold for an agency (for parallel execution).

    Args:
        args: Tuple of (df_with_emb, df_no_emb, similarity_matrix, threshold, output_dir, agency_name)

    Returns:
        Output path string
    """
    df_with_emb, df_no_emb, similarity_matrix, threshold, output_dir, agency_name = args

    logger.info(f"  t={threshold:.2f} ({agency_name})...")
    cluster_assignments = cluster_with_threshold(similarity_matrix, threshold)

    # Assign singletons unique IDs after the last embedding cluster ID
    max_cluster_id = max(cluster_assignments.values(), default=-1)
    singleton_assignments = {}
    for i, row_id in enumerate(df_no_emb['id']):
        singleton_assignments[row_id] = max_cluster_id + 1 + i

    # Build output DataFrames
    df_with_out = df_with_emb.copy()
    df_with_out['Parent Clusters'] = df_with_out.index.map(
        lambda idx: [cluster_assignments.get(idx, -1)]
    )

    df_no_out = df_no_emb.copy()
    df_no_out['Parent Clusters'] = df_no_out['id'].map(
        lambda i: [singleton_assignments.get(i, -1)]
    )

    # Merge and restore original row order
    df_output = pd.concat([df_with_out, df_no_out]).sort_values('id').reset_index(drop=True)

    # Save to agency subdir in ablations directory
    agency_dir = Path(output_dir) / agency_name.replace(' ', '_')
    filename = f"clustering_results_ablation_embeddings_t{threshold:.2f}.csv"
    output_path = agency_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    cluster_counts = pd.Series([c[0] for c in df_output['Parent Clusters']]).value_counts()
    logger.info(f"  t={threshold:.2f}: {len(cluster_counts)} clusters, "
                f"largest={cluster_counts.max()}, avg={cluster_counts.mean():.2f}")
    logger.info(f"  Saved: {output_path}")

    return str(output_path)


def cluster_agency(agency: dict, output_dir: str, thresholds: list):
    """Load embeddings for one agency, split singletons, cluster, save all thresholds."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    agency_name = agency['name']
    embeddings_csv = agency['embeddings_csv']
    start_time = time.time()

    if not Path(embeddings_csv).exists():
        logger.warning(f"  Embeddings CSV not found, skipping: {Path(embeddings_csv).name}")
        return

    # Load all rows; hybrid mode keeps null-embedding rows so we can track singletons
    df = load_embeddings(embeddings_csv, mode="hybrid")

    if len(df) == 0:
        logger.warning("  No rows found, skipping.")
        return

    # Split: docs with valid embeddings vs singletons (no OCR / zero vector)
    valid_mask = df.apply(has_valid_embedding, axis=1)
    df_with_emb = df[valid_mask].copy().reset_index(drop=True)
    df_no_emb = df[~valid_mask].copy().reset_index(drop=True)

    logger.info(f"  {len(df_with_emb)} docs with embeddings, "
                f"{len(df_no_emb)} singletons (no OCR / zero vector)")

    if len(df_with_emb) == 0:
        logger.warning("  No valid embeddings — all rows become singletons.")
        df_no_emb['Parent Clusters'] = [[i] for i in range(len(df_no_emb))]
        agency_dir = Path(output_dir) / agency_name.replace(' ', '_')
        agency_dir.mkdir(parents=True, exist_ok=True)
        for threshold in thresholds:
            filename = f"clustering_results_ablation_embeddings_t{threshold:.2f}.csv"
            out = agency_dir / filename
            df_no_emb.to_csv(out, index=False)
        return

    # Compute similarity matrix once (most expensive step)
    logger.info("  Computing cosine similarity matrix...")
    embeddings = df_with_emb['embedding'].tolist()
    similarity_matrix = compute_cosine_similarity_matrix(embeddings)
    logger.info(f"  Matrix shape: {similarity_matrix.shape}")

    # Dispatch all thresholds in parallel
    args_list = [
        (df_with_emb, df_no_emb, similarity_matrix, threshold, output_dir, agency_name)
        for threshold in thresholds
    ]
    with ProcessPoolExecutor(max_workers=len(thresholds)) as executor:
        list(executor.map(process_threshold_for_agency, args_list))

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")


def main():
    """Process all agencies sequentially; thresholds run in parallel within each agency."""
    import time
    start_time = time.time()

    logger.info("="*80)
    logger.info("EMBEDDINGS-BASED CLUSTERING (MULTI-AGENCY)")
    logger.info("="*80)
    logger.info(f"Agencies: {len(AGENCIES)}")
    logger.info(f"Thresholds: {THRESHOLDS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    for i, agency in enumerate(AGENCIES, 1):
        logger.info(f"\n[{i}/{len(AGENCIES)}] {agency['name']}")
        try:
            cluster_agency(agency, OUTPUT_DIR, THRESHOLDS)
        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("ALL AGENCIES COMPLETE")
    logger.info(f"Total time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*80}")


