import asyncio
import itertools
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from jinja2 import Template
from tqdm import tqdm

from prap_clustering._llm import get_llm
from prap_clustering.embeddings import cosine_similarity, embed_texts

from .directory_depth_analyzer import analyze_directory_depth, get_grouping_directory

# BLOCKLIST FILTERING DISABLED FOR ABLATION STUDIES
# from blocklist import filter_case_ids, filter_subject_names, filter_officer_names
from .frequency_filter import FrequencyFilter
from .helpers import (
    combine_features,
    normalize_case_ids,
    normalize_dates,
    normalize_names,
    parse_structured_case_ids,
    parse_structured_dates,
    parse_structured_officer_names,
    parse_structured_subject_names,
)
from .prompts import SUMMARY_COMPARISON_PROMPT, validate_summary_comparison_response
from .regex_extract_fp_fn import (
    extract_date_from_metadata,
    extract_ids_from_metadata,
    extract_names_from_metadata,
)
from .singleton_merge_validator import should_merge_singletons

# also try it after re-running regex

# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Stockton Police Department--2024-06-08_06-46-18 - autofolio_1.1.0_output--Stockton Police Department--2024-06-08_06-46-18_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Alameda County Sheriff--2025-02-13_07-19-00 - autofolio_1.1.0_output--Alameda County Sheriff--2025-02-13_07-19-00_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Del Norte County Sheriff--2024-10-08_14-59-22 - autofolio_1.1.0_output--Del Norte County Sheriff--2024-10-08_14-59-22_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Marin County Sheriff--2025-01-07_23-21-44 - autofolio_1.1.0_output--Marin County Sheriff--2025-01-07_23-21-44_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--San Mateo County Sheriff--2024-11-27_04-14-29 - autofolio_1.1.0_output--San Mateo County Sheriff--2024-11-27_04-14-29_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Anaheim Police Department--2024-11-06_01-21-49 - autofolio_1.1.0_output--Anaheim Police Department--2024-11-06_01-21-49_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Santa Cruz County Sheriff--2025-01-07_19-25-01 - autofolio_1.1.0_output--Santa Cruz County Sheriff--2025-01-07_19-25-01_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.2.0_output--San Jose Police Department--2025-04-08_00-13-55 - autofolio_1.2.0_output--San Jose Police Department--2025-04-08_00-13-55_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.2.0_output--Long Beach Police Department--2025-03-21_22-49-17 - autofolio_1.2.0_output--Long Beach Police Department--2025-03-21_22-49-17_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.1.0_output--Orange County Sheriff--2025-01-07_18-54-45 - autofolio_1.1.0_output--Orange County Sheriff--2025-01-07_18-54-45_ocr_col_dropped.csv"
# CSV_PATH = "../../data/output/structured_features/autofolio_1.2.0_output--Antioch Police Department--2025-04-24_19-06-53 - autofolio_1.2.0_output--Antioch Police Department--2025-04-24_19-06-53_ocr_col_dropped.csv"

OUTPUT_PATH = "../../data/output/clustering_results_hybrid_cascade_optimized.csv"


# ============================================================================
# ABLATION CONFIG
# ============================================================================


@dataclass
class DocumentFeatures:
    """Parsed and normalized features for one document."""

    # Tier 1: Filepath + Filename features (combined)
    tier1_case_ids: set[str]
    tier1_dates: set[str]
    tier1_names: set[str]

    # Tier 2: LLM-extracted features
    tier2_case_ids: set[str]
    tier2_dates: set[str]
    tier2_subject_names: set[str]
    tier2_officer_names: set[str]

    # Tier 3: Summaries
    feature_summaries: str | None  # Concatenated meaningful summaries
    meaningful_summary_types: set[str]  # Which summary types are meaningful (for blocking)

    # Tier 3: Embeddings (per summary type)
    case_ids_summary_embedding: np.ndarray | None  # (384,) embedding
    dates_summary_embedding: np.ndarray | None  # (384,) embedding
    subject_names_summary_embedding: np.ndarray | None  # (384,) embedding
    officer_names_summary_embedding: np.ndarray | None  # (384,) embedding

    # Metadata
    doc_id: str
    filepath: str  # Raw filepath for directory-based blocking
    has_tier1_features: bool
    has_tier2_features: bool
    has_tier3_summaries: bool


@dataclass
class MatchResult:
    """Result of comparing two documents."""

    matched: bool  # True if edge should be drawn
    tier: int  # Which tier found the match (1, 2, or 3)
    weight: float  # Edge weight (1.0 if matched, 0.0 if not)
    reason: str  # Description of why/how match was found
    doc_ids: tuple[str, str]  # (id1, id2) for tracking
    match_type: str  # Specific rule type: "case_id_only", "subject_name_only", etc.
    shared_features: dict[str, set[str]]  # Features that triggered the match
    hard_block: bool = False  # If True, both docs had case IDs but they didn't match — stop cascade


@dataclass
class AblationConfig:
    """Configuration for an ablation study run."""

    name: str
    description: str
    date_proximity_days: int
    require_dates_present: bool
    enabled_tiers: list[int]
    enabled_rules: dict[str, bool]
    use_learned_model: bool = False  # If True, use trained model instead of hand-coded rules
    model_path: str | None = None  # Path to trained model .pkl file
    feature_subset: str | None = None  # 'tier1_only', 'tier2_only', or 'both' for ML models

    # Feature-level disables (override enabled_rules for cleaner ablations)
    disable_case_ids: bool = False  # If True, ignore all case ID matching
    disable_dates: bool = False  # If True, ignore all date matching
    disable_subject_names: bool = False  # If True, ignore all subject name matching
    disable_officer_names: bool = False  # If True, ignore all officer name matching

    # Validation settings (override module-level ENABLE_VALIDATION)
    enable_validation: bool = True  # If False, skip connectivity threshold validation

    # Tier 3 embedding gate (override module-level EMBEDDING_SIMILARITY_THRESHOLD)
    embedding_similarity_threshold: float = 0.9  # Minimum embedding similarity to trigger Tier 3 LLM


def load_ablation_config(config_name: str) -> AblationConfig:
    """
    Load ablation configuration from YAML file.

    Checks both ablation_configs_handengineered.yaml (hand-coded rules) and
    ablation_configs_ml.yaml (learned rules).

    Args:
        config_name: Name of the ablation config to load

    Returns:
        AblationConfig object

    Raises:
        ValueError: If config_name not found in either YAML file
    """
    # Try both config files
    _config_root = Path(__file__).resolve().parent.parent / "configs"
    config_files = [
        _config_root / "ablation_configs_handengineered.yaml",
        _config_root / "ablation_configs_ml.yaml",
    ]

    all_available = []

    for config_path in config_files:
        if not config_path.exists():
            continue

        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        # Find the matching ablation config
        for ablation in yaml_data.get('ablations', []):
            all_available.append(ablation['name'])

            if ablation['name'] == config_name:
                return AblationConfig(
                    name=ablation['name'],
                    description=ablation['description'],
                    date_proximity_days=ablation['date_proximity_days'],
                    require_dates_present=ablation['require_dates_present'],
                    enabled_tiers=ablation['enabled_tiers'],
                    enabled_rules=ablation['enabled_rules'],
                    use_learned_model=ablation.get('use_learned_model', False),
                    disable_case_ids=ablation.get('disable_case_ids', False),
                    disable_dates=ablation.get('disable_dates', False),
                    disable_subject_names=ablation.get('disable_subject_names', False),
                    disable_officer_names=ablation.get('disable_officer_names', False),
                    model_path=ablation.get('model_path', None),
                    feature_subset=ablation.get('feature_subset', None),
                    enable_validation=ablation.get('enable_validation', True),
                    embedding_similarity_threshold=ablation.get('embedding_similarity_threshold', 0.9),
                )

    # If not found, raise error with available configs
    raise ValueError(f"Ablation config '{config_name}' not found. Available: {all_available}")


# ============================================================================
# LEARNED RULES SUPPORT
# ============================================================================


def compute_date_diff_days(dates1: set[str], dates2: set[str]) -> int:
    """
    Compute minimum date difference in days between two sets of dates.

    Returns:
        Minimum difference in days, or -1 if either set is empty.
        Using -1 (instead of 999999) makes it clear to ML models:
        negative = missing data, 0+ = valid distance.
    """
    if not dates1 or not dates2:
        return -1  # Changed from 999999 for cleaner ML semantics

    from datetime import datetime

    try:
        parsed_dates1 = [datetime.strptime(d, "%Y-%m-%d") for d in dates1]
        parsed_dates2 = [datetime.strptime(d, "%Y-%m-%d") for d in dates2]
    except ValueError:
        return -1  # Changed from 999999

    min_diff = float('inf')
    for d1 in parsed_dates1:
        for d2 in parsed_dates2:
            diff = abs((d1 - d2).days)
            min_diff = min(min_diff, diff)

    return int(min_diff)


def extract_pair_features_for_model(doc1: DocumentFeatures, doc2: DocumentFeatures) -> dict:
    """
    Extract features for a document pair (for trained model inference).

    This mirrors the feature extraction in learn_matching_rules.py.

    Args:
        doc1: DocumentFeatures for first document
        doc2: DocumentFeatures for second document

    Returns:
        Dictionary of features for this pair
    """
    features = {}

    # ========== TIER 1 FEATURES ==========

    # Case IDs
    case_id_overlap = doc1.tier1_case_ids & doc2.tier1_case_ids
    features["tier1_has_case_id_match"] = len(case_id_overlap) > 0
    features["tier1_num_shared_case_ids"] = len(case_id_overlap)

    # Dates
    date_overlap = doc1.tier1_dates & doc2.tier1_dates
    features["tier1_has_exact_date_match"] = len(date_overlap) > 0
    features["tier1_num_shared_dates"] = len(date_overlap)
    features["tier1_min_date_diff_days"] = compute_date_diff_days(
        doc1.tier1_dates, doc2.tier1_dates
    )
    # Use 0 <= diff <= N so that -1 (missing) is never treated as "within range"
    features["tier1_dates_within_30d"] = 0 <= features["tier1_min_date_diff_days"] <= 30
    features["tier1_dates_within_90d"] = 0 <= features["tier1_min_date_diff_days"] <= 90
    features["tier1_dates_within_180d"] = 0 <= features["tier1_min_date_diff_days"] <= 180
    features["tier1_dates_within_365d"] = 0 <= features["tier1_min_date_diff_days"] <= 365

    # Names
    name_overlap = doc1.tier1_names & doc2.tier1_names
    features["tier1_has_name_match"] = len(name_overlap) > 0
    features["tier1_num_shared_names"] = len(name_overlap)

    # Name characteristics
    if name_overlap:
        max_name_len = max(len(name) for name in name_overlap)
        has_full_name = any(" " in name for name in name_overlap)
        has_long_name = any(len(name) >= 8 for name in name_overlap)
    else:
        max_name_len = 0
        has_full_name = False
        has_long_name = False

    features["tier1_max_shared_name_length"] = max_name_len
    features["tier1_has_full_name_match"] = has_full_name
    features["tier1_has_long_name_match"] = has_long_name
    features["tier1_has_substantial_name_match"] = has_full_name or has_long_name

    # Directory overlap
    features["shared_dir_depth"] = get_shared_directory_depth(doc1.filepath, doc2.filepath)
    features["shared_2plus_dirs"] = features["shared_dir_depth"] >= 2

    # ========== TIER 2 FEATURES ==========

    # Case IDs
    tier2_case_id_overlap = doc1.tier2_case_ids & doc2.tier2_case_ids
    features["tier2_has_case_id_match"] = len(tier2_case_id_overlap) > 0
    features["tier2_num_shared_case_ids"] = len(tier2_case_id_overlap)

    # Dates
    tier2_date_overlap = doc1.tier2_dates & doc2.tier2_dates
    features["tier2_has_exact_date_match"] = len(tier2_date_overlap) > 0
    features["tier2_num_shared_dates"] = len(tier2_date_overlap)
    features["tier2_min_date_diff_days"] = compute_date_diff_days(
        doc1.tier2_dates, doc2.tier2_dates
    )
    # Use 0 <= diff <= N so that -1 (missing) is never treated as "within range"
    features["tier2_dates_within_30d"] = 0 <= features["tier2_min_date_diff_days"] <= 30
    features["tier2_dates_within_90d"] = 0 <= features["tier2_min_date_diff_days"] <= 90
    features["tier2_dates_within_180d"] = 0 <= features["tier2_min_date_diff_days"] <= 180
    features["tier2_dates_within_365d"] = 0 <= features["tier2_min_date_diff_days"] <= 365

    # Subject names
    subject_overlap = doc1.tier2_subject_names & doc2.tier2_subject_names
    features["tier2_has_subject_match"] = len(subject_overlap) > 0
    features["tier2_num_shared_subjects"] = len(subject_overlap)

    # Officer names
    officer_overlap = doc1.tier2_officer_names & doc2.tier2_officer_names
    features["tier2_has_officer_match"] = len(officer_overlap) > 0
    features["tier2_num_shared_officers"] = len(officer_overlap)

    # ========== COMBINED FEATURES ==========

    # Case ID from either tier
    features["any_case_id_match"] = (
        features["tier1_has_case_id_match"] or features["tier2_has_case_id_match"]
    )

    # Date from either tier
    features["any_date_match"] = (
        features["tier1_has_exact_date_match"] or features["tier2_has_exact_date_match"]
    )
    # Ignore -1 (missing) when taking the minimum: min(5, -1) should be 5, not -1
    _valid_diffs = [
        d for d in [features["tier1_min_date_diff_days"], features["tier2_min_date_diff_days"]]
        if d >= 0
    ]
    features["combined_min_date_diff_days"] = min(_valid_diffs) if _valid_diffs else -1
    features["combined_dates_within_365d"] = 0 <= features["combined_min_date_diff_days"] <= 365

    # Compound features (current hand-coded rules)
    features["tier1_case_id_and_close_dates"] = (
        features["tier1_has_case_id_match"] and features["tier1_dates_within_365d"]
    )
    features["tier1_name_and_date"] = (
        features["tier1_has_substantial_name_match"] and features["tier1_has_exact_date_match"]
    )
    features["tier2_case_id_and_close_dates"] = (
        features["tier2_has_case_id_match"] and features["tier2_dates_within_365d"]
    )
    features["tier2_subject_and_date"] = (
        features["tier2_has_subject_match"] and features["tier2_has_exact_date_match"]
    )
    features["tier2_subject_and_case_id"] = (
        features["tier2_has_subject_match"] and features["tier2_has_case_id_match"]
    )
    features["tier2_officer_date_and_case_id"] = (
        features["tier2_has_officer_match"]
        and features["tier2_has_exact_date_match"]
        and features["tier2_has_case_id_match"]
    )
    features["tier2_officer_date_and_subject"] = (
        features["tier2_has_officer_match"]
        and features["tier2_has_exact_date_match"]
        and features["tier2_has_subject_match"]
    )

    # ========== TIER AVAILABILITY FEATURES (Option B) ==========
    # Must match learn_matching_rules.py exactly so inference features == training features.
    features["tier1_has_any_data"] = (
        bool(doc1.tier1_case_ids or doc1.tier1_dates or doc1.tier1_names) and
        bool(doc2.tier1_case_ids or doc2.tier1_dates or doc2.tier1_names)
    )
    features["tier2_has_any_data"] = (
        bool(doc1.tier2_case_ids or doc1.tier2_dates or doc1.tier2_subject_names) and
        bool(doc2.tier2_case_ids or doc2.tier2_dates or doc2.tier2_subject_names)
    )

    return features


def filter_features_by_tier(feature_cols: list[str], tier_subset: str) -> list[str]:
    """
    Filter feature columns based on which tier(s) to use.

    Must match the filtering logic in learn_matching_rules.py for consistency.

    Args:
        feature_cols: All available feature column names
        tier_subset: One of 'tier1_only', 'tier2_only', 'both', or None (use all)

    Returns:
        Filtered list of feature columns
    """
    if tier_subset is None or tier_subset == "both":
        # Use all features (default behavior)
        return feature_cols

    if tier_subset == "tier1_only":
        # Only tier1_* features + directory features (those are tier1 structural)
        # NOTE: any_* and combined_* features use BOTH tiers, so exclude them
        return [col for col in feature_cols if
                col.startswith("tier1_") or
                col.startswith("shared_dir")]

    elif tier_subset == "tier2_only":
        # Only tier2_* features + directory features (shared_dir is a structural baseline for all tiers)
        return [col for col in feature_cols if
                col.startswith("tier2_") or
                col.startswith("shared_dir")]

    else:
        raise ValueError(f"Unknown tier_subset: {tier_subset}. Must be 'tier1_only', 'tier2_only', 'both', or None")


def extract_pair_features_wrapper(pair_tuple):
    """
    Wrapper for extract_pair_features_for_model to enable multiprocessing.

    Args:
        pair_tuple: Tuple of (doc1, doc2)

    Returns:
        Dictionary of features for this pair
    """
    doc1, doc2 = pair_tuple
    return extract_pair_features_for_model(doc1, doc2)


def load_trained_model(model_path: str):
    """
    Load a trained model from pickle file.

    Args:
        model_path: Path to .pkl file

    Returns:
        Trained model object
    """
    import pickle

    logger.info(f"Loading trained model from: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"  Model type: {type(model).__name__}")
    return model


def _build_feature_row(pair_features: dict, feature_names: list[str], use_nan: bool = False) -> pd.DataFrame:
    """Build a single-row DataFrame for model prediction.

    Args:
        pair_features: Raw feature dict from extract_pair_features_for_model()
        feature_names: Ordered feature names expected by the model
        use_nan: If True (LightGBM), replace -1 sentinels with NaN in date-diff columns
    """
    feature_vector = []
    for feat_name in feature_names:
        value = pair_features.get(feat_name, 0)
        if isinstance(value, bool):
            value = int(value)
        feature_vector.append(value)
    X = pd.DataFrame([feature_vector], columns=feature_names).astype(float)
    if use_nan:
        date_diff_cols = [c for c in X.columns if "min_date_diff_days" in c]
        for col in date_diff_cols:
            X[col] = X[col].replace(-1.0, float("nan"))
    return X


def compare_with_learned_model(
    doc1: DocumentFeatures,
    doc2: DocumentFeatures,
    model,
    feature_names: list[str]
) -> MatchResult | None:
    """
    Compare two documents using a trained model instead of hand-coded rules.

    Supports three model types:
    - Standard sklearn model (DecisionTreeClassifier, RandomForestClassifier)
    - LightGBM model (use_nan sentinel handling applied automatically)
    - Cascade OR bundle (dict with type="cascade_or", tier1_model, tier2_model)

    Args:
        doc1: First document features
        doc2: Second document features
        model: Trained model or cascade bundle dict
        feature_names: List of feature names (ignored for cascade — bundle contains its own)

    Returns:
        MatchResult if matched, None if no match
    """
    pair_features = extract_pair_features_for_model(doc1, doc2)

    # ---- Option A: Cascade OR bundle ----
    if isinstance(model, dict) and model.get("type") == "cascade_or":
        t1_model = model["tier1_model"]
        t1_feats = model["tier1_features"]
        t1_nan   = model.get("tier1_use_nan", False)
        t2_model = model["tier2_model"]
        t2_feats = model["tier2_features"]
        t2_nan   = model.get("tier2_use_nan", False)

        X1 = _build_feature_row(pair_features, t1_feats, use_nan=t1_nan)
        X2 = _build_feature_row(pair_features, t2_feats, use_nan=t2_nan)
        prediction = int(t1_model.predict(X1)[0]) or int(t2_model.predict(X2)[0])

        if prediction == 1:
            return MatchResult(
                matched=True,
                tier=1,
                weight=1.0,
                reason="cascade_or match",
                doc1_id=doc1.doc_id,
                doc2_id=doc2.doc_id,
            )
        return None

    # ---- Standard sklearn / LightGBM model ----
    # Detect LightGBM by class name (avoids hard import dependency)
    use_nan = type(model).__name__ == "LGBMClassifier"
    X = _build_feature_row(pair_features, feature_names, use_nan=use_nan)

    prediction = model.predict(X)[0]

    if prediction == 1:
        # Model predicts MATCH
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0][1]  # Probability of class 1 (match)
        else:
            proba = 1.0

        return MatchResult(
            matched=True,
            tier=1,  # Use tier 1 for learned rules (could be configurable)
            weight=1.0,
            reason=f"Learned model match (confidence={proba:.3f})",
            doc_ids=(doc1.doc_id, doc2.doc_id),
            match_type="learned_model",
            shared_features={},  # Model doesn't expose which features triggered match
        )
    else:
        # Model predicts NO MATCH
        return None


# ============================================================================
# CONFIG
# ============================================================================

TIER3_SIMILARITY_THRESHOLD = 1.0  # Strict matching for Tier 3
EMBEDDING_SIMILARITY_THRESHOLD = 0.9  # Minimum embedding similarity to trigger LLM
DEBUG = True

# Validation settings
ENABLE_VALIDATION = True  # Apply connectivity threshold validation
VALIDATION_THRESHOLD = .3

# Frequency filter settings
# Case IDs appearing in more than this percentage of documents are filtered out
# (likely metadata, template IDs, or other artifacts rather than real case IDs)
CASE_ID_FREQUENCY_THRESHOLD_PERCENT = 0.1  # 10% of corpus
DATE_FREQUENCY_THRESHOLD_PERCENT = 0.1  # 10% of corpus
NAME_FREQUENCY_THRESHOLD_PERCENT = 0.1  # 10% of corpus

# test 2
# Tier 1 blocking settings
TIER1_MIN_SHARED_DIRS = 2  # Minimum shared directory levels for Tier 1 comparison

# .66 oak, .79 lapd, .60 fresno, .84 laso, .75 san-bern-so

# test 1
### v3 .1 validation with .95 embedding threshold

# oak: .62, .79 lapd, .64 fresno

# Performance tuning
MAX_CONCURRENT_LLM = 100  # High concurrency for Tier 3 async calls

# Post-processing settings
ENABLE_DIRECTORY_FALLBACK = False  # Enable directory-based singleton clustering after main clustering
                                    # Set to False for clean ablations (see true feature performance)
                                    # Set to True to demonstrate utility of directory-based fallback

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    format="%(process)d\t%(asctime)s\t%(levelname)s\t| %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if DEBUG:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.WARNING)


# ============================================================================
# REGEX RE-EXTRACTION
# ============================================================================


def rerun_regex_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run regex extraction on filepath/filename columns to ensure latest logic is used.

    This overwrites the extracted_*_fp and extracted_*_fn columns with fresh extractions
    using the current regex patterns. This ensures any improvements to the regex logic
    are applied before clustering, without needing to re-run expensive LLM extraction.

    Args:
        df: DataFrame with gdrive_path and gdrive_name/file_name_from_json columns

    Returns:
        DataFrame with updated extraction columns
    """
    logger.info("=" * 80)
    logger.info("RE-RUNNING REGEX EXTRACTION (ensuring latest patterns are used)")
    logger.info("=" * 80)

    # Determine which column to use for filename
    fn_col = 'gdrive_name' if 'gdrive_name' in df.columns else 'file_name_from_json'
    fp_col = 'gdrive_path'

    logger.info(f"  Filepath column: {fp_col}")
    logger.info(f"  Filename column: {fn_col}")

    # Track counts for logging
    total = len(df)
    fp_case_ids_count = 0
    fn_case_ids_count = 0
    fp_dates_count = 0
    fn_dates_count = 0
    fp_names_count = 0
    fn_names_count = 0

    # Re-extract for each row
    for idx in range(total):
        filepath = df.iloc[idx].get(fp_col, '')
        filename = df.iloc[idx].get(fn_col, '')

        # Convert to string, handle NaN
        filepath = str(filepath) if pd.notna(filepath) else ''
        filename = str(filename) if pd.notna(filename) else ''

        # Extract from filepath
        fp_case_ids = extract_ids_from_metadata(filepath)
        fp_dates = extract_date_from_metadata(filepath)
        fp_names = extract_names_from_metadata(filepath)

        # Extract from filename
        fn_case_ids = extract_ids_from_metadata(filename)
        fn_dates = extract_date_from_metadata(filename)
        fn_names = extract_names_from_metadata(filename)

        # Update DataFrame
        df.at[df.index[idx], 'extracted_case_ids_fp'] = str(fp_case_ids) if fp_case_ids else '[]'
        df.at[df.index[idx], 'extracted_case_ids_fn'] = str(fn_case_ids) if fn_case_ids else '[]'
        df.at[df.index[idx], 'extracted_dates_fp'] = str(fp_dates) if fp_dates else '[]'
        df.at[df.index[idx], 'extracted_dates_fn'] = str(fn_dates) if fn_dates else '[]'
        df.at[df.index[idx], 'extracted_names_fp'] = str(fp_names) if fp_names else '[]'
        df.at[df.index[idx], 'extracted_names_fn'] = str(fn_names) if fn_names else '[]'

        # Track counts
        if fp_case_ids: fp_case_ids_count += 1
        if fn_case_ids: fn_case_ids_count += 1
        if fp_dates: fp_dates_count += 1
        if fn_dates: fn_dates_count += 1
        if fp_names: fp_names_count += 1
        if fn_names: fn_names_count += 1

    logger.info(f"\nRe-extraction complete for {total:,} documents:")
    logger.info(f"  Case IDs from filepath: {fp_case_ids_count:,} docs ({fp_case_ids_count/total*100:.1f}%)")
    logger.info(f"  Case IDs from filename: {fn_case_ids_count:,} docs ({fn_case_ids_count/total*100:.1f}%)")
    logger.info(f"  Dates from filepath:    {fp_dates_count:,} docs ({fp_dates_count/total*100:.1f}%)")
    logger.info(f"  Dates from filename:    {fn_dates_count:,} docs ({fn_dates_count/total*100:.1f}%)")
    logger.info(f"  Names from filepath:    {fp_names_count:,} docs ({fp_names_count/total*100:.1f}%)")
    logger.info(f"  Names from filename:    {fn_names_count:,} docs ({fn_names_count/total*100:.1f}%)")
    logger.info("=" * 80 + "\n")

    return df


# ============================================================================
# DATA STRUCTURES
# ============================================================================



# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def extract_document_features(row: pd.Series, freq_filter: FrequencyFilter = None) -> DocumentFeatures:
    """
    Extract and normalize all features from a document row.

    Args:
        row: DataFrame row with all extracted feature columns
        freq_filter: Optional frequency filter to remove high-frequency features

    Returns:
        DocumentFeatures object with normalized feature sets
    """

    # ========== TIER 1: Filepath + Filename (Combined) ==========

    # Combine fp + fn case IDs
    tier1_case_ids_raw = combine_features(
        row.get("extracted_case_ids_fp"), row.get("extracted_case_ids_fn")
    )
    tier1_case_ids = normalize_case_ids(tier1_case_ids_raw)  # Blocklist filtering disabled
    # Apply frequency filter to remove high-frequency case IDs
    if freq_filter:
        tier1_case_ids = freq_filter.filter_case_ids(tier1_case_ids)

    # Combine fp + fn dates
    tier1_dates_raw = combine_features(
        row.get("extracted_dates_fp"), row.get("extracted_dates_fn")
    )
    tier1_dates = normalize_dates(tier1_dates_raw)
    # Apply frequency filter to remove metadata dates
    if freq_filter:
        tier1_dates = freq_filter.filter_dates(tier1_dates)

    # Combine fp + fn names
    tier1_names_raw = combine_features(
        row.get("extracted_names_fp"), row.get("extracted_names_fn")
    )
    tier1_names = normalize_names(tier1_names_raw)
    # Apply frequency filter only (blocklist filtering disabled)
    if freq_filter:
        tier1_names = freq_filter.filter_names(tier1_names)

    # ========== TIER 2: LLM-Extracted Features (Structured) ==========

    # LLM case IDs from structured JSON: [{"id": "IA2018-0167"}, {"id": "2018-0167"}]
    tier2_case_ids_raw = parse_structured_case_ids(row.get("extracted_case_ids_llm_structured"))
    tier2_case_ids = normalize_case_ids(tier2_case_ids_raw)  # Blocklist filtering disabled
    # Apply frequency filter to remove high-frequency case IDs
    if freq_filter:
        tier2_case_ids = freq_filter.filter_case_ids(tier2_case_ids)

    # LLM dates from structured JSON: {"incident_date": "2024-01-26"}
    tier2_dates_raw = parse_structured_dates(row.get("extracted_dates_llm_structured"))
    tier2_dates = normalize_dates(tier2_dates_raw)

    # LLM subject names from structured JSON: [{"name": "Kevin Bushnell", "subject_type": "suspect"}]
    tier2_subject_names_raw = parse_structured_subject_names(row.get("extracted_subject_names_llm_structured"))
    tier2_subject_names = normalize_names(tier2_subject_names_raw)  # Blocklist filtering disabled

    # LLM officer names from structured JSON: [{"name": "Officer Butera", "context": "responded to scene"}]
    tier2_officer_names_raw = parse_structured_officer_names(row.get("extracted_officer_names_llm_structured"))
    tier2_officer_names = normalize_names(tier2_officer_names_raw)  # Blocklist filtering disabled

    # ========== TIER 3: Summaries ==========

    # Helper to check if summary is meaningful (not placeholder)
    def is_meaningful_summary(text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower().strip()
        # Check for common placeholder patterns
        placeholders = [
            "no officers on this page",
            "no civilians on this page",
            "no case identifiers on this page",
            "no dates on this page",
            "no officer",
            "no civilian",
            "no case",
            "no date",
        ]
        return not any(placeholder in text_lower for placeholder in placeholders)

    # Track which summary types are meaningful
    meaningful_summary_types = set()

    # First, check if document has at least one meaningful summary
    # from the critical features: case_ids, dates, subject_names
    has_meaningful_core_summary = False

    for summary_col in ["case_ids_summary", "dates_summary", "subject_names_summary"]:
        summary_val = row.get(summary_col)
        if pd.notna(summary_val) and summary_val:
            if is_meaningful_summary(str(summary_val).strip()):
                has_meaningful_core_summary = True
                meaningful_summary_types.add(summary_col)

    # If document has at least one meaningful core summary, build concatenated summary
    # including ALL meaningful summaries (case_ids, dates, subjects, officers)
    feature_summaries = None
    if has_meaningful_core_summary:
        feature_summaries_parts = []

        for summary_col in [
            "case_ids_summary",
            "dates_summary",
            "subject_names_summary",
            "officer_names_summary",
        ]:
            summary_val = row.get(summary_col)
            if pd.notna(summary_val) and summary_val:
                summary_str = str(summary_val).strip()
                if is_meaningful_summary(summary_str):
                    feature_summaries_parts.append(summary_str)
                    # Also track officer_names if meaningful
                    if summary_col == "officer_names_summary":
                        meaningful_summary_types.add(summary_col)

        feature_summaries = (
            "\n\n".join(feature_summaries_parts) if feature_summaries_parts else None
        )

    # ========== Tier 3: Embeddings ==========

    # Embed each meaningful summary type individually (for pairwise comparison)
    case_ids_summary_embedding = None
    dates_summary_embedding = None
    subject_names_summary_embedding = None
    officer_names_summary_embedding = None

    # Only embed if we have meaningful summaries
    if has_meaningful_core_summary:
        # Build a list of (summary_type, summary_text) for batch embedding
        summaries_to_embed = []
        summary_types_list = []

        for summary_col in [
            "case_ids_summary",
            "dates_summary",
            "subject_names_summary",
            "officer_names_summary",
        ]:
            if summary_col in meaningful_summary_types:
                summary_val = row.get(summary_col)
                if pd.notna(summary_val) and summary_val:
                    summary_str = str(summary_val).strip()
                    if is_meaningful_summary(summary_str):
                        summaries_to_embed.append(summary_str)
                        summary_types_list.append(summary_col)

        # Batch embed all summaries at once
        if summaries_to_embed:
            embeddings = embed_texts(summaries_to_embed)

            # Map embeddings back to their types
            for i, summary_type in enumerate(summary_types_list):
                if summary_type == "case_ids_summary":
                    case_ids_summary_embedding = embeddings[i]
                elif summary_type == "dates_summary":
                    dates_summary_embedding = embeddings[i]
                elif summary_type == "subject_names_summary":
                    subject_names_summary_embedding = embeddings[i]
                elif summary_type == "officer_names_summary":
                    officer_names_summary_embedding = embeddings[i]

    # ========== Metadata ==========

    has_tier1 = bool(tier1_case_ids or tier1_dates or tier1_names)
    has_tier2 = bool(
        tier2_case_ids or tier2_dates or tier2_subject_names or tier2_officer_names
    )
    # Only has Tier 3 if document has at least one meaningful core summary
    has_tier3 = has_meaningful_core_summary

    # Get filepath for directory-based blocking.
    # get_shared_directory_depth uses parts[:-1] to strip the filename, so the
    # path must include the filename.  gdrive_path is a directory path; append
    # gdrive_name so the depth calculation is correct.
    filepath = row.get("gdrive_path", "")
    if pd.isna(filepath):
        filepath = ""
    filepath = str(filepath)
    gdrive_name_val = row.get("gdrive_name", "")
    if not pd.isna(gdrive_name_val) and gdrive_name_val:
        gdrive_name_val = str(gdrive_name_val)
        if not filepath.endswith(gdrive_name_val):
            filepath = filepath.rstrip("/") + "/" + gdrive_name_val

    return DocumentFeatures(
        tier1_case_ids=tier1_case_ids,
        tier1_dates=tier1_dates,
        tier1_names=tier1_names,
        tier2_case_ids=tier2_case_ids,
        tier2_dates=tier2_dates,
        tier2_subject_names=tier2_subject_names,
        tier2_officer_names=tier2_officer_names,
        feature_summaries=feature_summaries,
        meaningful_summary_types=meaningful_summary_types,
        case_ids_summary_embedding=case_ids_summary_embedding,
        dates_summary_embedding=dates_summary_embedding,
        subject_names_summary_embedding=subject_names_summary_embedding,
        officer_names_summary_embedding=officer_names_summary_embedding,
        doc_id=str(row.get("gdrive_id", row.name)),  # Use gdrive_id or row index
        filepath=filepath,
        has_tier1_features=has_tier1,
        has_tier2_features=has_tier2,
        has_tier3_summaries=has_tier3,
    )


# ============================================================================
# TIER COMPARISON FUNCTIONS
# ============================================================================


def get_shared_directory_depth(path1: str, path2: str) -> int:
    """
    Calculate how many leading directory levels two paths share.

    Example:
        path1 = "Oakland Police Department/batch1/case_123/doc.pdf"
        path2 = "Oakland Police Department/batch1/case_456/doc.pdf"
        Returns: 2 (share "Oakland Police Department" and "batch1")

    Args:
        path1: First filepath
        path2: Second filepath

    Returns:
        Number of shared leading directory levels
    """
    # Split paths into directory components (exclude filename)
    parts1 = Path(path1).parts[:-1] if path1 else ()
    parts2 = Path(path2).parts[:-1] if path2 else ()

    # Count matching leading directories
    shared = 0
    for p1, p2 in zip(parts1, parts2, strict=False):
        if p1 == p2:
            shared += 1
        else:
            break

    return shared


def should_compare_tier1(
    doc1: DocumentFeatures, doc2: DocumentFeatures, min_shared_dirs: int = 2
) -> bool:
    """
    Check if two documents should be compared in Tier 1 based on directory overlap.

    Blocking rule: Only compare if they share at least min_shared_dirs directory levels.
    This reduces comparisons between documents in unrelated directory structures.

    Args:
        doc1: First document features
        doc2: Second document features
        min_shared_dirs: Minimum shared directory levels required (default: 2)

    Returns:
        True if documents should be compared in Tier 1, False to skip
    """
    # If either document has no filepath, don't block (allow comparison)
    if not doc1.filepath or not doc2.filepath:
        return True

    shared = get_shared_directory_depth(doc1.filepath, doc2.filepath)
    return shared >= min_shared_dirs


def should_compare_tier3(
    doc1: DocumentFeatures,
    doc2: DocumentFeatures,
    embedding_threshold: float = 0.7,
) -> bool:
    """
    Check if two documents should be compared in Tier 3 using embedding-based filtering.

    Blocking rules:
    1. Both documents must have tier3 summaries
    2. Both must share at least one meaningful summary type
    3. Max embedding cosine similarity across shared types must be >= threshold

    Args:
        doc1: First document features
        doc2: Second document features
        embedding_threshold: Minimum cosine similarity to trigger LLM comparison

    Returns:
        True if documents pass all blocking rules, False otherwise
    """
    # Both must have tier3 summaries
    if not (doc1.has_tier3_summaries and doc2.has_tier3_summaries):
        return False

    # Check if they share at least one meaningful summary type
    shared_types = doc1.meaningful_summary_types & doc2.meaningful_summary_types
    if len(shared_types) == 0:
        return False

    # Calculate max cosine similarity across shared summary types
    max_similarity = 0.0

    # Map summary types to embedding fields
    embedding_map = {
        "case_ids_summary": ("case_ids_summary_embedding", "case_ids_summary_embedding"),
        "dates_summary": ("dates_summary_embedding", "dates_summary_embedding"),
        "subject_names_summary": (
            "subject_names_summary_embedding",
            "subject_names_summary_embedding",
        ),
        "officer_names_summary": (
            "officer_names_summary_embedding",
            "officer_names_summary_embedding",
        ),
    }

    for summary_type in shared_types:
        if summary_type in embedding_map:
            emb1_field, emb2_field = embedding_map[summary_type]
            emb1 = getattr(doc1, emb1_field)
            emb2 = getattr(doc2, emb2_field)

            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity(emb1, emb2)
                max_similarity = max(max_similarity, similarity)

    # Only compare if max similarity >= threshold
    return max_similarity >= embedding_threshold


def dates_within_range(
    dates1: set[str],
    dates2: set[str],
    days: int = 365,
    require_dates_present: bool = False
) -> bool:
    """
    Check if two sets of dates have any pair within the specified range.

    Args:
        dates1: First set of dates (YYYY-MM-DD format)
        dates2: Second set of dates (YYYY-MM-DD format)
        days: Maximum allowed difference in days (default: 365)
        require_dates_present: If True, return False when dates are missing (default: False)

    Returns:
        True if any pair of dates is within range
        False if dates missing (when require_dates_present=True) or all pairs exceed range
    """
    # Handle missing dates based on require_dates_present setting
    if not dates1 or not dates2:
        # If dates are required to be present, return False (block match)
        # Otherwise, return True (missing dates don't block match)
        return not require_dates_present

    from datetime import datetime

    # Parse all dates
    try:
        parsed_dates1 = [datetime.strptime(d, "%Y-%m-%d") for d in dates1]
        parsed_dates2 = [datetime.strptime(d, "%Y-%m-%d") for d in dates2]
    except ValueError:
        # If parsing fails, treat same as missing dates
        return not require_dates_present

    # Check if any pair of dates is within range
    for d1 in parsed_dates1:
        for d2 in parsed_dates2:
            diff_days = abs((d1 - d2).days)
            if diff_days <= days:
                return True

    # All pairs exceed the range
    return False


# def compare_tier1(
#     doc1: DocumentFeatures,
#     doc2: DocumentFeatures,
#     config: AblationConfig
# ) -> Optional[MatchResult]:
#     """
#     Tier 1: Compare filepath + filename features.

#     Rule: (any_case_id_match AND dates_within_range) OR (any_name_match AND any_date_match)

#     Args:
#         doc1: First document features
#         doc2: Second document features
#         config: Ablation configuration controlling which rules are enabled

#     Returns:
#         MatchResult if matched, None if no match
#     """

#     # Track case ID conflict for potential hard block at the end
#     case_id_conflict = False

#     # Rule 1: Case ID match with date proximity check
#     if config.enabled_rules.get("tier1_case_id", True):
#         # Check feature-level disable flag
#         if config.disable_case_ids:
#             case_id_overlap = set()  # Force empty if disabled
#         else:
#             case_id_overlap = doc1.tier1_case_ids & doc2.tier1_case_ids

#         if case_id_overlap:
#             # Check if dates are within reasonable range (prevents ambiguous IDs across years)
#             # Only use tier1_dates — no fallback to tier2 (tiers are isolated)
#             dates1 = doc1.tier1_dates
#             dates2 = doc2.tier1_dates

#             if not dates_within_range(
#                 dates1,
#                 dates2,
#                 days=config.date_proximity_days,
#                 require_dates_present=config.require_dates_present
#             ):
#                 # Dates differ by more than allowed window - ambiguous case ID, don't match
#                 return None

#             return MatchResult(
#                 matched=True,
#                 tier=1,
#                 weight=1.0,
#                 reason=f"Tier 1: Case ID match ({', '.join(list(case_id_overlap)[:3])})",
#                 doc_ids=(doc1.doc_id, doc2.doc_id),
#                 match_type="case_id_only",
#                 shared_features={"tier1_case_ids": case_id_overlap},
#             )

#         elif not config.disable_case_ids and doc1.tier1_case_ids and doc2.tier1_case_ids:
#             # Both docs have case IDs but they don't overlap — record conflict, keep checking
#             # other rules before deciding whether to hard block
#             case_id_conflict = True

#     # Rule 2: Name match AND date match
#     if config.enabled_rules.get("tier1_name_date", True):
#         # Check feature-level disable flags
#         if config.disable_subject_names and config.disable_officer_names:
#             name_overlap = set()  # Force empty if all names disabled
#         else:
#             name_overlap = doc1.tier1_names & doc2.tier1_names

#         if config.disable_dates:
#             date_overlap = set()  # Force empty if dates disabled
#         else:
#             date_overlap = doc1.tier1_dates & doc2.tier1_dates

#         if name_overlap and date_overlap:
#             # Filter out short single-word names to avoid false positives from common last names
#             # Require either:
#             # - Full name (contains space, e.g., "john smith")
#             # - OR single name with >= 8 characters (e.g., "washington" but not "garcia")
#             substantial_names = {
#                 name for name in name_overlap
#                 if ' ' in name  # Full name (first + last)
#                 or len(name) >= 8  # Long single name (likely distinctive)
#             }

#             if substantial_names:
#                 return MatchResult(
#                     matched=True,
#                     tier=1,
#                     weight=1.0,
#                     reason=f"Tier 1: Name + Date match ({list(substantial_names)[0]}, {list(date_overlap)[0]})",
#                     doc_ids=(doc1.doc_id, doc2.doc_id),
#                     match_type="name_and_date",
#                     shared_features={"tier1_names": substantial_names, "tier1_dates": date_overlap},
#                 )
#             # Otherwise, no match - short single names are too ambiguous

#     # Rule 3: Exact directory path match with sufficient depth
#     # If both docs share an identical full directory path (excluding filename) and that
#     # path has more than 3 components, they almost certainly belong to the same incident folder.
#     if config.enabled_rules.get("tier1_exact_dir", True):
#         dir_parts1 = Path(doc1.filepath).parts[:-1] if doc1.filepath else ()
#         dir_parts2 = Path(doc2.filepath).parts[:-1] if doc2.filepath else ()

#         if len(dir_parts1) > 3 and dir_parts1 == dir_parts2:
#             return MatchResult(
#                 matched=True,
#                 tier=1,
#                 weight=1.0,
#                 reason=f"Tier 1: Exact directory match ({'/'.join(dir_parts1)})",
#                 doc_ids=(doc1.doc_id, doc2.doc_id),
#                 match_type="exact_dir_match",
#                 shared_features={},
#             )

#     # No rule matched. If case IDs conflicted, hard block — no point routing to Tier 2.
#     # If no conflict (one or both docs lacked case IDs), return None and let Tier 2 try.
#     if case_id_conflict:
#         return MatchResult(
#             matched=False,
#             hard_block=True,
#             tier=1,
#             weight=0.0,
#             reason="Tier 1: Case ID mismatch (both docs have IDs but none overlap)",
#             doc_ids=(doc1.doc_id, doc2.doc_id),
#             match_type="case_id_mismatch",
#             shared_features={},
#         )

#     return None


def compare_tier1(
    doc1: DocumentFeatures,
    doc2: DocumentFeatures,
    config: AblationConfig
) -> MatchResult | None:
    """
    Tier 1: Compare filepath + filename features.

    Three-outcome design:
      YES  (match)     — case ID overlap, OR (no ID conflict AND dir/name+date match)
      MAYBE (pass T2)  — ID conflict with corroborating signals, OR no IDs and no signal
      NO   (hard block)— case ID conflict with no corroborating signals

    Rule ordering (highest trust first):
      1. Case ID match         — always YES; IDs are isolated and stand alone
      2. Exact directory match — YES if no conflict; MAYBE if conflict
      3. Name + Date match     — YES if no conflict; MAYBE if conflict

    Fallthrough logic:
      - no case IDs, no signal → MAYBE (pass to T2, nothing to block on)
      - case_id_conflict + no signal → NO (hard block)

    Args:
        doc1: First document features
        doc2: Second document features
        config: Ablation configuration controlling which rules are enabled

    Returns:
        MatchResult(matched=True)       — YES
        None                            — MAYBE
        MatchResult(hard_block=True)    — NO
    """

    case_id_conflict = False
    any_signal = False

    # Rule 1: Case ID match — always YES
    if config.enabled_rules.get("tier1_case_id", True):
        if config.disable_case_ids:
            case_id_overlap = set()
        else:
            case_id_overlap = doc1.tier1_case_ids & doc2.tier1_case_ids

        if case_id_overlap:
            return MatchResult(
                matched=True,
                tier=1,
                weight=1.0,
                reason=f"Tier 1: Case ID match ({', '.join(list(case_id_overlap)[:3])})",
                doc_ids=(doc1.doc_id, doc2.doc_id),
                match_type="case_id_only",
                shared_features={"tier1_case_ids": case_id_overlap},
            )

        elif not config.disable_case_ids and doc1.tier1_case_ids and doc2.tier1_case_ids:
            case_id_conflict = True

    # Rule 2: Exact directory match — YES if no conflict, MAYBE if conflict
    if config.enabled_rules.get("tier1_exact_dir", True):
        dir_parts1 = Path(doc1.filepath).parts[:-1] if doc1.filepath else ()
        dir_parts2 = Path(doc2.filepath).parts[:-1] if doc2.filepath else ()

        if len(dir_parts1) > 3 and dir_parts1 == dir_parts2:
            if not case_id_conflict:
                return MatchResult(
                    matched=True,
                    tier=1,
                    weight=1.0,
                    reason=f"Tier 1: Exact directory match ({'/'.join(dir_parts1)})",
                    doc_ids=(doc1.doc_id, doc2.doc_id),
                    match_type="exact_dir_match",
                    shared_features={},
                )
            any_signal = True

    # Rule 3: Name + Date match — YES if no conflict, MAYBE if conflict
    if config.enabled_rules.get("tier1_name_date", True):
        if config.disable_subject_names and config.disable_officer_names:
            name_overlap = set()
        else:
            name_overlap = doc1.tier1_names & doc2.tier1_names

        if config.disable_dates:
            date_overlap = set()
        else:
            date_overlap = doc1.tier1_dates & doc2.tier1_dates

        if name_overlap and date_overlap:
            substantial_names = {
                name for name in name_overlap
                if ' ' in name
                or len(name) >= 8
            }

            if substantial_names:
                if not case_id_conflict:
                    return MatchResult(
                        matched=True,
                        tier=1,
                        weight=1.0,
                        reason=f"Tier 1: Name + Date match ({list(substantial_names)[0]}, {list(date_overlap)[0]})",
                        doc_ids=(doc1.doc_id, doc2.doc_id),
                        match_type="name_and_date",
                        shared_features={"tier1_names": substantial_names, "tier1_dates": date_overlap},
                    )
                any_signal = True

    # Fallthrough
    if case_id_conflict and not any_signal:
        return MatchResult(
            matched=False,
            hard_block=True,
            tier=1,
            weight=0.0,
            reason="Tier 1: Case ID mismatch (both docs have IDs but none overlap)",
            doc_ids=(doc1.doc_id, doc2.doc_id),
            match_type="case_id_mismatch",
            shared_features={},
        )

    # All other cases → MAYBE
    # Covers: conflict + signal, no IDs + signal, no IDs + no signal
    return None

def compare_tier2(
    doc1: DocumentFeatures,
    doc2: DocumentFeatures,
    config: AblationConfig
) -> MatchResult | None:
    """
    Tier 2: Compare LLM-extracted features.

    Rules (all require multiple signals to avoid false positives):
    1. any_case_id_match AND dates_within_range (case IDs strong, but validate with dates)
    2. subject_name_match AND (date_match OR case_id_match)
    3. date_match AND officer_match AND (case_id_match OR subject_match)

    Args:
        doc1: First document features
        doc2: Second document features
        config: Ablation configuration controlling which rules are enabled

    Returns:
        MatchResult if matched, None if no match
    """

    # Rule 1: Case ID match with date proximity check
    if config.enabled_rules.get("tier2_case_id", True):
        # Check feature-level disable flag
        if config.disable_case_ids:
            case_id_overlap = set()  # Force empty if disabled
        else:
            case_id_overlap = doc1.tier2_case_ids & doc2.tier2_case_ids

        if case_id_overlap:
            # Check if dates are within reasonable range (prevents ambiguous IDs across years)
            # Only use tier2_dates — no fallback to tier1 (tiers are isolated)
            dates1 = doc1.tier2_dates
            dates2 = doc2.tier2_dates

            if not dates_within_range(
                dates1,
                dates2,
                days=config.date_proximity_days,
                require_dates_present=config.require_dates_present
            ):
                # Dates differ by more than allowed window - ambiguous case ID, don't match
                return None

            return MatchResult(
                matched=True,
                tier=2,
                weight=1.0,
                reason=f"Tier 2: LLM Case ID match ({', '.join(list(case_id_overlap)[:3])})",
                doc_ids=(doc1.doc_id, doc2.doc_id),
                match_type="case_id_only",
                shared_features={"tier2_case_ids": case_id_overlap},
            )

    # Rule 2: Subject name match ONLY if also have date OR case_id match
    # (prevents false positives from common names, lawyers, repeat offenders)
    if config.enabled_rules.get("tier2_subject_date_or_case", True):
        # Check feature-level disable flags
        if config.disable_subject_names:
            subject_overlap = set()  # Force empty if disabled
        else:
            subject_overlap = doc1.tier2_subject_names & doc2.tier2_subject_names

        if subject_overlap:
            # Require additional signal to avoid false positives from common names
            if config.disable_dates:
                date_overlap = set()
            else:
                date_overlap = doc1.tier2_dates & doc2.tier2_dates

            if config.disable_case_ids:
                case_id_overlap = set()
            else:
                case_id_overlap = doc1.tier2_case_ids & doc2.tier2_case_ids

            if date_overlap or case_id_overlap:
                match_type = "subject_name_and_date" if date_overlap else "subject_name_and_case_id"
                reason = f"Tier 2: Subject + {'Date' if date_overlap else 'Case ID'} match"
                shared_features = {"tier2_subject_names": subject_overlap}

                if date_overlap:
                    shared_features["tier2_dates"] = date_overlap
                if case_id_overlap:
                    shared_features["tier2_case_ids"] = case_id_overlap

                return MatchResult(
                    matched=True,
                    tier=2,
                    weight=1.0,
                    reason=reason,
                    doc_ids=(doc1.doc_id, doc2.doc_id),
                    match_type=match_type,
                    shared_features=shared_features,
                )
            # Otherwise, no match - subject name alone is too ambiguous

    # Rule 3: Date + Officer match ONLY if also have case_id OR subject match
    # (prevents false positives from same officer responding to different incidents on same day)
    if config.enabled_rules.get("tier2_date_officer_case_or_subject", True):
        # Check feature-level disable flags
        if config.disable_dates:
            date_overlap = set()
        else:
            date_overlap = doc1.tier2_dates & doc2.tier2_dates

        if config.disable_officer_names:
            officer_overlap = set()
        else:
            officer_overlap = doc1.tier2_officer_names & doc2.tier2_officer_names

        if date_overlap and officer_overlap:
            # Require additional signal
            if config.disable_case_ids:
                case_id_overlap = set()
            else:
                case_id_overlap = doc1.tier2_case_ids & doc2.tier2_case_ids

            if config.disable_subject_names:
                subject_overlap = set()
            else:
                subject_overlap = doc1.tier2_subject_names & doc2.tier2_subject_names

            if case_id_overlap or subject_overlap:
                match_type = "date_officer_and_case_id" if case_id_overlap else "date_officer_and_subject"
                reason = f"Tier 2: Date + Officer + {'Case ID' if case_id_overlap else 'Subject'} match"
                shared_features = {
                    "tier2_dates": date_overlap,
                    "tier2_officer_names": officer_overlap
                }

                if case_id_overlap:
                    shared_features["tier2_case_ids"] = case_id_overlap
                if subject_overlap:
                    shared_features["tier2_subject_names"] = subject_overlap

                return MatchResult(
                    matched=True,
                    tier=2,
                    weight=1.0,
                    reason=reason,
                    doc_ids=(doc1.doc_id, doc2.doc_id),
                    match_type=match_type,
                    shared_features=shared_features,
                )
            # Otherwise, no match - date + officer alone is too ambiguous

    # No match
    return None



# ============================================================================
# TIER 3 ASYNC COMPARISON
# ============================================================================


async def compare_tier3_async(
    doc1: DocumentFeatures,
    doc2: DocumentFeatures,
    similarity_threshold: float = 1.0,
    semaphore: asyncio.Semaphore | None = None,
) -> MatchResult:
    """
    Tier 3: Compare LLM-generated summaries using async LLM comparison.

    Args:
        doc1: First document features
        doc2: Second document features
        similarity_threshold: Minimum score to draw edge (default: 1.0 for strict)
        semaphore: Semaphore to limit concurrency

    Returns:
        MatchResult with LLM comparison outcome
    """
    if semaphore:
        async with semaphore:
            return await _compare_tier3_core(doc1, doc2, similarity_threshold)
    else:
        return await _compare_tier3_core(doc1, doc2, similarity_threshold)


async def _compare_tier3_core(
    doc1: DocumentFeatures, doc2: DocumentFeatures, similarity_threshold: float
) -> MatchResult:
    """Core Tier 3 comparison logic."""

    # Get summaries for both documents (only high-quality feature summaries)
    summary1 = doc1.feature_summaries
    summary2 = doc2.feature_summaries

    # If either document has no meaningful summary, cannot compare
    if not summary1 or not summary2:
        return MatchResult(
            matched=False,
            tier=3,
            weight=0.0,
            reason="Tier 3: Missing summaries (document will be singleton)",
            doc_ids=(doc1.doc_id, doc2.doc_id),
            match_type="no_match",
            shared_features={},
        )

    # Track shared summary types
    shared_summary_types = doc1.meaningful_summary_types & doc2.meaningful_summary_types

    # Call LLM to compare summaries
    template = Template(SUMMARY_COMPARISON_PROMPT)
    prompt = template.render(summary_1=summary1, summary_2=summary2)

    try:
        result = await asyncio.to_thread(get_llm().complete, prompt)
        response = result.text
        similarity_score = validate_summary_comparison_response(response)

        if similarity_score is not None:
            matched = similarity_score >= similarity_threshold
            return MatchResult(
                matched=matched,
                tier=3,
                weight=1.0 if matched else 0.0,
                reason=f"Tier 3: LLM comparison (score={similarity_score})",
                doc_ids=(doc1.doc_id, doc2.doc_id),
                match_type="llm_summary_comparison",
                shared_features={"summary_types": shared_summary_types},
            )
    except Exception as e:
        logger.warning(f"Tier 3 comparison failed for {doc1.doc_id} vs {doc2.doc_id}: {e}")

    # Default: no match
    return MatchResult(
        matched=False,
        tier=3,
        weight=0.0,
        reason="Tier 3: LLM comparison failed",
        doc_ids=(doc1.doc_id, doc2.doc_id),
        match_type="no_match",
        shared_features={},
    )


# ============================================================================
# CLUSTER VALIDATION
# ============================================================================


def validate_cluster(cluster_nodes: set, graph: nx.Graph, threshold: float = 0.3) -> bool:
    """
    Check if a cluster is valid based on connectivity threshold.

    Each node must have edges to at least threshold% of other nodes in the cluster.
    This prevents transitive chains where documents are weakly connected through
    intermediaries (e.g., A→B→C where A and C don't connect).

    Args:
        cluster_nodes: Set of node IDs in the cluster
        graph: NetworkX graph with edges
        threshold: Minimum connectivity ratio (default 0.3 = 30%)

    Returns:
        True if cluster is valid, False if it should be split
    """
    cluster_size = len(cluster_nodes)

    # Small clusters (≤2 nodes) are always valid
    if cluster_size <= 2:
        return True

    # Check each node's connectivity
    for node in cluster_nodes:
        # Find neighbors within this cluster
        neighbors = set(graph.neighbors(node)) & cluster_nodes

        # Calculate connectivity ratio (excluding self)
        connectivity = len(neighbors) / (cluster_size - 1)

        # If any node fails threshold, cluster is invalid
        if connectivity < threshold:
            return False

    return True


def split_invalid_cluster(
    cluster_nodes: set, graph: nx.Graph, threshold: float = 0.3, features_map: dict = None
) -> list[set]:
    """
    Efficiently split an invalid cluster into valid sub-clusters using batch removal.

    Strategy (Fast Algorithm - O(n² log n)):
    1. Find ALL nodes below connectivity threshold in one pass (O(n²))
    2. Apply CASE ID CONSTRAINT: Never remove nodes that share case IDs with kept nodes
    3. Remove remaining low-connectivity nodes all at once (batch removal)
    4. Find connected components in remaining nodes
    5. Recursively validate each component
    6. Add removed nodes as singletons

    This is much faster than one-by-one removal (which is O(n³)).

    Args:
        cluster_nodes: Set of node IDs in the invalid cluster
        graph: NetworkX graph with edges
        threshold: Minimum connectivity ratio
        features_map: Dict mapping doc_id -> DocumentFeatures (for case ID constraints)

    Returns:
        List of valid sub-clusters (each is a set of node IDs)
    """
    # Base case: cluster is valid
    if validate_cluster(cluster_nodes, graph, threshold):
        return [cluster_nodes]

    # Base case: cluster is too small to split
    if len(cluster_nodes) <= 2:
        return [cluster_nodes]

    cluster_size = len(cluster_nodes)

    # CASE ID CONSTRAINT: Check if ALL nodes share at least one common case ID
    # If so, keep cluster together regardless of connectivity
    if features_map:
        # Collect all case IDs from first node
        first_node = next(iter(cluster_nodes))
        if first_node in features_map:
            common_case_ids = features_map[first_node].tier1_case_ids | features_map[first_node].tier2_case_ids

            # Check if all other nodes share at least one of these case IDs
            all_share_case_id = True
            for node in cluster_nodes:
                if node in features_map:
                    node_case_ids = features_map[node].tier1_case_ids | features_map[node].tier2_case_ids
                    if not (common_case_ids & node_case_ids):
                        all_share_case_id = False
                        break
                    # Update common_case_ids to intersection (case IDs shared by ALL)
                    common_case_ids &= node_case_ids

            # If all nodes share at least one common case ID, don't split
            if all_share_case_id and common_case_ids:
                if DEBUG and cluster_size > 10:
                    logger.info(f"      CASE ID CONSTRAINT: Cluster of {cluster_size} docs shares case ID(s) {common_case_ids} - keeping together")
                return [cluster_nodes]

    # Step 1: Find ALL nodes below threshold in one pass (O(n²))
    # This is the key optimization - batch identification instead of iterative
    nodes_to_remove = set()
    nodes_to_keep = set()

    for node in cluster_nodes:
        neighbors = set(graph.neighbors(node)) & cluster_nodes
        connectivity = len(neighbors) / (cluster_size - 1)

        if connectivity < threshold:
            nodes_to_remove.add(node)
        else:
            nodes_to_keep.add(node)

    # Step 1.5: CASE ID CONSTRAINT - Never split documents with matching case IDs
    # Move nodes from nodes_to_remove back to nodes_to_keep if they share case IDs with kept nodes
    if features_map:
        # Collect all case IDs from nodes we're keeping
        kept_case_ids = set()
        for node in nodes_to_keep:
            if node in features_map:
                doc_features = features_map[node]
                # Combine all case ID sources
                all_case_ids = doc_features.tier1_case_ids | doc_features.tier2_case_ids
                kept_case_ids.update(all_case_ids)

        # Check nodes marked for removal - if they share any case ID with kept nodes, DON'T remove
        nodes_to_rescue = set()
        for node in nodes_to_remove:
            if node in features_map:
                doc_features = features_map[node]
                all_case_ids = doc_features.tier1_case_ids | doc_features.tier2_case_ids
                # If this node shares ANY case ID with kept nodes, rescue it
                if all_case_ids & kept_case_ids:
                    nodes_to_rescue.add(node)

        # Move rescued nodes back to kept
        if nodes_to_rescue:
            if DEBUG and len(nodes_to_rescue) > 0:
                logger.info(f"      CASE ID CONSTRAINT: Rescued {len(nodes_to_rescue)} nodes (shared case IDs with kept nodes)")
            nodes_to_keep.update(nodes_to_rescue)
            nodes_to_remove -= nodes_to_rescue

    # Step 2: If no nodes to remove, cluster is valid (shouldn't happen due to validate_cluster check)
    if not nodes_to_remove:
        return [cluster_nodes]

    # Step 3: Find connected components in remaining nodes
    validated_clusters = []

    if nodes_to_keep:
        subgraph = graph.subgraph(nodes_to_keep)
        components = list(nx.connected_components(subgraph))

        # Step 4: Recursively validate each component (can be parallelized in future)
        for component in components:
            validated_clusters.extend(split_invalid_cluster(component, graph, threshold, features_map))

    # Step 5: Add all removed nodes as singletons
    for node in nodes_to_remove:
        validated_clusters.append({node})

    return validated_clusters


def validate_and_split_clusters(
    candidate_clusters: list[set],
    graph: nx.Graph,
    threshold: float = 0.3,
    debug: bool = False,
    features_map: dict = None,
) -> list[set]:
    """
    Validate all clusters and split invalid ones.

    Args:
        candidate_clusters: List of candidate clusters from connected components
        graph: NetworkX graph with edges
        threshold: Minimum connectivity ratio
        debug: Enable debug logging
        features_map: Dict mapping doc_id -> DocumentFeatures (for case ID constraints)

    Returns:
        List of validated clusters (may be more than input if splits occurred)
    """
    validated_clusters = []
    splits_count = 0

    # Sort clusters by size for progress visibility (largest first)
    sorted_clusters = sorted(candidate_clusters, key=len, reverse=True)

    # Track large clusters for reporting
    large_clusters_count = sum(1 for c in sorted_clusters if len(c) > 100)

    if debug and large_clusters_count > 0:
        logger.info(f"  Validating {len(sorted_clusters):,} clusters ({large_clusters_count} large clusters >100 docs)")

    for idx, cluster in enumerate(sorted_clusters):
        cluster_size = len(cluster)

        # Log progress for large clusters
        if debug and cluster_size > 100:
            logger.info(f"  Processing cluster {idx+1}/{len(sorted_clusters)}: size={cluster_size:,}")

        if validate_cluster(cluster, graph, threshold):
            # Cluster is valid, keep as-is
            validated_clusters.append(cluster)
            if debug and cluster_size > 100:
                logger.info(f"    ✓ Valid (all nodes have ≥{threshold*100:.0f}% connectivity)")
        else:
            # Cluster is invalid, split it
            if debug and cluster_size > 100:
                logger.info("    ✗ Invalid - splitting...")

            split_start = time.time()
            sub_clusters = split_invalid_cluster(cluster, graph, threshold, features_map)
            split_time = time.time() - split_start

            validated_clusters.extend(sub_clusters)
            splits_count += len(sub_clusters) - 1

            if debug and cluster_size > 100:
                logger.info(f"    → Split into {len(sub_clusters):,} clusters in {split_time:.1f}s")

    if debug and splits_count > 0:
        logger.info(f"  Total splits: {splits_count:,}")

    return validated_clusters


# ============================================================================
# MAIN CLUSTERING PIPELINE
# ============================================================================


async def cluster_documents(
    data: list[dict],
    config: AblationConfig,
    tier3_threshold: float = 1.0
) -> dict:
    """
    Main clustering pipeline using optimized two-phase approach.

    Phase 1: Multiprocessing for Tier 1 & 2 (use all 128 cores)
    Phase 2: Async batch for Tier 3 (500+ concurrent LLM calls)

    Args:
        data: List of document dictionaries from CSV
        config: Ablation configuration controlling matching rules and tiers
        tier3_threshold: Similarity threshold for Tier 3 LLM comparison

    Returns:
        Dictionary with clustering results and statistics
    """

    logger.info(f"Starting optimized clustering for {len(data)} documents")
    logger.info(f"Ablation config: {config.name}")
    logger.info(f"  Description: {config.description}")
    logger.info(f"  Date proximity: {config.date_proximity_days} days")
    logger.info(f"  Require dates present: {config.require_dates_present}")
    logger.info(f"  Enabled tiers: {config.enabled_tiers}")
    logger.info(f"Max LLM concurrency: {MAX_CONCURRENT_LLM}")

    # ========== STEP 0: Build frequency filter ==========
    # Identify high-frequency features (metadata dates, organizational fragments)
    # that cause false positive merges
    logger.info("\n" + "=" * 80)
    logger.info("FREQUENCY-BASED FILTERING")
    logger.info("=" * 80)
    df = pd.DataFrame(data)

    # Calculate dynamic thresholds based on corpus size
    # Features appearing in more than X% of documents are likely metadata/artifacts
    corpus_size = len(data)
    dynamic_case_id_threshold = int(corpus_size * CASE_ID_FREQUENCY_THRESHOLD_PERCENT)
    dynamic_date_threshold = int(corpus_size * DATE_FREQUENCY_THRESHOLD_PERCENT)
    dynamic_name_threshold = int(corpus_size * NAME_FREQUENCY_THRESHOLD_PERCENT)

    logger.info(f"Corpus size: {corpus_size:,} documents")
    logger.info(f"  Case ID threshold: {dynamic_case_id_threshold:,} documents")
    logger.info(f"  Date threshold: {dynamic_date_threshold:,} documents")
    logger.info(f"  Name threshold: {dynamic_name_threshold:,} documents")

    freq_filter = FrequencyFilter(
        date_threshold=dynamic_date_threshold,  # Dynamic: X% of corpus
        name_threshold=dynamic_name_threshold,  # Dynamic: X% of corpus
        case_id_threshold=dynamic_case_id_threshold  # Dynamic: X% of corpus
    )
    freq_filter.build_from_dataframe(df)
    logger.info("=" * 80 + "\n")

    # ========== STEP 1: Extract features for all documents ==========
    logger.info("Extracting features from all documents...")
    features_map = {}

    for doc in data:
        row = pd.Series(doc)
        features = extract_document_features(row, freq_filter=freq_filter)
        features_map[features.doc_id] = features

    logger.info(f"Extracted features for {len(features_map)} documents")

    # ========== LOAD LEARNED MODEL (if configured) ==========
    trained_model = None
    model_feature_names = None

    if config.use_learned_model:
        logger.info("\n" + "=" * 80)
        logger.info("LEARNED MODEL MODE")
        logger.info("=" * 80)

        if not config.model_path:
            raise ValueError("use_learned_model is True but model_path not specified in config")

        # Load model
        trained_model = load_trained_model(config.model_path)

        # Load feature names from the training data
        # The model was trained with specific feature names - we need to match them

        # Try to load pair features to get feature names
        model_dir = Path(config.model_path).parent
        pair_features_path = model_dir / "pair_features_balanced.csv"

        if pair_features_path.exists():
            logger.info(f"Loading feature names from: {pair_features_path}")
            df_features = pd.read_csv(pair_features_path, nrows=1)
            # Exclude non-feature columns
            model_feature_names = [col for col in df_features.columns if col not in ["agency", "label"]]
            logger.info(f"  Model expects {len(model_feature_names)} features")
        else:
            logger.warning(f"Could not find pair_features_balanced.csv in {model_dir}")
            logger.warning("Will attempt to infer feature names from model")
            # Try to get feature names from model if it's a tree-based model
            if hasattr(trained_model, 'feature_names_in_'):
                model_feature_names = list(trained_model.feature_names_in_)
            else:
                raise ValueError("Cannot determine feature names for model")

        # Filter features by tier subset if specified
        if config.feature_subset:
            logger.info(f"Filtering features to: {config.feature_subset}")
            original_count = len(model_feature_names)
            model_feature_names = filter_features_by_tier(model_feature_names, config.feature_subset)
            logger.info(f"  Filtered from {original_count} to {len(model_feature_names)} features")
            logger.info(f"  Sample features: {', '.join(model_feature_names[:10])}")

        logger.info("=" * 80 + "\n")

    # ========== TIER 3 STATISTICS ==========
    # Count documents with tier3 summaries
    tier3_docs = [doc for doc in features_map.values() if doc.has_tier3_summaries]
    logger.info(f"\n{'='*80}")
    logger.info("TIER 3 BLOCKING ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Documents with tier3 summaries: {len(tier3_docs):,} / {len(features_map):,}")

    # Count by summary type
    summary_type_counts = {
        "case_ids_summary": 0,
        "dates_summary": 0,
        "subject_names_summary": 0,
        "officer_names_summary": 0,
    }
    for doc in tier3_docs:
        for summary_type in doc.meaningful_summary_types:
            if summary_type in summary_type_counts:
                summary_type_counts[summary_type] += 1

    logger.info("\nDocuments by summary type:")
    logger.info(f"  case_ids_summary:        {summary_type_counts['case_ids_summary']:,}")
    logger.info(f"  dates_summary:           {summary_type_counts['dates_summary']:,}")
    logger.info(f"  subject_names_summary:   {summary_type_counts['subject_names_summary']:,}")
    logger.info(f"  officer_names_summary:   {summary_type_counts['officer_names_summary']:,}")

    # Estimate pairs under old logic (both have any tier3 summaries)
    old_logic_pairs = len(tier3_docs) * (len(tier3_docs) - 1) // 2

    # Compute exact pair counts at different embedding thresholds
    logger.info("\nEmbedding-based filtering analysis:")
    logger.info(f"  Baseline (no embedding filter): {old_logic_pairs:,} pairs")

    tier3_doc_list = [doc for doc in features_map.values() if doc.has_tier3_summaries]
    thresholds_to_test = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, .95]

    for threshold in thresholds_to_test:
        pairs_at_threshold = 0
        for i in range(len(tier3_doc_list)):
            for j in range(i + 1, len(tier3_doc_list)):
                if should_compare_tier3(tier3_doc_list[i], tier3_doc_list[j], threshold):
                    pairs_at_threshold += 1

        reduction = old_logic_pairs - pairs_at_threshold
        reduction_pct = (reduction / old_logic_pairs * 100) if old_logic_pairs > 0 else 0

        marker = " <-- SELECTED" if threshold == EMBEDDING_SIMILARITY_THRESHOLD else ""
        logger.info(
            f"  Threshold {threshold:.1f}:  {pairs_at_threshold:,} pairs "
            f"(reduction: {reduction:,} pairs, {reduction_pct:.1f}%){marker}"
        )

    logger.info(f"{'='*80}\n")

    # ========== STEP 2: Generate all pairs ==========
    doc_ids = list(features_map.keys())
    total_pairs = len(doc_ids) * (len(doc_ids) - 1) // 2
    logger.info(f"Generating {total_pairs:,} document pairs...")

    # ========== PHASE 1: Tier 1 & 2 comparisons ==========
    tier1_edges = []
    tier2_edges = []
    tier3_pairs = []
    tier1_blocked = 0  # Track pairs blocked by directory filtering

    # ========== LEARNED MODEL MODE: Batched processing ==========
    if config.use_learned_model:
        logger.info("Phase 1: Running learned model with batched inference...")

        BATCH_SIZE = 1000000
        batch_pairs = []  # Accumulate pairs for batching
        batch_docs = []   # Store doc tuples for creating MatchResults
        pairs_processed = 0
        tier1_blocked = 0  # Track pairs blocked by directory filtering
        report_interval = 1000000

        phase1_start = time.time()

        # Create tqdm progress bar for visual progress tracking
        pbar = tqdm(
            total=total_pairs,
            desc="Learned model",
            unit="pairs",
            unit_scale=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for id1, id2 in itertools.combinations(doc_ids, 2):
            doc1 = features_map[id1]
            doc2 = features_map[id2]

            # Apply directory-based blocking (same as hand-coded rules for fair comparison)
            if not should_compare_tier1(doc1, doc2, min_shared_dirs=TIER1_MIN_SHARED_DIRS):
                tier1_blocked += 1
                pairs_processed += 1
                pbar.update(1)
                continue

            # Accumulate pairs for batch processing
            batch_pairs.append((doc1, doc2))
            batch_docs.append((doc1.doc_id, doc2.doc_id))
            pairs_processed += 1
            pbar.update(1)

            # Process batch when full or at end
            if len(batch_pairs) >= BATCH_SIZE or pairs_processed == total_pairs:
                batch_start = time.time()

                # Extract features for entire batch IN PARALLEL
                num_workers = min(10, len(batch_pairs))
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    batch_features = list(executor.map(extract_pair_features_wrapper, batch_pairs))

                # Create DataFrame for batch (single operation)
                X = pd.DataFrame(batch_features, columns=model_feature_names)

                # Predict for entire batch (single model call)
                predictions = trained_model.predict(X)

                # Create MatchResults for positive predictions
                for idx, prediction in enumerate(predictions):
                    if prediction == 1:
                        doc_id1, doc_id2 = batch_docs[idx]

                        # Get confidence if available
                        if hasattr(trained_model, 'predict_proba'):
                            proba = trained_model.predict_proba(X.iloc[[idx]])[0][1]
                        else:
                            proba = 1.0

                        result = MatchResult(
                            matched=True,
                            tier=1,
                            weight=1.0,
                            reason=f"Learned model match (confidence={proba:.3f})",
                            doc_ids=(doc_id1, doc_id2),
                            match_type="learned_model",
                            shared_features={},
                        )
                        tier1_edges.append(result)

                batch_elapsed = time.time() - batch_start
                total_elapsed = time.time() - phase1_start
                rate = pairs_processed / total_elapsed if total_elapsed > 0 else 0

                # Update progress bar postfix with stats
                pbar.set_postfix({
                    'blocked': f"{tier1_blocked:,}",
                    'matches': f"{len(tier1_edges):,}",
                    'batch_rate': f"{len(batch_pairs)/batch_elapsed:.0f}/s"
                }, refresh=True)

                # Progress reporting (keep less frequent logging)
                if pairs_processed % report_interval == 0 or pairs_processed == total_pairs:
                    logger.info(
                        f"Phase 1 progress: {pairs_processed:,}/{total_pairs:,} pairs | "
                        f"Batch: {len(batch_pairs)} pairs in {batch_elapsed:.1f}s ({len(batch_pairs)/batch_elapsed:.0f} pairs/s) | "
                        f"Overall: {rate:.0f} pairs/s | Blocked: {tier1_blocked:,} | Matches: {len(tier1_edges)}"
                    )

                # Clear batch for next iteration
                batch_pairs = []
                batch_docs = []

        # Close progress bar
        pbar.close()

        logger.info("Phase 1 complete!")
        logger.info(f"  Total time: {time.time() - phase1_start:.1f}s")
        logger.info(f"  Average rate: {total_pairs / (time.time() - phase1_start):.0f} pairs/s")
        logger.info(f"  Tier 1 blocked (dir filter): {tier1_blocked:,}")
        logger.info(f"  Tier 1 matches (learned model): {len(tier1_edges)}")
        logger.info("  Tier 2 matches: 0 (learned model skips Tier 2/3)")
        logger.info("  Pairs needing Tier 3: 0 (learned model skips Tier 3)")

    # ========== HAND-CODED RULES MODE: Original single-threaded ==========
    else:
        logger.info("Phase 1: Running Tier 1 & 2 comparisons (single-threaded)...")

        pairs_processed = 0
        report_interval = 10000

        for id1, id2 in itertools.combinations(doc_ids, 2):
            doc1 = features_map[id1]
            doc2 = features_map[id2]

            result = None

            # Try Tier 1 (with directory-based blocking) - only if enabled in config
            if 1 in config.enabled_tiers and doc1.has_tier1_features and doc2.has_tier1_features:
                # Only compare if documents share at least TIER1_MIN_SHARED_DIRS directory levels
                if should_compare_tier1(doc1, doc2, min_shared_dirs=TIER1_MIN_SHARED_DIRS):
                    result = compare_tier1(doc1, doc2, config)
                    if result and result.matched:
                        tier1_edges.append(result)
                else:
                    tier1_blocked += 1

            # Try Tier 2 if no Tier 1 result (match or hard block) - only if enabled in config
            if result is None and 2 in config.enabled_tiers and doc1.has_tier2_features and doc2.has_tier2_features:
                result = compare_tier2(doc1, doc2, config)
                if result:
                    tier2_edges.append(result)

            # Check if we need Tier 3 (with embedding-based filtering) - only if enabled in config
            if result is None and 3 in config.enabled_tiers and should_compare_tier3(doc1, doc2, config.embedding_similarity_threshold):
                tier3_pairs.append((id1, id2, doc1, doc2))

            pairs_processed += 1
            if pairs_processed % report_interval == 0:
                logger.info(
                    f"Phase 1 progress: {pairs_processed:,}/{total_pairs:,} pairs processed "
                    f"(Tier1={len(tier1_edges)}, Tier2={len(tier2_edges)}, NeedTier3={len(tier3_pairs)})"
                )

        logger.info("Phase 1 complete!")
        logger.info(f"  Tier 1 blocked (dir filter): {tier1_blocked:,}")
        logger.info(f"  Tier 1 matches: {len(tier1_edges)}")
        logger.info(f"  Tier 2 matches: {len(tier2_edges)}")
        logger.info(f"  Pairs needing Tier 3: {len(tier3_pairs)}")

    # ========== PHASE 2: Async batch for Tier 3 ==========
    tier3_edges = []

    if tier3_pairs:
        logger.info(f"Phase 2: Running Tier 3 async comparisons with {MAX_CONCURRENT_LLM} concurrency...")
        phase2_start = time.time()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

        # Create async tasks for all Tier 3 pairs
        tasks = [
            compare_tier3_async(doc1, doc2, tier3_threshold, semaphore)
            for id1, id2, doc1, doc2 in tier3_pairs
        ]

        # Run all tasks concurrently with progress bar
        tier3_results = []
        for i in range(0, len(tasks), 1000):
            batch_start = time.time()
            batch = tasks[i : i + 1000]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            tier3_results.extend(batch_results)

            batch_elapsed = time.time() - batch_start
            total_completed = min(i + 1000, len(tasks))
            total_elapsed = time.time() - phase2_start
            rate = total_completed / total_elapsed if total_elapsed > 0 else 0

            logger.info(
                f"Phase 2 progress: {total_completed:,}/{len(tasks):,} pairs | "
                f"Batch: {len(batch)} in {batch_elapsed:.1f}s ({len(batch)/batch_elapsed:.1f} pairs/s) | "
                f"Overall: {rate:.1f} pairs/s"
            )

        # Filter out exceptions and collect matched edges
        for result in tier3_results:
            if isinstance(result, MatchResult) and result.matched:
                tier3_edges.append(result)

        total_elapsed = time.time() - phase2_start
        logger.info(
            f"Phase 2 complete! Tier 3 matches: {len(tier3_edges)} | "
            f"Total time: {total_elapsed:.1f}s | Average: {len(tasks)/total_elapsed:.1f} pairs/s"
        )

    # ========== STEP 3: Build graph from edges ==========
    logger.info("Building graph from edges...")
    G = nx.Graph()
    for doc_id in features_map.keys():
        G.add_node(doc_id)

    # Add all edges
    all_edges = tier1_edges + tier2_edges + tier3_edges
    for edge in all_edges:
        id1, id2 = edge.doc_ids
        G.add_edge(id1, id2, weight=edge.weight, reason=edge.reason)

    logger.info(f"Added {len(all_edges)} edges to graph")

    # ========== STEP 4: Find connected components (clusters) ==========
    logger.info("Finding connected components...")
    candidate_clusters = list(nx.connected_components(G))
    logger.info(f"Found {len(candidate_clusters)} candidate clusters")

    # ========== STEP 4.5: Validate and split clusters ==========
    if config.enable_validation:
        logger.info(f"\nApplying cluster validation (threshold={VALIDATION_THRESHOLD})...")
        logger.info(f"  Each node must connect to ≥{VALIDATION_THRESHOLD*100:.0f}% of cluster")

        validation_start = time.time()
        clusters = validate_and_split_clusters(
            candidate_clusters, G, threshold=VALIDATION_THRESHOLD, debug=DEBUG, features_map=features_map
        )
        validation_time = time.time() - validation_start

        # Calculate validation impact
        clusters_before = len(candidate_clusters)
        clusters_after = len(clusters)
        splits = clusters_after - clusters_before

        logger.info(f"  Validation complete in {validation_time:.2f}s")
        logger.info(f"  Clusters before: {clusters_before:,}")
        logger.info(f"  Clusters after:  {clusters_after:,}")
        logger.info(f"  Net splits:      {splits:,}")

        # Show size distribution change
        before_sizes = [len(c) for c in candidate_clusters]
        after_sizes = [len(c) for c in clusters]
        before_multi = sum(1 for s in before_sizes if s > 1)
        after_multi = sum(1 for s in after_sizes if s > 1)

        logger.info(f"  Multi-doc clusters: {before_multi:,} → {after_multi:,}")
    else:
        logger.info("\nCluster validation DISABLED - using connected components as-is")
        clusters = candidate_clusters

    # ========== STEP 5: Build node-to-cluster mapping ==========
    node_to_cluster = {}
    for cluster_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            node_to_cluster[node] = cluster_id

    # ========== STEP 6: Build output ==========
    results = []
    for doc in data:
        doc_id = str(doc.get("gdrive_id", doc.get("id")))
        cluster_id = node_to_cluster.get(doc_id, -1)

        doc_copy = doc.copy()
        doc_copy["Parent Clusters"] = [cluster_id]
        results.append(doc_copy)

    # ========== STEP 7: Statistics ==========
    cluster_sizes = [len(c) for c in clusters]
    singleton_count = sum(1 for size in cluster_sizes if size == 1)
    multi_doc_clusters = sum(1 for size in cluster_sizes if size > 1)

    logger.info(f"\n{'='*80}")
    logger.info("CLUSTERING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total documents:        {len(data):,}")
    logger.info(f"Total pairs compared:   {total_pairs:,}")
    logger.info(f"Total clusters:         {len(clusters):,}")
    logger.info(f"  Singleton clusters:   {singleton_count:,}")
    logger.info(f"  Multi-doc clusters:   {multi_doc_clusters:,}")
    logger.info("")
    logger.info("Tier 1 blocking:")
    logger.info(f"  Blocked by dir filter: {tier1_blocked:,} ({tier1_blocked/total_pairs*100:.1f}%)")
    logger.info("")
    logger.info(f"Edges added:            {len(all_edges):,}")
    logger.info(f"  Tier 1 matches:       {len(tier1_edges):,}")
    logger.info(f"  Tier 2 matches:       {len(tier2_edges):,}")
    logger.info(f"  Tier 3 matches:       {len(tier3_edges):,}")
    logger.info(f"  Tier 3 comparisons:   {len(tier3_pairs):,}")
    logger.info(f"{'='*80}\n")

    # Calculate validation statistics
    if config.enable_validation:
        validation_stats = {
            "validation_enabled": True,
            "validation_threshold": VALIDATION_THRESHOLD,
            "clusters_before_validation": len(candidate_clusters),
            "clusters_after_validation": len(clusters),
            "clusters_split": len(clusters) - len(candidate_clusters),
        }
    else:
        validation_stats = {
            "validation_enabled": False,
        }

    return {
        "results": results,
        "statistics": {
            "total_documents": len(data),
            "total_pairs": total_pairs,
            "total_clusters": len(clusters),
            "singleton_clusters": singleton_count,
            "multi_doc_clusters": multi_doc_clusters,
            "edges_added": len(all_edges),
            "tier1_blocked": tier1_blocked,
            "tier1_matches": len(tier1_edges),
            "tier2_matches": len(tier2_edges),
            "tier3_matches": len(tier3_edges),
            "tier3_comparisons": len(tier3_pairs),
            **validation_stats,
        },
        "edges": all_edges,  # All MatchResult objects for diagnostic analysis
        "features_map": features_map,  # DocumentFeatures for each document
    }


# ============================================================================
# POST-PROCESSING: DIRECTORY-BASED SINGLETON CLUSTERING
# ============================================================================


def merge_singletons_by_directory(results: list[dict], features_map: dict) -> tuple[list[dict], dict]:
    """
    Post-processing step: Merge singleton clusters that share directories.

    Uses LLM to analyze directory structure and determine optimal merging depth:
    - min_depth=0: Merge singletons sharing same parent directory (case-specific paths)
    - min_depth=1+: Require deeper nesting (generic directories like "Data", "Files")

    This prevents creating mega-clusters from generic dump directories while still
    merging related documents in case-specific directory structures.

    Args:
        results: List of document dicts with 'Parent Clusters' field
        features_map: Dict mapping doc_id to DocumentFeatures

    Returns:
        Tuple of (updated results, statistics dict with llm_min_depth)
    """
    logger.info(f"\n{'='*80}")
    logger.info("POST-PROCESSING: Directory-Based Singleton Clustering")
    logger.info(f"{'='*80}")

    # Step 1: Identify singletons and their cluster IDs
    cluster_counts = {}
    for doc in results:
        cluster_id = doc["Parent Clusters"][0]
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    singleton_cluster_ids = {cid for cid, count in cluster_counts.items() if count == 1}
    logger.info(f"Found {len(singleton_cluster_ids):,} singleton clusters")

    # Step 2: Collect singleton filepaths (full paths) and analyze with LLM
    # Note: gdrive_path format varies by agency:
    #   - Some agencies: gdrive_path = directory only, gdrive_name = filename
    #   - Other agencies: gdrive_path = full path with filename, gdrive_name = filename
    # We detect which format by checking if gdrive_path already ends with gdrive_name
    singleton_filepaths = []
    for doc in results:
        cluster_id = doc["Parent Clusters"][0]
        if cluster_id in singleton_cluster_ids:
            gdrive_path = doc.get("gdrive_path", "")
            gdrive_name = doc.get("gdrive_name", "") or doc.get("file_name_from_json", "")
            if gdrive_path and gdrive_name:
                # Check if gdrive_path already includes the filename
                if gdrive_path.endswith(gdrive_name):
                    full_path = gdrive_path  # Already has filename
                else:
                    full_path = f"{gdrive_path}/{gdrive_name}"  # Need to append
                singleton_filepaths.append(full_path)
            elif gdrive_path:
                singleton_filepaths.append(gdrive_path)

    # Use LLM to determine min depth
    min_depth = analyze_directory_depth(singleton_filepaths, sample_size=20)
    logger.info(f"Using min_depth = {min_depth} for directory grouping")

    # Step 3: Group singletons by directory (with min_depth)
    dir_to_singletons = {}
    for idx, doc in enumerate(results):
        cluster_id = doc["Parent Clusters"][0]
        if cluster_id not in singleton_cluster_ids:
            continue

        doc_id = str(doc.get("gdrive_id", doc.get("id", idx)))
        gdrive_path = doc.get("gdrive_path", "")
        gdrive_name = doc.get("gdrive_name", "") or doc.get("file_name_from_json", "")

        if not gdrive_path:
            continue

        # Construct full filepath so get_grouping_directory's .parent works correctly
        # Handle both data formats: gdrive_path may or may not include the filename
        if gdrive_name and gdrive_path.endswith(gdrive_name):
            full_filepath = gdrive_path  # Already has filename
        elif gdrive_name:
            full_filepath = f"{gdrive_path}/{gdrive_name}"  # Need to append
        else:
            full_filepath = gdrive_path

        grouping_dir = get_grouping_directory(full_filepath, min_depth)
        if not grouping_dir or grouping_dir == ".":
            continue

        if grouping_dir not in dir_to_singletons:
            dir_to_singletons[grouping_dir] = []
        dir_to_singletons[grouping_dir].append((idx, doc_id))

    # Step 4: Find directories with multiple singletons (merge candidates)
    potential_merges = {d: docs for d, docs in dir_to_singletons.items() if len(docs) > 1}

    logger.info(f"Directories with 2+ singletons (at depth {min_depth}): {len(potential_merges):,}")

    if not potential_merges:
        logger.info("No singletons to merge by directory")
        return results, {
            "singletons_before": len(singleton_cluster_ids),
            "singletons_merged": 0,
            "new_clusters_from_dirs": 0,
            "singletons_after": len(singleton_cluster_ids),
            "llm_min_depth": min_depth,
            "llm_validations": 0,
            "llm_approved_merges": 0,
            "llm_rejected_merges": 0,
        }

    # Step 4.5: Validate each directory with LLM (parallel)
    logger.info(f"Validating {len(potential_merges):,} directories with LLM in parallel...")

    # Create output log file
    log_output_path = "../../data/output/singleton_merge_validations.txt"
    Path(log_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_output_path, 'w') as f:
        f.write("SINGLETON MERGE VALIDATION LOG\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total directories to validate: {len(potential_merges)}\n")

    logger.info(f"Logging validation decisions to: {log_output_path}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def validate_directory(grouping_dir_and_list):
        grouping_dir, singleton_list = grouping_dir_and_list
        # Get the full filepaths for this directory
        # Handle both data formats: gdrive_path may or may not include the filename
        full_filepaths = []
        for idx, _doc_id in singleton_list:
            gdrive_path = results[idx].get("gdrive_path", "")
            gdrive_name = results[idx].get("gdrive_name", "") or results[idx].get("file_name_from_json", "")
            if gdrive_path and gdrive_name:
                if gdrive_path.endswith(gdrive_name):
                    full_path = gdrive_path  # Already has filename
                else:
                    full_path = f"{gdrive_path}/{gdrive_name}"  # Need to append
                full_filepaths.append(full_path)
            elif gdrive_path:
                full_filepaths.append(gdrive_path)

        # Ask LLM if these files should be merged
        should_merge = should_merge_singletons(grouping_dir, full_filepaths, output_log_path=log_output_path)
        return grouping_dir, singleton_list, should_merge

    merge_candidates = {}
    llm_approved = 0
    llm_rejected = 0

    # Run validations in parallel (max 20 concurrent threads for API calls)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(validate_directory, (grouping_dir, singleton_list)): grouping_dir
            for grouping_dir, singleton_list in potential_merges.items()
        }

        for future in as_completed(futures):
            try:
                grouping_dir, singleton_list, should_merge = future.result()
                if should_merge:
                    merge_candidates[grouping_dir] = singleton_list
                    llm_approved += 1
                else:
                    llm_rejected += 1
            except Exception as e:
                grouping_dir = futures[future]
                logger.error(f"Error validating {grouping_dir}: {e}")
                llm_rejected += 1

    logger.info("LLM validation results:")
    logger.info(f"  Approved for merging: {llm_approved:,} directories")
    logger.info(f"  Rejected (won't merge): {llm_rejected:,} directories")

    total_singletons_to_merge = sum(len(docs) for docs in merge_candidates.values())
    logger.info(f"Singletons that will be merged: {total_singletons_to_merge:,}")

    if not merge_candidates:
        logger.info("No directories approved for merging")
        return results, {
            "singletons_before": len(singleton_cluster_ids),
            "singletons_merged": 0,
            "new_clusters_from_dirs": 0,
            "singletons_after": len(singleton_cluster_ids),
            "llm_min_depth": min_depth,
            "llm_validations": len(potential_merges),
            "llm_approved_merges": llm_approved,
            "llm_rejected_merges": llm_rejected,
        }

    # Step 5: Assign new cluster IDs to merged groups
    # Find the max existing cluster ID
    max_cluster_id = max(doc["Parent Clusters"][0] for doc in results)
    next_cluster_id = max_cluster_id + 1

    # Track statistics
    singletons_merged = 0
    new_clusters_created = 0

    # Create a mapping from doc index to new cluster ID
    idx_to_new_cluster = {}

    for grouping_dir, singleton_list in merge_candidates.items():
        # All singletons in this directory get the same new cluster ID
        for idx, doc_id in singleton_list:
            idx_to_new_cluster[idx] = next_cluster_id
            singletons_merged += 1

        new_clusters_created += 1
        next_cluster_id += 1

    # Step 6: Update results with new cluster assignments
    for idx, new_cluster_id in idx_to_new_cluster.items():
        results[idx]["Parent Clusters"] = [new_cluster_id]

    # Calculate new singleton count
    new_cluster_counts = {}
    for doc in results:
        cluster_id = doc["Parent Clusters"][0]
        new_cluster_counts[cluster_id] = new_cluster_counts.get(cluster_id, 0) + 1

    new_singleton_count = sum(1 for count in new_cluster_counts.values() if count == 1)

    logger.info("\nDirectory clustering results:")
    logger.info(f"  Min depth used: {min_depth}")
    logger.info(f"  LLM validations: {len(potential_merges):,} directories")
    logger.info(f"    Approved: {llm_approved:,}")
    logger.info(f"    Rejected: {llm_rejected:,}")
    logger.info(f"  Singletons merged: {singletons_merged:,}")
    logger.info(f"  New clusters created: {new_clusters_created:,}")
    logger.info(f"  Singletons before: {len(singleton_cluster_ids):,}")
    logger.info(f"  Singletons after: {new_singleton_count:,}")
    logger.info(f"  Reduction: {len(singleton_cluster_ids) - new_singleton_count:,} ({(len(singleton_cluster_ids) - new_singleton_count) / len(singleton_cluster_ids) * 100:.1f}%)")

    stats = {
        "singletons_before": len(singleton_cluster_ids),
        "singletons_merged": singletons_merged,
        "new_clusters_from_dirs": new_clusters_created,
        "singletons_after": new_singleton_count,
        "llm_min_depth": min_depth,
        "llm_validations": len(potential_merges),
        "llm_approved_merges": llm_approved,
        "llm_rejected_merges": llm_rejected,
    }

    return results, stats


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main(csv_path: str, ablation_config_name: str, output_dir: str = "../../data/output/ablations"):
    """
    Main entry point for ablation clustering.

    Args:
        csv_path: Path to input CSV with document data
        ablation_config_name: Name of ablation config to load from YAML
        output_dir: Directory to save results (default: ../../data/output/ablations)
    """

    logger.info("=" * 80)
    logger.info("ABLATION STUDY CLUSTERING PIPELINE")
    logger.info("=" * 80)

    # Load ablation config
    logger.info(f"Loading ablation config: {ablation_config_name}")
    config = load_ablation_config(ablation_config_name)
    logger.info(f"  Description: {config.description}")
    logger.info("")

    # Load data
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} documents")
    df = df[~(df.provisional_case_name.fillna("?").str.contains(r"n\/a|\?"))]
    logger.info(f"num of {len(df)} documents after filtering out unassigned rows")

    # Re-run regex extraction to ensure latest patterns are used
    # This is fast (no LLM calls) and ensures any regex improvements are applied
    df = rerun_regex_extraction(df)

    # Convert to list of dicts
    data = df.to_dict("records")

    # Run clustering with ablation config
    output = await cluster_documents(data, config, tier3_threshold=TIER3_SIMILARITY_THRESHOLD)

    # Post-processing: Merge singletons by directory (if enabled)
    if ENABLE_DIRECTORY_FALLBACK:
        logger.info("Directory-based singleton fallback: ENABLED")
        output["results"], dir_clustering_stats = merge_singletons_by_directory(
            output["results"], output["features_map"]
        )
        output["statistics"]["directory_clustering"] = dir_clustering_stats
    else:
        logger.info("Directory-based singleton fallback: DISABLED")
        # Calculate singleton count for statistics (without merging)
        cluster_counts = {}
        for doc in output["results"]:
            cluster_id = doc["Parent Clusters"][0]
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        singleton_count = sum(1 for count in cluster_counts.values() if count == 1)
        output["statistics"]["directory_clustering"] = {
            "enabled": False,
            "singletons_before": singleton_count,
            "singletons_merged": 0,
            "new_clusters_from_dirs": 0,
            "singletons_after": singleton_count,
        }

    # Build output path with ablation name
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / f"clustering_results_ablation_{config.name}.csv"

    # Save results
    logger.info(f"Saving results to: {output_path}")
    results_df = pd.DataFrame(output["results"])
    results_df.to_csv(output_path, index=False)

    # Save edge metadata for diagnostic analysis
    import pickle
    edge_metadata_path = str(output_path).replace(".csv", "_edge_metadata.pkl")
    logger.info(f"Saving edge metadata to: {edge_metadata_path}")
    with open(edge_metadata_path, "wb") as f:
        pickle.dump({
            "edges": output["edges"],
            "features_map": output["features_map"],
        }, f)

    logger.info("=" * 80)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 80)

    return output


def run_clustering(csv_path: str = None, ablation_config_name: str = "baseline_365days"):
    """
    Synchronous wrapper for main().

    Args:
        csv_path: Path to input CSV (if None, uses CSV_PATH from config)
        ablation_config_name: Name of ablation config to use
    """
    if csv_path is None:
        # Use the hardcoded CSV_PATH from configuration section
        csv_path = CSV_PATH if 'CSV_PATH' in globals() else None
        if csv_path is None:
            raise ValueError("No CSV path provided and CSV_PATH not set in configuration")

    return asyncio.run(main(csv_path, ablation_config_name))


