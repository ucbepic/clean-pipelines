"""
Ablation study evaluation script.

Automatically discovers and evaluates all ablation clustering results for multiple agencies.
Compares against ground truth (provisional_case_name column) and generates:
- Per-agency reports showing F1 scores for each ablation
- Cross-agency summary showing how each ablation performs across all agencies

Usage:
    python evaluate.py
"""

import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .metrics import evaluate_clusters, get_summary_statistics
from .schemas import ScoreReport


def score(
    results_csv: Path | str,
    ground_truth_col: str = "provisional_case_name",
) -> ScoreReport:
    """Score one clustering result CSV against the embedded ground truth column."""
    results_path = Path(results_csv)
    df = pd.read_csv(results_path)
    if ground_truth_col not in df.columns:
        raise ValueError(f"Missing ground-truth column {ground_truth_col!r} in {results_path}")

    metrics = evaluate_clusters(df, df)
    overall = metrics["overall"]

    cluster_column = next(
        (
            c
            for c in ["Parent Clusters", "Clusters", "cluster", "parent_clusters"]
            if c in df.columns
        ),
        "",
    )

    return ScoreReport(
        results_csv=str(results_path),
        ground_truth_column=ground_truth_col,
        cluster_column=cluster_column,
        num_groundtruth_clusters=int(overall.get("num_groundtruth_clusters_evaluated", 0)),
        num_test_clusters=int(df[cluster_column].nunique()) if cluster_column else 0,
        avg_precision=float(overall["precision"]),
        avg_recall=float(overall["recall"]),
        avg_f1=float(overall["f1"]),
        clusters_with_splits=int(overall.get("clusters_with_splits", 0)),
        clusters_with_merges=int(overall.get("clusters_with_merges", 0)),
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

# User-friendly name mapping for ML ablations
ABLATION_NAME_MAPPING_ML = {
    # Decision Tree
    "dt_tier1_only": "ml_dt_tier1_only",
    "dt_tier2_only": "ml_dt_tier2_only",
    "dt_both": "ml_dt_both",
    "dt_cascade": "ml_dt_cascade",
    # Random Forest
    "rf_tier1_only": "ml_rf_tier1_only",
    "rf_tier2_only": "ml_rf_tier2_only",
    "rf_both": "ml_rf_both",
    "rf_cascade": "ml_rf_cascade",
    # LightGBM
    "lgbm_tier1_only": "ml_lgbm_tier1_only",
    "lgbm_tier2_only": "ml_lgbm_tier2_only",
    "lgbm_both": "ml_lgbm_both",
    "lgbm_cascade": "ml_lgbm_cascade",
    # Older config names (kept so existing output dirs still resolve)
    "learned_rules_tier1_only": "ml_dt_tier1_only",
    "learned_rules_tier2_only": "ml_dt_tier2_only",
    "learned_rules_tier1_and_tier2": "ml_dt_both",
    "learned_rules_rf_tier1_only": "ml_rf_tier1_only",
    "learned_rules_rf_tier2_only": "ml_rf_tier2_only",
    "learned_rules_rf_tier1_and_tier2": "ml_rf_both",
}

# User-friendly name mapping for V2 ablations
ABLATION_NAME_MAPPING_V2 = {
    # Architecture Ablations
    "baseline_v2": "regex_llm_semantic_full",
    "baseline_v2_no_validation": "regex_llm_semantic_full_no_validation",
    "tier1_and_tier2_only": "regex_llm_only",
    "tier2_and_tier3": "llm_semantic_only",
    "tier1_only_v2": "regex_only",
    "tier2_extraction_only_v2": "llm_only",
    # Feature Ablations (all use regex+llm base)
    "no_case_ids_v2": "regex_llm_without_case_ids",
    "no_subject_names_v2": "regex_llm_without_subject_names",
    "no_dates_v2": "regex_llm_without_dates",
    "no_all_names_v2": "regex_llm_without_names",
    "case_ids_dates_only_v2": "regex_llm_case_ids_dates_only",
    "names_dates_only_v2": "regex_llm_names_dates_only",
    # Date proximity window sweep
    "date_prox_90d": "baseline_date_window_90d",
    "date_prox_365d": "baseline_date_window_365d",
    # Embedding similarity gate sweep
    "emb_gate_085": "baseline_emb_gate_0.85",
    "emb_gate_095": "baseline_emb_gate_0.95",
}


def get_display_name(ablation_name: str, version: str = "v2") -> str:
    """Get user-friendly display name for an ablation.

    Args:
        ablation_name: Original ablation name from config
        version: 'v2' or 'ml' (default 'v2')
    """
    if version == "ml":
        return ABLATION_NAME_MAPPING_ML.get(ablation_name, ablation_name)
    else:
        return ABLATION_NAME_MAPPING_V2.get(ablation_name, ablation_name)


# Path to ablation configs (to get descriptions). Defaults point at the
# packaged YAMLs; override via env vars or by re-assigning these globals
# before calling :func:`main`.
import os as _os

_PKG_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _PKG_DIR / "configs"

ABLATION_CONFIG_PATH_V2 = Path(
    _os.environ.get(
        "PRAP_CLUSTERING_ABLATION_CONFIG_V2", _CONFIGS_DIR / "ablation_configs_handengineered.yaml"
    )
)
ABLATION_CONFIG_PATH_ML = Path(
    _os.environ.get("PRAP_CLUSTERING_ABLATION_CONFIG_ML", _CONFIGS_DIR / "ablation_configs_ml.yaml")
)

# Directory containing V2 ablation clustering results (hand-coded rules)
ABLATION_OUTPUT_DIR_V2 = Path(
    _os.environ.get(
        "PRAP_CLUSTERING_ABLATIONS_V2_DIR",
        Path.cwd() / "data" / "output" / "ablations_v2_dir_fallback",
    )
)

# Directory containing ML ablation clustering results (learned rules)
ABLATION_OUTPUT_DIR_ML = Path(
    _os.environ.get(
        "PRAP_CLUSTERING_ABLATIONS_ML_DIR", Path.cwd() / "data" / "output" / "ablations_ml"
    )
)

# Directory for evaluation reports
REPORTS_DIR = Path(
    _os.environ.get("PRAP_CLUSTERING_REPORTS_DIR", Path.cwd() / "reports" / "ablations")
)

# Ablations to skip during evaluation (e.g., deprecated or redundant ablations)
SKIP_ABLATIONS = {"strict_dates_0days", "no_date_validation"}

# Agencies to exclude from evaluation.
# Set to None to include all agencies, or provide a set of agency directory names to exclude.
EXCLUDED_AGENCIES = {
    "Brentwood_Police_Department",
    "Whittier_Police_Department",
    "San_Ramon_Police_Department",
    "CSU_San_Jose_Police_Department",
    "Ontario_Police_Department",
    "Salinas_Police_Department",
    "Orange_County_District_Attorney",
}

# Original evaluation agencies (used for V2 hand-coded rules tuning)
ORIGINAL_EVALUATION_AGENCIES = {
    "Alameda_County_Sheriff",
    "Anaheim_Police_Department",
    "Antioch_Police_Department",
    "Del_Norte_County_Sheriff",
    "Long_Beach_Police_Department",
    "Marin_County_Sheriff",
    "Orange_County_Sheriff",
    "San_Jose_Police_Department",
    "San_Mateo_County_Sheriff",
    "Santa_Cruz_County_Sheriff",
    "Stockton_Police_Department",
}

# Agencies used for training learned rules models
# These agencies were used to train the decision tree and random forest models
# They are excluded from "original evaluation" section and shown separately
LEARNED_RULES_TRAINING_AGENCIES = {
    "Oakland_Police_Department",
    "Los_Angeles_Police_Department",
    "San_Bernardino_Police_Department",
    "Los_Angeles_County_Sheriff",
    "Fresno_Police_Department",
    "San_Bernardino_County_Sheriff",
    "California_Highway_Patrol",
    "San_Diego_County_Sheriff",
    "San_Diego_Police_Department",
    "Bay_Area_Rapid_Transit_(BART)_Police_Department",
    "Redding_Police_Department",
    "King City Police Department",
    "Berkeley Police Department",
    "Monterey County Sheriff",
    "Humboldt County Sheriff",
    "California Department of Fish and Wildlife",
    "Sacramento Police Department",
    "Kern County District Attorney",
    "California State Personnel Board",
    "Ventura County Sheriff",
    "Riverside County District Attorney",
    "Sacramento County Coroner Office",
    "Alameda County District Attorney",
    "Porterville Police Department",
    "Fresno County District Attorney",
    "Tulare County Sheriff",
}

# Holdout agencies (not used for V2 tuning - true test set)
# NOTE: Must use underscored directory names to match actual output directories
HOLDOUT_AGENCIES = {
    "Folsom_Police_Department",
    "San_Leandro_Police_Department",
    "Santa_Clara_Police_Department",
    "Hayward_Police_Department",
    "Vallejo_Police_Department",
    "Santa_Ana_Police_Department",
    "Chula_Vista_Police_Department",
    "Irvine_Police_Department",
    "Pasadena_Police_Department",
    "San_Diego_County_Medical_Examiner",
    "Fresno_County_Sheriff",
    "Sacramento_County_Sheriff",
    "San_Francisco_County_Sheriff",
    "Richmond_Police_Department",
    "Los_Angeles_District_Attorney",
    "San_Francisco_Police_Commission",
    "Riverside_County_Department_of_Public_Social_Services",
    "Cal_State_East_Bay_University_Police_Department",
    "San_Joaquin_County_Medical_Examiner",
    "Santa_Monica_Police_Department",
    "Kern_County_Sheriff",
    "Santa_Clara_County_Sheriff",
    "Shasta_County_District_Attorney",
    "Contra_Costa_County_Sheriff",
    "Contra_Costa_County_District_Attorney",
    "UC_Davis_Police_Department",
    "Seal_Beach_Police_Department",
    "Office_of_Inspector_General_for_Prisons",
    "Bakersfield_Police_Department",
    "California_Department_of_Corrections_and_Rehabilitation",
    "California_Department_of_Justice",
}


# ============================================================================


def load_ablation_descriptions(config_path: Path) -> dict[str, str]:
    """Load ablation descriptions from config file.

    Args:
        config_path: Path to the YAML config file
    """
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    descriptions = {}
    for ablation in config.get("ablations", []):
        descriptions[ablation["name"]] = ablation["description"]

    return descriptions


def discover_agencies(output_dir: Path) -> list[str]:
    """Discover all agency subdirectories in ablation output directory.

    Args:
        output_dir: Path to the ablation output directory
    """
    if not output_dir.exists():
        print(f"Warning: Ablation output directory not found: {output_dir}")
        return []

    # Find all subdirectories (each represents an agency)
    agency_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    agency_names = [d.name for d in sorted(agency_dirs)]

    return agency_names


def discover_ablation_results_for_agency(
    agency_name: str, output_dir: Path, config_path: Path, version: str = "v2"
) -> list[dict]:
    """Auto-discover all ablation clustering result files for a specific agency.

    Args:
        agency_name: Name of the agency
        output_dir: Path to the ablation output directory
        config_path: Path to the config file
        version: 'v1', 'v2', or 'ml' to determine which ablations to include
    """
    agency_dir = output_dir / agency_name

    if not agency_dir.exists():
        print(f"Warning: Agency directory not found: {agency_dir}")
        return []

    # Find all CSV files matching pattern
    pattern = str(agency_dir / "clustering_results_ablation_*.csv")
    result_files = glob.glob(pattern)

    if not result_files:
        print(f"Warning: No ablation results found for {agency_name} in {agency_dir}")
        return []

    # Load descriptions
    descriptions = load_ablation_descriptions(config_path)

    # Build config list
    configs = []
    for file_path in sorted(result_files):
        # Extract ablation name from filename
        # Pattern: clustering_results_ablation_<name>.csv
        filename = Path(file_path).name
        ablation_name = filename.replace("clustering_results_ablation_", "").replace(".csv", "")

        # Skip ablations in SKIP_ABLATIONS set
        if ablation_name in SKIP_ABLATIONS:
            continue

        # Filter based on version
        if version == "embeddings":
            if not ablation_name.startswith("embeddings_t"):
                continue
        elif version == "ml":
            # Accept any ablation name defined in ablation_configs_ml.yaml (dynamic, not hardcoded)
            ml_config_names = set(load_ablation_descriptions(config_path).keys())
            if ablation_name not in ml_config_names:
                continue
        elif version == "v2":
            # Accept any ablation name defined in ablation_configs_handengineered.yaml (dynamic, not hardcoded)
            v2_config_names = set(load_ablation_descriptions(config_path).keys())
            is_old_learned_rule = ablation_name in [
                "learned_rules_decision_tree",
                "learned_rules_random_forest",
            ]

            if ablation_name not in v2_config_names:
                continue
            if is_old_learned_rule:
                # Skip old learned rules - these are replaced by new ML ablations with tier subsets
                continue

        # Get description if available
        description = descriptions.get(ablation_name, "No description available")

        configs.append(
            {
                "input_csv": file_path,
                "output_report": str(
                    REPORTS_DIR / agency_name / f"{version}_{ablation_name}_report.txt"
                ),
                "method_name": ablation_name,
                "description": description,
                "agency_name": agency_name,
                "version": version,
            }
        )

    return configs


# ============================================================================


def print_cluster_analysis(metrics: dict, method_name: str, input_file: str, output_file=None):
    """
    Print detailed analysis of clustering results.

    Args:
        metrics: Output from evaluate_clusters()
        method_name: Name of clustering method
        input_file: Path to input CSV
        output_file: Optional file handle to write to (if None, prints to stdout)
    """

    def output(text=""):
        """Helper to print to both stdout and file if provided."""
        print(text)
        if output_file:
            output_file.write(text + "\n")

    output("=" * 80)
    output("CLUSTERING EVALUATION REPORT")
    output("=" * 80)
    output(f"Method:         {method_name}")
    output(f"Input:          {Path(input_file).name}")
    output("Ground Truth:   provisional_case_name column")
    output(f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output()

    # Overall metrics
    output("OVERALL METRICS (Macro-Averaged):")
    output("-" * 80)
    overall = metrics["overall"]
    output(f"Precision:      {overall['precision']:.4f} ({overall['precision'] * 100:.2f}%)")
    output(f"Recall:         {overall['recall']:.4f} ({overall['recall'] * 100:.2f}%)")
    output(f"F1 Score:       {overall['f1']:.4f} ({overall['f1'] * 100:.2f}%)")
    output()

    # Summary statistics
    summary = get_summary_statistics(metrics)
    output("CLUSTER QUALITY DISTRIBUTION:")
    output("-" * 80)
    total_clusters = overall["num_groundtruth_clusters_evaluated"]
    output(f"Total ground truth clusters evaluated: {total_clusters}")
    output(
        f"  Perfect (F1 = 1.0):           {summary['perfect_clusters']:3d} ({summary['perfect_clusters'] / total_clusters * 100:.1f}%)"
    )
    output(
        f"  High quality (0.9 ≤ F1 < 1.0): {summary['high_quality_clusters']:3d} ({summary['high_quality_clusters'] / total_clusters * 100:.1f}%)"
    )
    output(
        f"  Medium quality (0.7 ≤ F1 < 0.9): {summary['medium_quality_clusters']:3d} ({summary['medium_quality_clusters'] / total_clusters * 100:.1f}%)"
    )
    output(
        f"  Low quality (F1 < 0.7):       {summary['low_quality_clusters']:3d} ({summary['low_quality_clusters'] / total_clusters * 100:.1f}%)"
    )
    output()

    # Cluster size distribution
    output("GROUND TRUTH CLUSTER SIZE DISTRIBUTION:")
    output("-" * 80)
    output(f"Min cluster size:  {summary['min_cluster_size']} documents")
    output(f"Max cluster size:  {summary['max_cluster_size']} documents")
    output(f"Avg cluster size:  {summary['avg_cluster_size']:.1f} documents")
    output()

    # Error analysis
    output("ERROR ANALYSIS:")
    output("-" * 80)
    output(
        f"Clusters with splits:  {overall['clusters_with_splits']} ({overall['clusters_with_splits'] / total_clusters * 100:.1f}%)"
    )
    output(
        f"Clusters with merges:  {overall['clusters_with_merges']} ({overall['clusters_with_merges'] / total_clusters * 100:.1f}%)"
    )
    output()

    # Top errors (worst F1 scores)
    output("TOP 10 WORST PERFORMING CLUSTERS:")
    output("-" * 80)
    cluster_details = sorted(metrics["cluster_details"], key=lambda x: x["f1"])[:10]

    for i, cluster in enumerate(cluster_details, 1):
        output(f"\n{i}. Ground Truth Cluster: {cluster['groundtruth_cluster']}")
        output(
            f"   F1 Score:    {cluster['f1']:.4f} (Precision: {cluster['precision']:.4f}, Recall: {cluster['recall']:.4f})"
        )
        output(f"   Should have: {cluster['num_files_should_be_together']} documents")
        output(
            f"   Best match:  {cluster['num_files_in_best_cluster']} documents in test cluster {cluster['best_matching_test_cluster']}"
        )

        if cluster["num_split_clusters"] > 0:
            output(f"   Split into:  {cluster['num_split_clusters']} additional test clusters")

        if cluster["num_incorrectly_merged_files"] > 0:
            output(
                f"   Merged with: {cluster['num_incorrectly_merged_files']} documents from other ground truth clusters"
            )

    output()
    output("=" * 80)
    output()


def evaluate_single_result(config: dict):
    """Evaluate a single clustering result and return metrics."""
    input_csv = config["input_csv"]
    output_report = config["output_report"]
    method_name = config["method_name"]
    description = config.get("description", "No description")
    agency_name = config.get("agency_name", "Unknown")
    version = config.get("version", "v2")

    print(f"\n{'=' * 80}")
    print(f"Agency: {agency_name}")
    print(f"Version: {version}")
    print(f"Evaluating: {method_name}")
    print(f"Description: {description}")
    print(f"{'=' * 80}\n")

    # Load clustering results (contains both clusters and ground truth)
    print(f"Loading: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"WARNING: File not found, skipping: {input_csv}\n")
        return None
    except Exception as e:
        print(f"WARNING: Error loading file, skipping: {e}\n")
        return None

    print(f"Data shape: {df.shape}\n")

    # Verify required columns
    if "provisional_case_name" not in df.columns:
        print(f"ERROR: Missing 'provisional_case_name' column in {input_csv}")
        return None

    if "gdrive_name" not in df.columns:
        print(f"ERROR: Missing 'gdrive_name' column in {input_csv}")
        return None

    # Evaluate (pass same dataframe for both ground truth and test results)
    print("Evaluating clusters...\n")

    try:
        metrics = evaluate_clusters(df, df)
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}\n")
        import traceback

        traceback.print_exc()
        return None

    # Generate report
    output_path = Path(output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        print_cluster_analysis(metrics, f"{agency_name} - {method_name}", input_csv, f)

    print(f"Report saved to: {output_report}\n")

    return metrics


def generate_comparison_report(all_metrics: list, agency_name: str, output_path: str = None):
    """Generate a combined comparison report ranking all approaches for a single agency."""

    if output_path is None:
        output_path = str(REPORTS_DIR / agency_name / "ablation_comparison_report.txt")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load V2 ablation configs to get configuration details
    ablation_configs = {}
    if ABLATION_CONFIG_PATH_V2.exists():
        with open(ABLATION_CONFIG_PATH_V2) as f:
            config_data = yaml.safe_load(f)
            for ablation in config_data.get("ablations", []):
                ablation_configs[ablation["name"]] = ablation

    # Sort by F1 score (best to worst)
    sorted_results = sorted(all_metrics, key=lambda x: x["metrics"]["overall"]["f1"], reverse=True)

    with open(output_path, "w") as f:

        def output(text=""):
            print(text)
            f.write(text + "\n")

        output("=" * 80)
        output(f"ABLATION STUDY COMPARISON REPORT - {agency_name}")
        output("=" * 80)
        output(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output(f"Ablations evaluated: {len(sorted_results)}")
        output()

        output("RANKING BY F1 SCORE (Best to Worst):")
        output("=" * 80)
        output()

        for rank, result in enumerate(sorted_results, 1):
            method_name = result["method_name"]
            friendly_name = get_display_name(method_name)
            description = result.get("description", "No description")
            overall = result["metrics"]["overall"]
            config = ablation_configs.get(method_name, {})

            output(f"{rank}. {friendly_name}")
            output(f"   Description: {description}")

            # Add configuration details
            if config:
                date_window = config.get("date_proximity_days", "N/A")
                if date_window == 999999:
                    date_window = "∞ (no validation)"
                else:
                    date_window = f"{date_window}d"

                require_dates = "Yes" if config.get("require_dates_present", False) else "No"
                enabled_tiers = config.get("enabled_tiers", [])
                tiers_str = ",".join(map(str, enabled_tiers)) if enabled_tiers else "N/A"

                use_learned = config.get("use_learned_model", False)
                use_cascade = config.get("use_cascade_model", False)
                if use_learned or use_cascade:
                    if method_name.startswith("dt_"):
                        model_type = "Decision Tree"
                    elif method_name.startswith("rf_"):
                        model_type = "Random Forest"
                    elif method_name.startswith("lgbm_"):
                        model_type = "LightGBM"
                    else:
                        model_type = method_name
                    mode = "cascade" if use_cascade else "joint"
                    output(f"   Config: Model={model_type} ({mode})")
                else:
                    output(
                        f"   Config: Date Window={date_window}, Require Dates={require_dates}, Tiers={tiers_str}"
                    )

            output(f"   Precision: {overall['precision']:.4f} ({overall['precision'] * 100:.2f}%)")
            output(f"   Recall:    {overall['recall']:.4f} ({overall['recall'] * 100:.2f}%)")
            output(f"   F1 Score:  {overall['f1']:.4f} ({overall['f1'] * 100:.2f}%)")
            output(f"   Clusters evaluated: {overall['num_groundtruth_clusters_evaluated']}")
            output(
                f"   Splits: {overall['clusters_with_splits']} ({overall['clusters_with_splits'] / overall['num_groundtruth_clusters_evaluated'] * 100:.1f}%)"
            )
            output(
                f"   Merges: {overall['clusters_with_merges']} ({overall['clusters_with_merges'] / overall['num_groundtruth_clusters_evaluated'] * 100:.1f}%)"
            )
            output()

        output("=" * 80)
        output("DETAILED METRICS TABLE")
        output("=" * 80)
        output()
        output(
            f"{'Ablation':<30} {'Date':>6} {'ReqD':>4} {'Tiers':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Split':>5} {'Merge':>5}"
        )
        output("-" * 80)

        for result in sorted_results:
            method_name = result["method_name"]
            friendly_name = get_display_name(method_name)
            overall = result["metrics"]["overall"]
            config = ablation_configs.get(method_name, {})

            # Truncate method name if too long
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            # Extract config details
            date_window = config.get("date_proximity_days", "N/A")
            if date_window == 999999:
                date_str = "∞"
            elif isinstance(date_window, int):
                date_str = f"{date_window}d"
            else:
                date_str = "N/A"

            require_dates = "Y" if config.get("require_dates_present", False) else "N"

            enabled_tiers = config.get("enabled_tiers", [])
            if config.get("use_learned_model", False):
                tiers_str = "Learn"
            elif enabled_tiers:
                tiers_str = ",".join(map(str, enabled_tiers))
            else:
                tiers_str = "N/A"

            output(
                f"{display_name:<30} {date_str:>6} {require_dates:>4} {tiers_str:>6} "
                f"{overall['precision']:>6.4f} {overall['recall']:>6.4f} {overall['f1']:>6.4f} "
                f"{overall['clusters_with_splits']:>5d} {overall['clusters_with_merges']:>5d}"
            )

        output()
        output("=" * 80)
        output("LEGEND:")
        output("  Date: Date proximity window (0d=exact, 90d/180d/365d=window, ∞=no validation)")
        output("  ReqD: Require dates present (Y=yes, N=no/missing dates allowed)")
        output("  Tiers: Enabled tiers (1=regex only, 1,2=regex+LLM, 1,2,3=all, Learn=ML model)")
        output("  Prec/Rec/F1: Precision, Recall, F1 Score")
        output("  Split: Number of ground truth clusters split across multiple test clusters")
        output("  Merge: Number of ground truth clusters with incorrectly merged documents")
        output("=" * 80)
        output()

    print(f"\nComparison report saved to: {output_path}")


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean of values.

    Resamples the list (agencies) with replacement n_bootstrap times and
    returns the (lower, upper) percentile bounds.

    Args:
        values: List of per-agency metric values
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (e.g. 0.95 for 95% CI)

    Returns:
        (ci_lower, ci_upper) tuple
    """
    if len(values) < 2:
        mean = float(np.mean(values)) if values else 0.0
        return mean, mean

    arr = np.array(values)
    rng = np.random.default_rng(seed=42)
    boot_means = np.mean(rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True), axis=1)
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


def calculate_statistics(values: list[float]) -> dict:
    """
    Calculate comprehensive statistics for a list of values.

    Returns dict with mean, median, std, variance, min, max, q1, q3, iqr,
    and bootstrap 95% CI (ci_lower, ci_upper).
    """
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "ci_lower": None,
            "ci_upper": None,
        }

    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    ci_lower, ci_upper = bootstrap_ci(values)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,  # Sample std
        "variance": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,  # Sample variance
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def print_category_statistics_table(ablation_averages: list[dict], category_name: str, output_func):
    """
    Print comprehensive statistics table for all ablations in a category.

    Shows statistics (mean, median, std, etc.) for each ablation across agencies in the category.

    Args:
        ablation_averages: List of dicts with ablation results
        category_name: 'original', 'learned_rules', or 'holdout'
        output_func: Function to call for outputting text
    """
    # Filter to ablations that have results in this category
    category_key = f"{category_name}_results"
    relevant_ablations = [
        a for a in ablation_averages if a.get(category_key) and len(a[category_key]) > 0
    ]

    if not relevant_ablations:
        return

    output_func("=" * 100)
    output_func("STATISTICAL SUMMARY - All Ablations in This Category")
    output_func("=" * 100)
    output_func()
    output_func(
        "This table shows statistics for each ablation across all agencies in this category."
    )
    output_func("Low std/variance = consistent performance across agencies")
    output_func("Large IQR/range = high variability across agencies")
    output_func()

    # Prepare data for CSV export
    stats_rows = []

    # F1 STATISTICS TABLE
    output_func("F1 SCORE STATISTICS:")
    output_func("-" * 115)
    output_func(
        f"{'Ablation':<30} {'Mean':>7} {'Median':>7} {'Std':>7} {'Var':>7} {'Min':>7} {'Max':>7} {'IQR':>7} {'CI_Lo':>7} {'CI_Hi':>7}"
    )
    output_func("-" * 115)

    for ablation in relevant_ablations:
        results = ablation[category_key]
        f1_values = [r["metrics"]["f1"] for r in results]
        stats = calculate_statistics(f1_values)

        name = ablation["method_name"]
        friendly_name = get_display_name(name)
        display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

        output_func(
            f"{display_name:<30} "
            f"{stats['mean']:>7.4f} {stats['median']:>7.4f} {stats['std']:>7.4f} "
            f"{stats['variance']:>7.4f} {stats['min']:>7.4f} {stats['max']:>7.4f} "
            f"{stats['iqr']:>7.4f} {stats['ci_lower']:>7.4f} {stats['ci_upper']:>7.4f}"
        )

        # Store for CSV (use friendly name)
        stats_rows.append(
            {
                "ablation": friendly_name,
                "metric": "F1",
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "variance": stats["variance"],
                "min": stats["min"],
                "max": stats["max"],
                "iqr": stats["iqr"],
                "ci_lower_95": stats["ci_lower"],
                "ci_upper_95": stats["ci_upper"],
            }
        )

    output_func()

    # PRECISION STATISTICS TABLE
    output_func("PRECISION STATISTICS:")
    output_func("-" * 115)
    output_func(
        f"{'Ablation':<30} {'Mean':>7} {'Median':>7} {'Std':>7} {'Var':>7} {'Min':>7} {'Max':>7} {'IQR':>7} {'CI_Lo':>7} {'CI_Hi':>7}"
    )
    output_func("-" * 115)

    for ablation in relevant_ablations:
        results = ablation[category_key]
        precision_values = [r["metrics"]["precision"] for r in results]
        stats = calculate_statistics(precision_values)

        name = ablation["method_name"]
        friendly_name = get_display_name(name)
        display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

        output_func(
            f"{display_name:<30} "
            f"{stats['mean']:>7.4f} {stats['median']:>7.4f} {stats['std']:>7.4f} "
            f"{stats['variance']:>7.4f} {stats['min']:>7.4f} {stats['max']:>7.4f} "
            f"{stats['iqr']:>7.4f} {stats['ci_lower']:>7.4f} {stats['ci_upper']:>7.4f}"
        )

        # Store for CSV (use friendly name)
        stats_rows.append(
            {
                "ablation": friendly_name,
                "metric": "Precision",
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "variance": stats["variance"],
                "min": stats["min"],
                "max": stats["max"],
                "iqr": stats["iqr"],
                "ci_lower_95": stats["ci_lower"],
                "ci_upper_95": stats["ci_upper"],
            }
        )

    output_func()

    # RECALL STATISTICS TABLE
    output_func("RECALL STATISTICS:")
    output_func("-" * 115)
    output_func(
        f"{'Ablation':<30} {'Mean':>7} {'Median':>7} {'Std':>7} {'Var':>7} {'Min':>7} {'Max':>7} {'IQR':>7} {'CI_Lo':>7} {'CI_Hi':>7}"
    )
    output_func("-" * 115)

    for ablation in relevant_ablations:
        results = ablation[category_key]
        recall_values = [r["metrics"]["recall"] for r in results]
        stats = calculate_statistics(recall_values)

        name = ablation["method_name"]
        friendly_name = get_display_name(name)
        display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

        output_func(
            f"{display_name:<30} "
            f"{stats['mean']:>7.4f} {stats['median']:>7.4f} {stats['std']:>7.4f} "
            f"{stats['variance']:>7.4f} {stats['min']:>7.4f} {stats['max']:>7.4f} "
            f"{stats['iqr']:>7.4f} {stats['ci_lower']:>7.4f} {stats['ci_upper']:>7.4f}"
        )

        # Store for CSV (use friendly name)
        stats_rows.append(
            {
                "ablation": friendly_name,
                "metric": "Recall",
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "variance": stats["variance"],
                "min": stats["min"],
                "max": stats["max"],
                "iqr": stats["iqr"],
                "ci_lower_95": stats["ci_lower"],
                "ci_upper_95": stats["ci_upper"],
            }
        )

    output_func()
    output_func("=" * 100)
    output_func()

    # Export statistics to CSV
    csv_path = REPORTS_DIR / f"{category_name}_statistics.csv"
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(csv_path, index=False)
    print(f"Exported statistics to: {csv_path}")


def generate_cross_agency_summary(all_results: dict[str, list[dict]], output_path: str = None):
    """
    Generate a cross-agency summary showing how each ablation performs across all agencies.

    Separates results into:
    - Original evaluation agencies (used for V2 tuning)
    - Holdout agencies (true test set for generalization)

    Args:
        all_results: Dict mapping agency_name -> list of {method_name, description, metrics}
        output_path: Path to save the summary report
    """

    if output_path is None:
        output_path = str(REPORTS_DIR / "cross_agency_summary.txt")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load V2 and ML ablation configs
    ablation_configs = {}
    if ABLATION_CONFIG_PATH_V2.exists():
        with open(ABLATION_CONFIG_PATH_V2) as f:
            config_data = yaml.safe_load(f)
            for ablation in config_data.get("ablations", []):
                ablation_configs[ablation["name"]] = ablation

    if ABLATION_CONFIG_PATH_ML.exists():
        with open(ABLATION_CONFIG_PATH_ML) as f:
            config_data = yaml.safe_load(f)
            for ablation in config_data.get("ablations", []):
                ablation_configs[ablation["name"]] = ablation

    # Separate agencies into three categories:
    # 1. Original evaluation agencies - explicitly defined, used for V2 hand-coded rules tuning
    # 2. Learned rules training agencies - used to train ML models (excluded from original eval by default)
    # 3. Holdout agencies - true test set
    original_agencies = []
    holdout_agencies = []
    learned_rules_agencies = []
    uncategorized_agencies = []  # Track any agencies not in our predefined lists

    for agency_name in all_results.keys():
        if agency_name in ORIGINAL_EVALUATION_AGENCIES:
            original_agencies.append(agency_name)
        elif agency_name in LEARNED_RULES_TRAINING_AGENCIES:
            learned_rules_agencies.append(agency_name)
        elif agency_name in HOLDOUT_AGENCIES:
            holdout_agencies.append(agency_name)
        else:
            # This shouldn't happen if our lists are complete
            uncategorized_agencies.append(agency_name)
            print(f"WARNING: Agency '{agency_name}' not found in any category list!")

    original_agencies.sort()
    holdout_agencies.sort()
    learned_rules_agencies.sort()

    # Aggregate results by ablation method across agencies
    ablation_summary = {}  # method_name -> list of (agency, metrics, category)

    for agency_name, agency_results in all_results.items():
        if agency_name in ORIGINAL_EVALUATION_AGENCIES:
            category = "original"
        elif agency_name in LEARNED_RULES_TRAINING_AGENCIES:
            category = "learned_rules"
        elif agency_name in HOLDOUT_AGENCIES:
            category = "holdout"
        else:
            category = "uncategorized"

        for result in agency_results:
            method_name = result["method_name"]
            if method_name not in ablation_summary:
                ablation_summary[method_name] = []
            ablation_summary[method_name].append(
                {
                    "agency": agency_name,
                    "metrics": result["metrics"]["overall"],
                    "category": category,
                }
            )

    with open(output_path, "w") as f:

        def output(text=""):
            print(text)
            f.write(text + "\n")

        output("=" * 80)
        output("CROSS-AGENCY ABLATION SUMMARY")
        output("=" * 80)
        output(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output(f"Total agencies: {len(all_results)}")
        output(
            f"  - Original evaluation agencies: {len(original_agencies)} (explicitly defined, used for V2 hand-coded rules tuning)"
        )
        output(
            f"  - Learned rules training agencies: {len(learned_rules_agencies)} (used to train ML models, excluded from original)"
        )
        output(f"  - Holdout test agencies: {len(holdout_agencies)} (true test set)")
        if uncategorized_agencies:
            output(
                f"  - Uncategorized agencies: {len(uncategorized_agencies)} (WARNING: not in any predefined list)"
            )
        output(f"Ablation methods: {len(ablation_summary)}")
        output()

        # Calculate average performance for each ablation (split by category)
        ablation_averages = []
        for method_name, agency_results in ablation_summary.items():
            if not agency_results:
                continue

            # Separate results by category
            original_results = [r for r in agency_results if r["category"] == "original"]
            holdout_results = [r for r in agency_results if r["category"] == "holdout"]
            learned_rules_results = [r for r in agency_results if r["category"] == "learned_rules"]

            # Calculate averages for ALL agencies
            avg_precision = sum(r["metrics"]["precision"] for r in agency_results) / len(
                agency_results
            )
            avg_recall = sum(r["metrics"]["recall"] for r in agency_results) / len(agency_results)
            avg_f1 = sum(r["metrics"]["f1"] for r in agency_results) / len(agency_results)

            # Calculate averages for ORIGINAL agencies
            if original_results:
                orig_avg_precision = sum(r["metrics"]["precision"] for r in original_results) / len(
                    original_results
                )
                orig_avg_recall = sum(r["metrics"]["recall"] for r in original_results) / len(
                    original_results
                )
                orig_avg_f1 = sum(r["metrics"]["f1"] for r in original_results) / len(
                    original_results
                )
            else:
                orig_avg_precision = orig_avg_recall = orig_avg_f1 = None

            # Calculate averages for HOLDOUT agencies
            if holdout_results:
                hold_avg_precision = sum(r["metrics"]["precision"] for r in holdout_results) / len(
                    holdout_results
                )
                hold_avg_recall = sum(r["metrics"]["recall"] for r in holdout_results) / len(
                    holdout_results
                )
                hold_avg_f1 = sum(r["metrics"]["f1"] for r in holdout_results) / len(
                    holdout_results
                )
            else:
                hold_avg_precision = hold_avg_recall = hold_avg_f1 = None

            # Calculate averages for LEARNED RULES TRAINING agencies
            if learned_rules_results:
                learned_avg_precision = sum(
                    r["metrics"]["precision"] for r in learned_rules_results
                ) / len(learned_rules_results)
                learned_avg_recall = sum(
                    r["metrics"]["recall"] for r in learned_rules_results
                ) / len(learned_rules_results)
                learned_avg_f1 = sum(r["metrics"]["f1"] for r in learned_rules_results) / len(
                    learned_rules_results
                )
            else:
                learned_avg_precision = learned_avg_recall = learned_avg_f1 = None

            # Calculate B-Cubed averages for ORIGINAL agencies
            if original_results:
                orig_avg_bcubed_precision = sum(
                    r["metrics"].get("bcubed_precision", 0) for r in original_results
                ) / len(original_results)
                orig_avg_bcubed_recall = sum(
                    r["metrics"].get("bcubed_recall", 0) for r in original_results
                ) / len(original_results)
                orig_avg_bcubed_f1 = sum(
                    r["metrics"].get("bcubed_f1", 0) for r in original_results
                ) / len(original_results)
            else:
                orig_avg_bcubed_precision = orig_avg_bcubed_recall = orig_avg_bcubed_f1 = None

            # Calculate B-Cubed averages for HOLDOUT agencies
            if holdout_results:
                hold_avg_bcubed_precision = sum(
                    r["metrics"].get("bcubed_precision", 0) for r in holdout_results
                ) / len(holdout_results)
                hold_avg_bcubed_recall = sum(
                    r["metrics"].get("bcubed_recall", 0) for r in holdout_results
                ) / len(holdout_results)
                hold_avg_bcubed_f1 = sum(
                    r["metrics"].get("bcubed_f1", 0) for r in holdout_results
                ) / len(holdout_results)
            else:
                hold_avg_bcubed_precision = hold_avg_bcubed_recall = hold_avg_bcubed_f1 = None

            # Calculate B-Cubed averages for LEARNED RULES agencies
            if learned_rules_results:
                learned_avg_bcubed_precision = sum(
                    r["metrics"].get("bcubed_precision", 0) for r in learned_rules_results
                ) / len(learned_rules_results)
                learned_avg_bcubed_recall = sum(
                    r["metrics"].get("bcubed_recall", 0) for r in learned_rules_results
                ) / len(learned_rules_results)
                learned_avg_bcubed_f1 = sum(
                    r["metrics"].get("bcubed_f1", 0) for r in learned_rules_results
                ) / len(learned_rules_results)
            else:
                learned_avg_bcubed_precision = learned_avg_bcubed_recall = learned_avg_bcubed_f1 = (
                    None
                )

            # Calculate ARI averages for ORIGINAL agencies
            if original_results:
                ari_values = [
                    r["metrics"].get("ari")
                    for r in original_results
                    if r["metrics"].get("ari") is not None
                ]
                orig_avg_ari = sum(ari_values) / len(ari_values) if ari_values else None
            else:
                orig_avg_ari = None

            # Calculate ARI averages for HOLDOUT agencies
            if holdout_results:
                ari_values = [
                    r["metrics"].get("ari")
                    for r in holdout_results
                    if r["metrics"].get("ari") is not None
                ]
                hold_avg_ari = sum(ari_values) / len(ari_values) if ari_values else None
            else:
                hold_avg_ari = None

            # Calculate ARI averages for LEARNED RULES agencies
            if learned_rules_results:
                ari_values = [
                    r["metrics"].get("ari")
                    for r in learned_rules_results
                    if r["metrics"].get("ari") is not None
                ]
                learned_avg_ari = sum(ari_values) / len(ari_values) if ari_values else None
            else:
                learned_avg_ari = None

            ablation_averages.append(
                {
                    "method_name": method_name,
                    "num_agencies": len(agency_results),
                    "num_original": len(original_results),
                    "num_holdout": len(holdout_results),
                    "num_learned_rules": len(learned_rules_results),
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_f1": avg_f1,
                    "orig_avg_precision": orig_avg_precision,
                    "orig_avg_recall": orig_avg_recall,
                    "orig_avg_f1": orig_avg_f1,
                    "hold_avg_precision": hold_avg_precision,
                    "hold_avg_recall": hold_avg_recall,
                    "hold_avg_f1": hold_avg_f1,
                    "learned_avg_precision": learned_avg_precision,
                    "learned_avg_recall": learned_avg_recall,
                    "learned_avg_f1": learned_avg_f1,
                    # B-Cubed metrics
                    "orig_avg_bcubed_precision": orig_avg_bcubed_precision,
                    "orig_avg_bcubed_recall": orig_avg_bcubed_recall,
                    "orig_avg_bcubed_f1": orig_avg_bcubed_f1,
                    "hold_avg_bcubed_precision": hold_avg_bcubed_precision,
                    "hold_avg_bcubed_recall": hold_avg_bcubed_recall,
                    "hold_avg_bcubed_f1": hold_avg_bcubed_f1,
                    "learned_avg_bcubed_precision": learned_avg_bcubed_precision,
                    "learned_avg_bcubed_recall": learned_avg_bcubed_recall,
                    "learned_avg_bcubed_f1": learned_avg_bcubed_f1,
                    # ARI metrics
                    "orig_avg_ari": orig_avg_ari,
                    "hold_avg_ari": hold_avg_ari,
                    "learned_avg_ari": learned_avg_ari,
                    # Results by category
                    "agency_results": agency_results,
                    "original_results": original_results,
                    "holdout_results": holdout_results,
                    "learned_rules_results": learned_rules_results,
                }
            )

        # Sort by average F1 on ORIGINAL agencies (best to worst)
        # This shows which approaches performed best on the tuning set
        ablation_averages.sort(
            key=lambda x: x["orig_avg_f1"] if x["orig_avg_f1"] is not None else 0, reverse=True
        )

        # ====================================================================
        # SECTION 1: ORIGINAL EVALUATION AGENCIES
        # ====================================================================
        output("=" * 80)
        output("SECTION 1: ORIGINAL EVALUATION AGENCIES")
        output("=" * 80)
        output(
            f"Explicitly defined agencies used for V2 hand-coded rules tuning: {len(original_agencies)}"
        )
        output()
        output("NOTE: These agencies are explicitly listed in ORIGINAL_EVALUATION_AGENCIES.")
        output("      Learned rules training agencies are excluded and shown separately.")
        output()
        if original_agencies:
            for agency in original_agencies:
                output(f"  - {agency}")
        output()
        if learned_rules_agencies:
            output(f"Excluded (learned rules training): {len(learned_rules_agencies)}")
            for agency in learned_rules_agencies:
                output(f"  - {agency}")
            output()

        output("RANKING BY AVERAGE F1 SCORE ON ORIGINAL AGENCIES (Best to Worst):")
        output("=" * 80)
        output()

        for rank, ablation in enumerate(ablation_averages, 1):
            if ablation["num_original"] == 0:
                continue

            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            description = ablation_configs.get(method_name, {}).get("description", "No description")

            output(f"{rank}. {friendly_name}")
            output(f"   Description: {description}")
            output(f"   Agencies evaluated: {ablation['num_original']}")
            output(
                f"   Average Precision: {ablation['orig_avg_precision']:.4f} ({ablation['orig_avg_precision'] * 100:.2f}%)"
            )
            output(
                f"   Average Recall:    {ablation['orig_avg_recall']:.4f} ({ablation['orig_avg_recall'] * 100:.2f}%)"
            )
            output(
                f"   Average F1 Score:  {ablation['orig_avg_f1']:.4f} ({ablation['orig_avg_f1'] * 100:.2f}%)"
            )
            output()

            # Show per-agency breakdown for ORIGINAL agencies only
            output("   Per-agency results:")
            for result in sorted(
                ablation["original_results"], key=lambda x: x["metrics"]["f1"], reverse=True
            ):
                agency = result["agency"]
                metrics = result["metrics"]
                output(
                    f"      {agency:<40s} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
                )
            output()

        # Print statistics table for all ablations in original evaluation category
        output()
        print_category_statistics_table(ablation_averages, "original", output)

        # ====================================================================
        # SECTION 2: LEARNED RULES TRAINING AGENCIES
        # ====================================================================
        output()
        output("=" * 80)
        output("SECTION 2: LEARNED RULES TRAINING AGENCIES")
        output("=" * 80)
        output(f"Agencies used to train ML models: {len(learned_rules_agencies)}")
        output()
        output(
            "NOTE: These agencies were used for manual feature engineering and ML model training."
        )
        output("      They are excluded from 'original evaluation' to avoid data leakage.")
        output()
        if learned_rules_agencies:
            for agency in learned_rules_agencies:
                output(f"  - {agency}")
        output()

        # Re-sort by average F1 on LEARNED RULES agencies (best to worst)
        ablation_averages_learned = [a for a in ablation_averages if a["num_learned_rules"] > 0]
        ablation_averages_learned.sort(
            key=lambda x: x["learned_avg_f1"] if x["learned_avg_f1"] is not None else 0,
            reverse=True,
        )

        output("RANKING BY AVERAGE F1 SCORE ON LEARNED RULES TRAINING AGENCIES (Best to Worst):")
        output("=" * 80)
        output()

        for rank, ablation in enumerate(ablation_averages_learned, 1):
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            description = ablation_configs.get(method_name, {}).get("description", "No description")

            output(f"{rank}. {friendly_name}")
            output(f"   Description: {description}")
            output(f"   Agencies evaluated: {ablation['num_learned_rules']}")
            output(
                f"   Average Precision: {ablation['learned_avg_precision']:.4f} ({ablation['learned_avg_precision'] * 100:.2f}%)"
            )
            output(
                f"   Average Recall:    {ablation['learned_avg_recall']:.4f} ({ablation['learned_avg_recall'] * 100:.2f}%)"
            )
            output(
                f"   Average F1 Score:  {ablation['learned_avg_f1']:.4f} ({ablation['learned_avg_f1'] * 100:.2f}%)"
            )
            output()

            # Show per-agency breakdown for LEARNED RULES agencies only
            output("   Per-agency results:")
            for result in sorted(
                ablation["learned_rules_results"], key=lambda x: x["metrics"]["f1"], reverse=True
            ):
                agency = result["agency"]
                metrics = result["metrics"]
                output(
                    f"      {agency:<40s} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
                )
            output()

        # Print statistics table for all ablations in learned rules category
        output()
        print_category_statistics_table(ablation_averages_learned, "learned_rules", output)

        # ====================================================================
        # SECTION 3: HOLDOUT TEST AGENCIES
        # ====================================================================
        output()
        output("=" * 80)
        output("SECTION 2: HOLDOUT TEST AGENCIES")
        output("=" * 80)
        output(f"Held-out agencies (not used for V2 tuning): {len(holdout_agencies)}")
        output()
        if holdout_agencies:
            for agency in holdout_agencies:
                output(f"  - {agency}")
        output()

        # Re-sort by average F1 on HOLDOUT agencies (best to worst)
        # This shows which approaches generalize best to unseen data
        ablation_averages_holdout = [a for a in ablation_averages if a["num_holdout"] > 0]
        ablation_averages_holdout.sort(
            key=lambda x: x["hold_avg_f1"] if x["hold_avg_f1"] is not None else 0, reverse=True
        )

        output("RANKING BY AVERAGE F1 SCORE ON HOLDOUT AGENCIES (Best to Worst):")
        output("=" * 80)
        output()

        for rank, ablation in enumerate(ablation_averages_holdout, 1):
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            description = ablation_configs.get(method_name, {}).get("description", "No description")

            output(f"{rank}. {friendly_name}")
            output(f"   Description: {description}")
            output(f"   Agencies evaluated: {ablation['num_holdout']}")
            output(
                f"   Average Precision: {ablation['hold_avg_precision']:.4f} ({ablation['hold_avg_precision'] * 100:.2f}%)"
            )
            output(
                f"   Average Recall:    {ablation['hold_avg_recall']:.4f} ({ablation['hold_avg_recall'] * 100:.2f}%)"
            )
            output(
                f"   Average F1 Score:  {ablation['hold_avg_f1']:.4f} ({ablation['hold_avg_f1'] * 100:.2f}%)"
            )
            output()

            # Show per-agency breakdown for HOLDOUT agencies only
            output("   Per-agency results:")
            for result in sorted(
                ablation["holdout_results"], key=lambda x: x["metrics"]["f1"], reverse=True
            ):
                agency = result["agency"]
                metrics = result["metrics"]
                output(
                    f"      {agency:<40s} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
                )
            output()

        # Print statistics table for all ablations in holdout category
        output()
        print_category_statistics_table(ablation_averages_holdout, "holdout", output)

        # ====================================================================
        # SECTION 4: COMPARISON - ORIGINAL vs HOLDOUT
        # ====================================================================
        output()
        output("=" * 80)
        output("SECTION 4: GENERALIZATION ANALYSIS - ORIGINAL vs HOLDOUT")
        output("=" * 80)
        output()
        output(
            "Comparison of performance on original agencies (hand-coded rules tuning) vs holdout agencies (unseen test data)."
        )
        output()
        output(
            f"Original agencies: {len(original_agencies)} (explicitly defined, used for V2 hand-coded rules tuning)"
        )
        output(
            f"Holdout agencies:  {len(holdout_agencies)} (true test set, never seen during tuning)"
        )
        output()
        output(
            f"NOTE: Learned rules training agencies ({len(learned_rules_agencies)}) are excluded from this comparison."
        )
        output("      They represent a different data relationship (used for ML model training).")
        output()
        output("Positive delta indicates better performance on holdout (good generalization).")
        output(
            "Negative delta indicates worse performance on holdout (possible overfitting to tuning set)."
        )
        output()

        # Create comparison table
        output(f"{'Ablation':<30} {'Orig F1':>8} {'Hold F1':>8} {'Delta':>8} {'% Change':>10}")
        output("-" * 80)

        comparison_data = []
        for ablation in ablation_averages:
            if ablation["num_original"] > 0 and ablation["num_holdout"] > 0:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                orig_f1 = ablation["orig_avg_f1"]
                hold_f1 = ablation["hold_avg_f1"]
                delta = hold_f1 - orig_f1
                pct_change = (delta / orig_f1) * 100 if orig_f1 > 0 else 0

                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )

                delta_str = f"{delta:+.4f}"
                pct_str = f"{pct_change:+.2f}%"

                output(
                    f"{display_name:<30} {orig_f1:>8.4f} {hold_f1:>8.4f} {delta_str:>8} {pct_str:>10}"
                )

                comparison_data.append(
                    {
                        "method_name": friendly_name,
                        "orig_f1": orig_f1,
                        "hold_f1": hold_f1,
                        "delta": delta,
                        "pct_change": pct_change,
                    }
                )

        output()

        # Summary statistics
        if comparison_data:
            avg_orig_f1 = sum(d["orig_f1"] for d in comparison_data) / len(comparison_data)
            avg_hold_f1 = sum(d["hold_f1"] for d in comparison_data) / len(comparison_data)
            avg_delta = sum(d["delta"] for d in comparison_data) / len(comparison_data)
            avg_pct_change = sum(d["pct_change"] for d in comparison_data) / len(comparison_data)

            output("SUMMARY STATISTICS:")
            output("-" * 80)
            output(f"Average F1 on original agencies: {avg_orig_f1:.4f}")
            output(f"Average F1 on holdout agencies:  {avg_hold_f1:.4f}")
            output(f"Average delta (holdout - orig):  {avg_delta:+.4f} ({avg_pct_change:+.2f}%)")
            output()

            # Count positive vs negative deltas
            positive_deltas = sum(1 for d in comparison_data if d["delta"] > 0)
            negative_deltas = sum(1 for d in comparison_data if d["delta"] < 0)
            neutral_deltas = sum(1 for d in comparison_data if d["delta"] == 0)

            output(
                f"Ablations with better holdout performance: {positive_deltas}/{len(comparison_data)}"
            )
            output(
                f"Ablations with worse holdout performance:  {negative_deltas}/{len(comparison_data)}"
            )
            output(
                f"Ablations with equal performance:         {neutral_deltas}/{len(comparison_data)}"
            )
            output()

            # Identify best generalizing ablation
            best_generalizer = max(comparison_data, key=lambda x: x["hold_f1"])
            output("BEST GENERALIZING ABLATION (highest F1 on holdout):")
            output(f"  Method: {best_generalizer['method_name']}")
            output(f"  Orig F1: {best_generalizer['orig_f1']:.4f}")
            output(f"  Hold F1: {best_generalizer['hold_f1']:.4f}")
            output(
                f"  Delta: {best_generalizer['delta']:+.4f} ({best_generalizer['pct_change']:+.2f}%)"
            )
            output()

        output()
        output("=" * 80)
        output("DETAILED F1 COMPARISON TABLE")
        output("=" * 80)
        output()

        # TABLE 1: Original agencies
        if original_agencies:
            output("ORIGINAL EVALUATION AGENCIES:")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in original_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(original_agencies) + 8))

            # Prepare data for CSV export
            original_csv_rows = []

            for ablation in ablation_averages:
                if ablation["num_original"] == 0:
                    continue

                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                agency_f1_map = {
                    r["agency"]: r["metrics"]["f1"] for r in ablation["original_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in original_agencies:
                    f1 = agency_f1_map.get(agency, None)
                    if f1 is not None:
                        row += f" {f1:>13.4f}"
                        csv_row[agency] = f1
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_f1 = ablation["orig_avg_f1"] if ablation["orig_avg_f1"] is not None else 0
                row += f" {mean_f1:>7.4f}"
                csv_row["Mean"] = mean_f1

                output(row)
                original_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if original_csv_rows:
                csv_path = REPORTS_DIR / "original_f1_comparison.csv"
                original_df = pd.DataFrame(original_csv_rows)
                original_df.to_csv(csv_path, index=False)
                print(f"Exported original F1 comparison to: {csv_path}")

                # Also create a summary table with means and bootstrap CIs
                summary_rows = []
                for ablation in ablation_averages:
                    if ablation["num_original"] == 0:
                        continue
                    f1_vals = [r["metrics"]["f1"] for r in ablation["original_results"]]
                    prec_vals = [r["metrics"]["precision"] for r in ablation["original_results"]]
                    rec_vals = [r["metrics"]["recall"] for r in ablation["original_results"]]
                    f1_ci = bootstrap_ci(f1_vals)
                    prec_ci = bootstrap_ci(prec_vals)
                    rec_ci = bootstrap_ci(rec_vals)
                    summary_rows.append(
                        {
                            "Ablation": get_display_name(ablation["method_name"]),
                            "Mean_Precision": ablation["orig_avg_precision"],
                            "Precision_CI_Lower_95": prec_ci[0],
                            "Precision_CI_Upper_95": prec_ci[1],
                            "Mean_Recall": ablation["orig_avg_recall"],
                            "Recall_CI_Lower_95": rec_ci[0],
                            "Recall_CI_Upper_95": rec_ci[1],
                            "Mean_F1": ablation["orig_avg_f1"],
                            "F1_CI_Lower_95": f1_ci[0],
                            "F1_CI_Upper_95": f1_ci[1],
                        }
                    )

                if summary_rows:
                    summary_path = REPORTS_DIR / "original_f1_means.csv"
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_csv(summary_path, index=False)
                    print(f"Exported original F1 means to: {summary_path}")

        # TABLE 2: Learned rules training agencies
        if learned_rules_agencies and ablation_averages_learned:
            output("LEARNED RULES TRAINING AGENCIES:")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in learned_rules_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(learned_rules_agencies) + 8))

            # Prepare data for CSV export
            learned_csv_rows = []

            for ablation in ablation_averages_learned:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                agency_f1_map = {
                    r["agency"]: r["metrics"]["f1"] for r in ablation["learned_rules_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in learned_rules_agencies:
                    f1 = agency_f1_map.get(agency, None)
                    if f1 is not None:
                        row += f" {f1:>13.4f}"
                        csv_row[agency] = f1
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_f1 = (
                    ablation["learned_avg_f1"] if ablation["learned_avg_f1"] is not None else 0
                )
                row += f" {mean_f1:>7.4f}"
                csv_row["Mean"] = mean_f1

                output(row)
                learned_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if learned_csv_rows:
                csv_path = REPORTS_DIR / "learned_rules_f1_comparison.csv"
                learned_df = pd.DataFrame(learned_csv_rows)
                learned_df.to_csv(csv_path, index=False)
                print(f"Exported learned rules F1 comparison to: {csv_path}")

                # Also create a summary table with means and bootstrap CIs
                summary_rows = []
                for ablation in ablation_averages_learned:
                    f1_vals = [r["metrics"]["f1"] for r in ablation["learned_rules_results"]]
                    prec_vals = [
                        r["metrics"]["precision"] for r in ablation["learned_rules_results"]
                    ]
                    rec_vals = [r["metrics"]["recall"] for r in ablation["learned_rules_results"]]
                    f1_ci = bootstrap_ci(f1_vals)
                    prec_ci = bootstrap_ci(prec_vals)
                    rec_ci = bootstrap_ci(rec_vals)
                    summary_rows.append(
                        {
                            "Ablation": get_display_name(ablation["method_name"]),
                            "Mean_Precision": ablation["learned_avg_precision"],
                            "Precision_CI_Lower_95": prec_ci[0],
                            "Precision_CI_Upper_95": prec_ci[1],
                            "Mean_Recall": ablation["learned_avg_recall"],
                            "Recall_CI_Lower_95": rec_ci[0],
                            "Recall_CI_Upper_95": rec_ci[1],
                            "Mean_F1": ablation["learned_avg_f1"],
                            "F1_CI_Lower_95": f1_ci[0],
                            "F1_CI_Upper_95": f1_ci[1],
                        }
                    )

                if summary_rows:
                    summary_path = REPORTS_DIR / "learned_rules_f1_means.csv"
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_csv(summary_path, index=False)
                    print(f"Exported learned rules F1 means to: {summary_path}")

        # TABLE 3: Holdout agencies
        if holdout_agencies and ablation_averages_holdout:
            output("HOLDOUT TEST AGENCIES:")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in holdout_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(holdout_agencies) + 8))

            # Prepare data for CSV export
            holdout_csv_rows = []

            for ablation in ablation_averages_holdout:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                agency_f1_map = {
                    r["agency"]: r["metrics"]["f1"] for r in ablation["holdout_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in holdout_agencies:
                    f1 = agency_f1_map.get(agency, None)
                    if f1 is not None:
                        row += f" {f1:>13.4f}"
                        csv_row[agency] = f1
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_f1 = ablation["hold_avg_f1"] if ablation["hold_avg_f1"] is not None else 0
                row += f" {mean_f1:>7.4f}"
                csv_row["Mean"] = mean_f1

                output(row)
                holdout_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if holdout_csv_rows:
                csv_path = REPORTS_DIR / "holdout_f1_comparison.csv"
                holdout_df = pd.DataFrame(holdout_csv_rows)
                holdout_df.to_csv(csv_path, index=False)
                print(f"Exported holdout F1 comparison to: {csv_path}")

                # Also create a summary table with means and bootstrap CIs
                summary_rows = []
                for ablation in ablation_averages_holdout:
                    f1_vals = [r["metrics"]["f1"] for r in ablation["holdout_results"]]
                    prec_vals = [r["metrics"]["precision"] for r in ablation["holdout_results"]]
                    rec_vals = [r["metrics"]["recall"] for r in ablation["holdout_results"]]
                    f1_ci = bootstrap_ci(f1_vals)
                    prec_ci = bootstrap_ci(prec_vals)
                    rec_ci = bootstrap_ci(rec_vals)
                    summary_rows.append(
                        {
                            "Ablation": get_display_name(ablation["method_name"]),
                            "Mean_Precision": ablation["hold_avg_precision"],
                            "Precision_CI_Lower_95": prec_ci[0],
                            "Precision_CI_Upper_95": prec_ci[1],
                            "Mean_Recall": ablation["hold_avg_recall"],
                            "Recall_CI_Lower_95": rec_ci[0],
                            "Recall_CI_Upper_95": rec_ci[1],
                            "Mean_F1": ablation["hold_avg_f1"],
                            "F1_CI_Lower_95": f1_ci[0],
                            "F1_CI_Upper_95": f1_ci[1],
                        }
                    )

                if summary_rows:
                    summary_path = REPORTS_DIR / "holdout_f1_means.csv"
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_csv(summary_path, index=False)
                    print(f"Exported holdout F1 means to: {summary_path}")

        output("=" * 80)
        output("LEGEND:")
        output("  F1 scores shown for each (ablation, agency) pair")
        output("  Mean: Average F1 across agencies in that group")
        output("  N/A: Ablation not run for this agency")
        output("=" * 80)
        output()

        # ====================================================================
        # DETAILED B-CUBED F1 COMPARISON TABLE
        # ====================================================================
        output()
        output("=" * 80)
        output("DETAILED B-CUBED F1 COMPARISON TABLE")
        output("=" * 80)
        output()

        # TABLE 1: Original agencies - B-Cubed F1
        if original_agencies:
            output("ORIGINAL EVALUATION AGENCIES (B-Cubed F1):")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in original_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(original_agencies) + 8))

            # Sort by B-Cubed F1
            ablation_bcubed_sorted = sorted(
                [a for a in ablation_averages if a["num_original"] > 0],
                key=lambda x: x.get("orig_avg_bcubed_f1", 0),
                reverse=True,
            )

            # Prepare data for CSV export
            original_bcubed_csv_rows = []

            for ablation in ablation_bcubed_sorted:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                # Build map of agency -> B-Cubed F1
                agency_bcubed_map = {
                    r["agency"]: r["metrics"].get("bcubed_f1", 0)
                    for r in ablation["original_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in original_agencies:
                    bcubed_f1 = agency_bcubed_map.get(agency, None)
                    if bcubed_f1 is not None:
                        row += f" {bcubed_f1:>13.4f}"
                        csv_row[agency] = bcubed_f1
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_bcubed_f1 = ablation.get("orig_avg_bcubed_f1", 0)
                row += f" {mean_bcubed_f1:>7.4f}"
                csv_row["Mean"] = mean_bcubed_f1

                output(row)
                original_bcubed_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if original_bcubed_csv_rows:
                csv_path = REPORTS_DIR / "original_bcubed_f1_comparison.csv"
                bcubed_df = pd.DataFrame(original_bcubed_csv_rows)
                bcubed_df.to_csv(csv_path, index=False)
                print(f"Exported original B-Cubed F1 comparison to: {csv_path}")

        # TABLE 2: Holdout agencies - B-Cubed F1
        if holdout_agencies:
            output("HOLDOUT TEST AGENCIES (B-Cubed F1):")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in holdout_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(holdout_agencies) + 8))

            # Sort by B-Cubed F1
            ablation_bcubed_sorted = sorted(
                [a for a in ablation_averages if a["num_holdout"] > 0],
                key=lambda x: x.get("hold_avg_bcubed_f1", 0),
                reverse=True,
            )

            # Prepare data for CSV export
            holdout_bcubed_csv_rows = []

            for ablation in ablation_bcubed_sorted:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                # Build map of agency -> B-Cubed F1
                agency_bcubed_map = {
                    r["agency"]: r["metrics"].get("bcubed_f1", 0)
                    for r in ablation["holdout_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in holdout_agencies:
                    bcubed_f1 = agency_bcubed_map.get(agency, None)
                    if bcubed_f1 is not None:
                        row += f" {bcubed_f1:>13.4f}"
                        csv_row[agency] = bcubed_f1
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_bcubed_f1 = ablation.get("hold_avg_bcubed_f1", 0)
                row += f" {mean_bcubed_f1:>7.4f}"
                csv_row["Mean"] = mean_bcubed_f1

                output(row)
                holdout_bcubed_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if holdout_bcubed_csv_rows:
                csv_path = REPORTS_DIR / "holdout_bcubed_f1_comparison.csv"
                bcubed_df = pd.DataFrame(holdout_bcubed_csv_rows)
                bcubed_df.to_csv(csv_path, index=False)
                print(f"Exported holdout B-Cubed F1 comparison to: {csv_path}")

        output("=" * 80)
        output("LEGEND:")
        output("  B-Cubed F1 scores (document-weighted) for each (ablation, agency) pair")
        output("  Mean: Average B-Cubed F1 across agencies in that group")
        output("=" * 80)
        output()

        # ====================================================================
        # DETAILED ARI COMPARISON TABLE
        # ====================================================================
        output()
        output("=" * 80)
        output("DETAILED ARI COMPARISON TABLE")
        output("=" * 80)
        output()

        # TABLE 1: Original agencies - ARI
        if original_agencies:
            output("ORIGINAL EVALUATION AGENCIES (ARI):")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in original_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(original_agencies) + 8))

            # Sort by ARI
            ablation_ari_sorted = sorted(
                [a for a in ablation_averages if a["num_original"] > 0],
                key=lambda x: (
                    x.get("orig_avg_ari", -1) if x.get("orig_avg_ari") is not None else -1
                ),
                reverse=True,
            )

            # Prepare data for CSV export
            original_ari_csv_rows = []

            for ablation in ablation_ari_sorted:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                # Build map of agency -> ARI
                agency_ari_map = {
                    r["agency"]: r["metrics"].get("ari") for r in ablation["original_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in original_agencies:
                    ari = agency_ari_map.get(agency)
                    if ari is not None:
                        row += f" {ari:>13.4f}"
                        csv_row[agency] = ari
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_ari = ablation.get("orig_avg_ari")
                if mean_ari is not None:
                    row += f" {mean_ari:>7.4f}"
                    csv_row["Mean"] = mean_ari
                else:
                    row += f" {'N/A':>7}"
                    csv_row["Mean"] = None

                output(row)
                original_ari_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if original_ari_csv_rows:
                csv_path = REPORTS_DIR / "original_ari_comparison.csv"
                ari_df = pd.DataFrame(original_ari_csv_rows)
                ari_df.to_csv(csv_path, index=False)
                print(f"Exported original ARI comparison to: {csv_path}")

        # TABLE 2: Holdout agencies - ARI
        if holdout_agencies:
            output("HOLDOUT TEST AGENCIES (ARI):")
            output("-" * 80)
            header = f"{'Ablation':<30}"
            for agency in holdout_agencies:
                agency_short = agency[:10] + "..." if len(agency) > 13 else agency
                header += f" {agency_short:>13}"
            header += f" {'Mean':>7}"
            output(header)
            output("-" * (30 + 14 * len(holdout_agencies) + 8))

            # Sort by ARI
            ablation_ari_sorted = sorted(
                [a for a in ablation_averages if a["num_holdout"] > 0],
                key=lambda x: (
                    x.get("hold_avg_ari", -1) if x.get("hold_avg_ari") is not None else -1
                ),
                reverse=True,
            )

            # Prepare data for CSV export
            holdout_ari_csv_rows = []

            for ablation in ablation_ari_sorted:
                method_name = ablation["method_name"]
                friendly_name = get_display_name(method_name)
                display_name = (
                    friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name
                )
                row = f"{display_name:<30}"

                # Build map of agency -> ARI
                agency_ari_map = {
                    r["agency"]: r["metrics"].get("ari") for r in ablation["holdout_results"]
                }

                # Build CSV row (use friendly name)
                csv_row = {"Ablation": friendly_name}

                for agency in holdout_agencies:
                    ari = agency_ari_map.get(agency)
                    if ari is not None:
                        row += f" {ari:>13.4f}"
                        csv_row[agency] = ari
                    else:
                        row += f" {'N/A':>13}"
                        csv_row[agency] = None

                mean_ari = ablation.get("hold_avg_ari")
                if mean_ari is not None:
                    row += f" {mean_ari:>7.4f}"
                    csv_row["Mean"] = mean_ari
                else:
                    row += f" {'N/A':>7}"
                    csv_row["Mean"] = None

                output(row)
                holdout_ari_csv_rows.append(csv_row)

            output()

            # Export to CSV
            if holdout_ari_csv_rows:
                csv_path = REPORTS_DIR / "holdout_ari_comparison.csv"
                ari_df = pd.DataFrame(holdout_ari_csv_rows)
                ari_df.to_csv(csv_path, index=False)
                print(f"Exported holdout ARI comparison to: {csv_path}")

        output("=" * 80)
        output("LEGEND:")
        output("  ARI scores (pair-based, -1 to 1) for each (ablation, agency) pair")
        output("  Mean: Average ARI across agencies in that group")
        output("=" * 80)
        output()

        # ====================================================================
        # SECTION 5: B-CUBED METRICS (DOCUMENT-WEIGHTED)
        # ====================================================================
        output()
        output("=" * 80)
        output("SECTION 5: B-CUBED METRICS (DOCUMENT-WEIGHTED)")
        output("=" * 80)
        output()
        output("B-Cubed metrics weight by document count - errors in large clusters affect")
        output("more documents' scores than errors in small clusters.")
        output()
        output("If rankings differ from macro-averaged F1, this indicates that approaches")
        output("perform differently on large vs. small clusters.")
        output()

        # Original agencies - B-Cubed ranking
        output("ORIGINAL EVALUATION AGENCIES - Ranked by B-Cubed F1:")
        output("-" * 80)

        # Sort by B-Cubed F1
        ablation_bcubed_orig = [a for a in ablation_averages if a["num_original"] > 0]
        ablation_bcubed_orig.sort(key=lambda x: x.get("orig_avg_bcubed_f1", 0), reverse=True)

        output(
            f"{'Ablation':<30} {'B³-Prec':>8} {'B³-Rec':>8} {'B³-F1':>8} {'Macro-F1':>9} {'Δ':>7}"
        )
        output("-" * 80)

        for ablation in ablation_bcubed_orig:
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            bcubed_f1 = ablation.get("orig_avg_bcubed_f1", 0)
            bcubed_prec = ablation.get("orig_avg_bcubed_precision", 0)
            bcubed_rec = ablation.get("orig_avg_bcubed_recall", 0)
            macro_f1 = ablation.get("orig_avg_f1", 0)
            delta = bcubed_f1 - macro_f1

            output(
                f"{display_name:<30} {bcubed_prec:>8.4f} {bcubed_rec:>8.4f} {bcubed_f1:>8.4f} {macro_f1:>9.4f} {delta:>+7.4f}"
            )

        output()
        output("Legend: Δ = B³-F1 - Macro-F1 (positive = better on document-weighted metric)")
        output()

        # Learned rules agencies - B-Cubed ranking
        output("LEARNED RULES TRAINING AGENCIES - Ranked by B-Cubed F1:")
        output("-" * 80)

        ablation_bcubed_learned = [a for a in ablation_averages if a["num_learned_rules"] > 0]
        ablation_bcubed_learned.sort(key=lambda x: x.get("learned_avg_bcubed_f1", 0), reverse=True)

        output(
            f"{'Ablation':<30} {'B³-Prec':>8} {'B³-Rec':>8} {'B³-F1':>8} {'Macro-F1':>9} {'Δ':>7}"
        )
        output("-" * 80)

        for ablation in ablation_bcubed_learned:
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            bcubed_f1 = ablation.get("learned_avg_bcubed_f1", 0)
            bcubed_prec = ablation.get("learned_avg_bcubed_precision", 0)
            bcubed_rec = ablation.get("learned_avg_bcubed_recall", 0)
            macro_f1 = ablation.get("learned_avg_f1", 0)
            delta = bcubed_f1 - macro_f1

            output(
                f"{display_name:<30} {bcubed_prec:>8.4f} {bcubed_rec:>8.4f} {bcubed_f1:>8.4f} {macro_f1:>9.4f} {delta:>+7.4f}"
            )

        output()
        output("Legend: Δ = B³-F1 - Macro-F1 (positive = better on document-weighted metric)")
        output()

        # Holdout agencies - B-Cubed ranking
        output("HOLDOUT TEST AGENCIES - Ranked by B-Cubed F1:")
        output("-" * 80)

        ablation_bcubed_hold = [a for a in ablation_averages if a["num_holdout"] > 0]
        ablation_bcubed_hold.sort(key=lambda x: x.get("hold_avg_bcubed_f1", 0), reverse=True)

        output(
            f"{'Ablation':<30} {'B³-Prec':>8} {'B³-Rec':>8} {'B³-F1':>8} {'Macro-F1':>9} {'Δ':>7}"
        )
        output("-" * 80)

        for ablation in ablation_bcubed_hold:
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            bcubed_f1 = ablation.get("hold_avg_bcubed_f1", 0)
            bcubed_prec = ablation.get("hold_avg_bcubed_precision", 0)
            bcubed_rec = ablation.get("hold_avg_bcubed_recall", 0)
            macro_f1 = ablation.get("hold_avg_f1", 0)
            delta = bcubed_f1 - macro_f1

            output(
                f"{display_name:<30} {bcubed_prec:>8.4f} {bcubed_rec:>8.4f} {bcubed_f1:>8.4f} {macro_f1:>9.4f} {delta:>+7.4f}"
            )

        output()

        # ====================================================================
        # SECTION 6: ADJUSTED RAND INDEX (PAIR-BASED)
        # ====================================================================
        output()
        output("=" * 80)
        output("SECTION 6: ADJUSTED RAND INDEX (PAIR-BASED)")
        output("=" * 80)
        output()
        output("ARI penalizes both splits and merges harshly by comparing all document pairs.")
        output("Range: -1 to 1 (1 = perfect, 0 = random, <0 = worse than random)")
        output()
        output("If ARI is lower than F1, this suggests merge errors that best-match")
        output("strategy partially masks.")
        output()

        # Original agencies - ARI ranking
        output("ORIGINAL EVALUATION AGENCIES - Ranked by ARI:")
        output("-" * 80)

        ablation_ari_orig = [a for a in ablation_averages if a["num_original"] > 0]
        ablation_ari_orig.sort(
            key=lambda x: x.get("orig_avg_ari", -1) if x.get("orig_avg_ari") is not None else -1,
            reverse=True,
        )

        output(f"{'Ablation':<30} {'ARI':>8} {'Macro-F1':>9} {'Δ':>7}")
        output("-" * 80)

        for ablation in ablation_ari_orig:
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            ari = ablation.get("orig_avg_ari", 0)
            macro_f1 = ablation.get("orig_avg_f1", 0)
            delta = ari - macro_f1 if ari is not None else None

            if delta is not None:
                output(f"{display_name:<30} {ari:>8.4f} {macro_f1:>9.4f} {delta:>+7.4f}")
            else:
                output(f"{display_name:<30} {'N/A':>8} {macro_f1:>9.4f} {'N/A':>7}")

        output()
        output("Legend: Δ = ARI - Macro-F1 (negative Δ suggests merge problems)")
        output()

        # Learned rules agencies - ARI ranking
        output("LEARNED RULES TRAINING AGENCIES - Ranked by ARI:")
        output("-" * 80)

        ablation_ari_learned = [a for a in ablation_averages if a["num_learned_rules"] > 0]
        ablation_ari_learned.sort(
            key=lambda x: (
                x.get("learned_avg_ari", -1) if x.get("learned_avg_ari") is not None else -1
            ),
            reverse=True,
        )

        output(f"{'Ablation':<30} {'ARI':>8} {'Macro-F1':>9} {'Δ':>7}")
        output("-" * 80)

        for ablation in ablation_ari_learned:
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            ari = ablation.get("learned_avg_ari", 0)
            macro_f1 = ablation.get("learned_avg_f1", 0)
            delta = ari - macro_f1 if ari is not None else None

            if delta is not None:
                output(f"{display_name:<30} {ari:>8.4f} {macro_f1:>9.4f} {delta:>+7.4f}")
            else:
                output(f"{display_name:<30} {'N/A':>8} {macro_f1:>9.4f} {'N/A':>7}")

        output()
        output("Legend: Δ = ARI - Macro-F1 (negative Δ suggests merge problems)")
        output()

        # Holdout agencies - ARI ranking
        output("HOLDOUT TEST AGENCIES - Ranked by ARI:")
        output("-" * 80)

        ablation_ari_hold = [a for a in ablation_averages if a["num_holdout"] > 0]
        ablation_ari_hold.sort(
            key=lambda x: x.get("hold_avg_ari", -1) if x.get("hold_avg_ari") is not None else -1,
            reverse=True,
        )

        output(f"{'Ablation':<30} {'ARI':>8} {'Macro-F1':>9} {'Δ':>7}")
        output("-" * 80)

        for ablation in ablation_ari_hold:
            method_name = ablation["method_name"]
            friendly_name = get_display_name(method_name)
            display_name = friendly_name[:27] + "..." if len(friendly_name) > 30 else friendly_name

            ari = ablation.get("hold_avg_ari", 0)
            macro_f1 = ablation.get("hold_avg_f1", 0)
            delta = ari - macro_f1 if ari is not None else None

            if delta is not None:
                output(f"{display_name:<30} {ari:>8.4f} {macro_f1:>9.4f} {delta:>+7.4f}")
            else:
                output(f"{display_name:<30} {'N/A':>8} {macro_f1:>9.4f} {'N/A':>7}")

        output()

        # Add CSV export documentation
        output()
        output("=" * 80)
        output("EXPORTED CSV FILES")
        output("=" * 80)
        output()
        output("The following CSV files have been exported to reports/ablations/:")
        output()
        output("Macro-Averaged F1 Comparison Tables (detailed, with all agency scores):")
        output("  - original_f1_comparison.csv")
        output("  - learned_rules_f1_comparison.csv")
        output("  - holdout_f1_comparison.csv")
        output()
        output("Macro-Averaged F1 Summary Tables (compact, mean scores only):")
        output("  - original_f1_means.csv")
        output("  - learned_rules_f1_means.csv")
        output("  - holdout_f1_means.csv")
        output()
        output("B-Cubed F1 Comparison Tables (document-weighted):")
        output("  - original_bcubed_f1_comparison.csv")
        output("  - holdout_bcubed_f1_comparison.csv")
        output()
        output("ARI Comparison Tables (pair-based):")
        output("  - original_ari_comparison.csv")
        output("  - holdout_ari_comparison.csv")
        output()
        output("Statistics Tables (mean, median, std, variance, min, max, IQR):")
        output("  - original_statistics.csv")
        output("  - learned_rules_statistics.csv")
        output("  - holdout_statistics.csv")
        output()
        output("=" * 80)
        output()

    print(f"\nCross-agency summary saved to: {output_path}")


def main():
    """Evaluate all ablation clustering results for all agencies."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY EVALUATION - MULTI-AGENCY (V2 + ML)")
    print("=" * 80)
    print(f"V2 Ablation output directory: {ABLATION_OUTPUT_DIR_V2}")
    print(f"ML Ablation output directory: {ABLATION_OUTPUT_DIR_ML}")
    print(f"Reports directory: {REPORTS_DIR}")
    print()

    # Discover agencies from ML directory and V2/embeddings directory (union)
    ml_agencies = set(discover_agencies(ABLATION_OUTPUT_DIR_ML))
    v2_agencies = set(discover_agencies(ABLATION_OUTPUT_DIR_V2))
    agencies = sorted(ml_agencies | v2_agencies)

    if not agencies:
        print("No agency directories found in ML or V2 output directories.")
        return

    print(
        f"Found {len(agencies)} agencies total ({len(ml_agencies)} with ML results, {len(v2_agencies)} with V2/embeddings results):"
    )
    for agency in agencies:
        tags = []
        if agency in ml_agencies:
            tags.append("ML")
        if agency in v2_agencies:
            tags.append("V2/embeddings")
        print(f"  - {agency} [{', '.join(tags)}]")
    print()

    # Filter out excluded agencies
    if EXCLUDED_AGENCIES is not None:
        excluded = [a for a in agencies if a in EXCLUDED_AGENCIES]
        agencies = [a for a in agencies if a not in EXCLUDED_AGENCIES]
        if excluded:
            print(f"Excluding {len(excluded)} agencies (EXCLUDED_AGENCIES):")
            for agency in excluded:
                print(f"  - {agency}")
            print()

    # Ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Track results for cross-agency summary
    all_agency_results = {}  # agency_name -> list of {method_name, description, metrics}

    # Evaluate each agency
    for agency_idx, agency_name in enumerate(agencies, 1):
        print("\n" + "=" * 80)
        print(f"AGENCY {agency_idx}/{len(agencies)}: {agency_name}")
        print("=" * 80)

        # Discover ML ablation results for this agency
        ml_configs = discover_ablation_results_for_agency(
            agency_name, ABLATION_OUTPUT_DIR_ML, ABLATION_CONFIG_PATH_ML, version="ml"
        )

        if ml_configs:
            print(f"Found {len(ml_configs)} ML ablation result(s) for {agency_name}")
        else:
            print(f"No ML ablation results found for {agency_name}")

        # Discover V2 ablation results for this agency (if available)
        v2_configs = discover_ablation_results_for_agency(
            agency_name, ABLATION_OUTPUT_DIR_V2, ABLATION_CONFIG_PATH_V2, version="v2"
        )

        if v2_configs:
            print(f"Found {len(v2_configs)} V2 ablation result(s) for {agency_name}")
        else:
            print(f"No V2 ablation results found for {agency_name}")

        # Discover embeddings ablation results for this agency (if available)
        embeddings_configs = discover_ablation_results_for_agency(
            agency_name,
            ABLATION_OUTPUT_DIR_V2,
            ABLATION_CONFIG_PATH_V2,  # config path unused for embeddings version
            version="embeddings",
        )

        if embeddings_configs:
            print(
                f"Found {len(embeddings_configs)} embeddings ablation result(s) for {agency_name}"
            )
        else:
            print(f"No embeddings ablation results found for {agency_name}")

        # Combine configs
        agency_configs = ml_configs + v2_configs + embeddings_configs

        # Skip agencies with no results at all
        if not agency_configs:
            print(f"No ablation results found for {agency_name}, skipping...\n")
            continue

        print(f"Total: {len(agency_configs)} ablation result(s) for {agency_name}\n")

        agency_metrics = []

        for config in agency_configs:
            try:
                result = evaluate_single_result(config)
                if result:
                    agency_metrics.append(
                        {
                            "method_name": config["method_name"],
                            "description": config.get("description", "No description"),
                            "metrics": result,
                        }
                    )
            except FileNotFoundError as e:
                print(f"WARNING: File not found, skipping - {e}")
                continue
            except Exception as e:
                print(f"WARNING: Error evaluating, skipping - {e}")
                import traceback

                traceback.print_exc()
                continue

        # Generate agency-specific comparison report
        if agency_metrics:
            print(f"\nGenerating comparison report for {agency_name}...\n")
            generate_comparison_report(agency_metrics, agency_name)
            all_agency_results[agency_name] = agency_metrics
        else:
            print(f"No valid results to compare for {agency_name}.")

    print("\n" + "=" * 80)
    print("AGENCY-LEVEL EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Agencies evaluated: {len(all_agency_results)}/{len(agencies)}")
    print()

    # Generate cross-agency summary
    if all_agency_results:
        print("Generating cross-agency summary report...\n")
        generate_cross_agency_summary(all_agency_results)
        print()
    else:
        print("No results to summarize across agencies.")

    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nReports saved to: {REPORTS_DIR}/")
    print()
    print("Report structure:")
    print(f"  - Per-agency reports: {REPORTS_DIR}/<agency_name>/")
    print(f"  - Cross-agency summary: {REPORTS_DIR}/cross_agency_summary.txt")
    print()


if __name__ == "__main__":
    main()
