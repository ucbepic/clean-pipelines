# prap-clustering

PRAP pipeline that clusters police-accountability files (PDFs, images, audio, video) into incidents. Inputs are CSVs of pre-OCR'd documents with metadata; outputs are CSVs with a `Parent Clusters` column assigning each file to an incident-level cluster.

This package is a structural port of the legacy `clustering-pipeline/` tree. It bundles three sub-pipelines:

- **hybrid** (`prap_clustering.clustering`) — three-tier cascade combining regex, structured-feature rules, and pairwise LLM comparison. This is the production pipeline.
- **embeddings** (`prap_clustering.embeddings_pipeline`) — sentence-transformers (`all-MiniLM-L6-v2`) embedding-based clustering baseline.
- **metadata** (`prap_clustering.metadata_pipeline`) — regex-only clustering baseline (no LLM, no embeddings).

Feature extraction modules (`prap_clustering.feature_extraction`) are shared across pipelines.

## Three-tier extraction strategy (hybrid pipeline)

The hybrid pipeline constructs a file graph (nodes = files, edges = same-incident matches). Connected components form candidate clusters. Each tier either matches a pair, hard-blocks it, or passes it to the next tier.

1. **Tier 1 — Filepath + filename regex.** Cheap, deterministic. Matches on case-ID overlap or shared deep directory paths; hard-blocks on conflicting case IDs.
2. **Tier 2 — LLM-extracted feature rules.** Structured fields (case IDs, dates, subject/officer names) parsed from per-PDF summaries. Requires two corroborating signals to match. Embedding cosine similarity (threshold 0.9, from local `sentence-transformers/all-MiniLM-L6-v2`) hard-blocks dissimilar pairs, filtering ~97% of candidates before Tier 3.
3. **Tier 3 — Pairwise semantic comparison.** LLM directly compares concatenated feature summaries for surviving pairs.

A cluster-refinement step then prunes weakly-connected nodes (each node must connect to ≥30 % of its cluster).

## Installation

```bash
uv sync   # from the prap/ workspace root
```

## CLI

```bash
# === Hybrid pipeline (production) ===

# 1. Extract features (regex + LLM) from a raw OCR CSV.
prap-clustering extract-features --input data/raw.csv --output data/features.csv

# 2a. Hand-engineered hybrid cascade against an ablation config.
prap-clustering cluster-handengineered \
    --input data/features.csv \
    --output-dir data/output/ablations \
    --ablation baseline_v2

# 2b. ML-based hybrid (joint or cascade learned models).
prap-clustering cluster-ml \
    --input data/features.csv \
    --output-dir data/output/ablations_ml \
    --ablation rf_both

# === Embeddings baseline ===

# Generate embeddings for the multi-agency AGENCIES list inside the script.
prap-clustering embeddings-features
# Run embedding-based clustering over the generated embeddings.
prap-clustering embeddings-cluster

# === Metadata-only baseline ===

# Extract regex-only features.
prap-clustering metadata-features
# Run metadata-only clustering (graph variant by default; --deterministic for
# the deterministic variant).
prap-clustering metadata-cluster
prap-clustering metadata-cluster --deterministic

# === Evaluation ===

# Single-CSV scoring: returns a ScoreReport JSON.
prap-clustering eval --results data/output/ablations/agency/ablation.csv

# Multi-agency batch evaluation (produces per-agency and cross-agency reports).
prap-clustering eval-batch \
    --results-dir data/output/ablations \
    --ml-results-dir data/output/ablations_ml \
    --reports-dir reports/ablations
```

## Configuration

LLM is configured via `prap-core` standard env vars:
- `PRAP_LLM_MODEL` — model name (e.g. `azure/gpt-4.1-mini-2025-04-14`)
- `PRAP_LLM_API_KEY`, `PRAP_LLM_API_BASE`, `PRAP_LLM_API_VERSION`

Clustering-specific env vars (used by `prap_clustering.evaluation` and feature extraction):
- `PRAP_CLUSTERING_ABLATIONS_V2_DIR` — directory of hand-engineered ablation outputs
- `PRAP_CLUSTERING_ABLATIONS_ML_DIR` — directory of ML ablation outputs
- `PRAP_CLUSTERING_REPORTS_DIR` — output directory for eval reports
- `PRAP_CLUSTERING_ABLATION_CONFIG_V2`, `PRAP_CLUSTERING_ABLATION_CONFIG_ML` — override the bundled YAML configs
- `PRAP_CLUSTERING_FEATURES_DIR` — default output directory for `extract-features` when `--output` is omitted
- `PRAP_CLUSTERING_COSTS_DIR` — output directory for the cost-estimation flow

## Ablation configs

Two YAML files ship in `src/prap_clustering/configs/`:
- `ablation_configs_handengineered.yaml` — hand-coded matching rules (Tiers 1/2/3, ablations over rule subsets and date windows)
- `ablation_configs_ml.yaml` — learned matching rules (decision tree, random forest, LightGBM; joint and cascade configurations)

Pass the desired config name to `cluster-handengineered --ablation <name>` or `cluster-ml --ablation <name>`. The full list of configs lives in those YAML files.

## Eval baselines (hybrid pipeline)

From the parent README (31 California agencies, 4,937 labeled cases, macro-averaged per-case metrics, 95% bootstrap CIs over 2,000 agency-level resamples):

| Configuration | Prec. | Rec. | F1 | 95% CI |
|---|---:|---:|---:|:---:|
| All tiers + cluster refinement | **0.92** | **0.76** | **0.76** | 0.71–0.81 |
| All tiers, no refinement | 0.86 | 0.80 | 0.75 | 0.69–0.80 |
| Tiers 1 + 2 only (no T3) | 0.92 | 0.74 | 0.74 | 0.70–0.80 |
| Tiers 2 + 3 only (no regex) | 0.94 | 0.71 | 0.73 | 0.67–0.79 |
| Tier 2 only | 0.94 | 0.68 | 0.71 | 0.65–0.78 |
| Tier 1 only | 0.96 | 0.63 | 0.67 | 0.60–0.74 |
| RF (T1+T2 features) | 0.88 | 0.73 | 0.71 | 0.66–0.77 |
| Embedding baseline (τ=0.85) | 0.64 | 0.63 | 0.43 | 0.35–0.52 |

Cost (legacy estimates):
- Feature extraction: ~$0.05/PDF (21 LLM calls at GPT-4.1-mini rates)
- Clustering Tier 3: most agencies < $1 (embedding pre-filter removes ~97% of candidates)
- Full corpus (~60k files): ~$4,200

## Behavioral divergences from `clustering-pipeline/`

- **Sync LLM.** The legacy `AsyncAzureOpenAI` client was replaced by `prap_core.llm.LLM` (sync). All `await prompt_gpt_async(...)` call sites in `hybrid_handengineered.py` are routed through `asyncio.to_thread(...)`, preserving the async surface but losing true async I/O.
- **Local sentence-transformers embeddings stay local.** `embed_texts` / `cosine_similarity` used for summary-similarity scoring inside the hybrid pipeline remain backed by `sentence-transformers/all-MiniLM-L6-v2` via `prap_clustering.embeddings` (loaded into-process, no API call). The name-similarity embeddings in `embeddings_pipeline/cluster/cluster.py` and `metadata_pipeline/clustering/cluster.py` go through `prap_core.llm.LLM.embed(...)` (which routes to `litellm.embedding` and honors `PRAP_EMBEDDING_MODEL` — defaults to `openai/text-embedding-3-large`).
- **Feature-extraction prompts kept as `.py` modules.** `feature_extraction/prompts/{dates,case_ids,officer_names,subject_names,structured}.py` retain their nested-dict structure (`PROMPTS = {"summarization": {...}, "extraction": {...}, "citations": {...}}`) as meaningful import surface, rather than being extracted into individual `.txt` files. The single inline triple-quoted prompt in `clustering/prompts.py` is extracted to `summary_comparison.txt` (matching the existing precedent).
- **CLI replaces hardcoded paths.** `CSV_PATH` / `OUTPUT_PATH` constants from the legacy `cluster.py` scripts are no longer dead globals — the CLI provides them, and dead `if __name__ == "__main__":` blocks that referenced undefined globals were removed.
- **`evaluation.py` is dual-mode.** A new `score(results_csv, ground_truth_col)` function returns a `ScoreReport` pydantic model for single-CSV evaluation; the legacy multi-agency `main()` is preserved behind `eval-batch`.
