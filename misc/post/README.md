# post

Rule-based **post-processing**, **record-linkage**, and **time-series
feature** utilities that sit downstream of the LLM extraction pipelines.
None of this is LLM-driven — it's `pandas` + `scikit-learn` + `xgboost`
plumbing.

## Subdirectories

- `clean/` — cleans extracted incident/officer CSV+JSON into normalized
  `df_post.csv` / `df_cpdp.csv`.
- `train-data/` — joins incident data against POST roster data using
  Jaro–Winkler and sentence-transformer embeddings to produce training
  labels (`entity_matches.xlsx`, `training_data.csv`).
- `ts-blocking/` — generates blocking keys (configurable prefix/suffix
  length) over the labeled record pairs.
- `ts-features/` — turns blocked record pairs into a tabular feature
  matrix.
- `ts-train/` — trains an XGBoost / Random Forest / Logistic Regression
  classifier over those features.
- `eda/` — one exploratory notebook (`eda.ipynb`, outputs cleared).
- `agencies/` — `Untitled.csv`, a reference mapping of agency-name spellings
  across drive / CRP-PRA datasets to a canonical name + agency type + county.
- `link/` — placeholder dir; no source code (the original tree had only an
  input CSV under `data/`, which is not shipped).

## Maturity / scope

Rule-based post-processing utilities; **not §4 pipeline-shaped** (no
`prap_core`, no `prap-<name>` CLIs, no JSONL I/O). Each subdir has its own
Makefile and `src/src.py` with `argparse` flags — runnable standalone, but
each step expects the previous step's output CSV on disk.

## Required environment

Python deps gathered from imports: `pandas`, `numpy`, `scikit-learn`,
`xgboost`, `sentence-transformers`, `jellyfish`, `openpyxl`.

## How to run

Each subdir has its own `make` target. Typical chain:

```bash
cd clean       && make all
cd ../train-data && make all
cd ../ts-blocking && make all
cd ../ts-features && make all
cd ../ts-train   && make all
```

Default input/output paths follow the relative-path layout in each
Makefile and assume `data/input/` / `data/output/` next to the script.
The `data/` directories are **not** shipped — you'll need to supply inputs
from upstream stages.

## Status

This is **not** a package — no `packages/` entry, no `prap-<name>` CLIs,
no `prap_core` deps. Standalone code kept here for review.
