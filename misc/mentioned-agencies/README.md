# mentioned-agencies (download + post-process notebooks)

Auxiliary notebooks for the mentioned-agencies workflow. **The actual
extraction pipeline lives at
[`packages/mentioned-agencies/`](../../packages/mentioned-agencies/)** —
these notebooks are the upstream download step and the downstream
CSV-merge / EDA step that wrap around it.

## Contents

- `download/src/src.ipynb` — downloads source case-file bundles
  (originally used a Google service-account token; the `token.json` file
  was **not** copied — supply your own OAuth credentials to re-run).
- `post-process/src/src.ipynb` — merges per-case `mentioned_agencies.csv`
  outputs (the extraction step itself runs out of
  `packages/mentioned-agencies/`).
- `post-process/src/eda.ipynb` — exploratory analysis of the merged
  results.

All three notebooks had their executed outputs cleared
(`jupyter nbconvert --clear-output --inplace`) before commit in case any
outputs contained PII or internal-only data.

## Maturity / scope

Exploratory notebooks; expect rough edges. No automated runner, no tests,
no pipeline contract. Renamed from the original `mentioned_agencies/`
(underscore) to `mentioned-agencies/` (hyphen) for repo-wide kebab-case
consistency.

## Required environment

Standard Jupyter + `pandas`. The download notebook originally used Google
Drive APIs.

## How to view

```bash
jupyter lab download/src/src.ipynb
jupyter lab post-process/src/src.ipynb
jupyter lab post-process/src/eda.ipynb
```

## Status

This is **not** a package — no `packages/` entry, no `prap-<name>` CLI,
no `prap_core` deps, no schemas. Standalone notebooks kept here for
review. For the real extraction pipeline (per-page LLM extraction,
case-level validation, fuzzy dedup) see `packages/mentioned-agencies/`.
