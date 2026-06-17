# sample

Notebook-only **sampling / EDA helper** used to choose review subsets from
autofolio-output CSVs (one CSV per agency).

## Contents

- `src/eda.ipynb` — exploratory analysis of the autofolio outputs.
- `src/concat.ipynb` — concatenation / sampling helper across the
  per-agency CSVs.
- `src/methodology.md` — short methodology note from the original author.

Both notebooks had their executed outputs cleared
(`jupyter nbconvert --clear-output --inplace`) before commit, in case any
outputs contained PII or internal-only data.

## Maturity / scope

Exploratory notebooks; expect rough edges. No automated runner, no tests,
no pipeline contract.

## Required environment

Standard Jupyter + `pandas`. Input data (the per-agency autofolio CSVs)
lived in `data/input/` in the original tree; that `data/` directory is
**not** shipped — provide your own inputs to re-run the notebooks.

## How to view

```bash
jupyter lab src/eda.ipynb
jupyter lab src/concat.ipynb
```

## Status

This is **not** a package — no `packages/` entry, no `prap-<name>` CLI, no
`prap_core` deps, no schemas. Standalone notebooks kept here for review.
