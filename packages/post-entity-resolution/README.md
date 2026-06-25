# prap-post-entity-resolution

Officer entity-resolution: resolve LLM-extracted officer mentions to POST
employment records via a trained XGBoost model, plus the offline pipeline
that trains that model.

> **Status:** a **full `uv` workspace member** (in root `members` /
> `dependencies` / `[tool.uv.sources]` / `testpaths`), lint- and
> format-clean (`ruff check .` / `ruff format --check .` report zero hits
> for this package; legacy `training/**` + prompt-text lines are carved
> out in `[tool.ruff.lint.per-file-ignores]`, mirroring `clustering/**`).
>
> **`resolve/` (inference) follows the §4 contract:**
> - Imports as `prap_post_entity_resolution.resolve.*` (no `PYTHONPATH`
>   hack); domain types in `schemas.py`; agency-validation LLM through
>   `prap_core.llm.LLM` (model-agnostic).
> - Typer CLI `prap-post-entity-resolution` with `prepare` / `run` /
>   `eval`; jsonl I/O via `prap_core.io`; `eval` scores link-level
>   precision/recall/F1 via `prap_core.eval` against a GT table.
> - The trained model + scaler + `common_last_names.csv` ship in the
>   wheel (`force-include`); `.env` does not.
> - Tested by `tests/` (9 offline tests, in `testpaths`) and verified
>   behavior-identical (4 auto-matched) against the live API.
>
> **`training/`** stays the offline Makefile chain (model-generation
> tool, not a §4 jsonl pipeline) — the two known bugs are fixed
> (`ts-train` now honors `--model-type` and creates its `--output-dir`).

## Layout

```
src/prap_post_entity_resolution/
├── training/            # OFFLINE: trains the XGBoost model (Makefile chain)
│   ├── clean/           #   preprocessing: raw POST + incident dumps -> df_post / df_cpdp
│   ├── train-data/      #   labeled-pair generation -> training_data.csv / labeled_data
│   ├── ts-blocking/     #   blocking on name prefix/suffix
│   ├── ts-features/     #   feature engineering -> features.csv
│   └── ts-train/        #   model training -> best_model_xgboost.pkl
├── resolve/             # INFERENCE: the production entity-resolution pipeline
│   ├── pipeline.py      #   PostMatcher orchestrator (stage 0-4)
│   ├── candidates.py scoring.py features.py validation.py agency.py
│   ├── client.py        #   NPIClient — HTTP client for the NPI employment API
│   ├── io.py cli.py llm.py explain.py
│   └── models/          #   trained artifacts (best_model_xgboost.pkl, features_scaler.pkl)
└── shared/              #   models.py/env.py dep (copied from npi-api/api/shared);
                         #   sibling of resolve/ so `from shared.X` resolves
data/
└── fixtures/            # small test CSVs (subsets of real inputs) for e2e checks
    ├── train_sample.csv     # 40-row balanced subset of labeled_data_10_1_2025.csv
    └── resolve_sample.csv   # 12-row subset of involved_officers.csv
```

## Provenance (where the code came from)

| Staged path | Source |
|---|---|
| `training/{clean,train-data,ts-blocking,ts-features,ts-train}/` | `bids-experimental/post/<same>/` (src + Makefile only) |
| `resolve/*.py`, `resolve/models/`, `resolve/data/input/common_last_names.csv` | `npi-api/api/resolve/` |
| `shared/` (sibling of `resolve/`) | `npi-api/api/shared/` (imported by resolve as `shared.models` / `shared.env`) |
| `training/train-data/data/output/labeled_data_10_1_2025.csv` | source for `train_sample.csv` |

**Deliberately NOT copied:** the large data dirs — `clean`'s 66 MB raw
inputs (`california-processed.csv`, fresno JSON), resolve's `df_lapd.csv`
(75 MB), `post_export.csv` (65 MB), `involved_officers.csv` (4.5 MB), and
all run outputs. Only small reference/fixture CSVs were brought over.

## Testing end-to-end (with the small fixtures)

- **Training (model-training core):** `train_sample.csv` →
  `ts-blocking` → `ts-features` → `ts-train`. **Verified to run
  end-to-end on the fixture** (blocking → features → train produces a
  model `.pkl`). The upstream `clean` / `train-data` stages need the
  66 MB raw POST dump and are not fixture-testable. Example:

  ```bash
  cd src/prap_post_entity_resolution/training
  uv run --no-project --with pandas python ts-blocking/src/src.py \
      --input ../../../../data/fixtures/train_sample.csv \
      --output /tmp/blocks.csv --prefix-len 1 --suffix-len 2
  uv run --no-project --with pandas --with numpy --with scikit-learn \
      --with jellyfish --with sentence-transformers \
      python ts-features/src/src.py --input /tmp/blocks.csv --output /tmp/features.csv
  mkdir -p /tmp/model   # NOTE: ts-train does not create --output-dir itself (refactor TODO)
  uv run --no-project --with pandas --with scikit-learn --with xgboost \
      python ts-train/src/src.py --input /tmp/features.csv --output-dir /tmp/model --test-size 0.2
  ```

  Caveats found (left verbatim, fix in refactor): `ts-train` ignores
  `--model-type` (always trains xgboost+rf+logistic and keeps the most
  accurate) and does not `mkdir` its `--output-dir`.
- **Resolve:** `resolve_sample.csv` via `resolve.cli from-csv
  --default-state CA`. Hits a live NPI API (`NPI_API_URL`); the
  `NPIClient`/scorer/validator are all injectable for offline tests.
  **Verified end-to-end against the live API** (candidate fetch →
  XGBoost scoring → OpenAI agency validation): the 12-row fixture
  resolves to **4 auto-matched, 8 routed to review** (auto-matches are
  same-agency / high-probability / agency-validated; reviews are the
  common-name and ambiguous gates firing correctly).

  The NPI API is **not** vendored here (it serves a Supabase-backed
  `all_npi_states` table — there's no local data to copy). Start it from
  its own repo and point the resolver at it:

  ```bash
  # in the npi-api repo: its venv has fastapi/uvicorn/supabase, and the
  # API's load_env() auto-loads SUPABASE_KEY from archive/legacy_ca/server/.env
  cd ~/Desktop/npi-api/api && venv/bin/python run.py api   # serves :8001

  # then here (run dir must be on PYTHONPATH so `resolve` + `shared` resolve):
  cd src/prap_post_entity_resolution
  export NPI_API_URL=http://localhost:8001
  export OPENAI_API_KEY=...    # for the agency-validation LLM stage
  PYTHONPATH="$PWD" python -m resolve.cli from-csv \
      --input ../../data/fixtures/resolve_sample.csv --default-state CA \
      --output-dir /tmp/resolve_out
  ```

  Gotchas hit while wiring this up (fix during refactor):
  - `shared/` was moved to sit **beside** `resolve/` (was nested under it)
    so the verbatim `from shared.models import ...` imports resolve. The
    run dir must be on `PYTHONPATH`.
  - `resolve`'s own `load_env()` looks for `.env`s relative to the old
    npi-api layout, so it does **not** find this package's
    `resolve/.env`; pass `OPENAI_API_KEY` via the environment for now.
    (Quote the key or not — `python-dotenv` strips quotes, but a raw
    shell `export` does not.)
  - The pickled model + scaler emit version warnings (saved under older
    xgboost / scikit-learn). Harmless here, but re-save them with pinned
    versions during the refactor.

## Refactor TODO

Done (verified by `tests/` + an unchanged 4-auto-matched live run):

- [x] Replaced `resolve/llm.py` (direct OpenAI) + `shared/env.py` with
      `prap_core.llm.LLM` for agency validation (model-agnostic via
      `PRAP_LLM_*`). `llm.py` and `shared/` deleted.
- [x] Re-homed the `shared/` models into `schemas.py` (pydantic); resolve
      imports are package-relative — imports as
      `prap_post_entity_resolution.resolve.*`, **no `PYTHONPATH` hack**.
- [x] Typer `app` (`prepare` / `run` / `eval`) + `prap-post-entity-resolution`
      console script, replacing the argparse CLI.
- [x] jsonl I/O via `prap_core.io` (`prepare`: CSV→mentions jsonl;
      `run`: mentions jsonl→results jsonl).
- [x] `eval` verb: link-level P/R/F1 via `prap_core.eval.prf`.
- [x] Registered as a full workspace member (root `members` /
      `dependencies` / `[tool.uv.sources]` / `testpaths`); lint+format clean.
- [x] Re-saved `models/*.pkl` under the pinned xgboost/sklearn — version
      warnings gone, predictions bit-identical.
- [x] Model + scaler + `common_last_names.csv` `force-include`d in the
      wheel (`.env` excluded).
- [x] `ts-train` honors `--model-type` and creates its `--output-dir`.

Remaining / by design:

- **`training/`** stays a Makefile-driven offline model-generation tool
  (not converted to jsonl/Typer) — its `clean` / `train-data` stages need
  the 66 MB raw POST dump and aren't fixture-testable.
- **`eval`** is implemented and unit-tested on synthetic GT, but a real
  pipeline-level GT table (`officer_uid, post_person_nbr`) isn't staged
  here — wire one in to produce real numbers.
- **`resolve`'s LLM `.env`**: config now flows through `prap_core`
  (`PRAP_LLM_*`); set those env vars (the old `load_env()`/`resolve/.env`
  path is gone).
