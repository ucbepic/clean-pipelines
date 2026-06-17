# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PRAP (Police Records Access Project) — modular, MIT-licensed pipelines that extract structured fields from police records using LLMs. Pre-release scaffolding (v0.1.0a0).

## Workspace layout

This is a **`uv` workspace**. The root `pyproject.toml` declares `[tool.uv.workspace] members = ["packages/*"]`; every pipeline is its own installable package under `packages/`, all depending on a shared `prap-core`.

```
packages/
├── core/                       # prap-core: shared library (no domain types)
├── incident-date/              # reference pipeline — copy this layout for new ones
├── case-type/
├── involved-agency/
├── location/
├── mentioned-agencies/
├── split-officer-names/
├── page-stream-segmentation/
├── clustering/
└── redactions/                 # WIP, currently exits non-zero
misc/                           # non-§4 code (CV classifiers, AWS-only utils, notebooks)
```

`prap-core` is **domain-free**: no `Officer`, no `IncidentDate`. Modules: `llm` (LiteLLM wrapper + retries + structured output + on-disk cache), `ocr` (`OCREngine` protocol + Azure/Tesseract/Unstructured adapters), `pdf`, `io` (jsonl streaming, manifests), `config` (pydantic-settings), `prompts` (versioned loader), `eval` (P/R/F1), `summary_filter`. Pipeline packages must not import provider SDKs directly — go through `prap_core.config.Settings` / `prap_core.llm.LLM`.

Each pipeline package follows the `incident-date` layout: `cli.py` (Typer `app`), `pipeline.py` (`run()` + per-case worker), `schemas.py` (pydantic), `prompts/` (txt files, force-included in the wheel via hatch), `data_loading.py`, `evaluation.py`, plus a Typer console-script entry point named `prap-<pipeline>` (e.g. `prap-incident-date = "prap_incident_date.cli:app"`).

The standard pipeline CLI verbs are `prepare` (parquet/csv → jsonl), `run` (LLM extraction → jsonl), `eval` (score against GT). Eval inputs/GT live in a separate private repo; this repo ships only the code.

## Commands

```bash
uv sync                                       # install full workspace
uv sync --all-packages                        # CI variant
uv run pytest                                 # all package test suites (per [tool.pytest.ini_options].testpaths)
uv run pytest packages/core                   # one package
uv run pytest packages/core/tests/test_llm.py::test_embed   # single test
uv run ruff check .                           # lint (CI gate)
uv run ruff format --check .                  # format (CI gate)
uv run ruff format .                          # apply formatting
uv run prap-incident-date run --input cases.jsonl --output results.jsonl   # any pipeline CLI works the same way
```

CI (`.github/workflows/test.yml`) runs ruff check, ruff format --check, and `pytest packages/core` only — full cross-package pytest is run locally.

## Configuration

All LLM/OCR/cache config flows through env vars (see `.env.example`), not code. Key vars: `PRAP_LLM_MODEL` (LiteLLM-style: `openai/...`, `anthropic/...`, `azure/...`, `ollama/...`), `PRAP_LLM_API_KEY`, `PRAP_LLM_API_BASE`, `PRAP_LLM_API_VERSION`, `PRAP_OCR_BACKEND` (`unstructured` | `tesseract`), `PRAP_CACHE_DIR` (default `~/.cache/prap`, lives outside repo). Pipelines are model-agnostic — never hardcode a provider.

## Conventions

- **Some "bugs" are load-bearing.** The `incident-date` verify prompt deliberately leaves a `$initial_dates` placeholder un-substituted (see the NOTE in `pipeline.py`); changing this changes model output and therefore eval numbers. Read the surrounding NOTE before "fixing" anything that looks anomalous.
- **Lint exemptions** in `pyproject.toml [tool.ruff.lint.per-file-ignores]` are intentional, especially the broad `clustering/**` carve-outs that allow bare `except:`, long lines, and `B007`. Don't tighten them.
- **`B008` is allowed in `**/cli.py`** because Typer requires `Option()` in default args.
- Three pipelines have no ground truth (`split-officer-names`, `mentioned-agencies`, `redactions`) and use a "three-check sanity gate" (schema validation, distribution sanity, tiny example fixture) instead of numeric eval.
- `redactions` currently depends on Azure Content Safety; CLI prints a notice and exits non-zero until an open-source classifier replaces it.
- New pipelines: copy `packages/incident-date/` as the template.
