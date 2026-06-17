# prap-core

Shared, domain-free library for PRAP pipelines.

Modules (planned, per `plans/2026-05-11-prap-refactor-plan.md` §3):

- `llm` — LiteLLM wrapper, retries, structured output, disk cache, accounting
- `ocr` — `OCREngine` protocol + Azure / Tesseract / Unstructured adapters
- `pdf` — splitting, page utilities
- `io` — path helpers, jsonl streaming, manifests
- `config` — pydantic-settings (env + `.env`)
- `prompts` — versioned prompt loader
- `eval` — precision / recall / F1 primitives

This package contains no domain types (no `Officer`, no `IncidentDate`); those
live in individual pipeline packages.
