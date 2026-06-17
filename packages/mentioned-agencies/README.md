# prap-mentioned-agencies

Extract law-enforcement agencies *mentioned* in police case documents
(distinct from the involved/investigating agency, which lives in
[`prap-involved-agency`](../involved-agency/)).

## How it works

Three stages per case:

1. **Per-page extraction** — for every page with OCR text, ask the LLM
   for a JSON array of agency names mentioned on that page
   (`prompts/extract_per_page.txt`). The prompt includes a near-complete
   list of California correctional agencies as a recognition aid.
2. **Case-level validation** — pool the per-page hits and ask the LLM to
   drop non-LEAs, complete abbreviations, and expand acronyms against a
   sample of the case text (`prompts/validate_agencies.txt`).
3. **Fuzzy deduplication** — `fuzzywuzzy` ratio at threshold 85,
   preferring the longer/more-complete name.

Structural refactor of `cpost/extract/src.py`. Algorithms, prompts, and
extraction order are preserved. Two intentional divergences:
- Per-call `time.sleep(1.0)` dropped (relies on `prap_core.llm`
  retry/backoff).
- `multiprocessing.Pool` → `ThreadPoolExecutor` (matches other prap
  packages and works with the shared `LLM` client).

## Input

Either:
- A directory of `agency_case_file_bundle-*.json` files (cpost format —
  top-level `HIDDEN_provisional_case_name`, `case_files: [{file_name,
  ocr_doc_text_per_page: {page_texts: [{page_number, text}]}}]`), **or**
- A jsonl where each line is a flattened `CaseBundle` (see `schemas.py`).

## Output

Jsonl, one `MentionedAgenciesResult` per case:

```json
{
  "provisional_case_name": "...",
  "mentioned_agencies": ["Fresno Police Department", "California Highway Patrol"],
  "n_pages_processed": 14,
  "n_raw_extractions": 22,
  "n_after_validation": 4,
  "validation_confidence": "high",
  "error": null
}
```

## Usage

```bash
prap-mentioned-agencies run \
  --input  /path/to/agency_case_file_bundles_dir \
  --output /tmp/prap-runs/mentioned-agencies.jsonl \
  --n-threads 16
```

LLM is configured via `PRAP_LLM_*` env vars (see `prap_core.config`).
Default model = `openai/gpt-4o-mini`.

## Evaluation

No ground-truth set in v1. Sanity gate is the §1 three-check pass:
schema validation, distribution sanity (fraction of cases with ≥1 agency,
top-K agencies), tiny example fixture in `examples/` (TODO).

## Not in scope

- Resolving mentioned agencies to canonical IDs (the cpost workflow does
  this downstream by joining against an agency master list).
- The Google Drive download step and the CSV merge — those live in
  `misc/mentioned-agencies/` as notebooks.
