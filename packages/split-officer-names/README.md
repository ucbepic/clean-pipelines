# prap-split-officer-names

Cleans, validates, and splits **officer-name strings** extracted from
police records. Given a list of raw officer-name strings (rank + name +
noise), emits structured records: `(officer_name, cleaned_name,
valid_name, first_name, last_name, middle_name, suffix)`.

This is name **post-processing**, not extraction — the upstream
pipeline produces the raw `officer_name_string` column that this
package consumes.

Structural port of `involved_officer/split_names/src/src.py`. (The
legacy parent dir name `involved_officer` is misleading; the actual
code lives in the `split_names/` subdir.)

## Pipeline

1. **Rule-based cleaning** (no LLM):
   - Strip multi-word and single-word rank prefixes
     (`MULTI_WORD_RANKS`, `SINGLE_WORD_RANKS` — preserved verbatim).
   - Remove badge numbers (`#3446`, `832.7`), `FNU` / `LNU`
     placeholders, bracketed `[...]` and parenthetical `(...)` content,
     trailing punctuation, extra whitespace.
   - Re-strip rank prefixes in case cleaning revealed more.
2. **Obvious-invalidity guard**: reject single-word, placeholder, or
   generic-reference strings before calling the LLM.
3. **LLM stage 1 — extract** (`prompts/extract.txt`): pydantic-typed
   structured output via `prap_core.llm.LLM.complete(...,
   response_format=NameExtractionResult)`. Returns
   `is_valid_name` + first / last / middle / suffix.
4. **LLM stage 2 — validate** (`prompts/validate.txt`): re-verifies
   stage 1 with the extracted parts; returns
   `NameValidationResult(final_decision=bool)`.
5. Unique cleaned names are classified once; results map back to every
   record that shares that cleaned name.

## Use

```bash
prap-split-officer-names prepare \
  --input-csv path/to/involved-officers.csv \
  --output /tmp/names.jsonl

prap-split-officer-names run --input /tmp/names.jsonl --output /tmp/validated.jsonl
```

`prepare` accepts a CSV with either `officer_name` (legacy column) or
`officer_name_string` (current export schema), plus optional `roles`
(rows whose role contains "mentioned" are dropped by default) and
`case_id` / `case_name` passthrough.

## Evaluation

No ground truth is shipped for this pipeline. Sanity for "the run
worked" is:

- output passes the `NameClassification` schema, and
- the `RunResult` summary line (`n_records`, `n_unique_names`,
  `n_valid`) is non-degenerate — i.e. some non-zero fraction is
  marked valid and the cleaned names are populated.

A 200-row smoke run from the real export
(`involved-officer-cases-export.csv`) produces 155 records (200 minus
"mentioned" roles), 154 unique cleaned names, 137 marked valid
(88.4%).

## Notes

- The original pipeline keeps a `time.sleep(0.1)` per name as
  rate-limiting before each worker call; preserved here.
- Default `--n-threads` is 20 (matches the original `max_workers`).
- The model is configured entirely through
  `prap_core.config.Settings`; this package never imports a provider SDK
  directly. The original code pinned `gpt-4.1-mini-2025-04-14`; that
  pinning now lives in `PRAP_LLM_MODEL`.
- The companion scripts in `involved_officer/split_names/`
  (`src_all_names.py`, `compare_names.py`, `identify_ranks.py`) and
  `involved_officer/merge/` are **not** ported in this pass — see the
  refactor plan §5 Phase 4 (deferred to v1.1).
