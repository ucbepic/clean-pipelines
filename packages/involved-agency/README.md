# prap-involved-agency

# Status

Incomplete. I believe that some of the groundtruth data is off. False positives/false negatives need to be reivewied and validated.

# Pipeline

Extracts the **investigating** and **responding** agencies from a police
case bundle. For each case, runs an extract → verify → cite chain over
per-document summaries plus per-page OCR, and emits one CSV row per
(case, agency, role).

Behavior preserved verbatim from
`involved_agency/extract/src/extract.py` in the private repo (the
2,363-LOC canonical entry point). Only the LLM/IO/prompt-loading plumbing
changed; every `run_structured_inference(...)` call now goes through
`prap_core.llm.LLM.complete(..., response_format=...)`.

## What it does

1. **Filter** the per-document summaries down to the most agency-dense ones
   (`prompts/filter_summaries.txt` + `extract_clean_indices.txt`).
2. **Extract** investigating and responding agencies from the filtered
   summaries (`prompts/extract_agencies.txt`) into an
   `AgencyExtraction` structured object.
3. **Verify** each extracted agency one-by-one against the same source
   text, with an opportunity to recover missed agencies
   (`prompts/verify_agency.txt`).
4. **Cite** each surviving agency by searching every page of the case
   bundle for a passage that names the agency and an action verb. Uses a
   two-stage primary → validator chain
   (`prompts/primary_citation.txt` + `validator_citation.txt`).
5. **Write** one CSV row per (case, agency, role). Dual-role agencies are
   split into two rows. Cases with no agencies still get a single
   `agency_found=False` row.

## Install

```bash
uv sync
```

## Configure

```bash
export PRAP_LLM_MODEL=azure/gpt-4.1-mini
export PRAP_LLM_API_KEY=...
# or any LiteLLM-supported model — see prap/.env.example
```

## Use

### Input shape

A jsonl where each line is a case bundle dict (the legacy
`agency_case_file_bundle-<case_name>.json` shape) with at minimum:

```jsonc
{
  "case_name": "1742345262313-hme",
  "provisional_case_name": "1742345262313-hme",
  "case_files": [
    {
      "file_name": "doc-1.pdf",
      "sha1": "...",
      "summary": "...per-document summary...",
      "page_range": { "gdrive_id": "..." },
      "ocr_doc_text_per_page": {
        "page_texts": [{ "page_number": 1, "text": "..." }, ...]
      }
    },
    ...
  ]
}
```

### Prepare

```bash
prap-involved-agency prepare \
  --input-dir  /path/to/summarize/data/output/random_sample_v2 \
  --output     /tmp/cases.jsonl \
  --ground-truth /path/to/extract/data/output/groundtruth_labeled.csv  # optional filter
```

### Run

```bash
prap-involved-agency run \
  --input    /tmp/cases.jsonl \
  --output   /tmp/agencies.csv \
  --n-threads 15 \
  --save-every 10
```

CSV columns include `case_name`, `agency_name`, `agency_type`
(`RESPONDING` / `INVESTIGATING`), `agency_found`, `verified`,
`verification_status`, `confidence_level`, `llm_reasoning`,
`role_description`, `evidence`, `has_dual_role`, `dual_role_note`,
`num_citations`, `citations`, `full_extraction_response`.

### Evaluate

```bash
prap-involved-agency eval \
  --results       /tmp/agencies.csv \
  --ground-truth  /path/to/extract/data/output/groundtruth_labeled.csv \
  --ground-truth-fp /path/to/extract/data/output/groundtruth_false_positives_labeled.csv \
  --out-dir       /tmp/involved-agency-eval \
  --model-name    gpt-4.1-mini
```

## Programmatic use

```python
from prap_involved_agency import run
from prap_core.llm import LLM

llm = LLM()  # reads PRAP_LLM_* env
result = run("cases.jsonl", "agencies.csv", llm=llm, n_threads=15)
```

## Layout

```
packages/involved-agency/
├── src/prap_involved_agency/
│   ├── pipeline.py      # run() + per-case worker (was extract.py)
│   ├── citations.py     # find_agency_citations (was citations.py)
│   ├── helpers.py       # filter_important_summaries + clean_summaries
│   ├── schemas.py       # Pydantic models (was pydantic_models.py)
│   ├── evaluation.py    # match-to-GT + binary_prf scoring
│   ├── prompts/         # all 7 prompt templates (byte-identical with legacy)
│   └── cli.py           # Typer entry: prepare / run / eval
└── pyproject.toml
```

## Evaluation

Ground truth lives at
`involved_agency/extract/data/output/groundtruth_labeled.csv` in the
private repo (with `case_name`, `agency_name`, `agency_type`, `correct`
columns). It is **not** shipped publicly (per the `data/` ban in §1 of
the refactor plan).

Each `(case, gt_agency)` pair is a record: gold is `True` when the GT
row's `correct == '1'`, predicted is `True` when an LLM-name-matched
extraction exists for that case. Extractions absent from the GT are
counted as `(gold=False, predicted=True)` rows. Scoring is record-level
binary precision / recall / F1 via `prap_core.eval.binary_prf`.

### Results (2026-05-12, gpt-4.1-mini)

Smoke-run on **149 cases** (GT-filtered from the 522-bundle
`summaries/random_sample` corpus), `n_threads=16`, no errors. Extraction
produced 695 `(case, agency, role)` rows; 631 verified True. Eval
scored 367 `(case, gt_agency)` pairs against
`groundtruth_labeled.csv` + `groundtruth_false_positives_labeled.csv`.

| Model | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| gpt-4.1-mini | 367 | **0.8394** | **0.8822** | **0.8602** |

This is the first end-to-end metric on the refactored package. Numbers
have not yet been compared to the pre-refactor baseline; the §1
scope-softening note defers strict numeric-match to Phase 7. Re-runs on
a stronger model (e.g. gpt-4o, sonnet) are not blocked by anything in
the pipeline — just swap `PRAP_LLM_MODEL` and re-invoke.

## Notes

- The verification prompt's `recovery_reasoning` recommendations
  (missed-agency suggestions) are not yet plumbed back into the output;
  the legacy pipeline used `verification['recommendation']` and discarded
  `missed_agencies`. Preserved verbatim.
- The `corrected_agency_type` field in `SingleAgencyVerification` is
  required-Optional per the legacy schema; pass `None` rather than
  omitting.
- The model is configured entirely through `prap_core.config.Settings`;
  this package never imports a provider SDK directly.
