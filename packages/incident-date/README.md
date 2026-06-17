# prap-incident-date

Extracts the **incident date** from a police case file. Given per-document
summaries (or OCR text for special cases), runs a three-stage LLM chain
— extract → verify → ISO-8601 — and writes one record per case.

This is the reference pipeline for PRAP. New pipelines copy this layout.

## What it does

1. Per case, optionally chunk-filter the summaries down to the most
   relevant ones using `prap_core.summary_filter`.
2. Concatenate the selected text and ask the LLM for the incident date
   with reasoning (`prompts/extract.txt`).
3. Re-prompt for verification (`prompts/verify.txt`).
4. Convert the natural-language answer to ISO-8601 via a third LLM call
   (`prompts/to_iso.txt`).

Behavior is preserved verbatim from
`casefile_extraction/case_extraction/extract_incident_date/` in the
private repo, including a deliberate `initial_date` / `$initial_dates`
naming mismatch in the verification prompt (documented in
`pipeline.py`).

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

A jsonl file where each line is a `CaseRecord`:

```jsonc
{
  "provisional_case_name": "1718081113360-ofq",
  "summaries": ["...summary 1...", "...summary 2..."],
  "ocr_pages": null
}
```

`ocr_pages` is for the special-case path: when set, the pipeline skips
summary filtering and joins the OCR text directly. See
`data_loading.py:SPECIAL_CASE_PREFIXES`.

### Run

```bash
prap-incident-date run --input cases.jsonl --output results.jsonl
```

Each output line is an `IncidentDateResult`:

```jsonc
{
  "provisional_case_name": "...",
  "extracted_date": ["2024-07-18"],
  "nl_date": "VERIFICATION RESULT: CONFIRMED ..."
}
```

### Prepare the input from the casefile_extraction parquet/csv tables

```bash
prap-incident-date prepare \
  --ground-truth path/to/gt-per-case.parquet \
  --documents path/to/gt-per-file-processed.parquet \
  --output cases.jsonl
```

## Programmatic use

```python
from prap_incident_date import run
from prap_core.llm import LLM

llm = LLM()  # reads PRAP_LLM_* env
result = run("cases.jsonl", "results.jsonl", llm=llm, n_threads=20)
```

## Layout

```
packages/incident-date/
├── src/prap_incident_date/
│   ├── pipeline.py        # run() + per-case worker
│   ├── steps/             # (reserved for future step-extraction)
│   ├── schemas.py         # CaseRecord, IncidentDateResult, RunResult
│   ├── prompts/           # filter, extract, verify, to_iso
│   ├── data_loading.py    # parquet/csv → jsonl
│   ├── evaluation.py      # match/PRF/F1, case-type breakdown
│   ├── helpers.py         # date-component formatting
│   └── cli.py             # Typer entry point
└── pyproject.toml
```

## End-to-end flow

```bash
# 1. Convert the casefile_extraction tables into the jsonl input
prap-incident-date prepare \
  --ground-truth /path/to/gt-per-case.parquet \
  --documents   /path/to/gt-per-file-processed.parquet \
  --output      /tmp/cases.jsonl

# 2. Run the LLM extraction (reads PRAP_LLM_* env vars or prap/.env)
prap-incident-date run \
  --input  /tmp/cases.jsonl \
  --output /tmp/results.jsonl

# 3. Score the output against the ground truth
prap-incident-date eval \
  --results      /tmp/results.jsonl \
  --ground-truth /path/to/gt-per-case.parquet \
  --out-dir      /tmp/incident-date-eval \
  --model-name   gpt-4.1-mini
```

## Evaluation

A case is a **true positive** when at least one extracted date matches
any of the ground-truth dates assembled from the `Start_*` / `End_*` /
`Misconduct_date_ranges` columns by `helpers.get_ground_truth_dates`.
A case with no GT date and no extracted date is a true negative;
otherwise the mismatch splits into FP / FN per
`evaluation._row_classify`.

Run against the 209-case parquet eval set with `openai/gpt-4.1-mini`:

| Total | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 209 | **0.9333** | **0.9529** | **0.9430** |

`eval` also writes per-case-type breakdowns (UOF / Misconduct / OIS)
into the metrics CSV so you can see where the misses concentrate.

### Open-source model comparison

Two in-house vLLM text models compared against an `openai/gpt-4.1-mini`
baseline on the same 209-case eval set. Endpoint:
`http://kj.hrdag.net/llm/v1` (OpenAI-compatible). Run date: 2026-04-22.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| **gemma4-31b-it** | **0.9531** | **0.9581** | **0.9556** |
| qwen3.5-27b | 0.9574 | 0.9424 | 0.9499 |

Both models match or exceed the proprietary numbers above and are
viable open-weights replacements for this task.

## Notes

- The verify prompt template references `$initial_dates` but the pipeline
  passes `initial_date` (singular). The placeholder is intentionally left
  un-substituted; changing it changes model output.
- The model is configured entirely through `prap_core.config.Settings`;
  this package never imports a provider SDK directly.
