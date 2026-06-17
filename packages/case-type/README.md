# prap-case-type

Classifies each police case as **use of force** / **misconduct** /
**officer-involved shooting** (tristate: True / False / Unclear).

## Pipeline

1. Per case, chunk-filter summaries to the most relevant via
   `prap_core.summary_filter` (with `add_missing_fallback_indices=True`).
2. Run the master analysis prompt (`prompts/master.txt`) on the
   concatenated text — long-form reasoning across all three categories.
3. Pass the master output through three boolean prompts:
   `uof`, `misconduct`, `ois`. Each must return `true` / `false`
   / `unclear`; the response is normalized to `True` / `False` /
   `Unclear` by `helpers.natural_language_to_tristate_enum`.
4. If every OCR text in the case is < 250 cl100k tokens, skip the LLM
   chain and return `Unclear` across the board (file-corruption guard).

Prompts use `string.Template` (`$var`) substitution.

## Use

```bash
prap-case-type prepare \
  --ground-truth /path/to/gt-per-case.parquet \
  --documents /path/to/gt-per-file-processed.parquet \
  --output /tmp/cases.jsonl

prap-case-type run --input /tmp/cases.jsonl --output /tmp/results.jsonl

prap-case-type eval \
  --results /tmp/results.jsonl \
  --ground-truth /path/to/gt-per-case.parquet \
  --out-dir /tmp/case-type-eval \
  --model-name gpt-4.1-mini
```

## Evaluation

Scoring is via `prap_core.eval.binary_prf`. Each tristate prediction
(`True` / `False` / `Unclear`) is normalized: `True` → 1, everything else
(including `Unclear`, NaN, and missing) → 0. The same rule applies to
both the model output and the ground-truth columns
(`UOF_case_type`, `Misconduct_case_type`, `OIS_case_type`). The
`overall` row is the micro-average across all three fields.

Run against the 214-case parquet eval set with `openai/gpt-4.1-mini`:

| Field | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| use_of_force | 214 | 0.9643 | 0.8940 | 0.9278 |
| misconduct | 214 | 0.9070 | 0.8864 | 0.8966 |
| officer_involved_shooting | 214 | 0.9677 | 0.9574 | 0.9626 |
| **overall (micro)** | 642 | **0.9565** | **0.9135** | **0.9345** |

### Open-source model comparison

Two in-house vLLM text models compared against an `openai/gpt-4.1-mini`
baseline on the same eval set. Binary true-vs-not-true,
micro-averaged across the three categories. Endpoint:
`http://kj.hrdag.net/llm/v1` (OpenAI-compatible). Run date: 2026-04-22.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| gemma4-31b-it | 0.9641 | 0.8374 | 0.8963 |
| qwen3.5-27b | 1.0000 | 0.2284 | 0.3718 |

`gemma4-31b-it` lands close to the proprietary numbers above.
`qwen3.5-27b` predicts True very rarely and needs prompt adjustment
(or a less-strict decision rule) before it's usable here.

## Layout

```
packages/case-type/
├── src/prap_case_type/
│   ├── pipeline.py        # run() + classify_case()
│   ├── schemas.py         # CaseRecord, CaseClassifications, CaseTypeResult, RunResult
│   ├── prompts/           # filter / master / uof / misconduct / ois
│   ├── data_loading.py    # parquet/csv → jsonl
│   ├── evaluation.py      # per-field + micro-averaged P/R/F1
│   ├── helpers.py         # tristate enum normalization
│   └── cli.py
└── pyproject.toml
```

## Behavior notes

- A case whose OCR docs are all < 250 cl100k tokens (`tiktoken.cl100k_base`)
  is auto-classified as `Unclear`. The corresponding log line says
  "< 50 tokens" — the threshold is 250; the log message is intentionally
  left as-is to match historical output formats.
- The model is configured entirely through `prap_core.config.Settings`;
  this package never imports a provider SDK directly.
