# prap-page-stream-segmentation

Segment a concatenated PDF (many distinct documents merged into one file)
into individual documents and extract per-document index metadata
(headline, date, people, agencies).

## How it works

Per source document (one input `DocText` record):

1. **Page-by-page classification** — for each page, the LLM classifies it
   as a new document (with `Document type: <TYPE>`) or a continuation of
   the previous document (`CONTINUATION`). The prompt includes a rolling
   history of recent page classifications (collapsed for old segments,
   detailed for the last `--recent-window` pages) plus the tail of the
   previous page for boundary disambiguation. Prompts:
   `prompts/classify_page.txt`. Fallback prompt for parsing the
   doctype out of the metadata block: `prompts/extract_doctype.txt`.
2. **TOC assembly** — runs of (start + continuations) are grouped into
   contiguous documents. For each document, a second LLM call
   (`prompts/toc_item.txt`) produces a canonical index entry:
   `Headline`, `Date` (ISO), `People` (name → role), `Agency or agencies`.

Output is one `DocumentTOC` per source PDF.

## Intentional divergences from the legacy `dc/`

- Azure-backed OCR fetcher (`dc/doctext.py`) **dropped**. Input is
  pre-OCR'd page text via the `DocText` schema. Use `prap_core.ocr`
  upstream if you need OCR.
- `openai.AsyncOpenAI` + asyncio + semaphore → `prap_core.llm.LLM` (sync)
  + `ThreadPoolExecutor` across documents. Page classification stays
  sequential within a document — it needs prior-page context.
- Per-prompt `temperature`/`top_p`/`max_tokens` overrides dropped; uses
  `prap_core.llm` defaults (`temperature=0.0`).
- Jinja2 prompts preserved (`{% if %}` and `{% for %}` conditionals) —
  same precedent as `prap-involved-agency`.

## Input

Jsonl, one `DocText` per line:

```json
{"sha1": "abc...", "pages": [{"page_number": 1, "text": "..."}, ...]}
```

## Output

Jsonl, one `DocumentTOC` per line. Each TOC has an ordered `entries` list
of `TOCEntry`, each with the page-level `page_classifications` array
preserved for downstream review or evaluation.

## Usage

```bash
prap-page-stream-segmentation run \
  --input  /path/to/docs.jsonl \
  --output /tmp/prap-runs/page-stream-seg.jsonl \
  --n-threads 8

prap-page-stream-segmentation eval \
  --results       /tmp/prap-runs/page-stream-seg.jsonl \
  --ground-truth  /path/to/labeled-sample.xlsx \
  --out-dir       /tmp/prap-runs/eval/ \
  --model-name    gpt-4.1-mini
```

LLM config via `PRAP_LLM_*` env vars (see `prap_core.config`).

## Evaluation

The eval reduces to per-page binary classification: GT label `1` = page
is the start of a new document, `0` = continuation. Predicted label `1`
when the page is the start-page of a `TOCEntry`. Metric = precision /
recall / F1 / accuracy via `prap_core.eval.binary_prf`. The plan
originally anticipated needing Pk / WindowDiff helpers — those aren't
needed because the legacy eval (`tasks/evaluate/src/compute-metrics.py`)
is itself a per-page binary metric.

Ground-truth file format: xlsx / csv / parquet with columns
`sha1`, `page_number`, `label`, and (optional) a `stratum` column for
per-group breakdown. The frozen GT lives in the legacy repo at
`page-stream-segmentation/tasks/evaluate/frozen/labeled-sample-*.xlsx`.

## Pre-refactor results (2026-02 / 2026-03)

Numbers below are from the legacy `ablations/` harness on
**155 files, 6,946 pages**, evaluated against
`labeled-sample-20260217.xlsx`. Model: `gpt-4.1-mini-2025-04-14` via
Azure OpenAI. Confidence intervals are bootstrap (2,000 resamples,
file-level). The refactored package has not yet been re-run on this
corpus — strict numeric comparison is deferred to Phase 7 of the PRAP
refactor plan.

### Config ablation

| Config | Description | Precision | Recall | F1 | 95% CI (F1) |
|---|---|---|---|---|---|
| `d1h1c1` | domain + history + context (default) | 87.2% | 85.1% | **86.1%** | 0.84–0.88 |
| `d0h1c0` | history only | 83.8% | 87.6% | 85.7% | 0.84–0.88 |
| `d0h1c1` | history + context | 91.0% | 80.4% | 85.4% | 0.83–0.88 |
| `d1h1c0` | domain + history | 79.1% | 90.3% | 84.4% | 0.81–0.87 |
| `vision_pairwise` | pairwise vision (no OCR, no history) | 84.1% | 73.2% | 78.3% | 0.75–0.82 |
| `full_doc` | full PDF in one LLM call | 92.7% | 38.9% | 54.8% | 0.43–0.69 |
| `d1h0c1` | domain + context | 31.1% | 95.7% | 46.9% | 0.42–0.53 |
| `d0h0c1` | context only | 30.7% | 97.1% | 46.6% | 0.41–0.53 |
| `embed_sim` | sentence-transformers cosine sim (τ=0.50) | 38.3% | 52.3% | 44.2% | 0.39–0.50 |
| `d1h0c0` | domain only | 21.9% | 98.9% | 35.9% | 0.30–0.43 |
| `d0h0c0` | bare | 21.4% | 99.3% | 35.2% | 0.29–0.42 |
| `always_split` | every page = new doc | 18.3% | 100.0% | 30.9% | 0.26–0.36 |
| `never_split` | entire PDF = one doc | 100.0% | 12.2% | 21.7% | 0.16–0.32 |

**Key findings.** History (H) is decisive — removing it drops F1 from
~86% to ~47%. Domain (D) and previous-page context (C) add smaller
additive gains. The page-by-page sequential architecture beats the
`full_doc` naive baseline by ~31 F1 points despite costing more —
confirming history is doing real work, not just adding tokens.

### `d1h1c1` (default config) breakdown

By file size:

| File size | Files | Precision | Recall | F1 | 95% CI |
|---|---|---|---|---|---|
| 1–9 pages | 69 (45%) | 92.4% | 94.2% | 93.3% | 0.90–0.96 |
| 10–49 pages | 54 (35%) | 91.3% | 84.6% | 87.8% | 0.83–0.92 |
| 50–99 pages | 16 (10%) | 84.4% | 93.8% | 88.9% | 0.86–0.92 |
| 100+ pages | 16 (10%) | 85.6% | 82.0% | 83.8% | 0.80–0.87 |

By case type:

| Case type | Files | Precision | Recall | F1 | 95% CI |
|---|---|---|---|---|---|
| Misconduct | 106 (68%) | 90.9% | 85.0% | 87.9% | 0.84–0.91 |
| Mixed | 15 (10%) | 81.0% | 94.6% | 87.3% | 0.85–0.91 |
| OIS | 19 (12%) | 84.3% | 82.3% | 83.3% | 0.75–0.86 |
| UOF only | 15 (10%) | 90.3% | 73.7% | 81.2% | 0.75–0.87 |

## Costs (pre-refactor, `gpt-4.1-mini` via Azure)

Rates: $0.40 / 1M input tokens, $1.60 / 1M output tokens.

### Default `d1h1c1` config — 155 files, 6,946 pages

| Metric | Value |
|---|---|
| Total input tokens | 98.8M |
| Total output tokens | 1.26M |
| **Cost (eval corpus)** | **$41.55** |
| Avg input tokens / page | 14,228 |
| Avg output tokens / page | 182 |
| **Cost per page** | **$0.0060** |

**Input-token breakdown:**

| Component | Avg / page | Share |
|---|---|---|
| System message + prompt template | 3,013 | 21% |
| Page text (OCR) | 428 | 3% |
| Running history (accumulated meta) | 10,680 | **75%** |
| Previous-page context (last 5 lines) | 41 | <1% |

History dominates cost and was originally quadratic in document length.
The hybrid history-truncation fix (now in `pipeline._build_hybrid_history`,
`recent_window=15`) collapses completed segments into one-liners and
keeps only the last 15 pages of detailed metadata — bringing avg input
tokens / page from ~14K down to ~3.5–4K.

### Wall-clock (eval-corpus run)

| Metric | Value |
|---|---|
| 155 files at 25-way parallelism | ~4 hours |
| Effective throughput | ~0.48 pages / second |
| Per-page latency | ~2.1 seconds |

Pages within a file are sequential (history-aware architecture);
parallelism is at the file level.

### Extrapolation

Full corpus is ~2.4M pages; ~40% need splitting.

| Scenario | Pages | Cost (pre-truncation) | Cost (with hybrid history cap) |
|---|---|---|---|
| Pages needing splitting (40%) | 960K | ~$5,760 | ~$1,500 |
| Full corpus (upper bound) | 2.4M | ~$14,400 | ~$3,750 |

### Vision pairwise baseline

The `vision_pairwise` ablation (rendering page pairs as 512×512 images
and asking `gpt-4.1-mini` whether they belong to the same document)
scored F1=78.3% at **$0.0003 / page** — 17× cheaper than `d1h1c1`, with
linear (not quadratic) scaling, but 9 F1 points lower. Not ported as a
package config; the source lives in `page-stream-segmentation/ablations/vision_pairwise/`.

## What did NOT come over from the legacy repo

- The `ablations/` harness itself (the 13-config dispatcher, bootstrap
  CI helper, vision-pairwise + embedding + full-doc baselines). The
  three configurable knobs survive as `--no-domain` / `--no-history` /
  `--no-context` CLI flags; the trivial controls (`always_split` /
  `never_split`) and the alternative baselines (vision, embedding,
  full-doc) are not in this package.
- `tasks/` (web explorer, classify-export builders, redact, doc-summary
  notebooks) — these are downstream consumers of the pipeline output,
  not part of the pipeline itself.
- The Azure-blob OCR fetcher (`dc/doctext.py`) and Azure-blob PDF cache
  (`vision_pairwise/pdf_downloader.py`). Input is pre-OCR'd `DocText`
  records; use `prap_core.ocr` upstream if you need OCR.

The legacy repo (`page-stream-segmentation/`) is preserved on the
`archive/pre-refactor` branch with all of the above intact.

## Flags (ablation toggles)

- `--no-domain` — drop the SB-1421 background preamble from the classifier
  prompt.
- `--no-history` — disable the rolling history block.
- `--no-context` — disable the previous-page tail context.
- `--recent-window N` — number of recent pages kept in detailed history
  (default 15).
