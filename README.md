# PRAP — Police Records Access Project

This repository contains code produced for the [Police Records Access Project](https://bids.berkeley.edu/california-police-records-access-project) by the Berkeley Institute for Data Science (BIDS)/UC Berkeley Epic Lab,
a collection of pipelines that turn raw government disclosures about
California police cases into structured data. Each pipeline lives in
its own package under `packages/` and shares a common `prap-core`
library for LLM calls, OCR, PDF handling, and evaluation.

## Contents

- [Quickstart](#quickstart-dev)
- [Layout](#layout)
- [Pipelines at a glance](#pipelines-at-a-glance)
- [Evaluations](#evaluations)
  - [`prap-clustering`](#prap-clustering)
  - [`prap-page-stream-segmentation`](#prap-page-stream-segmentation)
  - [`prap-incident-date`](#prap-incident-date)
  - [`prap-case-type`](#prap-case-type)
  - [`prap-location`](#prap-location)
  - [`prap-involved-agency`](#prap-involved-agency)
  - [Pipelines without ground truth](#pipelines-without-ground-truth)
- [`misc/`](#misc)
- [License](#license)

## Quickstart (dev)

```bash
cd prap
uv sync
uv run pytest
```

## Layout

```
prap/
├── packages/        # §4-compliant pipelines (LLM + jsonl + prap_core)
│   └── core/        # shared library (LLM, OCR, PDF, I/O, prompts, eval)
├── misc/            # standalone scripts, notebooks, non-§4 code
└── pyproject.toml   # uv workspace root
```

Shared LLM / OCR / embedding config (`PRAP_LLM_MODEL`,
`PRAP_LLM_API_KEY`, `PRAP_EMBEDDING_MODEL`, …) lives in
[`.env.example`](.env.example). Pipeline-specific env vars are
documented in each package's README.

## Pipelines at a glance

| Package | CLI | Headline result |
|---|---|---|
| `prap-core` | — | core library, 41 unit tests (incl. 5 for `LLM.embed()`) |
| [`prap-incident-date`](packages/incident-date/README.md) | `prap-incident-date` | F1 = **0.9430** on 209 cases |
| [`prap-case-type`](packages/case-type/README.md) | `prap-case-type` | F1 = **0.9345** micro on 214 cases |
| [`prap-involved-agency`](packages/involved-agency/README.md) | `prap-involved-agency` | F1 = **0.8602** on n=367 (gpt-4.1-mini) |
| [`prap-location`](packages/location/README.md) | `prap-location` | Post-stratified **P=92.8 / R=89.1**; targeted-CDP **P=97.2 %** |
| [`prap-page-stream-segmentation`](packages/page-stream-segmentation/README.md) | `prap-page-stream-segmentation` | F1 = **0.861** on 155 files / 6,946 pages (pre-refactor baseline) |
| [`prap-clustering`](packages/clustering/README.md) | `prap-clustering` | F1 = **0.76** on 31-agency / 4,937-case GT corpus (pre-refactor hybrid baseline) |
| [`prap-split-officer-names`](packages/split-officer-names/README.md) | `prap-split-officer-names` | 88.4 % valid on 155 records — no GT (three-check sanity gate) |
| [`prap-mentioned-agencies`](packages/mentioned-agencies/README.md) | `prap-mentioned-agencies` | no GT in v1; real-LLM corrections-bundle run pending |
| [`prap-redactions`](packages/redactions/README.md) | `prap-redactions` | 🚧 **work in progress** — relies on Azure Content Safety for violent-imagery classification; open-source replacement planned |

## Evaluations

Full eval results for the six pipelines with ground truth. See each
package's README for the underlying methodology.

### `prap-clustering`

Groups files (PDFs, images, audio, video) released by a single agency
into incident-level clusters — i.e., deciding which documents,
photos, and recordings all describe the same underlying case. A
three-tier cascade prioritizes cheap metadata signals before falling
back to more expensive LLM-based comparison, which outperforms
single-tier baselines (regex-only, embedding-only, or LLM-only).
Evaluated on a 31-agency / 4,937-case GT corpus.

The hybrid pipeline builds a graph (nodes = files, edges =
same-incident matches) by running each candidate pair through the
cascade. Each tier can match a pair, hard-block it, or pass it to
the next tier:

- **Tier 1 — filepath + filename regex.** Cheap, deterministic.
  Matches on case-ID overlap or shared deep directory paths;
  hard-blocks on conflicting case IDs.
- **Tier 2 — LLM-extracted feature rules.** Structured fields
  (case IDs, dates, subject/officer names) parsed from per-PDF
  summaries; requires two corroborating signals to match. A local
  sentence-transformers cosine-similarity check hard-blocks
  dissimilar pairs and filters ~97 % of candidates before Tier 3.
- **Tier 3 — pairwise semantic comparison.** An LLM directly
  compares concatenated feature summaries for the pairs that
  survived Tiers 1–2.

Connected components of the resulting match graph form clusters;
an optional cluster-refinement pass splits over-merged components.
The ablation rows below toggle which tiers are enabled, plus a
random-forest variant trained on T1+T2 features and an
embeddings-only baseline for reference.

| Configuration | Precision | Recall | F1 | 95% CI |
|---|---:|---:|---:|:---:|
| All tiers + cluster refinement | **0.92** | **0.76** | **0.76** | 0.71–0.81 |
| All tiers, no refinement | 0.86 | 0.80 | 0.75 | 0.69–0.80 |
| Tiers 1 + 2 only (no T3) | 0.92 | 0.74 | 0.74 | 0.70–0.80 |
| Tiers 2 + 3 only (no regex) | 0.94 | 0.71 | 0.73 | 0.67–0.79 |
| Tier 2 only | 0.94 | 0.68 | 0.71 | 0.65–0.78 |
| Tier 1 only | 0.96 | 0.63 | 0.67 | 0.60–0.74 |
| RF (T1+T2 features) | 0.88 | 0.73 | 0.71 | 0.66–0.77 |
| Embedding baseline (τ=0.85) | 0.64 | 0.63 | 0.43 | 0.35–0.52 |

See [`packages/clustering/README.md`](packages/clustering/README.md)
for the three-tier extraction strategy and CLI subcommands.

### `prap-page-stream-segmentation`

Government disclosures frequently arrive as **compound PDFs**: a
single file that concatenates many distinct documents (incident
reports, IA findings, witness statements, etc.) about one or more
cases. This pipeline segments a compound PDF into its constituent
documents by classifying each page as either the start of a new
document or a continuation of the previous one. Pages are classified
**sequentially with a running history** of prior decisions — so the
model can reason about which document is currently in progress
rather than judging each page in isolation, which is what drives the
gap over history-free baselines in the ablation below. Evaluated on
155 files / 6,946 pages with `gpt-4.1-mini`.

**Config ablation.** Each page is classified independently, and the
prompt can include up to three optional signals. The three-letter code
(`dXhYcZ`) toggles each one on (`1`) or off (`0`):

- **D — domain preamble.** An SB-1421 / police-records background blurb
  that tells the LLM what kinds of documents (use-of-force reports,
  IA investigations, etc.) it should expect at document boundaries.
- **H — running history.** A rolling record of how prior pages in the
  same PDF were classified (detailed for the last `--recent-window`
  pages, collapsed for older segments). Lets the model see "we are
  10 pages into an IA report" rather than judging each page in
  isolation.
- **C — previous-page context.** The tail of the previous page's OCR
  text, for boundary disambiguation when a page begins mid-sentence vs.
  with a new header.

The default is `d1h1c1` (all three on). Other rows are non-LLM
controls or alternative architectures: `vision_pairwise` shows two
consecutive page images to a vision model with no history;
`full_doc` puts the entire PDF in one call; `embed_sim` uses
sentence-transformers cosine similarity between adjacent pages;
`always_split` / `never_split` are trivial baselines.

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

**`d1h1c1` breakdown by file size:**

| File size | Files | Precision | Recall | F1 | 95% CI |
|---|---|---|---|---|---|
| 1–9 pages | 69 (45%) | 92.4% | 94.2% | 93.3% | 0.90–0.96 |
| 10–49 pages | 54 (35%) | 91.3% | 84.6% | 87.8% | 0.83–0.92 |
| 50–99 pages | 16 (10%) | 84.4% | 93.8% | 88.9% | 0.86–0.92 |
| 100+ pages | 16 (10%) | 85.6% | 82.0% | 83.8% | 0.80–0.87 |

**`d1h1c1` breakdown by case type:**

| Case type | Files | Precision | Recall | F1 | 95% CI |
|---|---|---|---|---|---|
| Misconduct | 106 (68%) | 90.9% | 85.0% | 87.9% | 0.84–0.91 |
| Mixed | 15 (10%) | 81.0% | 94.6% | 87.3% | 0.85–0.91 |
| OIS | 19 (12%) | 84.3% | 82.3% | 83.3% | 0.75–0.86 |
| UOF only | 15 (10%) | 90.3% | 73.7% | 81.2% | 0.75–0.87 |

**Cost — default `d1h1c1`, 155 files / 6,946 pages:**

| Metric | Value |
|---|---|
| Total input tokens | 98.8M |
| Total output tokens | 1.26M |
| **Cost (eval corpus)** | **$41.55** |
| Avg input tokens / page | 14,228 |
| Avg output tokens / page | 182 |
| **Cost per page** | **$0.0060** |

Running history dominates input cost (75% of per-page tokens). See
[`packages/page-stream-segmentation/README.md`](packages/page-stream-segmentation/README.md)
for the full input-token decomposition, wall-clock numbers, and
full-corpus cost extrapolation.

### `prap-incident-date`

Extracts the date the underlying incident occurred for each case.
Incident date is a load-bearing field for downstream **officer
resolution**: candidate officer mentions in a case are linked against
California's POST (Peace Officer Standards and Training) employment
database of ~77,000 certified officers, and the employment-timeline
check that disambiguates same-name officers requires knowing *when*
the incident occurred relative to each candidate's tenure at the
employing agency. Without a correct incident date, an officer who
accumulates uses of force at one agency and transfers to another
cannot reliably be tied back to a single POST record. End-to-end run
on a 209-case ground-truth set with `gpt-4.1-mini`.

| Total | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 209 | **0.9333** | **0.9529** | **0.9430** |

**Open-source models** (vLLM endpoint, OpenAI-compatible, run 2026-04-22)
on the same 209-case eval set:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| **gemma4-31b-it** | **0.9531** | **0.9581** | **0.9556** |
| qwen3.5-27b | 0.9574 | 0.9424 | 0.9499 |

Both models match or exceed the proprietary numbers above.

See [`packages/incident-date/README.md`](packages/incident-date/README.md).

### `prap-case-type`

Labels each case along three independent binary axes — use-of-force,
misconduct, and officer-involved-shooting — to support understanding
the distribution of case types across California agencies. Reported
as three per-field classifiers plus a micro-aggregate.

| Field | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| use_of_force | 214 | 0.9643 | 0.8940 | 0.9278 |
| misconduct | 214 | 0.9070 | 0.8864 | 0.8966 |
| officer_involved_shooting | 214 | 0.9677 | 0.9574 | 0.9626 |
| **overall (micro)** | 642 | **0.9565** | **0.9135** | **0.9345** |

**Open-source models** (vLLM endpoint, OpenAI-compatible, run 2026-04-22)
on the same eval set, micro-averaged across the three categories:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| gemma4-31b-it | 0.9641 | 0.8374 | 0.8963 |
| qwen3.5-27b | 1.0000 | 0.2284 | 0.3718 |

`gemma4-31b-it` lands close to the proprietary numbers above.
`qwen3.5-27b` shows much lower recall on the same prompt.

See [`packages/case-type/README.md`](packages/case-type/README.md).

### `prap-location`

Extracts the geographic location (city or unincorporated area, county)
where each incident occurred, to support understanding the geographic
distribution of cases across California.

> ⚠️ **Work in progress.** Like `prap-involved-agency`, this
> pipeline was never published as a public PRAP feature and was
> not iterated to completion. The numbers below reflect the state
> of the pipeline when work was paused, not a finalized result.

Three slices, all on the same underlying corpus.

**Initial sample (n=215, raw):**

| Metric | Value |
|---|---|
| Precision | 90.6 % |
| Recall    | 83.8 % |
| Accuracy  | 83.5 % |

**Post-stratified (reweighted to corpus county distribution):**

| Metric | Value | Δ vs raw |
|---|---|---|
| Precision | 92.8 % | +2.2 |
| Recall    | 89.1 % | +5.3 |
| Accuracy  | 87.6 % | +4.1 |

Post-stratification corrects for under-representation of very-large
urban counties in the sample relative to the full corpus.

**Targeted sample (census-designated places / unincorporated areas, n=36):**

| Metric | Value |
|---|---|
| Precision | **97.2 %** |

Recap across slices:

| Slice | Precision | Recall |
|---|---|---|
| Initial sample (215 cases, raw) | 90.6 % | 83.8 % |
| Post-stratified (corpus-weighted) | 92.8 % | 89.1 % |
| Targeted CDP / unincorporated (36 cases) | 97.2 % | — |

Recall numbers are treated as a **lower bound** — spot-checks suggest
residual GT mislabels from the initial student-labeled pass.
See [`packages/location/README.md`](packages/location/README.md)
for the stratification methodology and reproducibility steps.

### `prap-involved-agency`

Extracts all agencies involved in a case along with each agency's role
(e.g., employing agency, investigating agency, assisting agency), to
support understanding cross-agency patterns — for example, which
outside agencies investigate which departments' incidents.

> ⚠️ **Work in progress.** This pipeline was never published as a
> public PRAP feature, so it was not iterated to completion. The
> numbers below should be treated as a baseline, not a final result.

149 GT-filtered cases → 695 (case, agency, role) rows extracted;
evaluated at the (case, agency) pair level against 367 GT pairs with
`gpt-4.1-mini`.

| Model | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| gpt-4.1-mini | 367 | **0.8394** | **0.8822** | **0.8602** |

See [`packages/involved-agency/README.md`](packages/involved-agency/README.md).

### Pipelines without ground truth

Three pipelines ship without numeric eval. Each uses the
[three-check sanity gate](../plans/2026-05-11-prap-refactor-plan.md#eval-policy)
documented in the refactor plan: schema validation, distribution
sanity, and a tiny example fixture.

- **`prap-split-officer-names`** — name-splitting / validation
  pipeline. 88.4 % valid on a 155-record sample. See
  [`packages/split-officer-names/README.md`](packages/split-officer-names/README.md).
- **`prap-mentioned-agencies`** — three-stage agency extraction
  (per-page LLM → case-level validation → fuzzy dedup). Smoke tests
  pass; real-LLM run on corrections bundles still TODO. See
  [`packages/mentioned-agencies/README.md`](packages/mentioned-agencies/README.md).
- **`prap-redactions`** — work in progress. The pipeline currently
  relies on Azure Content Safety's out-of-the-box mode for the
  violent-imagery classifier; this will be swapped for an
  open-source classifier in a future release. Source preserved
  in-tree; CLI prints a notice and exits non-zero until the
  open-source path lands.

## `misc/`

Code that doesn't fit the §4 pipeline contract still ships in this
repo for review — see [`misc/README.md`](misc/README.md) for the
index. Includes: CV classifiers, AWS-only utilities, rule-based
post-processing, and exploratory notebooks.

## License

MIT.
