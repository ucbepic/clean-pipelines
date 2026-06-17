# prap-location

Extracts the **city** where each police incident actually occurred
(distinct from department names, autopsy locations, or administrative
addresses). LLM-only — no NER, no SpaCy.

Structural port of `launch/location/src/location_city.py`.

## Pipeline

1. **(Special cases only)** For a fixed list of case-name prefixes that
   ship per-page OCR text instead of `first_look_summary`, the per-doc
   OCR sections are first summarized via `prompts/ocr_summary.txt` so
   downstream stages see summary-like text.
2. **Filter** the most relevant document summaries via
   `prap_core.summary_filter.filter_important_summaries` using
   `prompts/filter.txt`. Original settings preserved:
   `chunk_size=4`, `final_count=2`, `max_allowed_summaries=5`.
3. **Initial analysis** (`prompts/initial.txt`): identify the primary
   incident city with strict false-positive guards (department names,
   autopsies, review offices all → `None`).
4. **Validation** (`prompts/validation.txt`): a second LLM pass that
   confirms / overrides / updates the initial answer and emits a JSON
   payload (`final_decision`, `verified_quote`, etc.). On JSON parse
   failure, an `override → NO` fallback is returned (mirrors the source).
5. **Structured conversion** (`prompts/structured.txt`): collapse the
   validation JSON into either `"[City], [State]"` or `None`.

Behavior is preserved verbatim. Prompts are byte-identical to the
source except `{{var}}` → `$var` (Python `string.Template`).

The original source's per-call `time.sleep(0.5)` rate-limit nudges are
intentionally dropped — `prap_core.llm.LLM` already does retry /
back-off / caching centrally, so the same purpose is served by the
shared client.

## Use

```bash
prap-location prepare \
  --ground-truth /path/to/location_data.csv \
  --documents /path/to/gt-per-file-processed.csv \
  --output /tmp/location-cases.jsonl

prap-location run --input /tmp/location-cases.jsonl --output /tmp/location-results.jsonl
```

`prepare` accepts either `.parquet` or `.csv` for both inputs.

## Targeted-sample CSV (manual validation)

Some location strata (census-designated places, small unincorporated
communities) are under-represented in the standard eval set, so a
separate **targeted-sample CSV** is generated for human review. Rather
than re-running extraction, this command reads the existing
`prap-location run` output and adds a citation pass over a filtered
subset:

```bash
prap-location targeted-sample \
  --run /tmp/location-results.jsonl \
  --documents /path/to/gt-per-file-processed.csv \
  --filter sd_cdp \
  --output /tmp/targeted-sample.csv
```

The filter is either a builtin name (resolved against
`src/prap_location/targeted_filters/<name>.txt`) or a path to a `.txt`
file of lowercase substring patterns, one per line (blank lines and
`#` comments ignored). The shipped `sd_cdp` list contains the 14 San
Diego County CDP / unincorporated community names used in the
2026-Q2 manual-validation pass.

For each filtered case the command:
1. Loads per-page OCR from the documents table (joined by
   `provisional_case_name`).
2. Runs a two-stage per-page LLM citation analysis
   (`citation_page_primary` → `citation_page_validator`) in parallel.
3. Groups matches into per-document citation summaries.
4. Runs an aggregate re-validation prompt pair
   (`citation_aggregate_reasoning` → `citation_aggregate_parse`)
   producing `True` / `False` / `Unclear`, written into the
   `citation_revalidation` CSV column (skip with `--no-revalidate`).
5. Writes a CSV with the original run row plus `citations`,
   `citations_summary`, `gdrive_urls`, `filenames`,
   `citation_revalidation`, and an empty `correct` column for the
   human reviewer to fill in.

Design rationale: citations are an extra LLM pass per page and only
useful for the ~36 review rows, so we deliberately do **not** run
citations across the full corpus. Main `location-results.jsonl` stays
the canonical artifact; the targeted CSV is a cheap derivative.

## Evaluation

**No ground-truth dataset is shipped with `packages/location/`.** The
original `bids-experimental/location/` directory contained SpaCy NER
training/eval data; per the 2026-05-11 scoping decision we explicitly
ported only `location_city.py` (the LLM-only extractor) and dropped
the NER training path.

### Approach

Evaluation was run privately against a 215-case ground-truth set
labeled by student annotators (`location_data.csv`). Because incidents
in California are unevenly distributed across counties — a handful of
very-large urban counties dominate the corpus, while small/rural
counties contribute single-digit case counts — raw aggregate P/R/F1
under-represents performance on the under-sampled strata.

**Post-stratification** corrects for this by:

1. Classifying each county along two dimensions:
   - **Population size:** `small` < 50k, `medium` 50k–200k, `large`
     200k–1M, `very_large` > 1M (California DOF population estimates).
   - **Urban / rural:** California Office of the Attorney General rural-areas
     codes — codes `1–2` map to urban, `3+` to rural.
   - Composite strata: `small_urban`, `small_rural`, `medium_urban`,
     `medium_rural`, `large_urban`, `large_rural`, `very_large_urban`,
     `very_large_rural`.
2. Measuring P/R/F1 **within** each stratum.
3. Reweighting each stratum by its share of the **full corpus** (not
   the labeled sample) to produce a population-adjusted point estimate.

### Results

**Initial sample (215 cases, raw aggregate):**

| Metric | Value |
|---|---|
| Precision | 90.6 % |
| Recall    | 83.8 % |
| Accuracy  | 83.5 % |

Recall is treated as a **lower bound** — spot-checks suggest a
non-trivial share of the student-labeled ground truth is mislabeled.

**Post-stratified (reweighted to corpus county distribution):**

| Metric | Value | Δ vs raw |
|---|---|---|
| Precision | 92.8 % | +2.2 |
| Recall    | 89.1 % | +5.3 |
| Accuracy  | 87.6 % | +4.1 |

Lift comes from very-large urban counties (which perform well) being
under-represented in the sample relative to the full corpus.

**Per-stratum (sample):**

- Precision: 90.5 – 100 % across all strata — the model rarely
  hallucinates a city.
- Recall: 50 – 100 % — varies substantially by county type. Small
  rural counties show 100 % P/R but only on n=2 cases. Medium-sized
  counties show 50 – 70 % recall, indicating the model misses cities
  more often there.
- Sample sizes are too small in several strata to call the
  inter-stratum differences statistically reliable.

### Targeted sample (census-designated places / unincorporated areas)

To reduce uncertainty in the under-represented small-county strata, a
**targeted sample** of 36 cases was drawn where the incident occurred
in a census-designated place (CDP) or unincorporated community in San
Diego County. (Filter list: the 14 keywords in
`targeted_filters/sd_cdp.txt` — Spring Valley, Lakeside, Ramona,
Valley Center, Fallbrook, Alpine, Rancho Santa Fe, Santa Ysabel,
Campo, Pala, Dulzura, San Onofre, Pine Valley, Borrego Springs.)

A human reviewer labels each row `correct = 1` (extraction matches the
actual incident location) or `correct = 0` (extraction is wrong),
using the page-level citations and the source PDF as reference. One
citation per case is sufficient.

**Targeted-sample result (gpt-4.1-mini, 36 cases):**

| Metric | Value |
|---|---|
| Precision | **97.2 %** |

### Recap

| Slice | Precision | Recall |
|---|---|---|
| Initial sample (215 cases, raw) | 90.6 % | 83.8 % |
| Post-stratified (corpus-weighted) | 92.8 % | 89.1 % |
| Targeted CDP / unincorporated (36 cases) | 97.2 % | — |

P/R on the initial and stratified slices are likely lower bounds (GT
mislabel residual). The targeted-sample precision is human-validated
and includes no GT noise.

### How to reproduce

1. `prap-location prepare` + `prap-location run` against your
   per-case GT + per-file documents tables → produces
   `location-results.jsonl`.
2. `prap-location targeted-sample --filter sd_cdp` → produces the
   36-row validation CSV. Citations and a `correct` column (left
   empty) are included so a human reviewer can fill it in.
3. Post-stratification is **not** wired into the package CLI yet —
   the data plumbing (`read_groundtruth_agencies`,
   `extract_county`, `create_county_strata`,
   `calculate_*_by_strata`, county-population CSV, urban/rural CSV)
   currently lives in `launch/location/census/`. Porting it under
   `prap-location stratify` is a Phase 7 follow-up.

### Sanity gate (in addition to the metrics above)

Until per-case city GT ships with the public release, the
three-check sanity gate from the other `prap` packages still applies:

1. **Schema sanity** — every output row deserializes as a
   `LocationResult` (validated automatically by `pipeline.run()`).
2. **Distribution sanity** — eyeball the breakdown of
   `extracted_location is None` vs non-null; the 2026-05-12 run on
   the internal 215-case set produced 68.8 % non-null (148/215). A
   run where ≥ 95 % of cases come back non-null indicates the
   false-positive guard regressed.
3. **Tiny example fixture** — hand-write 2–3 `CaseRecord`s spanning a
   clean positive ("shooting occurred on Main Street in Oakland"), a
   clean negative ("Anaheim Police Department officer …"), and an
   ambiguous case, and confirm the outputs match the obvious expected
   answers before each release.

## Layout

```
packages/location/
├── src/prap_location/
│   ├── pipeline.py        # run() + 3-stage extraction + OCR summarization branch
│   ├── citation.py        # targeted-sample citation pass + filter loader
│   ├── schemas.py         # CaseRecord, LocationResult, ValidationResult, RunResult
│   ├── prompts/           # filter / initial / validation / structured / compare /
│   │                      # ocr_summary / citation_page_primary / citation_page_validator /
│   │                      # citation_aggregate_reasoning / citation_aggregate_parse
│   ├── targeted_filters/  # sd_cdp.txt (and future per-county lists)
│   └── cli.py             # run / prepare / targeted-sample
└── pyproject.toml
```

## Behavior notes

- `compare.txt` is the LLM-judge prompt the source used at eval time
  to fuzzy-match `extracted_location` against the ground-truth `location_city`
  column. It is shipped here for completeness but is not invoked by
  `pipeline.run()`; it will be wired in when a GT-backed eval lands.
- The list of "special case" prefixes is preserved verbatim in
  `pipeline.SPECIAL_CASE_PREFIXES`; only documents whose case name
  begins with one of those prefixes take the OCR-summarization path.
- `summary_filter` parameters (`chunk_size=4`, `final_count=2`,
  `max_allowed_summaries=5`) match the source's `helpers.filter_important_summaries`
  defaults exactly. These differ from `prap-case-type`'s settings.
