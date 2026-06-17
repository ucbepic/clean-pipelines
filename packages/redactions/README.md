# prap-redactions

> **Status: PARKED for v1.1.**
> This pipeline depends on **Azure Content Safety** (graphic-imagery
> classifier) and **Azure Blob Storage** (PDF source). PRAP v1 is
> OpenAI-only — we removed the Azure surface from `prap_core` to keep
> the public release on a single LLM provider. The CLI (`prap-redactions
> run`) prints a parked notice and exits non-zero. Source is preserved
> in-tree for the eventual revival once the graphic-imagery classifier
> is swapped for a non-Azure alternative (or the Azure dep is
> deliberately re-introduced).

Identifies pages in police case files that require redaction. Structural port of
`redactions/complete_pipeline/src/`.

## Pipeline

For each case JSON (`agency_case_file_bundle-*.json`), pages are grouped by
SHA1 and each file is run through four classifiers:

1. **Sensitive persons** (`classifiers/sensitive_persons.py`) — multi-stage LLM
   pipeline that extracts victim/witness names (excluding law enforcement,
   defendants, legal professionals), with fuzzy deduplication and address
   lookup. Only runs if the file mentions a monitored California Penal Code
   (see `helpers.has_penal_code_mentions`).
2. **Graphic imagery** (`classifiers/graphic_imagery.py`) — downloads the PDF
   from Azure Blob Storage, converts each low-word-count page (<=20 words) to
   a 300 DPI JPEG via `pdftoppm`, and scores it with Azure Content Safety.
   Pages with violence severity >= 4 are flagged.
3. **Date of birth** (`classifiers/dob.py`) — high-precision regex: explicit
   DOB label within 5 characters of a date pattern (with 1900-2099 year range).
4. **Credit/debit cards** (`classifiers/cards.py`) — pattern + prefix + length
   + Luhn checksum (no mixed separators).
5. **SSN** (`classifiers/ssn.py`) — `XXX-XX-XXXX` regex, skipping pages with
   fewer than 15 characters of text.

Behavior is preserved verbatim from the original. The four LLM prompts
(`identify_names`, `remove_irrelevant_persons`, `format_output`,
`find_addresses`) live in `src/prap_redactions/prompts/` as `*.txt` and use
Python `string.Template` (`$var`) substitution.

## Use

```bash
prap-redactions run \
  --input /path/to/case-bundles/ \
  --output /tmp/redactions/
```

Outputs (matching the legacy main.py):

- `redaction_list.csv` — one row per file needing redaction with
  `sha1`, `provisional_case_name`, `page_numbers`, `page_numbers_structured`.
- `processed_cases.txt` — checkpoint file (one case name per line) used to
  resume an interrupted run.
- `redaction_errors.txt` — written only if any case raised an exception.

## Evaluation

There is no ground-truth labeled dataset for redaction decisions, so this
package ships no automated evaluation harness. The plan for a sanity gate
(per `plans/2026-05-11-prap-refactor-plan.md`) is three checks:

1. Schema-valid: every CSV row parses and every classifier list is `list[int]`.
2. Distribution sanity: rates of pages flagged per classifier are within
   expected bands (DOB and cards should be sparse; sensitive_persons should
   only fire on penal-code-bearing files).
3. Tiny example fixture: a hand-built case bundle with one known DOB, one
   known card, one known SSN, and one penal-code-bearing page round-trips
   through `pipeline.run()` and produces the expected `redaction_list.csv`.

None of these have been wired up yet.

## Environment

Standard PRAP settings (`prap_core.config.Settings`, `PRAP_` env prefix):

- `PRAP_LLM_MODEL`, `PRAP_LLM_API_KEY`, `PRAP_LLM_API_BASE`,
  `PRAP_LLM_API_VERSION` — for the sensitive-persons LLM stages.
- `PRAP_AZURE_CONTENT_SAFETY_ENDPOINT`, `PRAP_AZURE_CONTENT_SAFETY_API_KEY` —
  for the graphic-imagery classifier.
- `PRAP_AZURE_STORAGE_CONNECTION_STRING`, `PRAP_AZURE_STORAGE_CONTAINER` —
  for downloading PDFs by SHA1.

## System requirements

Requires `pdftoppm` from poppler-utils on `$PATH` (used by the graphic-imagery
classifier to rasterize PDF pages).

## Layout

```
packages/redactions/
├── src/prap_redactions/
│   ├── pipeline.py           # run() — orchestrator
│   ├── cli.py                # Typer CLI (one subcommand: `run`)
│   ├── schemas.py            # pydantic models + RunResult
│   ├── helpers.py            # penal code patterns + word counting
│   ├── pdf_download.py       # Azure Blob Storage downloader
│   ├── classifiers/
│   │   ├── sensitive_persons.py
│   │   ├── graphic_imagery.py
│   │   ├── dob.py
│   │   ├── cards.py
│   │   └── ssn.py
│   └── prompts/              # identify_names / remove_irrelevant_persons /
│                             # format_output / find_addresses (all .txt)
└── pyproject.toml
```

## Behavior notes

- The `extract_names` function overrides its `max_workers` parameter to 7
  inside the function body — preserved from the original.
- The graphic-imagery classifier also pins parallel workers to 7 internally.
- Classifier-level errors are swallowed (logged) in `pipeline.process_case`
  rather than failing the whole case — preserved from the original.
- LLM calls go through `prap_core.llm.LLM`; no `openai` SDK is imported here.
