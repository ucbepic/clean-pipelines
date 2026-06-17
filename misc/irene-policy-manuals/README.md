# irene-policy-manuals

Exploratory work on classifying law-enforcement PDFs as policy manuals,
case files with embedded policy language, or case files with no policy
content. Two scripts:

- `code/label.py` — extracts per-page text from labeled folders of PDFs
  (direct parse with PyMuPDF, falling back to Tesseract OCR when a page has
  almost no extractable text).
- `code/regex.py` — builds heuristic features (TF-IDF keywords, ALL-CAPS
  headers, numbered section patterns) and trains a Random Forest on them.

## Maturity / scope

Early, exploratory. The original author's own README is explicit:

> We know this approach is not strong enough for reliable classification.
> Current accuracy is around 50%, which is insufficient for production use.
> This script is included to document what has been tried.

Future directions the author suggested — LLM zero-shot classification,
full-document embeddings, fine-tuned transformers — are not implemented
here.

See `LEGACY_README.md` for the original, verbatim notes (input/output
folder layouts, columns produced by each script, etc.).

## Required environment

- Python: `pandas`, `pytesseract`, `Pillow`, `PyMuPDF` (`fitz`),
  `scikit-learn`.
- System: a working **Tesseract OCR** install (`tesseract` on PATH).

## How to run

```bash
# 1. Extract per-page text from labeled PDF folders.
python code/label.py --input-dir /path/to/files_sample

# 2. Train + score the Random Forest classifier on the extracted text.
python code/regex.py
```

`label.py` expects `--input-dir` to contain three subdirectories
(`policy_manuals/`, `cases_with_embedded_policy_manuals/`,
`cases_without_policy_manuals/`); each holds the PDFs for that class. The
output CSV defaults to `<input-dir>/../processed_data/sample_processed.csv`.

`regex.py` reads that same processed CSV from
`<repo>/processed_data/sample_processed.csv` (paths inside `regex.py` are
unchanged from the original — only `label.py` was parameterized).

## Status

This is **not** a package — no `packages/` entry, no `prap-<name>` CLI, no
`prap_core` dependency, no schemas. Standalone code kept here for review.
