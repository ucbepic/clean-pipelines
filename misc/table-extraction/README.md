# table-extraction

Single-script wrapper around **AWS Textract** that splits each input PDF
page-by-page, uploads each page to an S3 staging bucket, runs Textract's
`TABLES` analysis on it, and writes the recovered tables to CSV files
(`page_<N>_table.csv`) under the output directory.

## Maturity / scope

AWS Textract script. Not part of the LLM pipeline stack. Requires AWS
credentials and the env vars below. Kept here for review and occasional
ad-hoc use; not §4-pipeline-shaped (no `prap_core`, no `prap-<name>` CLI, no
JSONL I/O).

## Required environment

- **AWS credentials** discovered via the usual boto3 chain — env vars
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
  `AWS_PROFILE`) or `~/.aws/credentials`.
- `PRAP_TABLE_EXTRACTION_S3_BUCKET` — **required**, no default. Name of an
  S3 bucket the script can upload PDF pages into for Textract to read. The
  original private bucket name has been scrubbed from the source.
- `PRAP_TABLE_EXTRACTION_AWS_REGION` — optional, defaults to `us-west-2`.

Python deps: `boto3`, `PyPDF2`.

## How to run

```bash
export PRAP_TABLE_EXTRACTION_S3_BUCKET=my-textract-staging-bucket
export PRAP_TABLE_EXTRACTION_AWS_REGION=us-west-2  # optional

python src/src.py --input-dir /path/to/pdfs --output-dir /path/to/output
```

If `--input-dir` / `--output-dir` are omitted, the defaults (`../data/input`
and `../data/output`) are used, matching the original layout.

## Status

This is **not** a package — it does not live in `packages/`, has no
`prap-table-extraction` CLI, no `prap_core` dependency, and no pydantic
schemas. It is standalone code kept in the repo for transparency.
