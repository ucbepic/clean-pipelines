# misc/

Code that ships in the public PRAP repo for review but does **not** fit
the standard `packages/` pipeline contract (LLM-driven, jsonl I/O, uses
`prap_core`).

Things you'll find here:

- **Standalone scripts** that solve a narrow problem (e.g. an AWS
  Textract table extractor) but aren't shaped like a §4 pipeline.
- **Notebook-only exploratory work** that hasn't been translated into
  reusable modules.
- **CV / ML pipelines** that don't use an LLM.
- **Rule-based post-processing utilities** for record linkage,
  time-series feature engineering, etc.
- **Author-flagged experimental work** that the original author noted
  as incomplete or low-accuracy.

None of these are installed by `uv sync`. They have no `prap-<name>`
CLIs. They're plain Python files / notebooks you read or run with
your own interpreter. Each subdir's README documents what's there and
how to run it.

| Subdir | What it is |
|---|---|
| [`doc-classification/`](doc-classification/README.md) | CV/PyTorch document-image classifier |
| [`irene-policy-manuals/`](irene-policy-manuals/README.md) | Early heuristic / Random-Forest classifier (author notes ~50% accuracy) |
| [`mentioned-agencies/`](mentioned-agencies/README.md) | Notebook-only exploration of mentioned-agency extraction |
| [`post/`](post/README.md) | Rule-based post-processing, record linkage, time-series feature work |
| [`sample/`](sample/README.md) | Notebook-only sampling / EDA helper |
| [`table-extraction/`](table-extraction/README.md) | AWS Textract script for table extraction from PDFs |

If a `misc/` artifact later gets reshaped into a `prap_core`-using,
jsonl-emitting pipeline, it graduates to `packages/`.
