# doc-classification

A CV / PyTorch document classifier originally built to label scanned report
pages from PDF rendering (page → 224×224 image → CNN). The whole
`classify-reports/` subtree is preserved verbatim:

- `classify-reports/label/` — turn PDFs into a labeled per-page image
  dataset (Makefile-driven; `label/src/label.py`).
- `classify-reports/train/` — train a CNN on that dataset
  (`train/src/src.py`, plus a grid-search variant `Makefile-grid` and an
  `archive/` of older experiments).
- `classify-reports/train/logs/` — original training logs from 2025-02-06
  retained for reference.

## Maturity / scope

CV/PyTorch document classifier; **kept for review, not part of the LLM
pipeline stack**. Not §4-pipeline-shaped (no `prap_core`, no
`prap-<name>` CLI, no JSONL I/O, no pydantic schemas). The trained model
weights (`final_model.pth`, ~227 MB) were **intentionally not copied** —
this directory ships code only.

## Required environment

Major Python deps (gathered from the source imports): `torch`,
`torchvision`, `opencv-python` (`cv2`), `PyMuPDF` (`fitz`), `Pillow`,
`pandas`, `numpy`, `scikit-learn`. CUDA is optional but the default
`DEVICE` in the train Makefile is `cuda`; override with `make train
DEVICE=cpu` if you don't have a GPU.

## How to run

The original Makefiles are preserved. Working inside each subdir:

```bash
cd classify-reports/label
make init       # create data/input/{pos-train,neg-train} skeleton
# ...drop labeled PDFs into pos-train/ and neg-train/...
make process    # produces data/output/labeled_df.csv

cd ../train
make train      # trains the CNN; writes to models/ and logs/
make validate   # validates the saved checkpoint
```

The `data/` subdirectories are **not** shipped in this repo — `make init`
recreates the empty skeleton.

## Status

This is **not** a package — no `packages/` entry, no `prap-<name>` CLI, no
`prap_core` dependency. Standalone code kept here for transparency.
