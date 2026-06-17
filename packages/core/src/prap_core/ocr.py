from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from .config import Settings


class Page(BaseModel):
    page_number: int  # 1-indexed
    text: str


class OCREngine(Protocol):
    def extract(self, pdf_path: str | Path) -> list[Page]: ...


# ---- Adapters (heavy deps imported lazily) ----


class UnstructuredAdapter:
    def extract(self, pdf_path: str | Path) -> list[Page]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as e:
            raise RuntimeError(
                "unstructured is not installed; `pip install 'prap-core[ocr-unstructured]'`"
            ) from e
        elements = partition_pdf(filename=str(pdf_path))
        by_page: dict[int, list[str]] = {}
        for el in elements:
            page = int(getattr(getattr(el, "metadata", None), "page_number", 1) or 1)
            text = getattr(el, "text", "") or ""
            if text:
                by_page.setdefault(page, []).append(text)
        return [Page(page_number=p, text="\n".join(by_page[p])) for p in sorted(by_page)]


class TesseractAdapter:
    def extract(self, pdf_path: str | Path) -> list[Page]:
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError as e:
            raise RuntimeError(
                "tesseract deps missing; `pip install 'prap-core[ocr-tesseract]'` "
                "and install poppler + tesseract on the host"
            ) from e
        images = convert_from_path(str(pdf_path))
        return [
            Page(page_number=i + 1, text=pytesseract.image_to_string(img))
            for i, img in enumerate(images)
        ]


def get_ocr_engine(settings: Settings | None = None) -> OCREngine:
    if settings is None:
        from .config import Settings as _Settings

        settings = _Settings()
    backend = settings.ocr_backend
    if backend == "unstructured":
        return UnstructuredAdapter()
    if backend == "tesseract":
        return TesseractAdapter()
    raise ValueError(f"unknown OCR backend: {backend}")
