import pytest
from prap_core.config import Settings
from prap_core.ocr import (
    Page,
    TesseractAdapter,
    UnstructuredAdapter,
    get_ocr_engine,
)


def test_get_ocr_engine_unstructured():
    s = Settings(_env_file=None, ocr_backend="unstructured")
    assert isinstance(get_ocr_engine(s), UnstructuredAdapter)


def test_get_ocr_engine_tesseract():
    s = Settings(_env_file=None, ocr_backend="tesseract")
    assert isinstance(get_ocr_engine(s), TesseractAdapter)


def test_get_ocr_engine_rejects_unknown_backend():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, ocr_backend="azure")


def test_page_model():
    p = Page(page_number=1, text="hello")
    assert p.page_number == 1
    assert p.text == "hello"
