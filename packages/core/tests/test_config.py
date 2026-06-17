from pathlib import Path

from prap_core.config import Settings


def test_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("PRAP_"):
            monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_model.startswith("openai/")
    assert s.ocr_backend == "unstructured"
    assert isinstance(s.cache_dir, Path)
    assert s.cache_enabled is True


def test_env_override(monkeypatch):
    monkeypatch.setenv("PRAP_LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("PRAP_OCR_BACKEND", "tesseract")
    s = Settings(_env_file=None)
    assert s.llm_model == "openai/gpt-4o"
    assert s.ocr_backend == "tesseract"


def test_invalid_ocr_backend(monkeypatch):
    monkeypatch.setenv("PRAP_OCR_BACKEND", "bogus")
    try:
        Settings(_env_file=None)
    except Exception as e:
        assert "ocr_backend" in str(e).lower() or "literal" in str(e).lower()
    else:
        raise AssertionError("expected validation error")
