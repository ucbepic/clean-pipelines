import pytest
from prap_core.prompts import PromptDir


def test_load_and_render(tmp_path):
    (tmp_path / "greet.txt").write_text("Hello, $name.", encoding="utf-8")
    p = PromptDir(tmp_path)
    assert p.load("greet") == "Hello, $name."
    assert p.render("greet", name="Ayyub") == "Hello, Ayyub."


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        PromptDir(tmp_path / "nope")


def test_missing_prompt(tmp_path):
    p = PromptDir(tmp_path)
    with pytest.raises(FileNotFoundError):
        p.load("does_not_exist")
