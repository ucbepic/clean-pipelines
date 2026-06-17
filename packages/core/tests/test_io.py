from prap_core.io import append_jsonl, ensure_dir, read_jsonl, write_jsonl


def test_roundtrip(tmp_path):
    p = tmp_path / "out" / "records.jsonl"
    recs = [{"a": 1}, {"a": 2, "b": "x"}]
    n = write_jsonl(p, recs)
    assert n == 2
    assert list(read_jsonl(p)) == recs


def test_append(tmp_path):
    p = tmp_path / "a.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    assert list(read_jsonl(p)) == [{"a": 1}, {"a": 2}]


def test_ensure_dir(tmp_path):
    p = ensure_dir(tmp_path / "nested" / "deep")
    assert p.is_dir()


def test_read_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        list(read_jsonl(tmp_path / "missing.jsonl"))
