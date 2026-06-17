import pytest
from prap_core.pdf import page_count, slice_pages, split_pages
from pypdf import PdfWriter


def _make_pdf(path, n_pages: int) -> None:
    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=72, height=72)
    with open(path, "wb") as fh:
        w.write(fh)


def test_page_count(tmp_path):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, 5)
    assert page_count(pdf) == 5


def test_slice_pages(tmp_path):
    pdf = tmp_path / "x.pdf"
    out = tmp_path / "slice.pdf"
    _make_pdf(pdf, 5)
    slice_pages(pdf, 1, 4, out)
    assert page_count(out) == 3


def test_slice_pages_invalid_range(tmp_path):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, 3)
    with pytest.raises(ValueError):
        slice_pages(pdf, 2, 2, tmp_path / "o.pdf")
    with pytest.raises(ValueError):
        slice_pages(pdf, 0, 10, tmp_path / "o.pdf")


def test_split_pages(tmp_path):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, 3)
    out = tmp_path / "pages"
    written = split_pages(pdf, out)
    assert len(written) == 3
    assert all(p.exists() for p in written)
    assert all(page_count(p) == 1 for p in written)
