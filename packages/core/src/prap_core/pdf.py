from pathlib import Path

from pypdf import PdfReader, PdfWriter


def page_count(pdf_path: str | Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def slice_pages(pdf_path: str | Path, start: int, end: int, out_path: str | Path) -> Path:
    """Write pages [start, end) (0-indexed) of `pdf_path` to `out_path`."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    if start < 0 or end > total or start >= end:
        raise ValueError(f"invalid page range [{start}, {end}) for {total}-page pdf")
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        writer.write(fh)
    return out


def split_pages(pdf_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Split a PDF into one file per page. Returns the written paths in order."""
    reader = PdfReader(str(pdf_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    written: list[Path] = []
    for i, page in enumerate(reader.pages):
        target = out / f"{stem}_p{i:04d}.pdf"
        writer = PdfWriter()
        writer.add_page(page)
        with target.open("wb") as fh:
            writer.write(fh)
        written.append(target)
    return written
