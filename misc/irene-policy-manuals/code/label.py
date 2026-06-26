import argparse
import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image

# Logging setup
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def pdf_page_to_image(page, zoom=2):
    """Convert PDF page to image using PyMuPDF."""
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # Increase resolution
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))


def extract_text_from_pdf(pdf_path: Path, label: str):
    """Extract text from each page of the PDF, using OCR only if needed."""
    data = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc[i]
            direct_text = page.get_text().strip()  # type: ignore[attr-defined]

            # Use OCR if no text or very little text is extracted
            if len(direct_text) < 20:
                image = pdf_page_to_image(page)
                ocr_text = pytesseract.image_to_string(image)
                final_text = ocr_text.strip()
                method = "OCR"
            else:
                final_text = direct_text
                method = "direct"

            data.append(
                {
                    "filename": pdf_path.name,
                    "page_num": i,
                    "label": label,
                    "text": final_text,
                    "extraction_method": method,
                }
            )

    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
    return data


def process_pdfs_to_csv(folder_label_map, output_csv_path):
    all_data = []

    for folder, label in folder_label_map.items():
        pdf_files = list(folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in '{folder}' (label: {label})")

        for pdf_path in pdf_files:
            page_data = extract_text_from_pdf(pdf_path, label)
            all_data.extend(page_data)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False, escapechar="\\")
    logger.info(f"Saved dataset to {output_csv_path}")
    return df


# Main function
if __name__ == "__main__":
    default_input_dir = Path(__file__).resolve().parent.parent / "files_sample"

    parser = argparse.ArgumentParser(
        description=(
            "Extract per-page text from labeled PDF folders. The input directory "
            "must contain three subdirectories: policy_manuals/, "
            "cases_with_embedded_policy_manuals/, and cases_without_policy_manuals/."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=(
            "Root directory holding the three labeled PDF subfolders "
            f"(default: {default_input_dir})."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=("Output CSV path. Defaults to <input-dir>/../processed_data/sample_processed.csv."),
    )
    args = parser.parse_args()

    base_dir = args.input_dir

    folder_label_map = {
        base_dir / "policy_manuals": "policy_manual",
        base_dir / "cases_with_embedded_policy_manuals": "embedded_policy_manuals",
        base_dir / "cases_without_policy_manuals": "no_policy_manuals",
    }

    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else base_dir.parent / "processed_data" / "sample_processed.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    process_pdfs_to_csv(folder_label_map, output_csv)
