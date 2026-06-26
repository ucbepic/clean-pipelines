import argparse
import hashlib
import logging
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_page_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Multi-scale preprocessing pipeline for document images with content-aware processing.

    Args:
        image: Input image as numpy array
        size: Target size for final output
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Define processing scales
    scales = [0.5, 1.0, 2.0]
    processed_scales = []

    for scale in scales:
        # Calculate size for this scale
        work_size = (int(1024 * scale), int(1024 * scale))
        scaled = cv2.resize(gray, work_size, interpolation=cv2.INTER_CUBIC)

        # Denoise scaled version
        denoised = cv2.fastNlMeansDenoising(scaled, None, h=10, searchWindowSize=21)

        # Enhance contrast for this scale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Process based on local content density
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=15
        )

        # Clean noise specific to this scale
        kernel_size = max(2, int(3 * scale))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Resize back to original size for combining
        resized = cv2.resize(cleaned, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        processed_scales.append(resized)

    # Weight and combine the scales
    weights = [0.25, 0.5, 0.25]
    combined = np.zeros_like(processed_scales[1], dtype=np.float32)

    for scale_img, weight in zip(processed_scales, weights, strict=False):
        combined += scale_img * weight

    combined = np.clip(combined, 0, 255).astype(np.uint8)

    # Final cleanup
    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Resize to target size
    output = cv2.resize(final, size, interpolation=cv2.INTER_CUBIC)

    return output


def create_file_hash(file_path):
    """Create a hash based on file contents for deduplication"""
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def process_pdf(
    pdf_info: tuple[Path, int], output_dir: Path, image_size: tuple[int, int]
) -> list[dict]:
    """
    Process a single PDF file and generate preprocessed page images

    Args:
        pdf_info: Tuple of (pdf_path, label)
        output_dir: Directory for output files
        image_size: Target size for output images
    """
    pdf_path, label = pdf_info
    results = []

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_document = fitz.open(str(pdf_path))
        pdf_name = pdf_path.stem

        logger.info(f"Processing {pdf_path} with {len(pdf_document)} pages")

        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document[page_num]

                file_hash = create_file_hash(pdf_path)
                unique_dir = output_dir / file_hash[:8]
                unique_dir.mkdir(parents=True, exist_ok=True)

                png_filename = f"{pdf_name}_page_{page_num}_{file_hash}.png"
                png_path = unique_dir / png_filename

                if png_path.exists():
                    logger.info(f"Thumbnail exists: {png_path}")
                else:
                    # Get high-resolution image
                    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))

                    # Convert to numpy array
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )

                    # Preprocess image
                    processed_img = preprocess_page_image(img_array, size=image_size)

                    # Save processed image
                    cv2.imwrite(str(png_path), processed_img)
                    logger.info(f"Generated preprocessed image: {png_path}")

                results.append(
                    {
                        "filename": pdf_name,
                        "page_num": page_num,
                        "filehash": file_hash,
                        "img_filepath": str(png_path),
                        "label": label,
                        "unique_dir": str(unique_dir),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue

        pdf_document.close()

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process PDFs to create a labeled dataset of preprocessed page thumbnails",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pos-dir", type=str, required=True, help="Directory containing positive example PDFs"
    )

    parser.add_argument(
        "--neg-dir", type=str, required=True, help="Directory containing negative example PDFs"
    )

    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output files")

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Size of output thumbnails (width height)",
    )

    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive pages (default: 1.0)",
    )

    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.setLevel(args.verbosity)

    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    output_dir = Path(args.output_dir)

    # Validate input directories
    for path in [pos_dir, neg_dir]:
        if not path.exists():
            logger.error(f"Path not found: {path}")
            return

    # Get list of PDFs
    positive_pdfs = list(pos_dir.glob("*.pdf"))
    negative_pdfs = list(neg_dir.glob("*.pdf"))

    logger.info(f"Found {len(positive_pdfs)} positive and {len(negative_pdfs)} negative PDFs")

    if not positive_pdfs or not negative_pdfs:
        logger.error("No PDF files found in input directories")
        return

    # Prepare processing tasks
    pos_tasks = [(pdf, 1) for pdf in positive_pdfs]
    neg_tasks = [(pdf, 0) for pdf in negative_pdfs]

    # Process documents in parallel
    process_pdf_partial = partial(
        process_pdf, output_dir=output_dir, image_size=tuple(args.image_size)
    )

    num_workers = 500
    # Process files in parallel
    with Pool(processes=num_workers) as pool:
        # Process positive examples in parallel
        logger.info("Processing positive examples...")
        pos_results = list(pool.map(process_pdf_partial, pos_tasks))
        pos_results = [r for sublist in pos_results if sublist for r in sublist]
        total_pos_pages = len(pos_results)
        logger.info(f"Processed {total_pos_pages} positive pages")

        # Calculate negative examples needed and process them
        target_neg_pages = int(total_pos_pages * args.neg_ratio)
        logger.info(f"Target negative pages: {target_neg_pages}")
        random.shuffle(neg_tasks)

        # Process negative examples in parallel
        logger.info("Processing negative examples...")
        neg_results = list(pool.map(process_pdf_partial, neg_tasks))
        neg_results = [r for sublist in neg_results if sublist for r in sublist]
        neg_results = neg_results[:target_neg_pages]  # Trim to desired ratio
        logger.info(f"Processed {len(neg_results)} negative pages")

    # Combine and shuffle all results
    all_results = pos_results + neg_results
    random.shuffle(all_results)

    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = output_dir / "labeled_df_train.csv"
        df.to_csv(output_csv, index=False)

    return all_results


if __name__ == "__main__":
    main()
