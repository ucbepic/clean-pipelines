import os
import pandas as pd
import fitz  
import logging
from pathlib import Path
import hashlib
import argparse
import random
import cv2
import numpy as np
from typing import Tuple, List

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_page_image(image: np.ndarray, size: Tuple[int, int] = (212, 212)) -> np.ndarray:
    """
    Preprocess a page image with enhanced feature definition and larger size for better model performance.
    
    Args:
        image: Input image as numpy array
        size: Target size for the processed image (default increased to 512x512)
    
    Returns:
        Preprocessed image as numpy array with enhanced features
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initial resize to larger working size for better feature preservation
    working_size = (1024, 1024)
    gray = cv2.resize(gray, working_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise while preserving edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast using CLAHE with larger tile size
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(blurred)
    
    # Apply adaptive thresholding for better text/background separation
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=15
    )
    
    # Enhance edges using a more sophisticated edge detection
    edges = cv2.Canny(enhanced, 30, 150)
    
    # Dilate edges to make structural elements more prominent
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Use morphological operations to enhance text features
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_enhanced = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, text_kernel)
    
    # Combine binary image with edges for structural emphasis
    combined = cv2.addWeighted(text_enhanced, 0.7, dilated_edges, 0.3, 0)
    
    # Apply sharpening for better feature definition
    kernel_sharpen = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
    sharpened = cv2.filter2D(combined, -1, kernel_sharpen)
    
    # Final cleanup using morphological operations
    cleanup_kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, cleanup_kernel)
    
    # Resize to target size using cubic interpolation for better quality
    final = cv2.resize(cleaned, size, interpolation=cv2.INTER_CUBIC)
    
    # Normalize to ensure consistent pixel value range
    final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)
    
    return final

def create_file_hash(file_path, document_id, page_num):
    """Create a unique hash for each file+page combination"""
    content = f"{file_path}_{document_id}_{page_num}".encode()
    return hashlib.md5(content).hexdigest()

def create_unique_directory(base_dir, file_hash):
    """Create a unique directory based on the file hash"""
    unique_dir = Path(base_dir) / file_hash[:8]
    unique_dir.mkdir(parents=True, exist_ok=True)
    return unique_dir

def process_page_range(pdf_path, start_page, end_page, document_id, output_dir, label, size=(212, 212)):
    """
    Process a specific range of pages from a PDF with enhanced preprocessing
    """
    results = []
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        pdf_document = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).stem
        
        logger.info(f"Processing {pdf_path} pages {start_page}-{end_page} (document_id: {document_id})")
        
        for page_num in range(start_page, end_page + 1):
            try:
                page = pdf_document[page_num]
                
                file_hash = create_file_hash(pdf_path, document_id, page_num)
                unique_dir = create_unique_directory(output_dir, file_hash)
                
                png_filename = f"{pdf_name}_doc_{document_id}_page_{page_num}_{file_hash}.png"
                png_path = str(unique_dir / png_filename)
                
                if os.path.exists(png_path):
                    logger.info(f"Thumbnail exists: {png_path}")
                else:
                    # Get high-resolution image for better feature extraction
                    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
                    
                    # Convert PyMuPDF pixmap to numpy array
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    
                    # Apply preprocessing
                    processed_img = preprocess_page_image(img_array, size=size)
                    
                    # Save processed image
                    cv2.imwrite(png_path, processed_img)
                    logger.info(f"Generated preprocessed image: {png_path}")
                
                results.append({
                    'document_id': document_id,
                    'filename': pdf_name,
                    'page_num': page_num,
                    'page_index': page_num - start_page,  # 0-based index within document
                    'total_pages': end_page - start_page + 1,
                    'filehash': file_hash,
                    'img_filepath': png_path,
                    'label': label,
                    'unique_dir': str(unique_dir)
                })
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue
                
        pdf_document.close()
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process PDFs to create a labeled dataset of preprocessed page thumbnails',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--pos-dir',
        type=str,
        required=True,
        help='Directory containing positive example PDFs'
    )
    
    parser.add_argument(
        '--neg-dir',
        type=str,
        required=True,
        help='Directory containing negative example PDFs'
    )
    
    parser.add_argument(
        '--gt-tbl',
        type=str,
        required=True,
        help='Path to groundtruth table CSV'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Size of output thumbnails (width height)'
    )
    
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.setLevel(args.verbosity)
    
    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    output_dir = Path(args.output_dir)
    gt_path = Path(args.gt_tbl)
    
    for path in [pos_dir, neg_dir, gt_path]:
        if not path.exists():
            logger.error(f"Path not found: {path}")
            return
    
    logger.info(f"Reading groundtruth table: {gt_path}")
    gt_df = pd.read_csv(gt_path)
    
    # Get list of all PDFs
    positives = {p.name: p for p in pos_dir.glob("*.pdf")}
    negatives = list(neg_dir.glob("*.pdf"))
    
    logger.info(f"Found {len(positives)} positive and {len(negatives)} negative PDFs")
    
    if not positives or not negatives:
        logger.error("No PDF files found in input directories")
        return
    
    # If we have fewer negatives than needed, we'll need to oversample
    needed_negatives = len(gt_df)
    if len(negatives) < needed_negatives:
        logger.info(f"Need {needed_negatives} negatives but only have {len(negatives)}. Will oversample.")
        # Oversample negatives with replacement
        negatives = random.choices(negatives, k=needed_negatives)
        logger.info(f"Oversampled to {len(negatives)} negative PDFs")
    
    all_results = []
    used_negatives = set()  # Still track usage within one iteration to avoid immediate duplicates
    
    for _, row in gt_df.iterrows():
        pdf_name = row['gdrive_name']
        if pdf_name not in positives:
            logger.warning(f"PDF {pdf_name} not found in positive directory")
            continue
            
        # Process positive document
        start_page = int(row['start_page']) - 1
        end_page = int(row['end_page']) - 1
        
        pos_results = process_page_range(
            pdf_path=str(positives[pdf_name]),
            start_page=start_page,
            end_page=end_page,
            document_id=row['document_id'],
            output_dir=str(output_dir),
            label=1,
            size=tuple(args.image_size)
        )
        
        if pos_results:
            all_results.extend(pos_results)
            
            # Select a random unused negative PDF
            available_negatives = [pdf for pdf in negatives if pdf not in used_negatives]
            if not available_negatives:
                # If we've used all negatives in this round, reset the used set
                used_negatives.clear()
                available_negatives = negatives
            
            neg_pdf = random.choice(available_negatives)
            used_negatives.add(neg_pdf)
            
            # Open negative PDF to get page count
            pdf_document = fitz.open(str(neg_pdf))
            total_pages = len(pdf_document)
            pdf_document.close()
            
            # Process selected negative document (all pages)
            neg_results = process_page_range(
                pdf_path=str(neg_pdf),
                start_page=0,
                end_page=total_pages - 1,
                document_id=f"neg_{row['document_id']}",
                output_dir=str(output_dir),
                label=0,
                size=tuple(args.image_size)
            )
            
            if neg_results:
                all_results.extend(neg_results)
                logger.info(f"Paired positive {pdf_name} with negative {neg_pdf.name}")
            else:
                logger.warning(f"Failed to process negative document {neg_pdf.name}")
                used_negatives.remove(neg_pdf)
    
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = output_dir / "labeled_df_test.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Dataset saved to {output_csv}")
        
        logger.info("\nDataset Statistics:")
        logger.info(f"Total documents: {df['document_id'].nunique()}")
        logger.info(f"Total pages: {len(df)}")
        logger.info("\nLabel distribution:")
        logger.info(df.groupby('label')['document_id'].nunique())
        logger.info("\nPages per document distribution:")
        logger.info(df.groupby('document_id')['page_num'].count().describe())
        
        # Count how many times each negative PDF was used
        neg_usage = {}
        for result in all_results:
            if result['label'] == 0:
                doc_id = result['document_id']
                if doc_id not in neg_usage:
                    neg_usage[doc_id] = 1
                else:
                    neg_usage[doc_id] += 1
        
        logger.info("\nNegative PDF Usage Statistics:")
        logger.info(f"Number of unique negative PDFs used: {len(neg_usage)}")
        if neg_usage:
            usage_counts = list(neg_usage.values())
            logger.info(f"Times each negative PDF was used: min={min(usage_counts)}, max={max(usage_counts)}, avg={sum(usage_counts)/len(usage_counts):.2f}")
    else:
        logger.warning("No documents were successfully processed")

if __name__ == "__main__":
    main()
