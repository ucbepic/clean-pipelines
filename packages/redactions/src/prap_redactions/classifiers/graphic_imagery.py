"""Graphic imagery classifier.

Azure Content Safety credentials are sourced from prap_core.config.Settings.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image
from prap_core.config import Settings

from ..pdf_download import PdfDownloader

logger = logging.getLogger("prap.redactions.graphic_imagery")


class AzureContentSafetyService:
    """Azure Content Safety service for analyzing images"""

    def __init__(self, settings: Settings | None = None):
        s = settings or Settings()
        if not s.azure_content_safety_endpoint:
            raise ValueError("PRAP_AZURE_CONTENT_SAFETY_ENDPOINT not set in environment")
        if not s.azure_content_safety_api_key:
            raise ValueError("PRAP_AZURE_CONTENT_SAFETY_API_KEY not set in environment")

        self.api_endpoint = s.azure_content_safety_endpoint
        self.api_key = s.azure_content_safety_api_key
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def resize_image_if_needed(self, image_path: str) -> str:
        """Ensure image meets Azure's size limits (max 2000x2000, min 50x50, max 4MB)."""
        with Image.open(image_path) as img:
            width, height = img.size

            # Resize if too large
            if width > 2000 or height > 2000:
                logger.debug(
                    f"Resizing image {image_path} from {img.size} to 2000x2000..."
                )
                new_size = (2000, 2000)
                img.thumbnail(new_size, Image.LANCZOS)
                resized_path = image_path.replace(".jpg", "_resized.jpg")
                img.save(resized_path, "JPEG", quality=85)
                return resized_path

        return image_path

    def analyze_image(self, image_path: str) -> dict:
        """
        Analyze an image for content safety using Azure Content Safety API.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with categoriesAnalysis containing severity scores
        """
        # Ensure the image meets Azure's size requirements
        image_path = self.resize_image_if_needed(image_path)

        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")

        endpoint = (
            f"{self.api_endpoint}/contentsafety/image:analyze?api-version=2024-09-01"
        )
        logger.debug(f"Calling Azure Content Safety API: {endpoint}")

        payload = {"image": {"content": base64_encoded}}
        response = requests.post(endpoint, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Azure API Error {response.status_code}: {response.text}")


class GraphicImageryClassifier:
    """Classifies PDF pages for graphic imagery content"""

    def __init__(self, violence_threshold: int = 4, settings: Settings | None = None):
        """
        Initialize the classifier.

        Args:
            violence_threshold: Severity threshold (0-6) for flagging violent content
            settings: Optional prap_core Settings instance
        """
        self.downloader = PdfDownloader(settings=settings)
        self.content_safety = AzureContentSafetyService(settings=settings)
        self.violence_threshold = violence_threshold

    def pdf_to_images(
        self, pdf_path: str, output_dir: str, page_numbers: Optional[List[int]] = None
    ) -> List[str]:
        """
        Convert PDF pages to JPG images using pdftoppm.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            page_numbers: Optional list of specific page numbers to convert (1-indexed).
                         If None, converts all pages.

        Returns:
            List of image file paths
        """
        base_name = Path(pdf_path).stem

        # Build pdftoppm command
        cmd = [
            "pdftoppm",
            "-jpeg",
            "-r",
            "300",  # 300 DPI resolution
        ]

        # If specific pages requested, add page range arguments
        if page_numbers:
            # pdftoppm can convert specific pages using -f (first) and -l (last)
            # For multiple non-contiguous pages, we need to call it multiple times
            # or convert all and filter. For efficiency, we'll convert only requested pages
            # by calling pdftoppm multiple times for each page
            image_files = []
            for page_num in sorted(page_numbers):
                page_cmd = cmd + [
                    "-f",
                    str(page_num),
                    "-l",
                    str(page_num),
                    pdf_path,
                    os.path.join(output_dir, f"{base_name}-page-"),
                ]

                subprocess_result = subprocess.run(
                    page_cmd,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )

                if subprocess_result.returncode != 0:
                    error_msg = subprocess_result.stderr.decode("utf-8")
                    logger.error(
                        f"Error converting page {page_num} to image: {error_msg}"
                    )
                    raise Exception(
                        f"PDF conversion failed for page {page_num}: {error_msg}"
                    )

            # Get list of generated image files
            image_files = sorted(
                [
                    os.path.join(output_dir, f)
                    for f in os.listdir(output_dir)
                    if f.startswith(f"{base_name}-page-") and f.endswith(".jpg")
                ]
            )
        else:
            # Convert all pages
            cmd.extend([pdf_path, os.path.join(output_dir, f"{base_name}-page-")])

            subprocess_result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )

            if subprocess_result.returncode != 0:
                error_msg = subprocess_result.stderr.decode("utf-8")
                logger.error(f"Error converting PDF to images: {error_msg}")
                raise Exception(f"PDF conversion failed: {error_msg}")

            # Get list of generated image files
            image_files = sorted(
                [
                    os.path.join(output_dir, f)
                    for f in os.listdir(output_dir)
                    if f.startswith(f"{base_name}-page-") and f.endswith(".jpg")
                ]
            )

        logger.debug(f"Converted PDF to {len(image_files)} images")
        return image_files

    def analyze_page(self, image_path: str) -> Dict:
        """
        Analyze a single page image for graphic content.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with violence_score and other severity scores
        """
        try:
            result = self.content_safety.analyze_image(image_path)

            # Extract scores from the result
            scores = {
                "violence_score": None,
                "self_harm_score": None,
                "hate_score": None,
                "sexual_score": None,
            }

            for category_analysis in result.get("categoriesAnalysis", []):
                category = category_analysis["category"]
                severity = category_analysis["severity"]

                if category == "Violence":
                    scores["violence_score"] = severity
                elif category == "SelfHarm":
                    scores["self_harm_score"] = severity
                elif category == "Hate":
                    scores["hate_score"] = severity
                elif category == "Sexual":
                    scores["sexual_score"] = severity

            return scores

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                "violence_score": None,
                "self_harm_score": None,
                "hate_score": None,
                "sexual_score": None,
                "error": str(e),
            }

    def _analyze_image_with_metadata(self, image_path: str) -> tuple:
        """
        Analyze a single image and return page number with scores.
        Helper function for parallel processing.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (page_number, scores_dict)
        """
        # Extract page number from filename (e.g., "sha1-page-1.jpg" -> 1)
        filename = Path(image_path).stem
        page_number = int(filename.split("-")[-1])

        logger.debug(f"Analyzing page {page_number}")
        scores = self.analyze_page(image_path)

        return page_number, scores

    def classify_pdf(
        self, sha1: str, page_numbers: Optional[List[int]] = None
    ) -> Dict:
        """
        Classify pages in a PDF for graphic imagery.

        Args:
            sha1: SHA1 hash of the PDF file
            page_numbers: Optional list of specific page numbers to analyze (1-indexed).
                         If None, analyzes all pages.

        Returns:
            Dict with:
                - sha1: The file SHA1
                - pages_with_graphic_imagery: List of page numbers with graphic content
                - all_pages_scores: Dict mapping page numbers to their scores
                - success: Boolean indicating if processing was successful
                - error: Error message if processing failed
        """
        result = {
            "sha1": sha1,
            "pages_with_graphic_imagery": [],
            "all_pages_scores": {},
            "success": False,
            "error": None,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download PDF
                logger.info(f"Downloading PDF {sha1}")
                pdf_path = self.downloader.download_to_temp(sha1, temp_dir)

                if not pdf_path:
                    result["error"] = "Failed to download PDF"
                    return result

                # Convert to images
                if page_numbers:
                    logger.info(
                        f"Converting {len(page_numbers)} specific pages of PDF {sha1} to images"
                    )
                else:
                    logger.info(f"Converting PDF {sha1} to images")
                image_files = self.pdf_to_images(pdf_path, temp_dir, page_numbers)

                if not image_files:
                    result["error"] = "Failed to convert PDF to images"
                    return result

                # Analyze each page in parallel
                logger.info(f"Analyzing {len(image_files)} pages for graphic imagery in parallel")

                # Use up to 10 workers for parallel processing
                max_workers = 7

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_image = {
                        executor.submit(self._analyze_image_with_metadata, image_file): image_file
                        for image_file in image_files
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_image):
                        image_file = future_to_image[future]
                        try:
                            page_number, scores = future.result()
                            result["all_pages_scores"][page_number] = scores

                            # Check if page has graphic imagery based on violence threshold
                            if scores.get("violence_score") is not None:
                                if scores["violence_score"] >= self.violence_threshold:
                                    result["pages_with_graphic_imagery"].append(page_number)
                                    logger.info(
                                        f"Page {page_number}: Graphic imagery detected (violence score: {scores['violence_score']})"
                                    )
                        except Exception as e:
                            logger.error(f"Failed to analyze {image_file}: {str(e)}")

                result["success"] = True
                logger.info(
                    f"Successfully classified PDF {sha1}: {len(result['pages_with_graphic_imagery'])} pages with graphic imagery"
                )

            except Exception as e:
                logger.error(f"Error classifying PDF {sha1}: {e}", exc_info=True)
                result["error"] = str(e)

        return result


def classify_pdf_for_graphic_imagery(
    sha1: str,
    violence_threshold: int = 4,
    page_numbers: Optional[List[int]] = None,
    settings: Settings | None = None,
) -> Dict:
    """
    Convenience function to classify a PDF for graphic imagery.

    Args:
        sha1: SHA1 hash of the PDF file
        violence_threshold: Severity threshold (0-6) for flagging violent content
        page_numbers: Optional list of specific page numbers to analyze (1-indexed).
                     If None, analyzes all pages.
        settings: Optional prap_core Settings instance

    Returns:
        Dict with classification results
    """
    classifier = GraphicImageryClassifier(
        violence_threshold=violence_threshold, settings=settings
    )
    return classifier.classify_pdf(sha1, page_numbers=page_numbers)
