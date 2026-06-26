"""Azure Blob Storage PDF downloader.

Credentials are sourced from prap_core.config.Settings.
"""

from __future__ import annotations

import logging
import os

from azure.storage.blob import BlobServiceClient
from prap_core.config import Settings

logger = logging.getLogger("prap.redactions.pdf_download")


class PdfDownloader:
    """Downloads PDFs from Azure Blob Storage using SHA1 hash"""

    def __init__(self, settings: Settings | None = None):
        s = settings or Settings()
        if not s.azure_storage_connection_string:
            raise ValueError("PRAP_AZURE_STORAGE_CONNECTION_STRING not set in environment")
        if not s.azure_storage_container:
            raise ValueError("PRAP_AZURE_STORAGE_CONTAINER not set in environment")

        self.service_client = BlobServiceClient.from_connection_string(
            s.azure_storage_connection_string
        )
        self.container_name = s.azure_storage_container
        logger.debug("Connected to Azure Blob Storage")

    def construct_blob_path(self, sha1: str) -> str:
        """Construct the blob path from SHA1 hash"""
        return f"{sha1[0:2]}/{sha1[2:4]}/{sha1[4:6]}/{sha1}"

    def download_pdf(self, sha1: str, output_path: str) -> bool:
        """
        Download a PDF from Azure Blob Storage to a local file.

        Args:
            sha1: SHA1 hash of the file
            output_path: Local path where the PDF should be saved

        Returns:
            True if download successful, False otherwise
        """
        try:
            blob_path = self.construct_blob_path(sha1)
            logger.debug(f"Downloading blob: {blob_path}")

            blob_client = self.service_client.get_blob_client(
                container=self.container_name, blob=blob_path
            )

            # Download the blob
            with open(output_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            logger.debug(f"Successfully downloaded {sha1} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {sha1}: {e}")
            return False

    def download_to_temp(self, sha1: str, temp_dir: str) -> str | None:
        """
        Download a PDF to a temporary directory.

        Args:
            sha1: SHA1 hash of the file
            temp_dir: Temporary directory path

        Returns:
            Path to downloaded file if successful, None otherwise
        """
        output_path = os.path.join(temp_dir, f"{sha1}.pdf")

        if self.download_pdf(sha1, output_path):
            return output_path
        return None
