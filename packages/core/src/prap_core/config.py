from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRAP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str | None = None
    llm_api_base: str | None = None
    llm_api_version: str | None = None

    embedding_model: str = "openai/text-embedding-3-large"

    ocr_backend: Literal["unstructured", "tesseract"] = "unstructured"

    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "prap")
    cache_enabled: bool = True
