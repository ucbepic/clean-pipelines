"""Local sentence-transformers embeddings used for Tier 3 pre-filtering.

Lives in this package because `prap_core` has no embeddings surface yet.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np

logger = logging.getLogger("prap.clustering.embeddings")

# ----------------------------------------------------------------------------
# Model setup
# ----------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings

_embedding_model = None


def _get_embedding_model():
    """Lazy load the embedding model (cached globally)."""
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")

    return _embedding_model


def embed_texts(texts: list[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """Generate normalized embeddings for a list of texts (shape ``(N, 384)``)."""
    if not texts:
        return np.array([])

    model = _get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


async def embed_texts_async(
    texts: list[str], batch_size: int = 32, max_length: int = 256
) -> np.ndarray:
    """Async wrapper for :func:`embed_texts` (runs in thread pool)."""
    loop = asyncio.get_event_loop()
    encode_fn = partial(embed_texts, batch_size=batch_size, max_length=max_length)
    return await loop.run_in_executor(None, encode_fn, texts)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Cosine similarity between two pre-normalized 384-dim embeddings."""
    return float(np.dot(embedding1, embedding2))
