"""
vector_store/embedder.py
────────────────────────
Thin wrapper around SentenceTransformers that converts a list of text
strings into a NumPy embedding matrix.

Why SentenceTransformers?
  • Runs entirely locally – no API keys, no internet required after download.
  • ``all-MiniLM-L6-v2`` is tiny (~80 MB) yet very competitive on RAG tasks.
  • If you have an OpenAI key, swap the class below for OpenAIEmbeddings.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ─── Model identifier ────────────────────────────────────────────────────────
# Change this to any model on HuggingFace Hub, e.g.:
#   "BAAI/bge-small-en-v1.5"  – a bit larger but more accurate
#   "paraphrase-multilingual-MiniLM-L12-v2"  – multilingual support
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_st_model(model_name: str):
    """
    Load (and cache) the SentenceTransformer model.
    The @lru_cache decorator ensures we only load it once per process.
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    logger.info("Loading SentenceTransformer model: %s", model_name)
    return SentenceTransformer(model_name)


class LocalEmbedder:
    """
    Generates dense vector embeddings using a local SentenceTransformer model.

    Usage
    -----
    >>> embedder = LocalEmbedder()
    >>> vectors = embedder.embed(["clause one text", "clause two text"])
    >>> vectors.shape   # (2, 384) for MiniLM-L6
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # loaded lazily on first call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model the first time we need it."""
        if self._model is None:
            self._model = _get_st_model(self.model_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of strings.

        Parameters
        ----------
        texts      : Strings to embed (typically text chunks from the splitter).
        batch_size : How many strings to process at once.  Smaller = lower RAM.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim)
        """
        self._ensure_loaded()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        logger.info("Embedding %d texts with model '%s'", len(texts), self.model_name)
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,  # cosine distance works better when normalized
        )
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (returns a 1-D array)."""
        return self.embed([query])[0]

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimension of this model."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
