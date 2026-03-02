"""
vector_store/faiss_store.py
────────────────────────────
FAISS-based vector store – a drop-in replacement for chroma_store.py.

Why FAISS over ChromaDB here?
  ChromaDB relies on Pydantic V1 internally, which is incompatible with
  Python 3.14.  FAISS is a pure C++/numpy library with no Pydantic
  dependency and works on every Python version.

Persistence strategy
────────────────────
FAISS doesn't have a built-in server or SQL backend.  We persist the
index + metadata manually as two files:
  <persist_dir>/<collection_name>/index.faiss   – the vector index
  <persist_dir>/<collection_name>/index.pkl     – document metadata

Both files are created by LangChain's FAISS.save_local() and loaded by
FAISS.load_local().
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.text_splitter import TextChunk
from vector_store.embedder import LocalEmbedder

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


class ContractVectorStore:
    """
    FAISS-backed vector store for a single uploaded document.

    API is identical to the old ChromaDB version so the rest of the
    codebase (backend/api.py, QAChain, tests) needs zero changes.

    Parameters
    ----------
    collection_name : Sanitised document name used as the sub-directory.
    persist_dir     : Root directory where index files are written.
    embedder        : LocalEmbedder instance (SentenceTransformers).
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedder: Optional[LocalEmbedder] = None,
    ):
        self.collection_name = _sanitize_name(collection_name)
        self.persist_dir = str(Path(persist_dir).resolve())
        self.index_path = str(Path(self.persist_dir) / self.collection_name)
        self.embedder = embedder or LocalEmbedder()
        self._store = None          # loaded lazily

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Embed *chunks* and persist a new FAISS index.
        Previous index for this collection is deleted first.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list – nothing to store.")
            return

        # Build LangChain Document list
        documents = [
            Document(page_content=c.text, metadata=c.metadata)
            for c in chunks
        ]

        # Remove stale index
        self._delete_existing_index()

        logger.info(
            "Embedding %d chunks and building FAISS index '%s'…",
            len(documents), self.collection_name,
        )

        # LangChain's FAISS.from_documents handles embedding + indexing
        from langchain_community.vectorstores import FAISS as LCFaiss

        self._store = LCFaiss.from_documents(
            documents=documents,
            embedding=_LangChainEmbeddingAdapter(self.embedder),
        )

        # Persist to disk
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        self._store.save_local(self.index_path)
        logger.info("FAISS index saved to '%s'.", self.index_path)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Return the *top_k* most semantically similar chunks for *query*."""
        store = self._get_or_load_store()
        return store.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = 5):
        """Return a LangChain VectorStoreRetriever compatible with LCEL chains."""
        store = self._get_or_load_store()
        return store.as_retriever(search_kwargs={"k": top_k})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_load_store(self):
        """Return the in-memory store or load it from disk."""
        if self._store is not None:
            return self._store

        from langchain_community.vectorstores import FAISS as LCFaiss

        if not Path(self.index_path).exists():
            raise RuntimeError(
                f"No FAISS index found at '{self.index_path}'. "
                "Please upload and process a document first."
            )

        logger.info("Loading FAISS index from '%s'…", self.index_path)
        self._store = LCFaiss.load_local(
            self.index_path,
            embeddings=_LangChainEmbeddingAdapter(self.embedder),
            allow_dangerous_deserialization=True,   # needed for pickle
        )
        return self._store

    def _delete_existing_index(self) -> None:
        """Remove the on-disk index directory if it already exists."""
        index_dir = Path(self.index_path)
        if index_dir.exists():
            logger.info("Removing existing index at '%s'.", self.index_path)
            shutil.rmtree(index_dir, ignore_errors=True)


# ─── LangChain embedding adapter ──────────────────────────────────────────────

class _LangChainEmbeddingAdapter(Embeddings):
    """
    Adapts LocalEmbedder to the LangChain Embeddings interface.

    Inheriting from Embeddings ensures LangChain's FAISS takes the
    `isinstance(embeddings, Embeddings)` branch and calls .embed_query()
    / .embed_documents() directly, avoiding any __call__ ambiguity.
    """

    def __init__(self, embedder: LocalEmbedder):
        self._embedder = embedder

    # LangChain standard interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_query(text).tolist()

    # Some LangChain code paths call the object directly as a function
    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """Replace non-alphanumeric characters to make a safe directory name."""
    import re
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()
    return clean[:63] or "default_collection"
