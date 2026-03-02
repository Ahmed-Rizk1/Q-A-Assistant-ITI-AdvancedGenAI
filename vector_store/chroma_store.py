"""
vector_store/chroma_store.py
────────────────────────────
ChromaDB vector store adapter.

ChromaDB persists its data on disk so documents survive application restarts.
Each uploaded file gets its own *collection* (namespace) so different contracts
never pollute each other.

Architecture decision
─────────────────────
We wrap LangChain's ``Chroma`` class rather than calling ChromaDB directly.
This gives us free access to LangChain's retriever interface used in the
retrieval chain later.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from ingestion.text_splitter import TextChunk
from vector_store.embedder import LocalEmbedder

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
# Default persistence directory (can be overridden by env var)
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


class ContractVectorStore:
    """
    Manages a ChromaDB collection for a single uploaded document.

    Parameters
    ----------
    collection_name : Unique name for this document's collection.
                      Typically the sanitised filename without extension.
    persist_dir     : Directory where ChromaDB writes its SQLite files.
    embedder        : ``LocalEmbedder`` instance (uses SentenceTransformers).
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedder: Optional[LocalEmbedder] = None,
    ):
        self.collection_name = _sanitize_name(collection_name)
        self.persist_dir = str(Path(persist_dir).resolve())
        self.embedder = embedder or LocalEmbedder()
        self._store: Optional[Chroma] = None  # created lazily

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Embed *chunks* and store them in the ChromaDB collection.

        If the collection already exists (e.g. the same file was re-uploaded),
        it is deleted and recreated to keep the store in sync.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list – nothing to store.")
            return

        # Convert TextChunk → LangChain Document (standard interchange format)
        documents = [
            Document(page_content=c.text, metadata=c.metadata)
            for c in chunks
        ]

        # Delete existing collection to avoid stale data
        self._delete_existing_collection()

        logger.info(
            "Ingesting %d chunks into collection '%s' at '%s'",
            len(documents), self.collection_name, self.persist_dir,
        )

        # Build the Chroma store – this also computes embeddings and persists
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=_LangChainEmbeddingAdapter(self.embedder),
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        logger.info("Vector store ready with %d chunks.", len(documents))

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Return the *top_k* most semantically similar chunks for *query*.

        Parameters
        ----------
        query : Natural-language question from the user.
        top_k : Number of chunks to retrieve (default 5).
        """
        store = self._get_or_load_store()
        return store.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = 5):
        """
        Return a LangChain ``VectorStoreRetriever`` compatible with LCEL chains.
        """
        store = self._get_or_load_store()
        return store.as_retriever(search_kwargs={"k": top_k})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_load_store(self) -> Chroma:
        """Return cached store or load it from disk."""
        if self._store is not None:
            return self._store

        logger.info(
            "Loading existing Chroma collection '%s' from '%s'",
            self.collection_name, self.persist_dir,
        )
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=_LangChainEmbeddingAdapter(self.embedder),
            persist_directory=self.persist_dir,
        )
        return self._store

    def _delete_existing_collection(self) -> None:
        """Remove the collection from ChromaDB if it already exists."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            existing = [c.name for c in client.list_collections()]
            if self.collection_name in existing:
                logger.info("Deleting existing collection '%s'.", self.collection_name)
                client.delete_collection(self.collection_name)
        except Exception as exc:
            logger.warning("Could not delete existing collection: %s", exc)


# ─── LangChain embedding adapter ──────────────────────────────────────────────

class _LangChainEmbeddingAdapter:
    """
    Adapts our ``LocalEmbedder`` to the interface that LangChain's
    ``Chroma`` class expects (``embed_documents`` + ``embed_query``).
    """

    def __init__(self, embedder: LocalEmbedder):
        self._embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_query(text).tolist()


# ─── Utility ──────────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """
    ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens.
    We lowercase and replace forbidden characters with underscores.
    """
    import re
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()
    # Ensure it starts with a letter (ChromaDB requirement)
    if clean and clean[0].isdigit():
        clean = "doc_" + clean
    return clean[:63] or "default_collection"
