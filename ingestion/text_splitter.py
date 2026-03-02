"""
ingestion/text_splitter.py
──────────────────────────
Splits a large raw-text document into overlapping chunks that fit inside
the embedding model's context window.

Why overlap?
  Contract clauses often span paragraph boundaries.  Overlapping ensures
  that no sentence is silently cut in half and lost from retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single slice of document text sent to the embedding model.

    Attributes
    ----------
    text     : The chunk's raw text content.
    chunk_id : Zero-based index of this chunk within the document.
    metadata : Arbitrary key-value pairs (page number, source filename, …).
    """
    text: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


# ─── Splitter ─────────────────────────────────────────────────────────────────

def split_text(
    text: str,
    source_filename: str = "unknown",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[TextChunk]:
    """
    Split *text* into overlapping chunks suitable for embedding.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` which tries to split
    on paragraph boundaries (``\n\n``), then line breaks (``\n``), then
    sentences (``". "``), and finally individual characters as a last resort.
    This hierarchy keeps semantic units as intact as possible.

    Parameters
    ----------
    text            : Full document text (with optional [Page N] markers).
    source_filename : Name of the source file, stored in chunk metadata.
    chunk_size      : Target character count per chunk (default 800).
    chunk_overlap   : Number of characters shared between adjacent chunks.

    Returns
    -------
    List[TextChunk]
        Ordered list of text chunks, each with an id and metadata dict.
    """
    if not text.strip():
        logger.warning("split_text received empty text; returning empty list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        # Contract-aware separators: prefer section breaks before character limits
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,           # character-based (not token-based)
        is_separator_regex=False,
    )

    raw_chunks: list[str] = splitter.split_text(text)
    logger.info(
        "Split '%s' into %d chunks (size=%d, overlap=%d)",
        source_filename, len(raw_chunks), chunk_size, chunk_overlap,
    )

    chunks: list[TextChunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        # Try to extract a page number from the leading [Page N] marker
        page_num = _extract_page_number(chunk_text)
        chunks.append(TextChunk(
            text=chunk_text,
            chunk_id=idx,
            metadata={
                "source": source_filename,
                "chunk_id": idx,
                "page": page_num,
            },
        ))

    return chunks


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_page_number(text: str) -> int:
    """
    Try to parse a ``[Page N]`` marker from the beginning of a chunk.
    Returns the page number, or 0 if no marker is found.
    """
    import re
    match = re.search(r"\[Page (\d+)\]", text[:50])
    return int(match.group(1)) if match else 0
