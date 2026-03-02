"""
ingestion/__init__.py
─────────────────────
Package marker – makes `ingestion` a Python package and re-exports the
most-used symbols for convenience.
"""

from .document_loader import load_document, load_document_from_path
from .text_splitter import TextChunk, split_text

__all__ = [
    "load_document",
    "load_document_from_path",
    "split_text",
    "TextChunk",
]
