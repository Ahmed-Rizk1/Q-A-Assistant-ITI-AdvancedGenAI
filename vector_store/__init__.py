"""
vector_store/__init__.py
"""
from .faiss_store import ContractVectorStore
from .embedder import LocalEmbedder

__all__ = ["ContractVectorStore", "LocalEmbedder"]
