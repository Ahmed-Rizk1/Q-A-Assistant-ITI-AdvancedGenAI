"""
backend/api.py
──────────────
FastAPI backend that exposes the RAG pipeline as REST endpoints.

Endpoints
─────────
POST /upload          – Upload a PDF/DOCX, parse & ingest into ChromaDB.
POST /ask             – Ask a question about the currently loaded document.
POST /summarize       – Generate a structured summary of the loaded document.
GET  /health          – Simple liveness probe.
DELETE /reset         – Clear conversation memory + loaded document state.

The server holds a simple in-memory session state (one document at a time).
For multi-user production use, state should be moved to a database/cache.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion import load_document, split_text
from retrieval import QAChain, summarize_document
from vector_store import ContractVectorStore

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Contract Q&A API",
    description="Upload contracts and ask questions using RAG.",
    version="1.0.0",
)

# Allow the Gradio frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session (single-user mode) ─────────────────────────────────────

class _Session:
    """Holds the currently loaded document and QA chain."""
    full_text: Optional[str] = None
    filename: Optional[str] = None
    vector_store: Optional[ContractVectorStore] = None
    qa_chain: Optional[QAChain] = None

SESSION = _Session()


# ─── Request / Response models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]   # list of {source, page, excerpt}

class SummaryResponse(BaseModel):
    summary: str

class StatusResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    chunks: Optional[int] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=StatusResponse, tags=["System"])
def health() -> StatusResponse:
    """Liveness probe used by the Gradio frontend to check if the API is up."""
    return StatusResponse(
        status="ok",
        filename=SESSION.filename,
    )


@app.post("/upload", response_model=StatusResponse, tags=["Document"])
async def upload_document(file: UploadFile = File(...)) -> StatusResponse:
    """
    Upload a PDF or DOCX file.

    Steps:
      1. Read file bytes from the multipart form.
      2. Extract text using the appropriate parser.
      3. Split text into overlapping chunks.
      4. Embed chunks and store in ChromaDB.
      5. Initialise a fresh QA chain linked to the new vector store.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Use .pdf or .docx.",
        )

    logger.info("Receiving upload: %s", file.filename)
    file_bytes = await file.read()

    # ── Step 1: Parse ─────────────────────────────────────────────────────
    try:
        full_text = load_document(file_bytes, file.filename)
    except Exception as exc:
        logger.exception("Document parse error")
        raise HTTPException(status_code=422, detail=f"Parse error: {exc}") from exc

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────
    chunk_size = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

    chunks = split_text(
        full_text,
        source_filename=file.filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ── Step 3: Embed + Store ─────────────────────────────────────────────
    collection_name = Path(file.filename).stem   # e.g. "contract_2024"
    vs = ContractVectorStore(collection_name=collection_name)
    vs.add_chunks(chunks)

    # ── Step 4: Build QA chain ────────────────────────────────────────────
    top_k = int(os.getenv("TOP_K", 5))
    retriever = vs.as_retriever(top_k=top_k)
    qa = QAChain(retriever=retriever)

    # ── Step 5: Persist in session ────────────────────────────────────────
    SESSION.full_text = full_text
    SESSION.filename = file.filename
    SESSION.vector_store = vs
    SESSION.qa_chain = qa

    logger.info("Document '%s' ingested – %d chunks.", file.filename, len(chunks))
    return StatusResponse(
        status="ingested",
        filename=file.filename,
        chunks=len(chunks),
    )


@app.post("/ask", response_model=AskResponse, tags=["QA"])
def ask_question(body: AskRequest) -> AskResponse:
    """
    Ask a natural-language question about the uploaded contract.

    Returns the LLM answer plus the source passages that were used.
    """
    if SESSION.qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No document is loaded. Please upload a file first.",
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answer, docs = SESSION.qa_chain.ask(question)
    except Exception as exc:
        logger.exception("QA error")
        raise HTTPException(status_code=500, detail=f"QA error: {exc}") from exc

    # Serialise source documents for the response
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "?"),
            "excerpt": d.page_content[:300],   # truncate for readability
        }
        for d in docs
    ]

    return AskResponse(answer=answer, sources=sources)


@app.post("/summarize", response_model=SummaryResponse, tags=["Document"])
def summarize() -> SummaryResponse:
    """
    Generate a structured Markdown summary of the uploaded document.
    This endpoint may take 10-30 seconds for longer contracts.
    """
    if SESSION.full_text is None:
        raise HTTPException(
            status_code=400,
            detail="No document is loaded. Please upload a file first.",
        )

    try:
        summary = summarize_document(SESSION.full_text)
    except Exception as exc:
        logger.exception("Summarisation error")
        raise HTTPException(status_code=500, detail=f"Summary error: {exc}") from exc

    return SummaryResponse(summary=summary)


@app.delete("/reset", response_model=StatusResponse, tags=["System"])
def reset_session() -> StatusResponse:
    """
    Clear the current session (conversation history + loaded document).
    Call this before uploading a new document to start fresh.
    """
    if SESSION.qa_chain:
        SESSION.qa_chain.reset_memory()
    SESSION.full_text = None
    SESSION.filename = None
    SESSION.vector_store = None
    SESSION.qa_chain = None
    logger.info("Session reset.")
    return StatusResponse(status="reset")
