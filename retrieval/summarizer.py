"""
retrieval/summarizer.py
────────────────────────
Generates a high-level summary of the uploaded document.

Two strategies are implemented based on document length:

  • SHORT documents (< 3,000 chars):  One-shot – send the full text to the LLM.
  • LONG documents (≥ 3,000 chars):  Map-reduce – summarise each chunk, then
                                      combine the summaries into a final summary.

Both paths return a plain string with section headings so the UI can render
it as Markdown.
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

_CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a AI analyst. Summarise the document section below in "
     "exactly 2-3 bullet points. Each bullet must be one concise sentence. "
     "Focus only on: obligations, rights, deadlines, and monetary values. "
     "Use **bold** for key terms, dates, and amounts. No preamble."),
    ("human", "{text}"),
])

_COMBINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior AI analyst. Combine the section summaries below "
     "into a final structured document summary using exactly these Markdown headings.\n"
     "Keep each section to 2-4 bullet points. Bold key terms, amounts, and dates. "
     "Be concise — no filler sentences.\n\n"
     "## 👥 Parties Involved\n"
     "## ✅ Key Obligations\n"
     "## 💰 Payment Terms\n"
     "## 📅 Duration & Termination\n"
     "## 📌 Important Clauses\n"
     "## ⚠️ Risk Factors"),
    ("human", "Section summaries:\n\n{summaries}"),
])

_DIRECT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior AI analyst. Provide a structured Markdown summary "
     "of the contract below using exactly these headings. "
     "Keep each section to 2-4 bullet points. Bold key terms, amounts, and dates. "
     "Be concise — no filler sentences.\n\n"
     "## 👥 Parties Involved\n"
     "## ✅ Key Obligations\n"
     "## 💰 Payment Terms\n"
     "## 📅 Duration & Termination\n"
     "## 📌 Important Clauses\n"
     "## ⚠️ Risk Factors"),
    ("human", "{text}"),
])

# Threshold in characters below which we use the one-shot strategy
_SHORT_DOC_THRESHOLD = 3_000


# ─── Main function ────────────────────────────────────────────────────────────

def summarize_document(full_text: str, llm=None) -> str:
    """
    Generate a structured Markdown summary of the contract text.

    Parameters
    ----------
    full_text : The complete text extracted from the uploaded file.
    llm       : LangChain LLM instance.  If None, one is created automatically.

    Returns
    -------
    str  – Markdown-formatted summary string.
    """
    if llm is None:
        from retrieval.qa_chain import _build_llm
        llm = _build_llm(temperature=0.2)  # slight creativity for summaries

    parser = StrOutputParser()

    if len(full_text) <= _SHORT_DOC_THRESHOLD:
        # ── One-shot strategy ────────────────────────────────────────────────
        logger.info("Using one-shot summary strategy (%d chars).", len(full_text))
        chain = _DIRECT_SUMMARY_PROMPT | llm | parser
        return chain.invoke({"text": full_text})

    # ── Map-Reduce strategy ──────────────────────────────────────────────────
    logger.info("Using map-reduce summary strategy (%d chars).", len(full_text))

    # Split text into ~2000-char windows for the map step
    chunk_windows = _split_into_windows(full_text, window_size=2_000)
    logger.info("Map step: summarising %d windows.", len(chunk_windows))

    chunk_chain = _CHUNK_SUMMARY_PROMPT | llm | parser
    chunk_summaries: List[str] = []

    for i, window in enumerate(chunk_windows, start=1):
        logger.debug("Summarising window %d/%d", i, len(chunk_windows))
        summary = chunk_chain.invoke({"text": window})
        chunk_summaries.append(f"### Section {i}\n{summary}")

    combined = "\n\n".join(chunk_summaries)

    logger.info("Reduce step: combining %d summaries.", len(chunk_summaries))
    combine_chain = _COMBINE_PROMPT | llm | parser
    return combine_chain.invoke({"summaries": combined})


# ─── Internal helper ──────────────────────────────────────────────────────────

def _split_into_windows(text: str, window_size: int = 2_000) -> List[str]:
    """
    Naively split *text* into non-overlapping windows of *window_size* chars.
    Tries to break at paragraph boundaries to preserve context.
    """
    paragraphs = text.split("\n\n")
    windows: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > window_size and current:
            windows.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        windows.append("\n\n".join(current))

    return windows or [text]
