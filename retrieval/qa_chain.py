"""
retrieval/qa_chain.py
─────────────────────
Builds and runs the Retrieval-Augmented Generation (RAG) chain that:

  1. Retrieves the top-K most relevant document chunks from ChromaDB.
  2. Formats them into a prompt with citations.
  3. Calls the LLM (OpenAI by default, HuggingFace optional).
  4. Returns the answer plus the source passages used.

The chain keeps a conversation memory so follow-up questions can reference
earlier answers without repeating context.

LangChain Expression Language (LCEL) is used throughout for composability.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

logger = logging.getLogger(__name__)


# ─── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are DocBot AI, a friendly and expert AI analyst specialising in document analysis.\n\n"
    "GREETING & SMALL TALK:\n"
    "- If the user says hi, hello, hey, or any greeting, respond warmly and briefly introduce yourself."
    " Example: \"Hi there! I'm DocBot AI. Upload a document and ask me anything about it "
    "\u2014 obligations, payment terms, deadlines, risks, and more!\"\n"
    "- For off-topic questions, give a short helpful reply and guide them back to the contract.\n\n"
    "DOCUMENT QUESTIONS (when context is provided below):\n"
    "1. Answer using ONLY the provided context. Be concise \u2014 max 5 bullet points or 3 short paragraphs.\n"
    "2. Use Markdown: bullet lists, **bold** key terms, short headers where helpful.\n"
    "3. Bold important numbers, dates, amounts: **$50,000**, **30 days**.\n"
    "4. If the context truly does not contain the answer, say: "
    "\"I couldn't find that specific information in the uploaded document.\"\n"
    "5. Never repeat the question back.\n\n"
    "Context from the uploaded document (empty if no document uploaded yet):\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("human",
     "Given the conversation above, rewrite the follow-up question as a standalone "
     "question that can be understood without the chat history.  "
     "Return ONLY the rewritten question."),
])


# ─── LLM factory ─────────────────────────────────────────────────────────────

# OpenRouter base URL – drop-in OpenAI-compatible endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_llm(temperature: float = 0.0):
    """
    Instantiate the language model via OpenRouter.

    OpenRouter is 100 % OpenAI-compatible: we just point ChatOpenAI at
    https://openrouter.ai/api/v1 and swap the API key.

    Priority order:
      1. OpenRouter  (OPENROUTER_API_KEY  – preferred, cheap small models)
      2. HuggingFace (fully local, no API key, slower)

    Env vars:
      OPENROUTER_API_KEY  – your sk-or-v1-... key
      OPENROUTER_MODEL    – model slug (default: openai/gpt-4o-mini)
    """
    from langchain_openai import ChatOpenAI

    api_key    = os.getenv("OPENROUTER_API_KEY", "")
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if api_key and not api_key.startswith("sk-or-v1-..."):
        # ── OpenRouter path ──────────────────────────────────────────────────
        logger.info("Using OpenRouter LLM: %s", model_name)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,         # OpenRouter key passed here
            openai_api_base=OPENROUTER_BASE_URL,  # redirect to OpenRouter
            max_tokens=1024,
            default_headers={
                # OpenRouter strongly recommends these two headers for
                # attribution and routing; both are optional but good practice.
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "Document Q&A Assistant",
            },
        )

    # ── Local HuggingFace fallback ───────────────────────────────────────────
    logger.warning(
        "OPENROUTER_API_KEY not set or is a placeholder. "
        "Falling back to local HuggingFace model (this is slower)."
    )
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        hf_model = os.getenv("HF_MODEL", "facebook/opt-125m")  # tiny demo model
        logger.info("Loading HuggingFace model: %s", hf_model)

        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModelForCausalLM.from_pretrained(hf_model)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
        )
        return HuggingFacePipeline(pipeline=pipe)

    except ImportError:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set and 'transformers' is not installed.\n"
            "Either set OPENROUTER_API_KEY in your .env file, or run:\n"
            "  pip install transformers"
        )


# ─── Main QA chain ────────────────────────────────────────────────────────────

class QAChain:
    """
    Stateful question-answering chain with conversation memory.

    Parameters
    ----------
    retriever   : LangChain retriever (from ``ContractVectorStore.as_retriever``).
    temperature : LLM temperature – 0 for deterministic, higher for creative.
    """

    def __init__(self, retriever, temperature: float = 0.0):
        self.retriever = retriever
        self.llm = _build_llm(temperature)
        self.chat_history: List = []           # list of HumanMessage / AIMessage
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Chain construction (LCEL)
    # ------------------------------------------------------------------

    def _build_chain(self):
        """
        Build the full RAG chain using LangChain Expression Language.

        Flow:
          question + history
              → condense to standalone question
              → retrieve relevant chunks
              → format context
              → call LLM
              → parse answer string
        """

        # Step 1: Condense follow-up questions into standalone queries
        condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()

        def maybe_condense(inputs: dict) -> str:
            """Only run condensation if there IS chat history."""
            if inputs["chat_history"]:
                return condense_chain.invoke(inputs)
            return inputs["question"]

        # Step 2: Retrieve docs using the (possibly condensed) question
        def retrieve_docs(standalone_question: str) -> List[Document]:
            return self.retriever.invoke(standalone_question)

        # Step 3: Format retrieved docs into a single context string
        def format_context(docs: List[Document]) -> str:
            parts = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                parts.append(
                    f"[Excerpt {i} – {source}, Page {page}]\n{doc.page_content}"
                )
            return "\n\n---\n\n".join(parts)

        # Step 4: Assemble inputs for the QA prompt
        def build_qa_inputs(inputs: dict) -> dict:
            standalone_q = maybe_condense(inputs)
            docs = retrieve_docs(standalone_q)
            context = format_context(docs)
            # Attach retrieved docs so we can return them as citations
            inputs["_docs"] = docs
            inputs["context"] = context
            inputs["standalone_question"] = standalone_q
            return inputs

        # Step 5: Full chain
        chain = (
            RunnableLambda(build_qa_inputs)
            | RunnableParallel(
                answer=QA_PROMPT | self.llm | StrOutputParser(),
                docs=RunnableLambda(lambda x: x["_docs"]),
            )
        )
        return chain

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """
        Ask a question and return ``(answer, source_documents)``.

        Parameters
        ----------
        question : Natural-language question about the document.

        Returns
        -------
        answer   : LLM-generated response with inline citations.
        sources  : List of LangChain ``Document`` objects used as context.
        """
        inputs = {
            "question": question,
            "chat_history": self.chat_history,
        }

        result = self._chain.invoke(inputs)
        answer: str = result["answer"]
        docs: List[Document] = result["docs"]

        # Persist this turn in memory for future follow-ups
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        logger.info("Q: %s | A (excerpt): %.80s…", question, answer)
        return answer, docs

    def reset_memory(self) -> None:
        """Clear conversation history (useful when switching documents)."""
        self.chat_history = []
        logger.info("Conversation history cleared.")
