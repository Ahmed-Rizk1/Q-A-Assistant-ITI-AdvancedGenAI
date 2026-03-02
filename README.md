# 🏛️ Smart Contract Summary & Q&A Assistant

> **Upload any contract PDF or DOCX → Ask questions in natural language → Get cited, AI-powered answers.**

Built with **FastAPI · LangChain · ChromaDB · SentenceTransformers · Gradio**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-format upload** | PDF (PyMuPDF + pdfplumber fallback) and DOCX |
| 🧩 **Smart chunking** | Recursive text splitter with paragraph-aware overlapping |
| 🔍 **Semantic retrieval** | ChromaDB + `all-MiniLM-L6-v2` local embeddings |
| 💬 **Conversational Q&A** | Follow-up questions with chat memory |
| 📎 **Source citations** | Every answer cites the exact page and passage |
| 📋 **Auto-summarisation** | Map-reduce summary with legal section headings |
| 🤖 **Flexible LLM** | OpenAI GPT (recommended) or local HuggingFace fallback |
| 🌐 **Clean UI** | Gradio dark-mode interface with example prompts |

---

## � Application Screenshots

### 1. Document Upload and Processing
![Document Upload](pdf_to_test/images/Screenshot%20(97).png)

### 2. Intelligent Q&A
![Chat Interface](pdf_to_test/images/Screenshot%20(99).png)

### 3. Automatic Document Summarization
![Document Summary](pdf_to_test/images/Screenshot%20(103).png)

---

## �🗂️ Project Structure

```
RAGForITI/
│
├── app.py                    # ← Entry point (starts both servers)
├── requirements.txt          # All dependencies
├── .env.example              # Template for environment variables
│
├── ingestion/                # Document parsing & chunking
│   ├── document_loader.py    #   PDF (PyMuPDF + pdfplumber) & DOCX
│   └── text_splitter.py      #   LangChain RecursiveCharacterTextSplitter
│
├── vector_store/             # Embedding & vector DB
│   ├── embedder.py           #   SentenceTransformers (local, no API key)
│   └── chroma_store.py       #   ChromaDB adapter with LangChain interface
│
├── retrieval/                # RAG chain & summariser
│   ├── qa_chain.py           #   LCEL chain with memory & citations
│   └── summarizer.py         #   One-shot / Map-reduce summarisation
│
├── backend/                  # REST API
│   └── api.py                #   FastAPI: /upload /ask /summarize /reset
│
├── frontend/                 # Web UI
│   └── gradio_app.py         #   Gradio Blocks (dark glassmorphism theme)
│
└── tests/                    # Pytest test suite
    ├── test_ingestion.py
    └── test_api.py
```

---

## ⚡ Quick Start (Windows PowerShell)

### 1 · Clone or download the project

```powershell
cd C:\Users\Pro\Downloads\RAGForITI
```

### 2 · Create & activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Tip for PowerShell execution policy errors:**
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 3 · Install dependencies

```powershell
pip install -r requirements.txt
```

> First run downloads the SentenceTransformers model (~80 MB). Subsequent runs are instant.

### 4 · Configure your API key

```powershell
copy .env.example .env
```

Open `.env` in any text editor and set:

```
OPENAI_API_KEY=sk-your-real-key-here
```

> **No OpenAI key?** The system falls back to a tiny local HuggingFace model automatically (slower, less accurate). Set `HF_MODEL=facebook/opt-125m` or any other model in `.env`.

### 5 · Run the application

```powershell
python app.py
```

The terminal will show:

```
╔══════════════════════════════════════════════════════════╗
║   🏛️  Smart Contract Summary & Q&A Assistant            ║
║   Backend  →  http://127.0.0.1:8000   (FastAPI)         ║
║   Frontend →  http://127.0.0.1:7860   (Gradio)          ║
╚══════════════════════════════════════════════════════════╝
```

Open **http://127.0.0.1:7860** in your browser.

---

## 🖥️ Usage Guide

### Upload Tab

1. Click the upload area and select a `.pdf` or `.docx` contract.
2. Click **⚡ Process Document** – the file is parsed, chunked, embedded, and stored.
3. Optionally click **✨ Generate Summary** to get a structured Markdown summary.

### Chat Tab

- Type any question in the text box and press **Enter** or click **Send →**.
- Click one of the **Example Questions** buttons for instant prompts.
- Each answer includes **📎 Source passages** showing exactly which page was referenced.
- Click **🗑️ Clear Chat** to reset the conversation memory.
- Click **🔄 New Document** to upload a different contract.

---

## 🔧 Configuration Reference

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Model to use (`gpt-4o` for best results) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores data |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved per question |
| `GRADIO_SERVER_PORT` | `7860` | Gradio UI port |
| `BACKEND_URL` | `http://127.0.0.1:8000` | FastAPI backend URL |
| `HF_MODEL` | `facebook/opt-125m` | HuggingFace fallback model |

---

## 🧪 Running Tests

```powershell
pytest tests/ -v
```

Tests use mocks – they do **not** require OpenAI keys or a running ChromaDB.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  User (Browser)                                             │
│       │                                                     │
│       ▼                                                     │
│  Gradio UI (port 7860)          frontend/gradio_app.py      │
│       │  HTTP REST calls                                    │
│       ▼                                                     │
│  FastAPI API (port 8000)        backend/api.py              │
│       │                                                     │
│  ┌────┴────────────────────────────────────────┐           │
│  │                                             │           │
│  ▼                                             ▼           │
│  Ingestion Pipeline              QA Chain (RAG)            │
│  ├── document_loader.py          retrieval/qa_chain.py     │
│  └── text_splitter.py               │                      │
│       │                             │ LCEL                 │
│       ▼                             ▼                      │
│  ChromaDB (local)           OpenAI / HuggingFace LLM       │
│  vector_store/chroma_store.py                              │
│       ▲                                                     │
│       │ SentenceTransformers                               │
│       │ (all-MiniLM-L6-v2)                                 │
│       │ vector_store/embedder.py                           │
└─────────────────────────────────────────────────────────────┘
```

### RAG Flow (per question)

1. **User question** → Gradio → POST `/ask`
2. If follow-up: **Condense** question using chat history (LLM)
3. **Embed** the (condensed) question → SentenceTransformers
4. **Retrieve** top-K chunks from ChromaDB (cosine similarity)
5. **Format** context with page citations
6. **LLM** generates answer grounded in context
7. Answer + sources returned to Gradio → displayed in chat

---

## 🔒 Security Notes

- The `.env` file is **never committed** to Git (add `.env` to `.gitignore`).
- The app runs locally by default (no public exposure).
- To deploy in production: add authentication, rate limiting, and a persistent session store.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `langchain` | LLM orchestration (chains, memory, prompts) |
| `chromadb` | Local persistent vector database |
| `sentence-transformers` | Local embedding model (no API key) |
| `PyMuPDF` | Fast PDF text extraction |
| `pdfplumber` | Fallback PDF parser (better for tables) |
| `python-docx` | DOCX support |
| `gradio` | Browser UI |
| `openai` | OpenAI GPT integration |
| `python-dotenv` | `.env` file loading |

---

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Run tests: `pytest tests/ -v`
4. Open a PR with a clear description.

---

## 📜 License

MIT – free to use, modify, and distribute.
