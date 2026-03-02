# Document Q&A Assistant – Comprehensive Project Documentation

## 1. Introduction
This project is an advanced Retrieval-Augmented Generation (RAG) system built to allow users to ask natural language questions about uploaded documents (PDF and DOCX). The application consists of a backend API server built with FastAPI and a modern frontend interface built with Gradio. Use this document to understand the underlying architecture, tool selection, and technical trade-offs.

## 2. Architecture & Components
The system is divided into clearly separated modules to ensure maintainability and scalability.

- **Frontend (Gradio 6.8+):** A clean, two-panel user interface providing an elegant user experience.
- **Backend (FastAPI):** A high-performance REST API handling file uploads, text parsing, embeddings, and chat interaction.
- **Vector Database (FAISS):** High-speed local similarity search engine.
- **Embeddings Pipeline (Sentence Transformers):** Local embedding generation model.
- **Language Model (OpenRouter/GPT-4o-mini):** Remote LLM integration for response synthesis.

## 3. Technology Preferences & Rationale

### Why FastAPI?
We chose FastAPI over Flask or Django because:
1. **Asynchronous by Default:** Essential for handling multiple user requests simultaneously—like generating embeddings or calling external APIs (OpenRouter).
2. **Speed:** It is one of the fastest Python frameworks available, sitting closely behind NodeJS and Go implementations.
3. **Automatic Documentation:** FastAPI automatically generates Swagger UI (`/docs`), which makes testing the separated backend infinitely easier.

### Why Gradio over React/Vue?
For a backend-heavy Python ML application, building a full Node.js frontend (React/Vue) adds massive overhead. Gradio allows us to build a rich, interactive, reactive UI completely in Python using simple event handlers and direct backend bridging. The recent update to Gradio 6.0+ provides enough CSS flexibility to create a premium visual experience without touching Javascript.

### Why FAISS over ChromaDB?
- **Stability Issues:** ChromaDB has known incompatibility issues with newer Python versions (specifically 3.12+ and Python-Pydantic V2 conflicts). 
- **Lightweight & High-Speed:** FAISS (Facebook AI Similarity Search) is extremely lean. It relies on a local index file on disk rather than spinning up a sqlite/telemetry instance like Chroma DB does. For stateless API ingestion, FAISS is vastly superior and robust.

### Why OpenRouter & GPT-4o-mini?
- **Cost Efficiency:** `gpt-4o-mini` is extremely fast and inexpensive, making it ideal for high-volume RAG tasks.
- **Flexibility:** Using OpenRouter acts as an abstraction layer. If OpenAI goes down, or if the user wants to switch to Anthropic Claude 3 Haiku or Meta Llama 3, they only need to change the model identifier string in the environment variable. No API code needs rewriting.

### Why SentenceTransformers `all-MiniLM-L6-v2`?
Instead of paying OpenAI to generate embeddings for every chunk of text, we run `all-MiniLM-L6-v2` locally via HuggingFace Sentence Transformers. This model is small (~80MB), loads instantly into memory, and is highly optimized for semantic search relevance without racking up API bills.

## 4. System Workflow Breakdown

### A. Document Ingestion (`ingestion/document_loader.py` & `text_splitter.py`)
1. **Parsing:** When a user uploads a PDF or DOCX file, the backend saves it temporarily to `/tmp_uploads`.
2. **Extraction:** Python libraries `PyMuPDF (fitz)` for PDFs and `python-docx` for Word files extract raw text.
3. **Chunking:** The raw text is passed into `RecursiveCharacterTextSplitter`. Text is split into chunks of ~1500 characters with an overlap of 200 characters. Overlaps ensure that context isn't lost if a sentence is split exactly on a chunk boundary.

### B. Vector Storage (`vector_store/faiss_store.py`)
1. **Embedding:** The 1500-char chunks are fed into the `SentenceManager`, creating high-dimensional mathematical vectors representing the semantic meaning.
2. **Indexing:** FAISS stores these vectors in a local `.faiss` file along with metadata tracking which source page the text came from.

### C. Chat & Retrieval (`retrieval/qa_chain.py`)
1. **Question Refinement:** When a user asks a follow-up question (e.g., "What did they say about the penalty?"), a `CONDENSE_PROMPT` sends chat history to the LLM to rewrite it into a standalone question (e.g., "What penalty is outlined in the contract?").
2. **Similarity Search:** The standalone question is mathematically vectorized. FAISS searches the database for the top 5 most similar text chunks.
3. **Synthesis:** The retrieved chunks and the user's question are pushed into a strict `SYSTEM_PROMPT`. The LLM synthesizes an answer using *only* those provided chunks.

## 5. Deployment Considerations
The project includes a `Dockerfile` and a `docker-compose.yml` file. This is crucial for distributing the system. Users don't need to manually configure virtual environments or `apt-get` system dependencies. By running `docker-compose up -d`, Docker builds an isolated Linux container, installs all necessary libraries from `requirements.txt`, and safely maps ports 7860 (UI) and 8000 (API) to the host machine.
