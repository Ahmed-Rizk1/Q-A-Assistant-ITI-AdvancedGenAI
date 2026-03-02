# Personal Developer Guide — DocBot AI

Use this cheat sheet to quickly answer any technical questions about exactly how this project works under the hood. 

## The TL;DR Pitch
"This is a decoupled RAG application. I used FastAPI for a highly scalable backend API endpoint, and wrapped it with a modern Gradio web frontend. For the core logic, I'm using LangChain to manage pure FAISS for local lightweight vector indexing, SentenceTransformers for free local embeddings, and OpenRouter to bridge out to GPT-4o-mini for fast language synthesis."

---

## 🏗️ 1. Project Structure & Why I Built It This Way

If someone asks you: *"Why didn't you put everything in one `app.py` file?"*
**Your Answer:** "I separated concerns so the backend can scale independently. `api.py` handles the routes. The core logic is split into `ingestion` (handling parsing/chunking) and `retrieval` (handling database search and LLM prompts). `gradio_app.py` is purely visual. This means if we ever want to build a React native app or iOS app, our backend is entirely ready—we don't need to decouple anything."

### Key Files:
- **`app.py`**: The main runner. Bootstraps both Uvicorn (FastAPI) on thread 1 and Gradio on thread 2.
- **`backend/api.py`**: Contains the REST endpoints (`/upload`, `/ask`, `/summarize`, `/reset`).
- **`ingestion/document_loader.py`**: Where text extraction happens (via `fitz` for PDF and `docx` for Word).
- **`ingestion/text_splitter.py`**: Configures how big the document chunks are (1500 chars).
- **`retrieval/qa_chain.py`**: This is the "Brain". Holds the System Prompts and constructs the conversational retrieval chain.
- **`vector_store/faiss_store.py`**: The exact class handling saving/loading mathematical vectors to the local disk.

---

## 🧠 2. The Model Pipeline (How stuff flows)

If someone asks you: *"How does the document actually get turned into answers?"*
**Your Answer:** 
1. The user uploads a file via the Gradio UI.
2. The UI sends a POST request to the `/upload` endpoint in `api.py`.
3. The uploaded bytes are saved to a `tmp_uploads` folder.
4. `document_loader.py` parses the text into one giant string.
5. Langchain's `RecursiveCharacterTextSplitter` breaks that string into chunks.
6. Local `all-MiniLM-L6-v2` embedding model turns those chunks into numbers.
7. FAISS saves the numbers locally to a `./vector_store_data/` folder.
8. When the user asks a question, we embed their question, measure the distance between the question-vector and document-vectors using FAISS, grab the top 4/5 matches, and hand them to GPT-4o-mini through OpenRouter with strict instructions to *only* answer using that text!

---

## 🛠️ 3. Troubleshooting & "Gotchas"

**If the chat bubbles are missing or the UI looks broke:**
- Gradio versions past v5 handle CSS differently. The classes update dynamically. We avoid aggressive custom CSS targeting `.user` or `.bot` because it crushes Gradio's internal flexbox layout.

**If FAISS throws a typing error (`TypeError: Chatbot.__init__() got an unexpected keyword...`):**
- Langchain updated their core embeddings class. FAISS in Langchain expects a child of `Embeddings`. We wrote a custom adapter `_LangChainEmbeddingAdapter` in `faiss_store.py` to fix this so we don't need OpenAI's paid LLM for embeddings.

**If the LLM says "Sorry, I can't find that":**
- It means the sentence similarity search failed to pull the right chunk. FAISS didn't think the question was semantically similar to the answer paragraph.

## 🐳 4. Docker Deployment
If they ask: *"Is this containerized?"*
**Your Answer:** Yes! We built a standard `Dockerfile` using `python:3.11-slim`. I also included a `docker-compose.yml` that correctly wires up the dual-port system (7860 and 8000) and binds the environment variables like the OpenRouter API key securely. It's ready for any cloud virtual machine like AWS EC2 or DigitalOcean.
