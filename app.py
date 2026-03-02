"""
app.py  –  Application entry-point
────────────────────────────────────
Starts both the FastAPI backend and the Gradio frontend as two parallel
threads so the user only needs to run ONE command:

    python app.py

How it works
────────────
1. A background daemon thread spins up uvicorn (FastAPI) on port 8000.
2. The main thread waits until the backend is accepting connections.
3. The Gradio UI is then launched on port 7860 (default).
4. Press Ctrl-C to stop everything.

Environment
───────────
Copy `.env.example` to `.env` and set OPENROUTER_API_KEY.
All settings can be overridden via environment variables.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

# ─── Load environment variables from .env ─────────────────────────────────────
from dotenv import load_dotenv

load_dotenv(override=False)   # does NOT override variables already set in shell

# ─── Project-root on the path ─────────────────────────────────────────────────
# Ensure modules in the project root (ingestion/, retrieval/, …) are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


# ─── Backend thread ───────────────────────────────────────────────────────────

def _run_backend() -> None:
    """Start the FastAPI server in a daemon thread."""
    import uvicorn
    from backend.api import app as fastapi_app

    port = int(os.getenv("BACKEND_PORT", 8000))
    logger.info("Starting FastAPI backend on http://0.0.0.0:%d", port)
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info",   # show startup errors clearly
    )


def _wait_for_backend(timeout: int = 30) -> bool:
    """
    Poll the backend /health endpoint until it responds or we time out.
    Returns True if the backend is up, False on timeout.
    """
    import urllib.request

    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    health_url = f"{backend_url}/health"
    deadline = time.time() + timeout

    logger.info("Waiting for backend at %s …", health_url)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    logger.info("Backend is up! ✓")
                    return True
        except Exception:
            pass
        time.sleep(1)

    logger.error("Backend did not start within %d seconds.", timeout)
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║   🏛️  Document Q&A Assistant            ║\n"
        "║                                                          ║\n"
        "║   Backend  →  http://127.0.0.1:8000   (FastAPI)         ║\n"
        "║   Frontend →  http://127.0.0.1:7860   (Gradio)          ║\n"
        "║                                                          ║\n"
        "║   Press Ctrl-C to stop.                                  ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )

    # Start backend in a daemon thread (dies when main thread exits)
    backend_thread = threading.Thread(target=_run_backend, daemon=True, name="backend")
    backend_thread.start()

    # Wait until the API is ready before opening the UI
    # 90 s gives enough headroom for the Pydantic/Python 3.14 slow-import path
    if not _wait_for_backend(timeout=90):
        logger.error("Exiting – backend failed to start.")
        sys.exit(1)

    # Launch the Gradio frontend (blocks until Ctrl-C)
    from frontend.gradio_app import build_ui

    gradio_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    gradio_host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

    demo, css = build_ui()
    demo.launch(
        server_name=gradio_host,
        server_port=gradio_port,
        share=False,
        show_error=True,
        quiet=False,
        css=css,
    )


if __name__ == "__main__":
    main()
