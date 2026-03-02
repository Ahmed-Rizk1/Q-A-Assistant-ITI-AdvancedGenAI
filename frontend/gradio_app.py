"""
frontend/gradio_app.py
──────────────────────
Original UI: Document Q&A Assistant
Light theme, two-column layout:
  LEFT  — Document Upload + Status + Document Summary
  RIGHT — Chat with Your Document
"""

from __future__ import annotations
import os
import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


# ── Backend helper ────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs) -> dict:
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except requests.exceptions.ConnectionError:
        raise ValueError("❌ Cannot reach the backend. Make sure app.py is running.")
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = exc.response.text or str(exc)
        raise ValueError(f"API error: {detail}")


# ── Event handlers (logic unchanged) ─────────────────────────────────────────

def handle_upload(file):
    if file is None:
        yield "⚠️ Please select a file first.", ""
        return
    filepath = file.name if hasattr(file, "name") else file
    filename = os.path.basename(filepath)
    yield f"⏳ Uploading and processing **{filename}** …", ""
    try:
        with open(filepath, "rb") as fh:
            data = _api("POST", "/upload",
                        files={"file": (filename, fh, "application/octet-stream")})
        chunks = data.get("chunks", "?")
        yield (
            f"✅ **{filename}** processed successfully!\n"
            f"- Chunks indexed: **{chunks}**\n"
            f"- Vector store: FAISS\n"
            f"- Ready to answer questions."
        ), ""
    except ValueError as exc:
        yield f"❌ {exc}", ""


def handle_summarize(status_text):
    if not status_text or "✅" not in status_text:
        return "⚠️ Please upload and process a document before summarising."
    try:
        data = _api("POST", "/summarize")
        return data.get("summary", "No summary returned.")
    except ValueError as exc:
        return f"❌ {exc}"


def handle_question(question, history, status_text):
    if not question.strip():
        return "", history
    if not status_text or "✅" not in status_text:
        return "", history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": "⚠️ Please upload a document before asking questions."},
        ]
    thinking = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": "⏳ Thinking…"},
    ]
    yield "", thinking
    try:
        data = _api("POST", "/ask", json={"question": question})
    except ValueError as exc:
        yield "", thinking[:-1] + [{"role": "assistant", "content": f"❌ {exc}"}]
        return
    answer = data.get("answer", "No answer returned.")
    yield "", thinking[:-1] + [{"role": "assistant", "content": answer}]


def handle_clear_chat():
    try:
        _api("DELETE", "/reset")
    except Exception:
        pass
    return "", []


def handle_new_document():
    try:
        _api("DELETE", "/reset")
    except Exception:
        pass
    return None, "📂 Upload a document to get started.", "", "", []


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():

    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    body, .gradio-container {
        font-family: 'Inter', system-ui, sans-serif !important;
        background: #f8f9fc !important;
    }
    footer { display: none !important; }
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; padding: 24px 20px !important; }

    /* ─── Header ────────────────────────────────────────────── */
    .app-title { font-size: 26px; font-weight: 700; color: #1e1b4b; margin-bottom: 4px; }
    .app-subtitle { font-size: 13px; color: #64748b; margin-bottom: 24px; }
    .app-subtitle a { color: #7c3aed; text-decoration: none; }

    /* ─── Panels ─────────────────────────────────────────────── */
    .panel-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ─── File upload ────────────────────────────────────────── */
    #file-drop {
        border: 1.5px dashed #c4b5fd !important;
        border-radius: 10px !important;
        background: #faf5ff !important;
        transition: border-color 0.2s !important;
    }
    #file-drop:hover { border-color: #7c3aed !important; }

    /* ─── Buttons ────────────────────────────────────────────── */
    #btn-process {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        height: 42px !important;
        box-shadow: 0 2px 8px rgba(124,58,237,0.3) !important;
        transition: opacity 0.18s !important;
    }
    #btn-process:hover { opacity: 0.88 !important; }

    #btn-new {
        background: #fff !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        height: 42px !important;
    }
    #btn-new:hover { background: #f8fafc !important; color: #1e293b !important; }

    #btn-summarize {
        background: #fff !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        height: 40px !important;
        width: 100% !important;
    }
    #btn-summarize:hover { border-color: #a78bfa !important; color: #7c3aed !important; }

    /* ─── Status ─────────────────────────────────────────────── */
    #status-box {
        font-size: 13px !important;
        color: #374151 !important;
        line-height: 1.65 !important;
        padding: 2px 0 !important;
    }

    /* ─── Summary ────────────────────────────────────────────── */
    #summary-box {
        font-size: 13px !important;
        color: #374151 !important;
        line-height: 1.7 !important;
        max-height: 200px;
        overflow-y: auto;
        padding: 2px 0 !important;
    }
    #summary-box::-webkit-scrollbar { width: 4px; }
    #summary-box::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 2px; }

    /* ─── Chatbot window ──────────────────────────────────────── */
    #chatbot-window {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background: #f8f9fc !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07) !important;
    }

    /* We let Gradio's native theme handle the chat bubbles, 
       removing our aggressive CSS that stretched and deformed them. */


    /* ─── Chat input bar ──────────────────────────────────────── */
    #chat-input-bar {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #fff;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 16px 16px;
        padding: 12px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    /* The textbox inside the input bar */
    #question-input {
        flex: 1;
    }
    #question-input > label > span { display: none !important; }  /* hide label text */
    #question-input textarea {
        background: #f1f5f9 !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 24px !important;
        font-size: 14px !important;
        color: #1e293b !important;
        padding: 10px 18px !important;
        resize: none !important;
        min-height: 42px !important;
        line-height: 1.4 !important;
        transition: border-color 0.18s, box-shadow 0.18s !important;
        font-family: 'Inter', sans-serif !important;
    }
    #question-input textarea:focus {
        border-color: #7c3aed !important;
        background: #fff !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
        outline: none !important;
    }
    #question-input textarea::placeholder {
        color: #94a3b8 !important;
        font-size: 14px !important;
    }

    #btn-send {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 50% !important;
        width: 42px !important;
        min-width: 42px !important;
        height: 42px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        box-shadow: 0 3px 10px rgba(124,58,237,0.35) !important;
        flex-shrink: 0 !important;
        padding: 0 !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    #btn-send:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 5px 16px rgba(124,58,237,0.5) !important;
    }

    /* ─── Two-column gap ─────────────────────────────────────── */
    #two-col > .gap { gap: 28px !important; }
    """

    with gr.Blocks(title="Document Q&A Assistant") as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div>
          <div class="app-title">🏛️ Document Q&amp;A Assistant</div>
          <div class="app-subtitle">
            Upload any PDF or DOCX &nbsp;·&nbsp;
            Ask questions &nbsp;·&nbsp;
            <a>Get AI-powered answers</a>
          </div>
        </div>
        """)

        with gr.Row(elem_id="two-col", equal_height=False):

            # ════════ LEFT — Upload + Summary ══════════════════════════════
            with gr.Column(scale=2, min_width=320):

                gr.HTML('<div class="panel-label">📄 Document Upload</div>')

                file_input = gr.File(
                    label="Drop your PDF or DOCX here",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                    elem_id="file-drop",
                )

                with gr.Row():
                    process_btn = gr.Button(
                        "⚡ Process Document",
                        variant="primary",
                        scale=1,
                        elem_id="btn-process",
                    )
                    new_doc_btn = gr.Button(
                        "🔄 New Document",
                        scale=1,
                        elem_id="btn-new",
                    )

                status_box = gr.Markdown(
                    value="📂 Upload a document to get started.",
                    elem_id="status-box",
                )

                gr.HTML('<div class="panel-label" style="margin-top:18px;">📋 Document Summary</div>')

                summarize_btn = gr.Button(
                    "✨ Generate Summary",
                    elem_id="btn-summarize",
                )

                summary_output = gr.Markdown(
                    value="*Summary will appear here after processing.*",
                    elem_id="summary-box",
                )

            # ════════ RIGHT — Chat ══════════════════════════════════════════
            with gr.Column(scale=3, min_width=400):

                gr.HTML('<div class="panel-label">💬 Chat with Your Document</div>')

                chatbot = gr.Chatbot(
                    value=[],
                    label="",
                    height=430,
                    show_label=False,
                    elem_id="chatbot-window",
                    avatar_images=(None, None),
                )

                # Input bar — styled as a messenger-style bar
                with gr.Row(elem_id="chat-input-bar"):
                    question_box = gr.Textbox(
                        placeholder="Ask anything about the document…  (Enter to send)",
                        label="Message",
                        lines=1,
                        max_lines=4,
                        scale=8,
                        show_label=False,
                        elem_id="question-input",
                    )
                    send_btn = gr.Button(
                        "➤",
                        scale=1,
                        variant="primary",
                        elem_id="btn-send",
                    )

        # ── Wire events ─────────────────────────────────────────────────────

        process_btn.click(
            fn=handle_upload,
            inputs=[file_input],
            outputs=[status_box, summary_output],
        )

        summarize_btn.click(
            fn=handle_summarize,
            inputs=[status_box],
            outputs=[summary_output],
        )

        send_btn.click(
            fn=handle_question,
            inputs=[question_box, chatbot, status_box],
            outputs=[question_box, chatbot],
        )

        question_box.submit(
            fn=handle_question,
            inputs=[question_box, chatbot, status_box],
            outputs=[question_box, chatbot],
        )

        new_doc_btn.click(
            fn=handle_new_document,
            outputs=[file_input, status_box, summary_output, question_box, chatbot],
        )

    return demo, CSS


if __name__ == "__main__":
    demo, css = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        css=css,
    )
