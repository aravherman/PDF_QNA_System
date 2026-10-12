import os
import time
from functools import lru_cache
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from src.pdf_extractor import extract_text_from_pdf
from src.preprocessor import clean_text
from src.chunker import create_chunks
from src.retriever import create_faiss_index, search_faiss


# ---------------------------------------------------------------------------
# Model Management & Singletons
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful and precise document assistant. "
    "Answer the user's question using ONLY the provided document context. "
    "Keep your answer clear, accurate, and concise. "
    "If the answer cannot be determined from the context, respond strictly with: "
    "\"I could not find the answer in the uploaded document.\""
)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the sentence-transformers embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_qa_pipeline():
    """Load and cache the question answering chat pipeline."""
    print(f"Loading QA model: {QA_MODEL_NAME}...")
    use_gpu = torch.cuda.is_available()
    return pipeline(
        "text-generation",
        model=QA_MODEL_NAME,
        dtype=torch.float16 if use_gpu else torch.float32,
        device=0 if use_gpu else -1
    )


def generate_answer(qa_model, question: str, context: str) -> str:
    """Generate answer from context using the chat model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    output = qa_model(
        messages,
        max_new_tokens=250,
        do_sample=False,
        return_full_text=False
    )

    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        generated = generated[-1]["content"]
    return generated.strip()


# ---------------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------------

def process_documents(files, chunk_size, chunk_overlap):
    """
    Extract text from uploaded PDFs, chunk them, embed, and build FAISS index.
    """
    if not files:
        return (
            "<div class='status-box status-warning'>⚠️ No files uploaded. Please select one or more PDF files.</div>",
            "**No documents loaded yet.**",
            gr.update(choices=[], value=None),
            None
        )

    t0 = time.time()
    all_pages = []
    file_list = files if isinstance(files, list) else [files]

    for file_obj in file_list:
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        pages = extract_text_from_pdf(file_path)
        for p in pages:
            p["text"] = clean_text(p["text"])
        all_pages.extend(pages)

    if not all_pages:
        return (
            "<div class='status-box status-error'>❌ Could not extract any readable text from the provided PDF(s).</div>",
            "**0 pages extracted.**",
            gr.update(choices=[], value=None),
            None
        )

    # Chunking
    chunks = create_chunks(
        all_pages,
        chunk_size=int(chunk_size),
        overlap=int(chunk_overlap)
    )

    if not chunks:
        return (
            "<div class='status-box status-error'>❌ No text chunks were generated.</div>",
            "**0 chunks generated.**",
            gr.update(choices=[], value=None),
            None
        )

    # Embedding & Indexing
    emb_model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss_index = create_faiss_index(embeddings)

    elapsed = time.time() - t0
    unique_pdfs = len(set(p["pdf_name"] for p in all_pages))

    status_html = (
        f"<div class='status-box status-success'>"
        f"✅ <b>Successfully indexed {len(chunks)} chunks</b> from {unique_pdfs} PDF(s) "
        f"({len(all_pages)} pages) in {elapsed:.1f}s."
        f"</div>"
    )

    stats_md = (
        f"| Metric | Count |\n"
        f"| :--- | :--- |\n"
        f"| 📄 **Documents** | `{unique_pdfs}` |\n"
        f"| 📑 **Total Pages** | `{len(all_pages)}` |\n"
        f"| 🧩 **Text Chunks** | `{len(chunks)}` |\n"
        f"| 📐 **Chunk Size / Overlap** | `{int(chunk_size)} / {int(chunk_overlap)}` |\n"
    )

    chunk_choices = [
        f"Chunk #{c['chunk_id']} | {c['pdf_name']} (p.{c['page_number']})"
        for c in chunks
    ]
    initial_preview = chunk_choices[0] if chunk_choices else None

    state_data = {
        "chunks": chunks,
        "faiss_index": faiss_index,
        "pages": all_pages,
        "pdf_count": unique_pdfs
    }

    return status_html, stats_md, gr.update(choices=chunk_choices, value=initial_preview), state_data


def preview_chunk(selected_label, state_data):
    """Show details of the selected chunk."""
    if not selected_label or not state_data or "chunks" not in state_data:
        return "Select a chunk to view details."

    try:
        chunk_id_str = selected_label.split("|")[0].replace("Chunk #", "").strip()
        chunk_id = int(chunk_id_str)
        chunk = state_data["chunks"][chunk_id]
        return (
            f"**PDF File:** `{chunk['pdf_name']}`  \n"
            f"**Page Number:** `{chunk['page_number']}`  \n"
            f"**Word Count:** `{len(chunk['text'].split())}` words  \n\n"
            f"```text\n{chunk['text']}\n```"
        )
    except Exception as e:
        return f"Error retrieving chunk: {str(e)}"


def chat_qa(user_message, chat_history, top_k, state_data):
    """
    Retrieve relevant chunks and generate response using Qwen.
    """
    if not user_message or not user_message.strip():
        return "", chat_history, "Enter a question to inspect retrieved sources.", state_data

    user_query = user_message.strip()

    if not state_data or state_data.get("faiss_index") is None:
        bot_response = (
            "⚠️ **No documents indexed yet.**\n\n"
            "Please upload your PDF(s) in the left panel and click **⚡ Process & Index Documents** first."
        )
        chat_history = (chat_history or []) + [[user_query, bot_response]]
        return "", chat_history, "No documents indexed.", state_data

    emb_model = get_embedding_model()
    qa_model = get_qa_pipeline()

    # Retrieve
    query_emb = emb_model.encode([user_query], convert_to_numpy=True)
    results = search_faiss(
        state_data["faiss_index"],
        query_emb,
        state_data["chunks"],
        top_k=int(top_k)
    )

    if not results:
        bot_response = "I could not find any relevant content in the uploaded document."
        chat_history = (chat_history or []) + [[user_query, bot_response]]
        return "", chat_history, "No matching chunks found.", state_data

    # Generate answer
    context = "\n\n".join([r["text"] for r in results])
    raw_answer = generate_answer(qa_model, user_query, context)

    # Format answer with citations summary
    citations = [
        f"📄 **{r['pdf_name']}** (Page {r['page_number']})"
        for r in results
    ]
    citations_text = " • ".join(list(dict.fromkeys(citations)))

    bot_response = (
        f"{raw_answer}\n\n"
        f"---\n"
        f"🔍 *Sources:* {citations_text}"
    )

    chat_history = (chat_history or []) + [[user_query, bot_response]]

    # Format detailed source inspection
    sources_md = f"### 📚 Retrieved Context ({len(results)} Chunks for Query: *\"{user_query}\"*)\n\n"
    for i, r in enumerate(results, start=1):
        sources_md += (
            f"#### Source {i} — `{r['pdf_name']}` (Page {r['page_number']}) "
            f"| Distance: `{r['distance']:.4f}`\n\n"
            f"> {r['text']}\n\n"
        )

    return "", chat_history, sources_md, state_data


def clear_chat():
    """Clear conversation history."""
    return [], "Chat cleared. Ask a new question."


def set_sample_question(question_text):
    """Set the text input from a sample question pill."""
    return question_text


# ---------------------------------------------------------------------------
# Gradio Modern UI Definition
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Container & Base Styles */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header Banner */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.25);
}
.app-header h1 {
    font-size: 26px;
    font-weight: 800;
    margin: 0 0 8px 0;
    color: #ffffff !important;
}
.app-header p {
    font-size: 14px;
    margin: 0;
    opacity: 0.92;
}

/* Status Notifications */
.status-box {
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13.5px;
    margin-top: 10px;
    margin-bottom: 10px;
}
.status-success {
    background-color: #ecfdf5;
    border: 1px solid #10b981;
    color: #065f46;
}
.status-warning {
    background-color: #fffbeb;
    border: 1px solid #f59e0b;
    color: #92400e;
}
.status-error {
    background-color: #fef2f2;
    border: 1px solid #ef4444;
    color: #991b1b;
}

/* Sidebar Styling - Sleek Dark Obsidian & Slate */
.sidebar-panel {
    border-radius: 16px !important;
    background: linear-gradient(180deg, #090d16 0%, #0f172a 40%, #1e293b 100%) !important;
    padding: 20px !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    box-shadow: 0 12px 30px -5px rgba(2, 6, 23, 0.6) !important;
    color: #f8fafc !important;
}

.sidebar-panel h3 {
    color: #38bdf8 !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-panel label, 
.sidebar-panel span, 
.sidebar-panel p,
.sidebar-panel .label-wrap {
    color: #cbd5e1 !important;
}

/* Dark File Dropzone */
.sidebar-panel .file-preview,
.sidebar-panel [data-testid="file-upload"],
.sidebar-panel .upload-container,
.sidebar-panel .file,
.sidebar-panel .gr-file,
.sidebar-panel [class*="upload_"] {
    background: #090e1a !important;
    border: 1.5px dashed #0284c7 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    transition: all 0.2s ease-in-out !important;
}

.sidebar-panel [data-testid="file-upload"]:hover {
    border-color: #38bdf8 !important;
    background: #0d1527 !important;
}

.sidebar-panel .file-preview-holder,
.sidebar-panel .file-parts {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

/* Accordions inside Dark Sidebar */
.sidebar-panel .accordion,
.sidebar-panel details {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 10px !important;
    margin-top: 10px !important;
}

.sidebar-panel summary {
    color: #7dd3fc !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}

/* Sliders, Inputs, Dropdowns inside Dark Sidebar */
.sidebar-panel input[type="range"] {
    accent-color: #38bdf8 !important;
}

.sidebar-panel input,
.sidebar-panel textarea,
.sidebar-panel select,
.sidebar-panel .gr-dropdown {
    background: #0b1120 !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Process Documents Button */
.sidebar-panel button.primary,
.sidebar-panel .gr-button-primary {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 12px 18px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
    transition: all 0.2s ease !important;
}

.sidebar-panel button.primary:hover,
.sidebar-panel .gr-button-primary:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.6) !important;
}

/* Stats Markdown Table inside Sidebar */
.sidebar-panel table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin-top: 12px !important;
    font-size: 13.5px !important;
}

.sidebar-panel th {
    background: #090e1a !important;
    color: #38bdf8 !important;
    padding: 8px 10px !important;
    border-bottom: 1px solid #334155 !important;
    text-align: left !important;
}

.sidebar-panel td {
    padding: 8px 10px !important;
    border-bottom: 1px solid rgba(51, 65, 85, 0.6) !important;
    color: #e2e8f0 !important;
}

.sidebar-panel code {
    background: #090e1a !important;
    color: #38bdf8 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid #1e293b !important;
}

/* Chatbot styling */
.chatbot-box {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

/* Sample pills */
.sample-pill {
    font-size: 12px !important;
    padding: 4px 10px !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="PDF Cognitive Assistant") as demo:

    # Global State
    doc_state = gr.State(None)

    # Header
    gr.HTML(
        """
        <div class="app-header">
            <h1>📚 PDF Cognitive Assistant</h1>
            <p>High-Precision Retrieval Augmented Generation (RAG) with FAISS Vector Search & Qwen Instruct</p>
        </div>
        """
    )

    with gr.Row():
        # -------------------------------------------------------------------
        # Left Column: Document Studio
        # -------------------------------------------------------------------
        with gr.Column(scale=4, elem_classes=["sidebar-panel"]):
            gr.Markdown("### 📄 Document Studio")
            pdf_input = gr.File(
                label="Upload PDF Files",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )

            with gr.Accordion("⚙️ RAG Hyperparameters", open=False):
                chunk_size_slider = gr.Slider(
                    label="Chunk Size (words)",
                    minimum=50,
                    maximum=400,
                    value=150,
                    step=25
                )
                overlap_slider = gr.Slider(
                    label="Overlap (words)",
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5
                )
                top_k_slider = gr.Slider(
                    label="Top-K Retrieved Chunks",
                    minimum=1,
                    maximum=6,
                    value=3,
                    step=1
                )

            process_btn = gr.Button("⚡ Process & Index Documents", variant="primary", size="lg")
            status_output = gr.HTML(
                value="<div class='status-box status-warning'>Waiting for PDF documents to be uploaded.</div>"
            )

            stats_display = gr.Markdown(value="*Upload and index PDFs to view document metrics.*")

            with gr.Accordion("🔍 Chunks & Content Inspector", open=False):
                chunk_selector = gr.Dropdown(
                    label="Select Chunk",
                    choices=[],
                    value=None,
                    interactive=True
                )
                chunk_details = gr.Markdown(value="*No chunk selected.*")

        # -------------------------------------------------------------------
        # Right Column: Interactive Chat & Evidence
        # -------------------------------------------------------------------
        with gr.Column(scale=6):
            gr.Markdown("### 💬 Interactive Q&A")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                show_label=False,
                elem_classes=["chatbot-box"]
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the uploaded document(s)...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                send_btn = gr.Button("Ask 🚀", variant="primary", scale=1)

            # Sample Prompts
            gr.Markdown("**Quick Prompts:**")
            with gr.Row():
                sample_1 = gr.Button("📌 What is the main summary?", size="sm")
                sample_2 = gr.Button("🎯 What are the key conclusions?", size="sm")
                sample_3 = gr.Button("🔬 What methodology was used?", size="sm")
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm", variant="secondary")

            with gr.Accordion("🔎 Source Evidence & Distance Scores", open=True):
                evidence_display = gr.Markdown(
                    value="*Ask a question above to see the retrieved text passages and vector distances.*"
                )

    # -----------------------------------------------------------------------
    # Event Listeners
    # -----------------------------------------------------------------------

    # Process Documents
    process_btn.click(
        fn=process_documents,
        inputs=[pdf_input, chunk_size_slider, overlap_slider],
        outputs=[status_output, stats_display, chunk_selector, doc_state]
    )

    # Chunk Inspector selection
    chunk_selector.change(
        fn=preview_chunk,
        inputs=[chunk_selector, doc_state],
        outputs=[chunk_details]
    )

    # Submit Question via Send Button or Enter
    msg_input.submit(
        fn=chat_qa,
        inputs=[msg_input, chatbot, top_k_slider, doc_state],
        outputs=[msg_input, chatbot, evidence_display, doc_state]
    )

    send_btn.click(
        fn=chat_qa,
        inputs=[msg_input, chatbot, top_k_slider, doc_state],
        outputs=[msg_input, chatbot, evidence_display, doc_state]
    )

    # Clear Chat
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, evidence_display]
    )

    # Sample Buttons
    sample_1.click(fn=lambda: "What is the main summary?", inputs=[], outputs=[msg_input])
    sample_2.click(fn=lambda: "What are the key conclusions?", inputs=[], outputs=[msg_input])
    sample_3.click(fn=lambda: "What methodology was used in this document?", inputs=[], outputs=[msg_input])


# ---------------------------------------------------------------------------
# App Launch Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )
