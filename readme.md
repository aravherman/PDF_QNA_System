---
title: PDF Question Answering System
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: gradio_app.py
pinned: false
license: mit
---

# 📚 PDF Cognitive Assistant (RAG Q&A System)

A high-performance Question Answering system for PDF documents using **Retrieval-Augmented Generation (RAG)**, powered by **FAISS vector search**, **all-MiniLM-L6-v2** embeddings, and the **Qwen2.5-Instruct** chat model.

---

## 🚀 Features

- 📑 **Multi-PDF Text Extraction**: Ingest and process multiple PDF files simultaneously.
- 🧩 **Smart Text Chunking**: Configurable chunk size and overlap for context retention.
- ⚡ **Dense Vector Retrieval**: Fast semantic similarity search with FAISS and `sentence-transformers`.
- 🤖 **Context-Grounded Q&A**: Answers generated strictly from the uploaded PDF context using `Qwen2.5-1.5B-Instruct`.
- 🔎 **Evidence & Citation Explorer**: Inspect the exact source PDF, page number, and similarity score for each generated response.
- 🎨 **Modern Gradio Interface**: Clean dual-panel layout with real-time status indicators, sample question prompts, and chunk inspectors.

---

## 🛠️ Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio Web App
```bash
python gradio_app.py
```
This will launch the app locally at `http://127.0.0.1:7860` and automatically output a **public live link** (`https://xxxx.gradio.live`) for sharing!

---

## ☁️ Permanent Hosting on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Set **SDK** to **Gradio**.
3. Push your repository files to the Hugging Face Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/<SPACE-NAME>
   git add .
   git commit -m "Deploy Gradio app"
   git push space main
   ```
4. Hugging Face Spaces will build and host your app 24/7 for free!
