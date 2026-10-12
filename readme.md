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
