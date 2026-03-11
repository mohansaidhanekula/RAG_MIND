<div align="center">

<img src="https://img.shields.io/badge/RAG%20Mind%20AI-v5-6366f1?style=for-the-badge&logo=brain&logoColor=white" alt="RAG Mind AI"/>

# ✦ RAG Mind AI

### Query your documents with intelligence. Get grounded answers, not hallucinations.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Free%20API-f97316?style=flat-square&logo=lightning&logoColor=white)](https://console.groq.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-6366f1?style=flat-square)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)](LICENSE)

**[🚀 Live Demo](#) · [📖 Docs](#installation) · [🐛 Report Bug](issues) · [💡 Request Feature](issues)**

---

![RAG Mind AI Screenshot](https://placehold.co/900x480/0d0f1a/818cf8?text=RAG+Mind+AI+%E2%80%94+Dark+%2B+Light+Theme)

</div>

---

## 🧠 What is RAG Mind AI?

**RAG Mind AI** is a full-stack, production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit and Groq's free LLM API. Upload your PDFs, Word docs, Excel sheets, or any web page — then ask questions and get accurate, source-cited answers powered by Llama 3.3 70B.

> **Zero hallucinations.** Every answer is grounded in *your* documents, not the model's training data.

---

## ✨ Features

### 🔍 Hybrid Search Engine
Combines **BM25 keyword search** with **semantic vector search**, fused via **Reciprocal Rank Fusion (RRF)** — capturing both exact matches and conceptually similar content for dramatically better retrieval than either method alone.

### 📄 Universal Document Support
| Format | Details |
|--------|---------|
| 📕 PDF | Full text extraction with page-level citations |
| 📝 Word (.docx) | Paragraph-level extraction |
| 📊 Excel (.xlsx / .xls) | Sheet-by-sheet ingestion + smart cleaning |
| 📄 CSV | Auto-parsed tabular data |
| 🌐 Web Pages | URL-based scraping and chunking |
| 📋 Text / Markdown | Plain text files |

### 🧹 Excel Data Cleaning Engine
12 smart, toggleable cleaning operations applied per-sheet:

| Operation | What it does |
|-----------|-------------|
| Remove empty rows/cols | Strips fully blank rows and columns |
| Strip whitespace | Cleans leading/trailing spaces in all text cells |
| Standardize headers | Converts column names to clean `snake_case` |
| Remove duplicates | Drops identical rows |
| Fix numeric strings | Converts `$1,234.56` and `45%` to real numbers |
| Fix date columns | Normalizes all date formats to `YYYY-MM-DD` |
| Forward-fill merged cells | Handles Excel merged-cell patterns |
| Drop unnamed columns | Removes `Unnamed: 0`, `Unnamed: 1`, etc. |
| Normalize booleans | Maps `yes/no/true/false/y/n` → Python booleans |
| Drop sparse columns | Removes columns with >80% missing values |

Download your cleaned data as a professionally formatted `.xlsx` with frozen headers, zebra striping, and auto-sized columns.

### 🎙️ Voice I/O
- **Voice Input** — speak your question directly in the chat (uses `SpeechRecognition` + Google STT)
- **Voice Output** — answers are read aloud via gTTS text-to-speech

### 📊 Answer Quality Evaluation
Every response is automatically scored on 4 dimensions — all computed locally, no external API:

| Metric | Measures |
|--------|---------|
| 🔒 Faithfulness | Does the answer stay within the retrieved context? |
| 🔍 Context Relevance | Are the retrieved chunks actually relevant? |
| 💬 Answer Relevance | Does the answer address the question? |
| 📝 ROUGE-L | Lexical overlap with source context |

### 📋 Auto Document Summarization
Per-document summaries + a cross-document synthesis overview, all generated via Groq with structured output (summary, key topics, document type).

### 📥 Export Q&A Sessions
Export your full conversation — questions, answers, and source citations — to:
- **PDF** — formatted with ReportLab, page-ready
- **Word (.docx)** — editable document with styled Q&A pairs

### 🌗 Light / Dark Theme
A floating `☀️/🌙` toggle button switches between themes **instantly** — no page reload. Persists across sessions via `localStorage`. Auto-detects your OS theme preference on first visit.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 ┌─────────────────────────────────────────┐
 │           HYBRID RETRIEVAL              │
 │  BM25 keyword  +  ChromaDB semantic     │
 │        ↘              ↙                │
 │    Reciprocal Rank Fusion (RRF)         │
 └──────────────┬──────────────────────────┘
                │  Top-K relevant chunks
                ▼
 ┌─────────────────────────────────────────┐
 │        GROQ LLM (Llama 3.3 70B)        │
 │   System prompt + Context + Question   │
 └──────────────┬──────────────────────────┘
                │
                ▼
      Answer + Source Citations
      + Eval Scores (local)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | [Groq](https://console.groq.com) — Llama 3.3 70B, Mixtral 8x7B, DeepSeek R1, Gemma 2 (free) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` — runs locally, no API key |
| **Vector DB** | ChromaDB — local, persisted |
| **Keyword Search** | BM25 via `rank-bm25` |
| **RAG Framework** | LangChain |
| **UI** | Streamlit 1.38+ |
| **Excel** | pandas + openpyxl |
| **Voice** | SpeechRecognition + gTTS |
| **Export** | ReportLab (PDF) + python-docx (Word) |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-mind-ai.git
cd rag-mind-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up — no credit card required
3. Create an API key

### 5. Configure your key

```bash
cp .env.example .env
# Add your key to .env:
# GROQ_API_KEY=gsk_your_key_here
```

Or just paste it directly in the sidebar when the app loads.

### 6. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
rag-mind-ai/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── .env.example
└── utils/
    ├── __init__.py
    ├── loaders.py                  # PDF, DOCX, TXT, Web loaders + chunking
    ├── vectorstore.py              # ChromaDB + HuggingFace embeddings
    ├── rag_chain.py                # Groq LLM chain with streaming
    ├── hybrid_search.py            # BM25 + semantic + RRF fusion
    ├── summarizer.py               # Auto document summarization
    ├── voice.py                    # Voice input (STT) + output (TTS)
    ├── exporter.py                 # PDF + Word export
    ├── evaluator.py                # RAG evaluation metrics
    └── excel_processor.py          # Excel/CSV loading, cleaning, indexing
```

---

## 🎯 Usage Guide

### Chatting with documents

1. Enter your Groq API key in the sidebar
2. Upload files (PDF, DOCX, XLSX, TXT, MD) or paste a URL
3. Click **🚀 Ingest Documents**
4. Ask questions in the Chat tab — type or speak

### Cleaning Excel files

1. Switch to the **📊 Excel** tab
2. Upload your `.xlsx` or `.csv` file
3. Review the null/duplicate stats
4. Select cleaning operations and click **🧹 Run Cleaning**
5. Download the cleaned file or index it for Q&A

### Evaluating answer quality

Enable **📊 Eval scores** in the sidebar — quality scores appear under every answer. View the full dashboard in the **📊 Evaluation** tab.

---

## 🤖 Available Models (All Free via Groq)

| Model | Best for |
|-------|---------|
| `llama-3.3-70b-versatile` | Best overall quality |
| `llama-3.1-8b-instant` | Fastest responses |
| `mixtral-8x7b-32768` | Long context (32K) |
| `deepseek-r1-distill-llama-70b` | Reasoning tasks |
| `gemma2-9b-it` | Lightweight, Google |

---

## ⚙️ Configuration

All advanced settings are in the sidebar **⚙️ Advanced** expander:

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk size | 1000 | Tokens per document chunk |
| Chunk overlap | 200 | Token overlap between chunks |
| Top-k chunks | 5 | Retrieved chunks per query |
| Temperature | 0.0 | LLM randomness (0 = deterministic) |

---

## 🔮 Roadmap

- [ ] Multi-user workspaces with isolated knowledge bases
- [ ] Reranking via cross-encoder models
- [ ] Image/diagram understanding (multimodal RAG)
- [ ] RAGAS evaluation integration
- [ ] Docker deployment
- [ ] Conversation memory across sessions

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Fork → Clone → Branch → PR
git checkout -b feature/your-feature-name
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ using Groq · LangChain · Streamlit · ChromaDB**

⭐ Star this repo if you found it useful!

</div>
