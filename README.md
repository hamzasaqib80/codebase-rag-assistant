# 🧠  Codebase RAG Assistant

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![React](https://img.shields.io/badge/react-18.x-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple)

**A production-ready, full-stack RAG assistant that ingests, searches, and reasons over any GitHub repository on demand.**

</div>

---

## 📸 Platform Previews

### Dynamic Ingestion & Real-Time Pipeline Logs
*Paste any public GitHub URL directly into the sidebar, hit "Trigger Ingestion", and watch the live pipeline logs stream in real-time as the backend clones, chunks, embeds, and indexes the codebase into Pinecone.*

![Dynamic Ingestion & Real-Time Logs](Screenshot%20From%202026-07-09%2010-02-16.png)

### Expert-Level Codebase Q&A
*Ask complex architectural questions about any indexed repository. The assistant generates precise, markdown-formatted answers and cites the exact source files it referenced.*

![Expert Codebase Q&A](Screenshot%20From%202026-07-09%2010-02-32.png)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔗 **Dynamic Repository Ingestion** | Paste any public GitHub URL in the UI — no `.env` edits required. The backend handles cloning, chunking, embedding, and indexing. |
| 🔍 **Multi-Query Retrieval** | The LLM generates multiple query variants to cast a wider semantic net, improving recall from the vector store. |
| 🏆 **Cohere Reranking** | Retrieved chunks are reranked by Cohere's cross-encoder for maximum relevance before being sent to the LLM. |
| 📄 **Parent-Document Retrieval** | Searches on small, semantic child chunks but feeds the full parent document to the LLM — preventing context loss. |
| ☁️ **Cloud & GPU Ready** | Run the backend on a free Google Colab T4 GPU and expose it via Ngrok. The notebook is included. |
| 🌑 **Modern Dark UI** | Sleek React dashboard with real-time log streaming, markdown rendering, and source citations. |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       React Frontend                         │
│   (Vite + React, dynamic URL input, real-time log polling)   │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP / REST
┌─────────────────────▼───────────────────────────────────────┐
│                  FastAPI Backend (api.py)                     │
│    Async background ingestion │ LCEL chain assembly           │
└──────┬────────────────────────────────────┬─────────────────┘
       │                                    │
┌──────▼──────────┐               ┌─────────▼────────────────┐
│  ingest.py       │               │     rag_pipeline.py       │
│                  │               │                           │
│ • git clone/pull │               │ • HuggingFace Embeddings   │
│ • File splitter  │               │ • Pinecone VectorStore     │
│ • Commit history │               │ • MultiQueryRetriever      │
│ • Pinecone upsert│               │ • Cohere Reranker         │
│ • Docstore pkl   │               │ • Groq LLM (Llama 3.3)    │
└──────────────────┘               └───────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Free API keys for **[Pinecone](https://pinecone.io)**, **[Groq](https://groq.com)**, and **[Cohere](https://cohere.com)**

### Step 1 — Clone & Configure

```bash
git clone https://github.com/hamzasaqib80/codebase-rag-assistant.git
cd codebase-rag-assistant

# Copy the example env and fill in your API keys
cp .env.example .env
```

Edit `.env` with your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
GITHUB_REPO_URL=https://github.com/any/repo.git   # Default repo to pre-index
```

### Step 2 — Backend Setup

**Option A: Local (CPU)**
```bash
pip install -r requirements-api.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Option B: Google Colab (Free T4 GPU) — Recommended**
> 💡 Open `RAG_Ingestion_Colab.ipynb` in Google Colab, fill in your API keys, and run all cells. The notebook will start the FastAPI server and expose it via Ngrok automatically.

### Step 3 — Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create frontend environment file
echo "VITE_API_URL=http://localhost:8000" > .env
# (Replace with your Ngrok URL if using Colab)

# Start the dev server
npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)** in your browser.

### Step 4 — Index a Repository

1. Paste a GitHub URL in the sidebar input field (e.g. `https://github.com/psf/requests.git`).
2. Click **"Trigger Ingestion"**.
3. Watch the live pipeline logs as the repository is indexed.
4. Once complete, start asking questions in the chat!

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — Llama 3.3 70B (ultra-fast inference) |
| **Vector DB** | Pinecone (serverless) |
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` (local, no API cost) |
| **Reranker** | Cohere Rerank v3 |
| **Framework** | LangChain LCEL |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 18, Vite |
| **Cloud Hosting** | Google Colab + Ngrok |

---

## 📁 Project Structure

```
codebase-rag-assistant/
├── api.py                     # FastAPI backend — ingestion & query endpoints
├── ingest.py                  # Full ingestion pipeline (clone → chunk → embed → upsert)
├── rag_pipeline.py            # LangChain LCEL chain assembly
├── retrievers.py              # Custom ParentFetchingRetriever implementation
├── utils.py                   # Shared logging utilities
├── query.py                   # Standalone CLI query script
├── requirements-api.txt       # Backend Python dependencies
├── RAG_Ingestion_Colab.ipynb  # Google Colab notebook for cloud deployment
├── .env.example               # Environment variable template
└── frontend/                  # React + Vite frontend
    ├── src/
    │   └── App.jsx            # Main UI component
    └── package.json
```

---

## 📄 License

This project is licensed under the **MIT License**.
