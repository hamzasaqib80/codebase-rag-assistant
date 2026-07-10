# 🧠 Codebase RAG Assistant

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![React](https://img.shields.io/badge/react-18.x-cyan)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)

A production-ready, full-stack Retrieval-Augmented Generation (RAG) assistant designed to ingest, search, and reason over entire GitHub repositories. By combining multi-query expansion, context reranking, and parent-document retrieval, this assistant provides highly accurate technical onboarding and codebase Q&A.

## 📸 Platform Previews

### Dynamic Ingestion & Real-Time Logs
*Paste any GitHub URL into the control panel to dynamically clone and index the repository. Watch the ingestion pipeline logs stream in real-time.*
![Dynamic Ingestion & Real-Time Logs]<img width="1366" height="697" alt="Screenshot From 2026-07-09 10-02-16" src="https://github.com/user-attachments/assets/069e9df5-69d0-46c4-a776-fa2d17586f57" />


### Expert Codebase Q&A
*Ask complex questions about the architecture, commit history, or specific implementation details. The assistant cites the exact source files it used.*
![Expert Codebase Q&A]<img width="1366" height="697" alt="Screenshot From 2026-07-09 10-02-32" src="https://github.com/user-attachments/assets/7fef5494-9f4a-4c98-9831-13c3c4eb98c3" />


---

## ✨ Key Features

- **Dynamic Repository Ingestion:** No need to hardcode URLs. Paste any public GitHub URL directly into the frontend UI, and the backend handles the rest (cloning, chunking, embedding, and indexing).
- **Advanced Retrieval Pipeline:** 
  - **Multi-Query Retriever:** Uses an LLM to generate multiple variants of a user's question to retrieve a broader set of relevant documents.
  - **Cohere Reranking:** Re-ranks the retrieved chunks using Cohere's powerful reranking models to surface the absolute most relevant context.
  - **Parent-Document Retrieval:** Fetches small, semantic chunks during vector search but feeds the *entire parent document* to the LLM to prevent context loss.
- **Cloud & GPU Ready:** The backend is designed to run seamlessly on Google Colab (utilizing free T4 GPUs) for high-speed embeddings, exposing the API securely via Ngrok.
- **Modern Technical Dashboard:** A sleek, dark-mode React interface featuring markdown rendering, isolated source citations, and real-time backend polling.

## 🏗️ Architecture

The system is split into three core layers:

1. **Ingestion Engine (`ingest.py`)**: 
   - Clones the target GitHub repo and wipes old clones automatically.
   - Extracts all code source files and recent commit histories.
   - Uses `RecursiveCharacterTextSplitter` to optimally chunk files.
   - Embeds chunks using local HuggingFace embeddings (`all-mpnet-base-v2`).
   - Upserts vectors to **Pinecone**, while storing full documents in a local `docstore.pkl` (with isolated namespaces per repository).

2. **FastAPI Backend (`api.py`)**:
   - Manages asynchronous background ingestion tasks to keep the API responsive.
   - Assembles the LangChain LCEL pipeline (`rag_pipeline.py`).
   - Powered by **Groq (Llama 3.3 70B)** for lightning-fast inference.

3. **React Frontend (`frontend-v2/App.jsx`)**:
   - Built with Vite and React.
   - Polls the backend for ingestion status and console logs.
   - Manages the chat session and renders markdown responses beautifully.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- API Keys for Pinecone, Groq, and Cohere.

### 1. Backend Setup (Local or Cloud)
1. Copy `.env.example` to `.env` and fill in your keys.
2. Install the backend dependencies:
   ```bash
   pip install -r requirements-api.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   *(💡 **Pro Tip**: Use the provided `RAG_Ingestion_Colab.ipynb` to run the backend on a free Google Colab GPU and expose it via Ngrok!)*

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the `frontend` folder and point it to your backend:
   ```env
   VITE_API_URL=http://localhost:8000
   # Or your Ngrok URL if running on Colab
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```

## 🛠️ Built With
- **LangChain** - Framework for developing LLM applications.
- **Pinecone** - Vector Database for semantic search.
- **Groq** - Ultra-fast LLM inference engine.
- **Cohere** - State-of-the-art context reranking.
- **Sentence-Transformers** - Local embeddings.
- **FastAPI & React** - Core stack.

## 📄 License
This project is licensed under the MIT License.
