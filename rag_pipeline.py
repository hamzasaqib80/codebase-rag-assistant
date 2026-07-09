"""
rag_pipeline.py — Assembles the full RAG chain:

  PineconeVectorStore
    → ParentFetchingRetriever   (child search → parent doc fetch)
    → MultiQueryRetriever       (LLM expands query into N variants)
    → ContextualCompressionRetriever (Cohere Rerank selects top-5)
    → create_stuff_documents_chain  (stuffs context into prompt)
    → create_retrieval_chain        (LCEL orchestration)

Embedding model: sentence-transformers/all-mpnet-base-v2 (768 dims, local).
Uses modern LCEL composition only — no deprecated RetrievalQA chain.
Invoked via qa_chain.invoke({"input": "..."}).
"""

import logging
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryStore
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from retrievers import ParentFetchingRetriever
from utils import retry_on_failure, setup_logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logger = setup_logging()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "codebase-rag")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
DOCSTORE_PATH: str = "docstore.pkl"


PARENT_FETCH_K: int = 20       # child chunks retrieved per sub-query
COHERE_TOP_N: int = 5          # documents kept after Cohere rerank


# ---------------------------------------------------------------------------
# Builder helpers — all wrapped in @retry_on_failure for free-tier resilience
# ---------------------------------------------------------------------------

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def _build_embeddings() -> CohereEmbeddings:
    logger.info("Loading Cohere embeddings …")
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def _build_vectorstore(embeddings: CohereEmbeddings) -> PineconeVectorStore:
    """
    Connect to the existing Pinecone index.
    `text_key="page_content"` is mandatory (pitfall #2) — without it, Pinecone
    silently returns blank page_content on retrieval even though vectors exist.
    """
    logger.info("Connecting to Pinecone index '%s' …", PINECONE_INDEX_NAME)
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="page_content",
        pinecone_api_key=PINECONE_API_KEY,
    )


def _load_docstore() -> InMemoryStore:
    """
    Load the persisted InMemoryStore docstore from disk.
    Raises FileNotFoundError with a clear, instructive message if missing.
    """
    if not Path(DOCSTORE_PATH).exists():
        raise FileNotFoundError(
            f"'{DOCSTORE_PATH}' not found. "
            "Run the ingestion pipeline first:\n\n"
            "    python ingest.py\n\n"
            "This file is written at the end of ingestion and contains all parent "
            "documents (full file contents and commit messages) needed for retrieval."
        )
    logger.info("Loading docstore from '%s' …", DOCSTORE_PATH)
    with open(DOCSTORE_PATH, "rb") as fh:
        docstore: InMemoryStore = pickle.load(fh)
    logger.info("Docstore loaded.")
    return docstore


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def _build_llm() -> ChatGroq:
    logger.info("Initialising ChatGroq LLM (model=%s) …", GROQ_MODEL)
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior software architect conducting a technical onboarding session "
    "for a new engineer joining the team. Your role is to help them deeply understand "
    "the codebase by answering their questions accurately and thoroughly.\n\n"
    "Guidelines:\n"
    "• Cite specific files, function names, class names, and line numbers whenever "
    "  they appear in the context below. Format citations as `file.py:LineN`.\n"
    "• When referencing git commits, cite the short SHA and the key change described "
    "  in the commit message.\n"
    "• Synthesise across multiple files or commits when answering architectural questions.\n"
    "• If the answer cannot be derived from the context provided, say so explicitly: "
    "  \"The provided context does not contain enough information to answer this question.\"\n"
    "  Do NOT fabricate code, filenames, or commit history.\n"
    "• Keep responses structured: use headers, bullet points, and code blocks where "
    "  they aid clarity.\n\n"
    "Context from the codebase:\n"
    "{context}"
)

HUMAN_PROMPT = "{input}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_qa_chain():
    """
    Build and return the full LCEL RAG chain.

    Chain structure
    ---------------
    create_retrieval_chain(
        ContextualCompressionRetriever(
            compressor=CohereRerank(top_n=5),
            base_retriever=MultiQueryRetriever(
                retriever=ParentFetchingRetriever(vectorstore, docstore, k=20),
                llm=ChatGroq(...),
            ),
        ),
        create_stuff_documents_chain(llm, prompt),
    )

    Invocation
    ----------
    response = chain.invoke({"input": "How does X work?"})
    answer   = response["answer"]
    context  = response["context"]   # list of LangChain Documents
    """

    # --- Build components ---
    embeddings = _build_embeddings()
    vectorstore = _build_vectorstore(embeddings)
    docstore = _load_docstore()
    llm = _build_llm()

    # --- ParentFetchingRetriever (pitfall #3) ---
    parent_retriever = ParentFetchingRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="parent_id",
        k=PARENT_FETCH_K,
    )
    logger.info("ParentFetchingRetriever ready (k=%d).", PARENT_FETCH_K)

    # --- MultiQueryRetriever ---
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=parent_retriever,
        llm=llm,
    )
    logger.info("MultiQueryRetriever ready.")

    # --- Cohere Rerank compressor ---
    cohere_reranker = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        top_n=COHERE_TOP_N,
        model="rerank-english-v3.0",
    )

    # --- ContextualCompressionRetriever ---
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=multi_query_retriever,
    )
    logger.info(
        "ContextualCompressionRetriever ready (Cohere rerank top_n=%d).", COHERE_TOP_N
    )

    # --- Prompt template ---
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    # --- LCEL chain assembly (architecture requirement #6) ---
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(compression_retriever, document_chain)

    logger.info("RAG chain assembled successfully.")
    return qa_chain
