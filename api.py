import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from collections import deque

from rag_pipeline import get_qa_chain
from ingest import run_ingestion
from utils import setup_logging

# Setup memory log handler to capture ingestion logs
class InMemoryHandler(logging.Handler):
    def __init__(self, limit=100):
        super().__init__()
        self.logs = deque(maxlen=limit)

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

    def get_logs(self):
        return list(self.logs)

in_memory_handler = InMemoryHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
in_memory_handler.setFormatter(formatter)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logging.getLogger().addHandler(in_memory_handler)
    # Attempt to initialize chain on startup. If docstore doesn't exist, it will log a warning.
    # We do it asynchronously to not block startup.
    asyncio.create_task(asyncio.to_thread(init_chain))
    yield

app = FastAPI(title="RAG Backend API", lifespan=lifespan)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for RAG chain
qa_chain = None
ingestion_status = "idle"  # idle, running, success, failed

class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    label: str
    annotation: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

class IngestStatusResponse(BaseModel):
    status: str
    logs: List[str]

class IngestRequest(BaseModel):
    repo_url: Optional[str] = None

def init_chain():
    global qa_chain
    if qa_chain is None:
        try:
            qa_chain = get_qa_chain()
        except Exception as e:
            logging.error(f"Failed to initialize QA chain: {e}")

# Run ingestion in background
def bg_ingestion(repo_url: Optional[str] = None):
    global ingestion_status
    ingestion_status = "running"
    in_memory_handler.logs.clear()
    logging.info(f"Starting background ingestion pipeline for {repo_url or 'default repo'}...")
    try:
        run_ingestion(repo_url)
        ingestion_status = "success"
        logging.info("Ingestion completed successfully!")
        # Re-initialize chain after ingestion atomically
        global qa_chain
        new_chain = get_qa_chain()
        qa_chain = new_chain
    except Exception as e:
        ingestion_status = "failed"
        logging.error(f"Ingestion failed: {e}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global qa_chain
    if qa_chain is None:
        # Try to initialize again (maybe docstore was created recently)
        init_chain()
        if qa_chain is None:
            raise HTTPException(status_code=400, detail="RAG Pipeline not ready. Please run ingestion first.")
    
    try:
        # Run query in a separate thread to avoid blocking FastAPI event loop
        response = await asyncio.to_thread(qa_chain.invoke, {"input": request.question})
        answer = response.get("answer", "(No answer returned)")
        context_docs = response.get("context", [])
        
        sources = []
        for doc in context_docs:
            meta = doc.metadata or {}
            file_path = meta.get("file_path", "")
            sha = meta.get("sha", "")
            parent_id = meta.get("parent_id", "")
            
            if file_path:
                label = file_path
            elif sha:
                label = f"commit:{sha}"
            elif parent_id:
                label = parent_id
            else:
                label = doc.page_content[:60].strip().replace("\n", " ") + " …"
                
            node_type = meta.get("node_type", "")
            symbol_name = meta.get("symbol_name", "")
            
            annotation_parts = []
            if node_type and node_type not in ("full_file", "prose"):
                annotation_parts.append(node_type)
            if symbol_name and symbol_name != "unnamed":
                annotation_parts.append(symbol_name)
            annotation = " / ".join(annotation_parts) if annotation_parts else ""
            
            sources.append(SourceDoc(label=label, annotation=annotation))
            
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logging.error(f"Error invoking chain: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def start_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    global ingestion_status
    if ingestion_status == "running":
        return {"status": "already_running"}
    
    background_tasks.add_task(bg_ingestion, request.repo_url)
    return {"status": "started"}

@app.get("/ingest/status", response_model=IngestStatusResponse)
async def get_ingest_status():
    return IngestStatusResponse(
        status=ingestion_status,
        logs=in_memory_handler.get_logs()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
