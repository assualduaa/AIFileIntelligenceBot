"""
api.py - FastAPI application: routes, middleware, lifecycle
v2: Mistral + LangChain + FAISS

Endpoints:
  POST /upload          - ingest file -> FAISS index
  POST /query           - RAG query (legacy name)
  POST /chat            - RAG query (new canonical name)
  POST /summary         - document summarization
  POST /recommendations - suggested questions
  GET  /documents       - list indexed documents
  DELETE /documents/{f} - remove document
  GET  /stats           - vector store stats
  GET  /health          - health check
  DELETE /session/{id}  - clear session memory
  POST /memory          - update user memory
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import CORS_ORIGINS, BASE_DIR
from ingestion import ingest_file, save_upload
from retrieval import retrieve_context, list_documents, delete_document, get_store_stats
from memory import memory_manager, append_session_memory, clear_session_memory
from llm import generate_response, generate_summary, generate_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI File Intelligence Bot",
    description="RAG-powered document intelligence - Mistral + LangChain + FAISS",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
def on_startup():
    try:
        from langchain_pipeline import get_vector_store, get_embeddings
        get_embeddings()
        store = get_vector_store()
        total = store.index.ntotal if store else 0
        logger.info(f"Startup: FAISS index loaded ({total} vectors).")
    except Exception as e:
        logger.warning(f"Startup pre-warm skipped: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ======================================================================
# REQUEST / RESPONSE MODELS
# ======================================================================

class QueryRequest(BaseModel):
    query:      str
    user_id:    str           = "default_user"
    session_id: str           = "default_session"
    source:     Optional[str] = None
    top_k:      int           = 5


class SummaryRequest(BaseModel):
    source: Optional[str] = None
    top_k:  int           = 20


class RecommendationsRequest(BaseModel):
    source: Optional[str] = None
    top_k:  int           = 10


class MemoryUpdateRequest(BaseModel):
    user_id: str
    key:     str
    value:   str


# ======================================================================
# HEALTH
# ======================================================================

@app.get("/health")
def health():
    stats = get_store_stats()
    return {"status": "ok", "version": "2.0.0", "llm": "mistral",
            "vector_store": "faiss", **stats}


# ======================================================================
# FILE UPLOAD
# ======================================================================

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a document -> ingest -> embed -> FAISS index."""
    filename = file.filename or "unknown_file"
    logger.info(f"Upload received: {filename} ({file.content_type})")

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        saved_path = save_upload(tmp_path, filename)
        result     = ingest_file(saved_path, filename)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if result["status"] != "success":
        raise HTTPException(status_code=422, detail=result.get("message", "Ingestion failed"))

    return JSONResponse(content=result)


# ======================================================================
# RAG CHAT  (/query legacy + /chat new)
# ======================================================================

def _run_rag_query(req: QueryRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    context = memory_manager.build_context(
        user_id=req.user_id,
        session_id=req.session_id,
        query=req.query,
        top_k=req.top_k,
    )

    if not context["semantic"]:
        return {"answer": "No documents indexed yet. Please upload a file first.",
                "chunks": [], "mode": "no_context", "model": "none"}

    prompt = memory_manager.build_prompt_context(context, req.query)
    result = generate_response(
        prompt_context=prompt,
        query=req.query,
        chat_history=context["session"],
    )

    append_session_memory(req.session_id, "user",      req.query)
    append_session_memory(req.session_id, "assistant", result["answer"])

    return {
        "answer":     result["answer"],
        "model":      result["model"],
        "mode":       result["mode"],
        "tokens":     result.get("tokens", 0),
        "chunks":     context["semantic"],
        "session_id": req.session_id,
    }


@app.post("/query")
def query_documents(req: QueryRequest):
    """RAG query (legacy endpoint - preserved for backward compat)."""
    return _run_rag_query(req)


@app.post("/chat")
def chat(req: QueryRequest):
    """
    RAG chat endpoint (canonical v2 name).

    Pipeline:
      1. Embed query via HuggingFace sentence-transformers
      2. FAISS similarity search -> top-k chunks
      3. Build structured prompt with session + user memory context
      4. Mistral LLM generates context-grounded answer
      5. Update session history
    """
    return _run_rag_query(req)


# ======================================================================
# DOCUMENT SUMMARIZATION
# ======================================================================

@app.post("/summary")
def summarize_document(req: SummaryRequest):
    """
    Generate a structured document summary using Mistral + LangChain.

    If source is specified, summarizes only that document.
    If omitted, uses all indexed content.

    Response: {"summary": "...", "key_points": [...], "source": "...", "chunks_used": N}
    """
    if req.source:
        from langchain_pipeline import fetch_all_chunks_for_source
        chunks = fetch_all_chunks_for_source(req.source)
        if not chunks:
            raise HTTPException(status_code=404,
                                detail=f"Document '{req.source}' not found in index.")
    else:
        chunks = retrieve_context("document overview summary main content", top_k=req.top_k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents indexed yet.")

    result = generate_summary(chunks[: req.top_k])
    return {**result, "source": req.source or "all_documents",
            "chunks_used": len(chunks[: req.top_k])}


# ======================================================================
# RECOMMENDED QUESTIONS
# ======================================================================

@app.post("/recommendations")
def recommended_questions(req: RecommendationsRequest):
    """
    Generate 5-8 exploration questions from document content via Mistral + LangChain.

    Response: {"recommended_questions": [...], "source": "...", "chunks_used": N}
    """
    if req.source:
        from langchain_pipeline import fetch_all_chunks_for_source
        chunks = fetch_all_chunks_for_source(req.source)
        if not chunks:
            raise HTTPException(status_code=404,
                                detail=f"Document '{req.source}' not found in index.")
    else:
        chunks = retrieve_context("key topics discussed main content overview", top_k=req.top_k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents indexed yet.")

    result = generate_recommendations(chunks[: req.top_k])
    return {**result, "source": req.source or "all_documents",
            "chunks_used": len(chunks[: req.top_k])}


# ======================================================================
# DOCUMENT MANAGEMENT
# ======================================================================

@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.delete("/documents/{filename}")
def remove_document(filename: str):
    try:
        deleted = delete_document(filename)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in index.")

    # Also remove the physical file from uploads folder (best-effort)
    uploads_dir = BASE_DIR / "uploads"
    for candidate in uploads_dir.glob("**/*"):
        if candidate.name == filename or candidate.stem == filename:
            try:
                candidate.unlink()
                logger.info(f"Deleted upload file: {candidate}")
            except Exception as e:
                logger.warning(f"Could not delete upload file {candidate}: {e}")

    return {"deleted_chunks": deleted, "filename": filename}


# ======================================================================
# STATS, SESSION, MEMORY
# ======================================================================

@app.get("/stats")
def store_stats():
    return get_store_stats()


@app.delete("/session/{session_id}")
def reset_session(session_id: str):
    clear_session_memory(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/memory")
def update_memory(req: MemoryUpdateRequest):
    from memory import set_user_memory
    set_user_memory(req.user_id, req.key, req.value)
    return {"status": "saved", "key": req.key}
