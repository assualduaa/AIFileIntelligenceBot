"""
api.py — FastAPI application: routes, middleware, lifecycle
"""
import os
import uuid
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
from embeddings import load_tfidf_if_available, fit_tfidf_on_corpus
from ingestion import ingest_file, save_upload
from retrieval import retrieve_context, list_documents, delete_document, get_store_stats
from memory import memory_manager, append_session_memory, clear_session_memory
from llm import generate_response

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI File Intelligence Bot",
    description="RAG-powered document intelligence system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.on_event("startup")
def on_startup():
    """Always refit TF-IDF on stored corpus so relevance scores work correctly."""
    try:
        from retrieval import get_collection
        col = get_collection()
        if col.count() > 0:
            docs  = col.get(include=["documents"])
            texts = docs.get("documents", [])
            if texts:
                fit_tfidf_on_corpus(texts)
                logger.info(f"Startup: TF-IDF fitted on {len(texts)} stored chunks.")
            else:
                load_tfidf_if_available()
        else:
            load_tfidf_if_available()
            logger.info("Startup: no stored chunks yet.")
    except Exception as e:
        logger.warning(f"Startup TF-IDF fit failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve React frontend ───────────────────────────────────────────────
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Request/Response Models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:      str
    user_id:    str  = "default_user"
    session_id: str  = "default_session"
    source:     Optional[str] = None  # filter by specific doc
    top_k:      int  = 5


class MemoryUpdateRequest(BaseModel):
    user_id: str
    key:     str
    value:   str


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    stats = get_store_stats()
    return {
        "status":  "ok",
        "version": "1.0.0",
        **stats,
    }


# ── File Upload ────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a file → triggers full ingestion pipeline in background.
    Supported: PDF, DOCX, TXT, PNG/JPG (OCR), MP4/WAV (Whisper)
    """
    filename = file.filename or "unknown_file"
    logger.info(f"Upload received: {filename} ({file.content_type})")

    # Save to temp
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Run ingestion synchronously (for demo reliability)
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

    # Refit TF-IDF on updated corpus so queries use consistent embedding space
    try:
        from retrieval import get_collection
        col = get_collection()
        if col.count() > 0:
            docs = col.get(include=["documents"])
            fit_tfidf_on_corpus(docs.get("documents", []))
    except Exception as e:
        logger.warning(f"TF-IDF refit skipped: {e}")

    return JSONResponse(content=result)


# ── Chat / Query ───────────────────────────────────────────────────────
@app.post("/query")
def query_documents(req: QueryRequest):
    """
    RAG query pipeline:
    1. Retrieve top-k relevant chunks
    2. Build memory context (session + user + semantic)
    3. Construct grounded prompt
    4. Generate LLM response
    5. Update session memory
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Build memory context
    context = memory_manager.build_context(
        user_id=req.user_id,
        session_id=req.session_id,
        query=req.query,
        top_k=req.top_k,
    )

    # Check if any docs are indexed
    if not context["semantic"]:
        return {
            "answer":   "No documents indexed yet. Please upload a file first.",
            "chunks":   [],
            "mode":     "no_context",
        }

    # Build structured prompt
    prompt = memory_manager.build_prompt_context(context, req.query)

    # Generate response
    result = generate_response(
        prompt_context=prompt,
        query=req.query,
        chat_history=context["session"],
    )

    # Update session memory
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


# ── Documents List ─────────────────────────────────────────────────────
@app.get("/documents")
def get_documents():
    """List all indexed documents with metadata."""
    return {"documents": list_documents()}


# ── Delete Document ────────────────────────────────────────────────────
@app.delete("/documents/{filename}")
def remove_document(filename: str):
    """Remove a document and all its chunks from the vector store."""
    deleted = delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    return {"deleted_chunks": deleted, "filename": filename}


# ── Vector Store Stats ─────────────────────────────────────────────────
@app.get("/stats")
def store_stats():
    return get_store_stats()


# ── Session Reset ──────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
def reset_session(session_id: str):
    clear_session_memory(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── User Memory Update ─────────────────────────────────────────────────
@app.post("/memory")
def update_memory(req: MemoryUpdateRequest):
    from memory import set_user_memory
    set_user_memory(req.user_id, req.key, req.value)
    return {"status": "saved", "key": req.key}
