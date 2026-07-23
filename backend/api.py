"""
api.py - FastAPI application: routes, middleware, lifecycle
v3: Offline (Ollama) + Auth + Multi-user isolation + Persistent chat history

Endpoints:
  Auth
    POST   /auth/register        - create a user account
    POST   /auth/login           - OAuth2 password login -> JWT
    POST   /auth/logout          - stateless (client discards token)
    GET    /users/me             - current user profile

  Admin
    GET    /admin/users          - list all users
    POST   /admin/users          - create a user
    PATCH  /admin/users/{id}/disable - enable/disable a user
    GET    /admin/stats          - system-wide usage counts

  Models (Ollama dynamic model management)
    GET    /models                - installed Ollama models + active model
    POST   /models/refresh        - re-sync installed models from Ollama
    POST   /models/active         - set the active model

  Conversations / Chat History
    GET    /conversations                    - list current user's conversations
    POST   /conversations                    - create a new conversation
    DELETE /conversations/{id}                - delete a conversation
    GET    /conversations/{id}/messages       - list messages in a conversation

  Documents (per-user isolated)
    POST   /upload                - ingest file -> this user's FAISS index
    GET    /documents             - list this user's indexed documents
    DELETE /documents/{filename}  - remove a document (this user only)

  RAG
    POST   /query, /chat          - RAG query, scoped to user + conversation
    POST   /summary               - document summarization
    POST   /recommendations       - suggested questions

  Misc
    GET    /stats                 - this user's vector store stats
    GET    /health                - health check (public)
"""
import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from config import CORS_ORIGINS, BASE_DIR, OLLAMA_BASE_URL
from database import get_db, init_db
from models_db import User, Document, Conversation
from auth import (
    hash_password, authenticate_user, create_access_token,
    get_current_user, require_admin,
)
import model_manager
import chat_history

from ingestion import ingest_file, save_upload
from retrieval import retrieve_context, list_documents, delete_document, get_store_stats
from llm import generate_response, generate_summary, generate_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI File Intelligence Bot",
    description="RAG-powered document intelligence — fully offline via Ollama, multi-user, persistent chat history",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    try:
        from langchain_pipeline import get_embeddings
        get_embeddings()
        logger.info("Startup: embedding model pre-warmed.")
    except Exception as e:
        logger.warning(f"Startup pre-warm skipped: {e}")

    try:
        reachable = model_manager.ollama_is_reachable()
        logger.info(f"Ollama reachable at {OLLAMA_BASE_URL}: {reachable}")
    except Exception as e:
        logger.warning(f"Ollama reachability check failed: {e}")


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

class RegisterRequest(BaseModel):
    username: str
    email:    EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class AdminCreateUserRequest(BaseModel):
    username: str
    email:    EmailStr
    password: str
    role:     str = "user"


class QueryRequest(BaseModel):
    query:           str
    conversation_id: Optional[int] = None
    top_k:           int = 5


class SummaryRequest(BaseModel):
    source: Optional[str] = None
    top_k:  int           = 20


class RecommendationsRequest(BaseModel):
    source: Optional[str] = None
    top_k:  int           = 10


class MemoryUpdateRequest(BaseModel):
    key:   str
    value: str


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = "New Conversation"


class ModelActivateRequest(BaseModel):
    model_name: str
    provider:   str = "ollama"


# ======================================================================
# HEALTH  (public — useful for status widgets before login)
# ======================================================================

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "version":         "3.0.0",
        "llm_provider":    "ollama (primary) + mistral/openai (optional) + local fallback",
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_reachable": model_manager.ollama_is_reachable(),
    }


# ======================================================================
# AUTH
# ======================================================================

@app.post("/auth/register", response_model=UserOut)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == req.username) | (User.email == req.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already registered.")

    # First user ever created becomes admin (bootstrap convenience); everyone after is role="user".
    role = "admin" if db.query(User).count() == 0 else "user"

    user = User(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 password flow. `username` field accepts either username or email."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username/email or password.")

    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token(user)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id": user.id, "username": user.username, "email": user.email,
            "role": user.role,
        },
    }


@app.post("/auth/logout")
def logout(current_user: User = Depends(get_current_user)):
    """JWTs are stateless — logout is a client-side token discard. This endpoint
    exists so the frontend has a clean call to make and confirms the token was valid."""
    return {"status": "logged_out"}


@app.get("/users/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user


# ======================================================================
# ADMIN — USER MANAGEMENT
# ======================================================================

@app.get("/admin/users", response_model=List[UserOut])
def admin_list_users(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    return db.query(User).order_by(User.created_at.desc()).all()


@app.post("/admin/users", response_model=UserOut)
def admin_create_user(
    req: AdminCreateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")
    if db.query(User).filter((User.username == req.username) | (User.email == req.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already registered.")

    user = User(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
        role=req.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.patch("/admin/users/{user_id}/disable", response_model=UserOut)
def admin_toggle_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Toggles is_active. Admins cannot disable themselves (avoids locking everyone out)."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot disable your own account.")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.is_active = not user.is_active
    db.commit()
    db.refresh(user)
    return user


@app.get("/admin/stats")
def admin_stats(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    return {
        "total_users":         db.query(User).count(),
        "active_users":        db.query(User).filter(User.is_active == True).count(),  # noqa: E712
        "total_documents":     db.query(Document).count(),
        "total_conversations": db.query(Conversation).count(),
        "ollama_reachable":    model_manager.ollama_is_reachable(),
    }


# ======================================================================
# MODEL MANAGEMENT (dynamic Ollama model selection)
# ======================================================================

@app.get("/models")
def get_models(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    from llm_service import _PROVIDERS  # provider registry (ollama/mistral/openai/gemini)

    installed = model_manager.list_ollama_models()
    active = model_manager.get_active_model(db)

    external_providers = []
    for name in ("mistral", "openai", "gemini"):
        provider = _PROVIDERS[name]
        available = provider.is_available()
        external_providers.append({
            "provider":  name,
            "available": available,   # False = no API key configured for this provider
            "models":    provider.list_models() if available else [],
        })

    return {
        "installed_ollama_models": installed,
        "ollama_reachable":        model_manager.ollama_is_reachable(),
        "external_providers":      external_providers,
        "active_model": {
            "model_name": active.model_name,
            "provider":   active.provider,
        } if active else None,
        "configured_models": [
            {"id": m.id, "model_name": m.model_name, "provider": m.provider, "active_status": m.active_status}
            for m in model_manager.list_all_models(db)
        ],
    }


@app.post("/models/refresh")
def refresh_models(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    rows = model_manager.refresh_models(db)
    return {
        "status": "refreshed",
        "models": [
            {"id": m.id, "model_name": m.model_name, "provider": m.provider, "active_status": m.active_status}
            for m in rows
        ],
    }


@app.post("/models/active")
def set_active_model(
    req: ModelActivateRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    row = model_manager.set_active_model(db, req.model_name, req.provider)
    return {"status": "activated", "model_name": row.model_name, "provider": row.provider}


# ======================================================================
# CONVERSATIONS / CHAT HISTORY
# ======================================================================

@app.get("/conversations")
def get_conversations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    convs = chat_history.list_conversations(db, current_user.id)
    return {"conversations": [
        {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat(),
         "updated_at": c.updated_at.isoformat()}
        for c in convs
    ]}


@app.post("/conversations")
def create_conversation(
    req: ConversationCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = chat_history.create_conversation(db, current_user.id, req.title or "New Conversation")
    return {"id": conv.id, "title": conv.title, "created_at": conv.created_at.isoformat()}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = chat_history.delete_conversation(db, current_user.id, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = chat_history.get_conversation(db, current_user.id, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    msgs = chat_history.list_messages(db, current_user.id, conversation_id)
    return {"conversation_id": conversation_id, "messages": [
        {
            "id": m.id, "question": m.question, "response": m.response,
            "model_used": m.model_used,
            "referenced_documents": json.loads(m.referenced_documents or "[]"),
            "timestamp": m.timestamp.isoformat(),
        }
        for m in msgs
    ]}


# ======================================================================
# FILE UPLOAD  (per-user isolated)
# ======================================================================

def _ingest_in_background(user_id: int, saved_path: str, filename: str, db_document_id: int):
    """Background task: extract text -> chunk -> embed -> this user's FAISS index."""
    from database import SessionLocal
    try:
        result = ingest_file(user_id, saved_path, filename)
        if result["status"] == "success":
            logger.info(f"BG ingestion done: user_id={user_id} '{filename}' ({result['chunks']} chunks)")
            db = SessionLocal()
            try:
                doc = db.query(Document).filter(Document.id == db_document_id).first()
                if doc:
                    doc.file_type    = result.get("file_type")
                    doc.embedding_id = filename
                    db.commit()
            finally:
                db.close()
        else:
            logger.error(f"BG ingestion failed: user_id={user_id} '{filename}' — {result.get('message')}")
    except Exception as e:
        logger.error(f"BG ingestion exception for user_id={user_id} '{filename}': {e}")


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a document -> save to this user's folder -> schedule ingestion -> return immediately."""
    filename = file.filename or "unknown_file"
    logger.info(f"Upload received: user_id={current_user.id} '{filename}' ({file.content_type})")

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        saved_path = save_upload(current_user.id, tmp_path, filename)
    except Exception as e:
        logger.error(f"Save upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    doc = Document(
        user_id=current_user.id,
        filename=filename,
        filepath=saved_path,
        file_type=None,
        embedding_id=None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(_ingest_in_background, current_user.id, saved_path, filename, doc.id)

    return JSONResponse(content={
        "status":   "processing",
        "filename": filename,
        "document_id": doc.id,
        "message":  "File saved. Indexing in background — poll /documents to track progress.",
    })


# ======================================================================
# RAG CHAT  (/query legacy + /chat new) — per-user + per-conversation
# ======================================================================

def _run_rag_query(req: QueryRequest, db: Session, current_user: User) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    conversation_id = req.conversation_id
    if conversation_id is None:
        conv = chat_history.create_conversation(db, current_user.id)
        conversation_id = conv.id
    else:
        conv = chat_history.get_conversation(db, current_user.id, conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    chunks = retrieve_context(current_user.id, req.query, top_k=req.top_k)
    if not chunks:
        return {"answer": "No documents indexed yet. Please upload a file first.",
                "chunks": [], "mode": "no_context", "model": "none",
                "conversation_id": conversation_id}

    recent_history = chat_history.get_recent_messages(db, conversation_id, limit=6)

    chunk_lines = [
        f"[{i}] Source: {c.get('source','')} (relevance: {c.get('score',0)})\n{c.get('text','')[:600]}"
        for i, c in enumerate(chunks, 1)
    ]
    prompt = "RETRIEVED KNOWLEDGE:\n" + "\n\n".join(chunk_lines) + f"\n\nCURRENT QUESTION:\n{req.query}"

    result = generate_response(db, prompt_context=prompt, query=req.query, chat_history=recent_history)

    referenced_docs = list(dict.fromkeys(c.get("source", "") for c in chunks if c.get("source")))
    chat_history.append_message(
        db, conversation_id, current_user.id,
        question=req.query, response=result["answer"],
        model_used=result.get("model", "unknown"),
        referenced_documents=referenced_docs,
    )

    return {
        "answer":          result["answer"],
        "model":           result["model"],
        "mode":            result["mode"],
        "tokens":          result.get("tokens", 0),
        "chunks":          chunks,
        "conversation_id": conversation_id,
    }


@app.post("/query")
def query_documents(req: QueryRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """RAG query (legacy endpoint name - preserved for backward compat)."""
    return _run_rag_query(req, db, current_user)


@app.post("/chat")
def chat(req: QueryRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    RAG chat endpoint (canonical v3 name).

    Pipeline:
      1. Embed query (HuggingFace sentence-transformers)
      2. FAISS similarity search, scoped to current_user's index -> top-k chunks
      3. Build structured prompt with recent conversation history
      4. LLM Service Layer generates a context-grounded answer (Ollama by default)
      5. Persist the turn to this user's conversation in the DB
    """
    return _run_rag_query(req, db, current_user)


# ======================================================================
# DOCUMENT SUMMARIZATION
# ======================================================================

@app.post("/summary")
def summarize_document(
    req: SummaryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if req.source:
        from langchain_pipeline import fetch_all_chunks_for_source
        chunks = fetch_all_chunks_for_source(current_user.id, req.source)
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document '{req.source}' not found in your index.")
    else:
        chunks = retrieve_context(current_user.id, "document overview summary main content", top_k=req.top_k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents indexed yet.")

    result = generate_summary(db, chunks[: req.top_k])
    return {**result, "source": req.source or "all_documents", "chunks_used": len(chunks[: req.top_k])}


# ======================================================================
# RECOMMENDED QUESTIONS
# ======================================================================

@app.post("/recommendations")
def recommended_questions(
    req: RecommendationsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if req.source:
        from langchain_pipeline import fetch_all_chunks_for_source
        chunks = fetch_all_chunks_for_source(current_user.id, req.source)
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document '{req.source}' not found in your index.")
    else:
        chunks = retrieve_context(current_user.id, "key topics discussed main content overview", top_k=req.top_k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents indexed yet.")

    result = generate_recommendations(db, chunks[: req.top_k])
    return {**result, "source": req.source or "all_documents", "chunks_used": len(chunks[: req.top_k])}


# ======================================================================
# DOCUMENT MANAGEMENT  (per-user)
# ======================================================================

@app.get("/documents")
def get_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return {"documents": list_documents(current_user.id)}


@app.delete("/documents/{filename}")
def remove_document(
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        deleted = delete_document(current_user.id, filename)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in your index.")

    from config import user_upload_dir
    uploads_dir = user_upload_dir(current_user.id)
    for candidate in uploads_dir.glob("**/*"):
        if candidate.name == filename or candidate.stem == filename:
            try:
                candidate.unlink()
            except Exception as e:
                logger.warning(f"Could not delete upload file {candidate}: {e}")

    db.query(Document).filter(Document.user_id == current_user.id, Document.filename == filename).delete()
    db.commit()

    return {"deleted_chunks": deleted, "filename": filename}


# ======================================================================
# STATS, MEMORY  (per-user)
# ======================================================================

@app.get("/stats")
def store_stats(current_user: User = Depends(get_current_user)):
    return get_store_stats(current_user.id)


@app.post("/memory")
def update_memory(req: MemoryUpdateRequest, current_user: User = Depends(get_current_user)):
    from memory import set_user_memory
    set_user_memory(str(current_user.id), req.key, req.value)
    return {"status": "saved", "key": req.key}
