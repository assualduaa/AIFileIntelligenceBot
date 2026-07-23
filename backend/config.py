"""
config.py - Central configuration for AI File Intelligence Bot
v2: Mistral + LangChain + FAISS
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -- Paths ---------------------------------------------------------------
BASE_DIR            = Path(__file__).resolve().parent.parent
UPLOAD_DIR          = BASE_DIR / "uploads"
VECTOR_DIR          = BASE_DIR / "vector_store"
FAISS_INDEX_DIR     = VECTOR_DIR / "faiss_index"
FAISS_METADATA_PATH = VECTOR_DIR / "faiss_metadata.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -- Mistral (Primary LLM) -----------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# -- OpenAI (Legacy / optional fallback) ---------------------------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# -- Gemini (Optional external provider) ----------------------------------
# Uses the raw REST API (https://ai.google.dev/api) via httpx rather than a
# Python SDK — deliberately, since Google's SDK package name/methods have
# churned (google-generativeai -> google-genai) and the REST surface is the
# one thing I could implement with confidence without a live key to test
# against. Verify against https://ai.google.dev/api if a call ever 400s.
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

# -- Shared LLM settings -------------------------------------------------
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# -- Embeddings ----------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# -- RAG / Chunking ------------------------------------------------------
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# -- Legacy ChromaDB (no longer primary, kept for compat) ----------------
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "file_intelligence")

# -- Redis (optional short-term memory) ----------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"

# -- Server --------------------------------------------------------------
HOST         = os.getenv("HOST", "0.0.0.0")
PORT         = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# -- Ollama (Local LLM — CR-01: offline-first primary provider) ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# A first inference call has to load the model into memory before it can
# generate anything — on CPU that alone can take well over a minute for an
# 8B model. 60s was too tight and caused real requests to time out. Later
# calls are much faster since Ollama keeps the model warm in memory.
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

# -- Database (SQLite by default — no external service required) --------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}")

# -- Auth / JWT ------------------------------------------------------------
# IMPORTANT: JWT_SECRET_KEY has an insecure default so the app boots out of
# the box. Set a real random secret in .env before exposing this on a LAN.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-insecure-secret-change-me")
JWT_ALGORITHM  = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# -- Default admin seed (first-run bootstrap only, see database.py) ------
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_EMAIL    = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@local")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "ChangeMe123!")

# -- Per-user storage helpers (Requirement 4: document isolation) --------
def user_upload_dir(user_id: int) -> Path:
    d = UPLOAD_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_vector_dir(user_id: int) -> Path:
    d = VECTOR_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_faiss_index_dir(user_id: int) -> Path:
    d = user_vector_dir(user_id) / "faiss_index"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_faiss_metadata_path(user_id: int) -> Path:
    return user_vector_dir(user_id) / "faiss_metadata.json"

# -- Supported file types ------------------------------------------------
SUPPORTED_TYPES = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".doc":  "docx",
    ".txt":  "txt",
    ".png":  "image",
    ".jpg":  "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".mp4":  "video",
    ".mp3":  "audio",
    ".wav":  "audio",
}
