"""
config.py — Central configuration for AI File Intelligence Bot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
UPLOAD_DIR     = BASE_DIR / "uploads"
VECTOR_DIR     = BASE_DIR / "vector_store"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM ────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── RAG ────────────────────────────────────────────────────────────────
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL  = int(os.getenv("TOP_K_RETRIEVAL", "5"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "file_intelligence")

# ── Redis (optional short-term memory) ─────────────────────────────────
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379")
USE_REDIS        = os.getenv("USE_REDIS", "false").lower() == "true"

# ── Server ─────────────────────────────────────────────────────────────
HOST             = os.getenv("HOST", "0.0.0.0")
PORT             = int(os.getenv("PORT", "8000"))
CORS_ORIGINS     = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Supported file types ────────────────────────────────────────────────
SUPPORTED_TYPES  = {
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
