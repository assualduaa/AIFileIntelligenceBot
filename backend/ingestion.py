"""
ingestion.py — End-to-end file ingestion pipeline
v3: Scoped by user_id (CR-01 Requirement 4) — files land in each user's own
uploads directory and are indexed into that user's own FAISS store.
Upload → Detect type → Extract → Chunk → Embed → Store (per-user)
"""
import shutil
import logging
from pathlib import Path
from typing import Dict, Any

from config import SUPPORTED_TYPES, user_upload_dir
from processing import extract_text
from embeddings import build_chunk_documents
from retrieval  import store_documents

logger = logging.getLogger(__name__)


def ingest_file(user_id: int, file_path: str, original_filename: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single file, scoped to one user.

    Steps:
    1. Detect file type
    2. Extract text (with fallback)
    3. Semantic chunking
    4. Embed chunks
    5. Store in this user's FAISS index

    Returns status dict.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    file_type = SUPPORTED_TYPES.get(ext)
    if not file_type:
        return {
            "status":   "error",
            "message":  f"Unsupported file type: {ext}",
            "filename": original_filename,
        }

    logger.info(f"Ingesting '{original_filename}' for user_id={user_id} as type={file_type}")

    extraction = extract_text(str(path), file_type)
    if extraction["status"] != "success" or not extraction["text"]:
        return {
            "status":   "error",
            "message":  extraction["status"],
            "filename": original_filename,
            "chars":    0,
            "chunks":   0,
        }

    text     = extraction["text"]
    language = extraction["language"]
    logger.info(f"Extracted {len(text)} chars | lang={language}")

    documents = build_chunk_documents(
        text=text,
        filename=original_filename,
        file_type=file_type,
        language=language,
    )

    if not documents:
        return {
            "status":   "error",
            "message":  "No content chunks generated",
            "filename": original_filename,
        }

    stored = store_documents(user_id, documents)

    logger.info(f"Ingested '{original_filename}' for user_id={user_id}: {stored} chunks stored")
    return {
        "status":    "success",
        "filename":  original_filename,
        "file_type": file_type,
        "language":  language,
        "chars":     len(text),
        "chunks":    stored,
    }


def save_upload(user_id: int, tmp_path: str, original_filename: str) -> str:
    """Save uploaded file to this user's persistent uploads directory."""
    upload_dir = user_upload_dir(user_id)
    dest = upload_dir / original_filename
    counter = 1
    stem    = dest.stem
    suffix  = dest.suffix
    while dest.exists():
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(tmp_path, dest)
    return str(dest)
