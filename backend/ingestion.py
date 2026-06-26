"""
ingestion.py — End-to-end file ingestion pipeline
Upload → Detect type → Extract → Chunk → Embed → Store
"""
import shutil
import logging
from pathlib import Path
from typing import Dict, Any

from config import UPLOAD_DIR, SUPPORTED_TYPES
from processing import extract_text
from embeddings import build_chunk_documents
from retrieval  import store_documents, delete_document

logger = logging.getLogger(__name__)


def ingest_file(file_path: str, original_filename: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single file.

    Steps:
    1. Detect file type
    2. Extract text (with fallback)
    3. Semantic chunking
    4. Embed chunks
    5. Store in ChromaDB

    Returns status dict.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    # Step 1: Detect type
    file_type = SUPPORTED_TYPES.get(ext)
    if not file_type:
        return {
            "status":   "error",
            "message":  f"Unsupported file type: {ext}",
            "filename": original_filename,
        }

    logger.info(f"Ingesting '{original_filename}' as type={file_type}")

    # Step 2: Extract text
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

    # Step 3+4: Chunk + Embed
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

    # Step 5: Store
    stored = store_documents(documents)

    logger.info(f"✓ Ingested '{original_filename}': {stored} chunks stored")
    return {
        "status":    "success",
        "filename":  original_filename,
        "file_type": file_type,
        "language":  language,
        "chars":     len(text),
        "chunks":    stored,
    }


def save_upload(tmp_path: str, original_filename: str) -> str:
    """Save uploaded file to persistent uploads directory."""
    dest = UPLOAD_DIR / original_filename
    # Handle naming conflicts
    counter = 1
    stem    = dest.stem
    suffix  = dest.suffix
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(tmp_path, dest)
    return str(dest)
