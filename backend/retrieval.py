"""
retrieval.py - FAISS-backed semantic retrieval engine
v3: Every function is scoped by user_id (CR-01 Requirement 4 — multi-user
document isolation). Delegates to langchain_pipeline.py's per-user stores.

Public API:
  store_documents(user_id, documents)     -> int
  retrieve_context(user_id, query, top_k) -> List[Dict]
  list_documents(user_id)                 -> List[Dict]
  delete_document(user_id, source)        -> int
  get_store_stats(user_id)                -> Dict
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import TOP_K_RETRIEVAL
from langchain_pipeline import (
    add_documents_to_faiss,
    search_faiss,
    rebuild_faiss_without,
    get_total_vectors,
    upsert_file_metadata,
    remove_file_metadata,
    list_file_metadata,
    get_embeddings,
)

logger = logging.getLogger(__name__)


def store_documents(user_id: int, documents: List[Dict[str, Any]]) -> int:
    """Index chunk documents into this user's FAISS store. Returns number of chunks stored."""
    if not documents:
        return 0

    texts = [doc["text"] for doc in documents]
    metadatas = [
        {
            "source":      doc["source"],
            "file_type":   doc["file_type"],
            "language":    doc["language"],
            "timestamp":   doc["timestamp"],
            "chunk_index": doc["chunk_index"],
        }
        for doc in documents
    ]

    precomputed = [doc["embedding"] for doc in documents if "embedding" in doc]
    stored = add_documents_to_faiss(
        user_id, texts, metadatas,
        precomputed_embeddings=precomputed if len(precomputed) == len(texts) else None,
    )

    if documents:
        sample = documents[0]
        upsert_file_metadata(
            user_id,
            filename    = sample["source"],
            file_type   = sample["file_type"],
            language    = sample["language"],
            chunk_count = stored,
            timestamp   = sample.get("timestamp", datetime.utcnow().isoformat()),
        )

    logger.info(f"Stored {stored} chunks in FAISS for user_id={user_id}, source='{documents[0]['source']}'")
    return stored


def retrieve_context(
    user_id: int,
    query:  str,
    top_k:  int = TOP_K_RETRIEVAL,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic similarity search using this user's FAISS index. Returns top-k chunks."""
    if get_total_vectors(user_id) == 0:
        logger.info(f"FAISS index empty for user_id={user_id}.")
        return []

    chunks = search_faiss(user_id, query, top_k=top_k, source=source)
    logger.info(f"Retrieved {len(chunks)} chunks for user_id={user_id}: '{query[:60]}'")
    return chunks


def list_documents(user_id: int) -> List[Dict[str, Any]]:
    """Return all indexed files with metadata, for this user only."""
    return list_file_metadata(user_id)


def delete_document(user_id: int, source: str) -> int:
    """Remove all chunks belonging to `source` for this user. Rebuilds their FAISS index.
    Raises RuntimeError if the rebuild fails."""
    deleted = rebuild_faiss_without(user_id, source)
    if deleted > 0:
        remove_file_metadata(user_id, source)
        logger.info(f"Deleted '{source}' for user_id={user_id}: {deleted} chunks removed.")
    else:
        logger.warning(f"Document '{source}' not found in FAISS for user_id={user_id}.")
    return deleted


def get_store_stats(user_id: int) -> Dict[str, Any]:
    """Return vector store statistics for this user."""
    docs  = list_documents(user_id)
    total = get_total_vectors(user_id)
    return {
        "total_chunks":    total,
        "total_documents": len(docs),
        "vector_store":    "FAISS",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
