"""
retrieval.py - FAISS-backed semantic retrieval engine
v2: Replaces ChromaDB with FAISS via langchain_pipeline.py

Public API (backward compatible):
  store_documents(documents)     -> int
  retrieve_context(query, top_k) -> List[Dict]
  list_documents()               -> List[Dict]
  delete_document(source)        -> int
  get_store_stats()              -> Dict
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
)

logger = logging.getLogger(__name__)


def store_documents(documents: List[Dict[str, Any]]) -> int:
    """Index chunk documents into FAISS. Returns number of chunks stored."""
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

    stored = add_documents_to_faiss(texts, metadatas)

    if documents:
        sample = documents[0]
        upsert_file_metadata(
            filename    = sample["source"],
            file_type   = sample["file_type"],
            language    = sample["language"],
            chunk_count = stored,
            timestamp   = sample.get("timestamp", datetime.utcnow().isoformat()),
        )

    logger.info(f"Stored {stored} chunks in FAISS for '{documents[0]['source']}'")
    return stored


def retrieve_context(
    query:  str,
    top_k:  int = TOP_K_RETRIEVAL,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic similarity search using FAISS. Returns top-k chunks."""
    if get_total_vectors() == 0:
        logger.warning("FAISS index is empty.")
        return []

    chunks = search_faiss(query, top_k=top_k, source=source)
    logger.info(f"Retrieved {len(chunks)} chunks for: '{query[:60]}'")
    return chunks


def list_documents() -> List[Dict[str, Any]]:
    """Return all indexed files with metadata."""
    return list_file_metadata()


def delete_document(source: str) -> int:
    """Remove all chunks for a source file. Rebuilds FAISS index.
    Raises RuntimeError if the rebuild fails."""
    deleted = rebuild_faiss_without(source)  # raises RuntimeError on failure
    if deleted > 0:
        remove_file_metadata(source)
        logger.info(f"Deleted '{source}': {deleted} chunks removed.")
    else:
        logger.warning(f"Document '{source}' not found in FAISS.")
    return deleted


def get_store_stats() -> Dict[str, Any]:
    """Return vector store statistics."""
    docs  = list_documents()
    total = get_total_vectors()
    return {
        "total_chunks":    total,
        "total_documents": len(docs),
        "vector_store":    "FAISS",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
