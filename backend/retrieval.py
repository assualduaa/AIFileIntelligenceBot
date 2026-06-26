"""
retrieval.py — ChromaDB vector store + semantic retrieval engine
Uses vector similarity when embeddings work; falls back to keyword
scoring when all similarity scores are near zero (TF-IDF cold start).
"""
import re
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from config import VECTOR_DIR, CHROMA_COLLECTION, TOP_K_RETRIEVAL
from embeddings import embed_text

logger = logging.getLogger(__name__)

# ── ChromaDB Client (singleton) ────────────────────────────────────────
_client     = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(VECTOR_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION}' ready. "
                    f"Count: {_collection.count()}")
    return _collection


# ── Store Documents ────────────────────────────────────────────────────
def store_documents(documents: List[Dict[str, Any]]) -> int:
    """
    Insert chunk documents into ChromaDB.
    Returns number of chunks stored.
    """
    if not documents:
        return 0

    collection = get_collection()

    ids        = [doc["chunk_id"]  for doc in documents]
    embeddings = [doc["embedding"] for doc in documents]
    texts      = [doc["text"]      for doc in documents]
    metadatas  = [
        {
            "source":      doc["source"],
            "file_type":   doc["file_type"],
            "language":    doc["language"],
            "timestamp":   doc["timestamp"],
            "chunk_index": doc["chunk_index"],
        }
        for doc in documents
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

    count = len(documents)
    logger.info(f"Stored {count} chunks in ChromaDB")
    return count


# ── Retrieve Relevant Chunks ───────────────────────────────────────────
def retrieve_context(
    query:    str,
    top_k:    int = TOP_K_RETRIEVAL,
    source:   Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic similarity search.
    Optionally filter by source filename.
    Returns top-k ranked chunks with metadata.
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Vector store is empty — no documents indexed yet.")
        return []

    query_embedding = embed_text(query)

    where_filter = {"source": source} if source else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results.get("documents"):
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":       text,
                "source":     meta.get("source", "unknown"),
                "file_type":  meta.get("file_type", ""),
                "language":   meta.get("language", "en"),
                "timestamp":  meta.get("timestamp", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "score":      round(1 - dist, 4),  # cosine similarity
            })

    # If all scores are ~0 (TF-IDF cold start / zero-vector query),
    # fall back to keyword-based retrieval which always works
    if chunks and all(c["score"] < 0.05 for c in chunks):
        logger.warning("Vector scores all near 0 — switching to keyword retrieval.")
        return _keyword_retrieve(query, top_k, source)

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}'")
    return chunks


def _keyword_retrieve(
    query:  str,
    top_k:  int = TOP_K_RETRIEVAL,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    BM25-style keyword retrieval — no embeddings needed.
    Scores each chunk by how many query words it contains,
    with bonus for exact phrase matches.
    """
    collection = get_collection()
    where_filter = {"source": source} if source else None

    if where_filter:
        all_docs = collection.get(where=where_filter, include=["documents", "metadatas"])
    else:
        all_docs = collection.get(include=["documents", "metadatas"])

    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    query_words = [w for w in query_clean.split() if len(w) > 2]

    scored = []
    for text, meta in zip(all_docs.get("documents", []), all_docs.get("metadatas", [])):
        text_lower = text.lower()
        # Word-level match score
        word_score = sum(1 for w in query_words if w in text_lower)
        # Bonus: exact phrase match
        phrase_bonus = 3 if query_clean.strip() in text_lower else 0
        total = word_score + phrase_bonus

        scored.append({
            "text":        text,
            "source":      meta.get("source", "unknown"),
            "file_type":   meta.get("file_type", ""),
            "language":    meta.get("language", "en"),
            "timestamp":   meta.get("timestamp", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "score":       round(total / max(len(query_words), 1), 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Keyword retrieval: top score={scored[0]['score'] if scored else 0}")
    return scored[:top_k]


# ── List Indexed Documents ─────────────────────────────────────────────
def list_documents() -> List[Dict[str, Any]]:
    """Return unique indexed files with chunk counts."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    seen    = {}

    for meta in results.get("metadatas", []):
        src = meta.get("source", "unknown")
        if src not in seen:
            seen[src] = {
                "source":     src,
                "file_type":  meta.get("file_type", ""),
                "language":   meta.get("language", ""),
                "timestamp":  meta.get("timestamp", ""),
                "chunk_count": 0,
            }
        seen[src]["chunk_count"] += 1

    return sorted(seen.values(), key=lambda x: x["timestamp"], reverse=True)


# ── Delete Document ────────────────────────────────────────────────────
def delete_document(source: str) -> int:
    """Remove all chunks belonging to a source file."""
    collection = get_collection()
    results    = collection.get(where={"source": source}, include=["metadatas"])
    ids        = results.get("ids", [])

    if ids:
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for '{source}'")

    return len(ids)


# ── Store Stats ────────────────────────────────────────────────────────
def get_store_stats() -> Dict[str, Any]:
    """Return vector store statistics."""
    collection = get_collection()
    docs       = list_documents()
    return {
        "total_chunks":    collection.count(),
        "total_documents": len(docs),
        "collection":      CHROMA_COLLECTION,
    }
