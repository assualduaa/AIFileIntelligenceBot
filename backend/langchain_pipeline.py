"""
langchain_pipeline.py — LangChain Workflow Orchestration Engine
v3: Per-user FAISS vector stores (multi-tenant isolation) + pluggable LLM
    Service Layer (Ollama primary / Mistral / OpenAI) + RAG / Summary /
    Recommendations chains.

Architecture:
  Document text
       |
  LangChain RecursiveCharacterTextSplitter
       |
  HuggingFace Embeddings (all-MiniLM-L6-v2)  -- shared across users
       |
  FAISS Vector Store (persistent, one index per user_id)
       |
  Similarity Search (Top-K, scoped to the requesting user)
       |
  LLM Service Layer -> Ollama (local, default) / External API (future)
       |
  Structured Response

CR-01 Requirement 4 (document isolation): every FAISS index, its metadata
file, and its on-disk directory are keyed by user_id. There is no shared
global index anymore — a user's similarity search can only ever see vectors
written under their own user_id directory.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    user_faiss_index_dir, user_faiss_metadata_path,
)

logger = logging.getLogger(__name__)

# ── Singleton / per-user cache state ───────────────────────────────────
_vector_stores: Dict[int, Any] = {}   # user_id -> LangChain FAISS wrapper
_embeddings = None                     # HuggingFace embeddings model (shared, stateless)


# ══════════════════════════════════════════════════════════════════════
# EMBEDDINGS  (shared across users — the model itself holds no document data)
# ══════════════════════════════════════════════════════════════════════

def get_embeddings():
    """Return singleton HuggingFace embedding model."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"HuggingFace embeddings loaded: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"HuggingFaceEmbeddings unavailable ({e}). Trying SentenceTransformer...")
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            _embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info(f"SentenceTransformerEmbeddings loaded: {EMBEDDING_MODEL}")
        except Exception as e2:
            logger.error(f"All embedding models failed: {e2}")
            raise RuntimeError(f"Could not load any embedding model: {e2}")

    return _embeddings


# ══════════════════════════════════════════════════════════════════════
# FAISS VECTOR STORE (per-user)
# ══════════════════════════════════════════════════════════════════════

def get_vector_store(user_id: int):
    """Return this user's singleton FAISS vector store (loaded from disk if available)."""
    if user_id not in _vector_stores:
        _vector_stores[user_id] = _load_faiss_from_disk(user_id)
    return _vector_stores[user_id]


def _load_faiss_from_disk(user_id: int):
    """Attempt to load this user's persisted FAISS index. Returns None if not found."""
    from langchain_community.vectorstores import FAISS

    index_dir = user_faiss_index_dir(user_id)
    index_file = index_dir / "index.faiss"
    if not index_file.exists():
        logger.info(f"No persisted FAISS index for user_id={user_id} — will create on first upload.")
        return None

    try:
        store = FAISS.load_local(
            str(index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        total = store.index.ntotal
        logger.info(f"FAISS index loaded for user_id={user_id} ({total} vectors) from {index_dir}")
        return store
    except Exception as e:
        logger.warning(f"Could not load FAISS index for user_id={user_id}: {e}")
        return None


def _persist_faiss(user_id: int):
    """Save this user's current FAISS index to disk."""
    store = _vector_stores.get(user_id)
    if store is None:
        return
    try:
        index_dir = user_faiss_index_dir(user_id)
        store.save_local(str(index_dir))
        logger.info(f"FAISS index saved for user_id={user_id} ({store.index.ntotal} vectors).")
    except Exception as e:
        logger.error(f"FAISS save failed for user_id={user_id}: {e}")


def add_documents_to_faiss(
    user_id: int,
    texts: List[str],
    metadatas: List[Dict],
    precomputed_embeddings: Optional[List[List[float]]] = None,
) -> int:
    """
    Add chunked documents to this user's FAISS store and persist.
    If precomputed_embeddings are provided they are used directly (no re-embedding).
    Creates the store if it doesn't exist yet. Returns the number of chunks indexed.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    if not texts:
        return 0

    emb_model = get_embeddings()
    store = _vector_stores.get(user_id)

    if precomputed_embeddings and len(precomputed_embeddings) == len(texts):
        text_emb_pairs = list(zip(texts, precomputed_embeddings))
        if store is None:
            store = FAISS.from_embeddings(text_emb_pairs, emb_model, metadatas=metadatas)
            logger.info(f"Created new FAISS index for user_id={user_id} with {len(texts)} pre-embedded docs.")
        else:
            store.add_embeddings(text_emb_pairs, metadatas=metadatas)
            logger.info(f"Added {len(texts)} pre-embedded docs to FAISS index for user_id={user_id}.")
    else:
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        if store is None:
            store = FAISS.from_documents(docs, emb_model)
            logger.info(f"Created new FAISS index for user_id={user_id} with {len(docs)} documents.")
        else:
            store.add_documents(docs)
            logger.info(f"Added {len(docs)} documents to FAISS index for user_id={user_id}.")

    _vector_stores[user_id] = store
    _persist_faiss(user_id)
    return len(texts)


def search_faiss(
    user_id: int,
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic similarity search scoped to this user's FAISS index only."""
    store = get_vector_store(user_id)
    if store is None:
        return []

    try:
        fetch_k = top_k * 4 if source else top_k
        results = store.similarity_search_with_score(query, k=min(fetch_k, store.index.ntotal))

        chunks = []
        for doc, dist in results:
            if source and doc.metadata.get("source") != source:
                continue
            similarity = max(0.0, round(1.0 - float(dist) / 2.0, 4))
            chunks.append({
                "text":        doc.page_content,
                "source":      doc.metadata.get("source", "unknown"),
                "file_type":   doc.metadata.get("file_type", ""),
                "language":    doc.metadata.get("language", "en"),
                "timestamp":   doc.metadata.get("timestamp", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score":       similarity,
            })
            if len(chunks) >= top_k:
                break

        return chunks

    except Exception as e:
        logger.error(f"FAISS search failed for user_id={user_id}: {e}")
        return []


def fetch_all_chunks_for_source(user_id: int, source: str) -> List[Dict[str, Any]]:
    """Return all indexed chunks for a specific source file, scoped to this user."""
    store = get_vector_store(user_id)
    if store is None:
        return []

    chunks = []
    try:
        for idx, doc_id in store.index_to_docstore_id.items():
            doc = store.docstore.search(doc_id)
            if hasattr(doc, "page_content") and doc.metadata.get("source") == source:
                chunks.append({
                    "text":        doc.page_content,
                    "source":      doc.metadata.get("source", "unknown"),
                    "file_type":   doc.metadata.get("file_type", ""),
                    "language":    doc.metadata.get("language", "en"),
                    "timestamp":   doc.metadata.get("timestamp", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "score":       1.0,
                })
        chunks.sort(key=lambda x: x["chunk_index"])
    except Exception as e:
        logger.error(f"fetch_all_chunks_for_source failed for user_id={user_id}: {e}")

    return chunks


def _cleanup_faiss_files(user_id: int):
    """
    Windows-safe cleanup of this user's FAISS index files.
    Retries per-file deletion with exponential back-off to handle
    memory-mapped file locks that Windows holds briefly after the
    Python GC releases the FAISS store object.
    """
    import gc
    import time

    gc.collect()
    time.sleep(0.15)

    index_dir = user_faiss_index_dir(user_id)
    for fname in ["index.faiss", "index.pkl"]:
        fpath = index_dir / fname
        for attempt in range(6):
            try:
                if fpath.exists():
                    fpath.unlink()
                break
            except (PermissionError, OSError):
                time.sleep(0.25 * (attempt + 1))

    try:
        index_dir.rmdir()
    except Exception:
        pass

    index_dir.mkdir(parents=True, exist_ok=True)


def rebuild_faiss_without(user_id: int, source: str) -> int:
    """
    Delete all chunks belonging to `source` (for this user only) by rebuilding
    their FAISS index. FAISS does not support in-place deletion, so we rebuild.
    Returns number of chunks deleted. Raises RuntimeError on failure.
    """
    from langchain_community.vectorstores import FAISS

    store = get_vector_store(user_id)
    if store is None:
        return 0

    kept_docs: List = []
    kept_vecs: List = []
    deleted = 0

    for idx in range(store.index.ntotal):
        doc_id = store.index_to_docstore_id.get(idx)
        if doc_id is None:
            continue
        doc = store.docstore.search(doc_id)
        if not hasattr(doc, "page_content"):
            continue
        if doc.metadata.get("source") == source:
            deleted += 1
        else:
            kept_docs.append(doc)
            try:
                kept_vecs.append(store.index.reconstruct(idx).tolist())
            except Exception:
                kept_vecs.append(None)

    if deleted == 0:
        return 0

    try:
        _vector_stores[user_id] = None  # release mmap references before file ops

        if not kept_docs:
            _cleanup_faiss_files(user_id)
        else:
            valid_pairs = [(d, v) for d, v in zip(kept_docs, kept_vecs) if v is not None]
            if not valid_pairs:
                raise RuntimeError("Could not reconstruct any vectors from FAISS index.")

            valid_docs, valid_vecs = zip(*valid_pairs)
            text_emb_pairs = [(d.page_content, v) for d, v in zip(valid_docs, valid_vecs)]
            _vector_stores[user_id] = FAISS.from_embeddings(
                text_emb_pairs,
                get_embeddings(),
                metadatas=[d.metadata for d in valid_docs],
            )
            _persist_faiss(user_id)

        logger.info(f"FAISS rebuilt for user_id={user_id}: removed {deleted} chunks from '{source}'.")
    except Exception as e:
        logger.error(f"FAISS rebuild failed for user_id={user_id}: {e}")
        raise RuntimeError(f"Failed to rebuild FAISS index after removing '{source}': {e}")

    return deleted


def get_total_vectors(user_id: int) -> int:
    """Return total number of vectors in this user's FAISS index."""
    store = get_vector_store(user_id)
    if store is None:
        return 0
    try:
        return store.index.ntotal
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════
# FILE-LEVEL METADATA  (per-user JSON — FAISS stores chunk-level only)
# ══════════════════════════════════════════════════════════════════════

def load_file_metadata(user_id: int) -> Dict[str, Any]:
    path = user_faiss_metadata_path(user_id)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_file_metadata(user_id: int, meta: Dict[str, Any]):
    path = user_faiss_metadata_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def upsert_file_metadata(
    user_id: int,
    filename: str,
    file_type: str,
    language: str,
    chunk_count: int,
    timestamp: str,
):
    meta = load_file_metadata(user_id)
    existing = meta.get(filename, {})
    meta[filename] = {
        "source":      filename,
        "file_type":   file_type,
        "language":    language,
        "chunk_count": existing.get("chunk_count", 0) + chunk_count,
        "timestamp":   timestamp,
    }
    _save_file_metadata(user_id, meta)


def remove_file_metadata(user_id: int, filename: str):
    meta = load_file_metadata(user_id)
    meta.pop(filename, None)
    _save_file_metadata(user_id, meta)


def list_file_metadata(user_id: int) -> List[Dict[str, Any]]:
    meta = load_file_metadata(user_id)
    return sorted(meta.values(), key=lambda x: x.get("timestamp", ""), reverse=True)


# ══════════════════════════════════════════════════════════════════════
# RAG / SUMMARY / RECOMMENDATIONS CHAINS  (via LLM Service Layer)
# ══════════════════════════════════════════════════════════════════════

_RAG_SYSTEM = """You are an AI Document Intelligence Assistant.

RULES:
- Answer ONLY using the retrieved document context provided. Do NOT use outside knowledge.
- Be concise and factual. No hallucination.
- If context is insufficient: respond exactly "The answer is not available in the provided document context."
- Always respond in English.
- Short factual answers: 1-2 sentences. Explanatory answers: 3-5 sentences max.
"""

def run_rag_chain(db, query: str, context_chunks: List[Dict]) -> str:
    """Retrieved chunks -> structured prompt -> active LLM provider -> answer."""
    if not context_chunks:
        return "No documents indexed yet. Please upload a file first."

    context_str = "\n\n".join([
        f"[Source: {c['source']} | Chunk #{c['chunk_index']}]\n{c['text']}"
        for c in context_chunks
    ])
    user_content = f"Document Context:\n{context_str}\n\nQuestion: {query}"

    from llm_service import invoke_raw
    answer = invoke_raw(db, _RAG_SYSTEM, user_content)
    if answer:
        return answer

    try:
        from llm import _smart_synthesize, _build_prompt_str, _is_readable
        prompt_str = _build_prompt_str(context_chunks, query)
        answer = _smart_synthesize(query, prompt_str)
        if answer and _is_readable(answer):
            return answer
    except Exception as e:
        logger.warning(f"Local synthesizer failed: {e}")

    return (
        "No configured LLM provider is currently reachable and the document content could not be "
        "synthesized locally. Check that Ollama is running (OLLAMA_BASE_URL) or that a MISTRAL_API_KEY "
        "/ OPENAI_API_KEY is set."
    )


_SUMMARY_SYSTEM = """You are a professional document summarization expert.
Always respond with valid JSON only — no extra text, no markdown fences.
JSON schema:
{
  "summary": "<3-5 sentence summary of the document>",
  "key_points": ["<point 1>", "<point 2>", ...(5-7 points)]
}
"""

def run_summary_chain(db, context_chunks: List[Dict]) -> Dict[str, Any]:
    if not context_chunks:
        return {"summary": "No document content available.", "key_points": []}

    combined = "\n\n".join(c["text"] for c in context_chunks[:15])
    user_content = f"Summarize the following document:\n\n{combined}"

    from llm_service import invoke_raw
    raw = invoke_raw(db, _SUMMARY_SYSTEM, user_content)
    if raw:
        result = _parse_json_response(raw)
        if result:
            return result

    return _fallback_summary(context_chunks)


def _fallback_summary(chunks: List[Dict]) -> Dict[str, Any]:
    """Simple extractive summary when no LLM provider is reachable."""
    all_text = " ".join(c["text"] for c in chunks[:10])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", all_text) if len(s.strip()) > 30]
    return {
        "summary": " ".join(sentences[:3]),
        "key_points": sentences[3:10],
    }


_RECOMMENDATIONS_SYSTEM = """You are an expert at generating insightful questions from document content.
Always respond with valid JSON only — no extra text, no markdown fences.
JSON schema:
{
  "recommended_questions": [
    "<specific question 1>",
    "<specific question 2>",
    ...
  ]
}
Generate 5-8 meaningful, specific questions that are directly answerable from the document.
"""

def run_recommendations_chain(db, context_chunks: List[Dict]) -> Dict[str, Any]:
    if not context_chunks:
        return {"recommended_questions": []}

    combined = "\n\n".join(c["text"] for c in context_chunks[:10])
    user_content = (
        "Based on the following document content, generate 5-8 specific, "
        "insightful questions that a reader might want answered:\n\n"
        + combined
    )

    from llm_service import invoke_raw
    raw = invoke_raw(db, _RECOMMENDATIONS_SYSTEM, user_content)
    if raw:
        result = _parse_json_response(raw)
        if result:
            return result

    source = context_chunks[0].get("source", "this document") if context_chunks else "this document"
    return {
        "recommended_questions": [
            f"What is the main purpose of {source}?",
            "Who is the primary subject or author of this document?",
            "What are the key qualifications or skills mentioned?",
            "What timeline or dates are referenced in the document?",
            "What conclusions or summaries are presented?",
        ]
    }


# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _parse_json_response(raw: str) -> Optional[Dict]:
    """Extract and parse JSON from LLM response text."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def get_langchain_text_splitter():
    """Return a configured LangChain RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
