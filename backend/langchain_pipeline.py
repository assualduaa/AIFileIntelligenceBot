"""
langchain_pipeline.py — LangChain Workflow Orchestration Engine
v2: FAISS vector store + Mistral LLM + RAG / Summary / Recommendations chains

Architecture:
  Document text
       ↓
  LangChain RecursiveCharacterTextSplitter
       ↓
  HuggingFace Embeddings (all-MiniLM-L6-v2)
       ↓
  FAISS Vector Store (persistent)
       ↓
  Similarity Search (Top-K)
       ↓
  Mistral API (ChatMistralAI)
       ↓
  Structured Response
"""
import json
import logging
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import (
    MISTRAL_API_KEY, MISTRAL_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR, FAISS_METADATA_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
)

logger = logging.getLogger(__name__)

# ── Singleton state ────────────────────────────────────────────────────
_vector_store = None   # LangChain FAISS wrapper
_embeddings   = None   # HuggingFace embeddings model
_llm          = None   # Mistral LLM


# ══════════════════════════════════════════════════════════════════════
# EMBEDDINGS
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
# FAISS VECTOR STORE
# ══════════════════════════════════════════════════════════════════════

def get_vector_store():
    """Return singleton FAISS vector store (loaded from disk if available)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = _load_faiss_from_disk()
    return _vector_store


def _load_faiss_from_disk():
    """Attempt to load persisted FAISS index. Returns None if not found."""
    from langchain_community.vectorstores import FAISS

    index_file = FAISS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        logger.info("No persisted FAISS index found — will create on first upload.")
        return None

    try:
        store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        total = store.index.ntotal
        logger.info(f"FAISS index loaded ({total} vectors) from {FAISS_INDEX_DIR}")
        return store
    except Exception as e:
        logger.warning(f"Could not load FAISS index: {e}")
        return None


def _persist_faiss():
    """Save current FAISS index to disk."""
    global _vector_store
    if _vector_store is None:
        return
    try:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _vector_store.save_local(str(FAISS_INDEX_DIR))
        logger.info(f"FAISS index saved ({_vector_store.index.ntotal} vectors).")
    except Exception as e:
        logger.error(f"FAISS save failed: {e}")


def add_documents_to_faiss(texts: List[str], metadatas: List[Dict]) -> int:
    """
    Add chunked documents to FAISS store and persist.
    Creates the store if it doesn't exist yet.
    Returns the number of chunks indexed.
    """
    global _vector_store
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    if not texts:
        return 0

    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    embeddings = get_embeddings()

    if _vector_store is None:
        _vector_store = FAISS.from_documents(docs, embeddings)
        logger.info(f"Created new FAISS index with {len(docs)} documents.")
    else:
        _vector_store.add_documents(docs)
        logger.info(f"Added {len(docs)} documents to existing FAISS index.")

    _persist_faiss()
    return len(docs)


def search_faiss(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic similarity search on FAISS.
    Optionally filter by source filename (post-filter).
    Returns list of chunk dicts with 'text', 'source', 'score', etc.
    """
    store = get_vector_store()
    if store is None:
        return []

    try:
        fetch_k = top_k * 4 if source else top_k
        results = store.similarity_search_with_score(query, k=min(fetch_k, store.index.ntotal))

        chunks = []
        for doc, dist in results:
            if source and doc.metadata.get("source") != source:
                continue
            # FAISS L2 distance with normalized vectors → cosine similarity = 1 - dist²/2
            # LangChain's default fn: score = 1 - dist/2  (approximate)
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
        logger.error(f"FAISS search failed: {e}")
        return []


def fetch_all_chunks_for_source(source: str) -> List[Dict[str, Any]]:
    """
    Return all indexed chunks for a specific source file.
    Used by summary/recommendations endpoints.
    """
    store = get_vector_store()
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
        # Sort by chunk_index for coherent reading order
        chunks.sort(key=lambda x: x["chunk_index"])
    except Exception as e:
        logger.error(f"fetch_all_chunks_for_source failed: {e}")

    return chunks


def _rmtree_windows_safe(path: Path):
    """
    Remove a directory tree safely on Windows.
    Windows memory-maps FAISS files, so we must force GC before rmtree,
    and fall back to per-file deletion if the directory remove still fails.
    """
    import gc
    gc.collect()
    try:
        shutil.rmtree(path)
    except PermissionError:
        # GC wasn't enough — delete files individually, ignore locked ones
        for f in path.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        try:
            path.rmdir()
        except Exception:
            pass


def rebuild_faiss_without(source: str) -> int:
    """
    Delete all chunks belonging to `source` by rebuilding the FAISS index.
    FAISS does not support in-place deletion, so we rebuild.
    Returns number of chunks deleted.
    Raises RuntimeError on failure so callers get a meaningful error.
    """
    global _vector_store
    from langchain_community.vectorstores import FAISS

    store = get_vector_store()
    if store is None:
        return 0

    kept, deleted = [], 0
    for idx, doc_id in store.index_to_docstore_id.items():
        doc = store.docstore.search(doc_id)
        if not hasattr(doc, "page_content"):
            continue
        if doc.metadata.get("source") == source:
            deleted += 1
        else:
            kept.append(doc)

    if deleted == 0:
        return 0  # source not in index — caller handles 404

    try:
        if not kept:
            # Release the store *before* touching the files on disk
            _vector_store = None
            if FAISS_INDEX_DIR.exists():
                _rmtree_windows_safe(FAISS_INDEX_DIR)
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        else:
            embeddings = get_embeddings()
            _vector_store = FAISS.from_documents(kept, embeddings)
            _persist_faiss()

        logger.info(f"FAISS rebuilt: removed {deleted} chunks from '{source}'.")
    except Exception as e:
        logger.error(f"FAISS rebuild failed: {e}")
        raise RuntimeError(f"Failed to rebuild FAISS index after removing '{source}': {e}")

    return deleted


def get_total_vectors() -> int:
    """Return total number of vectors in the FAISS index."""
    store = get_vector_store()
    if store is None:
        return 0
    try:
        return store.index.ntotal
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════
# FILE-LEVEL METADATA  (separate JSON — FAISS stores chunk-level only)
# ══════════════════════════════════════════════════════════════════════

def load_file_metadata() -> Dict[str, Any]:
    if FAISS_METADATA_PATH.exists():
        try:
            with open(FAISS_METADATA_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_file_metadata(meta: Dict[str, Any]):
    FAISS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAISS_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def upsert_file_metadata(
    filename: str,
    file_type: str,
    language: str,
    chunk_count: int,
    timestamp: str,
):
    meta = load_file_metadata()
    existing = meta.get(filename, {})
    meta[filename] = {
        "source":      filename,
        "file_type":   file_type,
        "language":    language,
        "chunk_count": existing.get("chunk_count", 0) + chunk_count,
        "timestamp":   timestamp,
    }
    _save_file_metadata(meta)


def remove_file_metadata(filename: str):
    meta = load_file_metadata()
    meta.pop(filename, None)
    _save_file_metadata(meta)


def list_file_metadata() -> List[Dict[str, Any]]:
    meta = load_file_metadata()
    return sorted(meta.values(), key=lambda x: x.get("timestamp", ""), reverse=True)


# ══════════════════════════════════════════════════════════════════════
# MISTRAL LLM
# ══════════════════════════════════════════════════════════════════════

def get_mistral_llm():
    """Return singleton Mistral LLM via LangChain. None if API key missing."""
    global _llm
    if _llm is not None:
        return _llm
    if not MISTRAL_API_KEY:
        return None

    try:
        from langchain_mistralai import ChatMistralAI
        _llm = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model=MISTRAL_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        logger.info(f"Mistral LLM initialized: {MISTRAL_MODEL}")
    except Exception as e:
        logger.warning(f"LangChain Mistral init failed: {e}. Trying mistralai SDK directly...")
        _llm = None  # will retry on next call if needed

    return _llm


def _invoke_mistral(system_prompt: str, user_content: str) -> Optional[str]:
    """
    Call Mistral via LangChain ChatMistralAI.
    Falls back to direct mistralai SDK if LangChain wrapper fails.
    Returns response string or None on failure.
    """
    # Try LangChain wrapper first
    llm = get_mistral_llm()
    if llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LangChain Mistral call failed: {e}. Trying direct SDK...")

    # Direct SDK fallback
    if MISTRAL_API_KEY:
        try:
            from mistralai import Mistral
            client = Mistral(api_key=MISTRAL_API_KEY)
            chat_response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Direct Mistral SDK call failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════
# RAG CHAIN
# ══════════════════════════════════════════════════════════════════════

_RAG_SYSTEM = """You are an AI Document Intelligence Assistant.

RULES:
- Answer ONLY using the retrieved document context provided. Do NOT use outside knowledge.
- Be concise and factual. No hallucination.
- If context is insufficient: respond exactly "The answer is not available in the provided document context."
- Always respond in English.
- Short factual answers: 1-2 sentences. Explanatory answers: 3-5 sentences max.
"""

def run_rag_chain(query: str, context_chunks: List[Dict]) -> str:
    """
    LangChain RAG pipeline:
      retrieved chunks → structured prompt → Mistral → answer
    """
    if not context_chunks:
        return "No documents indexed yet. Please upload a file first."

    context_str = "\n\n".join([
        f"[Source: {c['source']} | Chunk #{c['chunk_index']}]\n{c['text']}"
        for c in context_chunks
    ])
    user_content = f"Document Context:\n{context_str}\n\nQuestion: {query}"

    answer = _invoke_mistral(_RAG_SYSTEM, user_content)
    if answer:
        return answer

    # Local synthesizer fallback (from llm.py)
    try:
        from llm import _smart_synthesize, _build_prompt_str
        prompt_str = _build_prompt_str(context_chunks, query)
        return _smart_synthesize(query, prompt_str)
    except Exception:
        return context_chunks[0]["text"][:400] if context_chunks else "Unable to generate answer."


# ══════════════════════════════════════════════════════════════════════
# SUMMARY CHAIN
# ══════════════════════════════════════════════════════════════════════

_SUMMARY_SYSTEM = """You are a professional document summarization expert.
Always respond with valid JSON only — no extra text, no markdown fences.
JSON schema:
{
  "summary": "<3-5 sentence summary of the document>",
  "key_points": ["<point 1>", "<point 2>", ...(5-7 points)]
}
"""

def run_summary_chain(context_chunks: List[Dict]) -> Dict[str, Any]:
    """
    Generate a structured document summary using Mistral.
    Input: list of chunk dicts (uses first 15 for token budget).
    Output: {"summary": str, "key_points": [str, ...]}
    """
    if not context_chunks:
        return {"summary": "No document content available.", "key_points": []}

    # Use first 15 chunks (≈ 12000 chars) to stay within token limits
    combined = "\n\n".join(c["text"] for c in context_chunks[:15])
    user_content = f"Summarize the following document:\n\n{combined}"

    raw = _invoke_mistral(_SUMMARY_SYSTEM, user_content)
    if raw:
        result = _parse_json_response(raw)
        if result:
            return result

    # Fallback: extract sentences manually
    return _fallback_summary(context_chunks)


def _fallback_summary(chunks: List[Dict]) -> Dict[str, Any]:
    """Simple extractive summary when LLM is unavailable."""
    all_text = " ".join(c["text"] for c in chunks[:10])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", all_text) if len(s.strip()) > 30]
    return {
        "summary": " ".join(sentences[:3]),
        "key_points": sentences[3:10],
    }


# ══════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS CHAIN
# ══════════════════════════════════════════════════════════════════════

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

def run_recommendations_chain(context_chunks: List[Dict]) -> Dict[str, Any]:
    """
    Generate recommended exploration questions from document content.
    Output: {"recommended_questions": [str, ...]}
    """
    if not context_chunks:
        return {"recommended_questions": []}

    combined = "\n\n".join(c["text"] for c in context_chunks[:10])
    user_content = (
        "Based on the following document content, generate 5-8 specific, "
        "insightful questions that a reader might want answered:\n\n"
        + combined
    )

    raw = _invoke_mistral(_RECOMMENDATIONS_SYSTEM, user_content)
    if raw:
        result = _parse_json_response(raw)
        if result:
            return result

    # Fallback: generic questions
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
    # Try to extract JSON block
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
