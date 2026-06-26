"""
embeddings.py - Semantic chunking + vector embedding generation
Supports sentence-transformers (preferred) and TF-IDF fallback (no GPU required).
"""
import re
import uuid
import pickle
import logging
from typing import List, Dict, Any
from datetime import datetime

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, BASE_DIR

TFIDF_VOCAB_PATH = BASE_DIR / "vector_store" / "tfidf_vocab.pkl"
logger = logging.getLogger(__name__)

_model = None
_model_type = None
_tfidf_vec = None
_tfidf_dim = 512


def _init_tfidf_model():
    global _tfidf_vec
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf_vec = TfidfVectorizer(
        max_features=_tfidf_dim,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
        ngram_range=(1, 2),
    )
    logger.info("TF-IDF embedding model initialised (fallback mode).")


def get_embedding_model():
    global _model, _model_type
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
            _model_type = "sentence_transformers"
            logger.info("Sentence-transformers model loaded.")
        except Exception as e:
            logger.warning(f"sentence-transformers unavailable ({e}). Using TF-IDF fallback.")
            _init_tfidf_model()
            _model = _tfidf_vec
            _model_type = "tfidf"
    return _model, _model_type


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        s_len = len(sentence)
        if s_len > chunk_size:
            words = sentence.split()
            word_chunk = []
            wlen = 0
            for word in words:
                if wlen + len(word) + 1 > chunk_size and word_chunk:
                    chunks.append(" ".join(word_chunk))
                    overlap_words = word_chunk[-(overlap // 6):]
                    word_chunk = overlap_words + [word]
                    wlen = sum(len(w) + 1 for w in word_chunk)
                else:
                    word_chunk.append(word)
                    wlen += len(word) + 1
            if word_chunk:
                current.extend(word_chunk)
                current_len += wlen
            continue
        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_text = " ".join(current)[-overlap:]
            current = [overlap_text, sentence]
            current_len = len(overlap_text) + s_len
        else:
            current.append(sentence)
            current_len += s_len + 1
    if current:
        chunks.append(" ".join(current))
    logger.info(f"Text chunked into {len(chunks)} chunks")
    return chunks


def embed_text(text: str) -> List[float]:
    return embed_batch([text])[0]


def embed_batch(texts: List[str]) -> List[List[float]]:
    import numpy as np
    model, model_type = get_embedding_model()
    if model_type == "sentence_transformers":
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vectors]
    global _tfidf_vec
    from sklearn.preprocessing import normalize as sk_normalize
    if _tfidf_vec is None:
        _init_tfidf_model()
    try:
        mat = _tfidf_vec.transform(texts)
    except Exception:
        _tfidf_vec.fit(texts)
        mat = _tfidf_vec.transform(texts)
    dense = mat.toarray().astype(np.float32)
    if dense.shape[1] < _tfidf_dim:
        pad = np.zeros((dense.shape[0], _tfidf_dim - dense.shape[1]), dtype=np.float32)
        dense = np.hstack([dense, pad])
    normed = sk_normalize(dense, norm="l2")
    return [row.tolist() for row in normed]


def fit_tfidf_on_corpus(texts: List[str]):
    global _tfidf_vec, _model, _model_type
    if not texts:
        return
    if _tfidf_vec is None:
        _init_tfidf_model()
    _tfidf_vec.fit(texts)
    _model = _tfidf_vec
    _model_type = "tfidf"
    try:
        TFIDF_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TFIDF_VOCAB_PATH, "wb") as f:
            pickle.dump(_tfidf_vec, f)
        logger.info(f"TF-IDF vocabulary saved.")
    except Exception as e:
        logger.warning(f"Could not save TF-IDF vocab: {e}")


def load_tfidf_if_available():
    global _tfidf_vec, _model, _model_type
    if TFIDF_VOCAB_PATH.exists() and _model is None:
        try:
            with open(TFIDF_VOCAB_PATH, "rb") as f:
                _tfidf_vec = pickle.load(f)
            _model = _tfidf_vec
            _model_type = "tfidf"
            logger.info("Loaded persisted TF-IDF vocabulary.")
        except Exception as e:
            logger.warning(f"Could not load TF-IDF vocab: {e}")


def build_chunk_documents(text: str, filename: str, file_type: str, language: str) -> List[Dict[str, Any]]:
    chunks = chunk_text(text)
    if not chunks:
        logger.warning(f"No chunks generated for {filename}")
        return []
    embeddings = embed_batch(chunks)
    timestamp = datetime.utcnow().isoformat()
    documents = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        documents.append({
            "chunk_id": str(uuid.uuid4()),
            "source": filename,
            "file_type": file_type,
            "language": language,
            "timestamp": timestamp,
            "chunk_index": i,
            "text": chunk,
            "embedding": vector,
        })
    logger.info(f"Built {len(documents)} chunk documents for '{filename}'")
    return documents
