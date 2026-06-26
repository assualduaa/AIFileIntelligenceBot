"""
memory.py — Multi-tier memory manager
Tiers: Session (in-memory) | User (JSON file) | Semantic (ChromaDB via retrieval.py)
"""
import json
import logging
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import BASE_DIR

logger      = logging.getLogger(__name__)
MEMORY_DIR  = BASE_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ── Session Memory (in-process dict, TTL-like via maxlen) ──────────────
_sessions: Dict[str, deque] = {}

def get_session_memory(session_id: str, max_turns: int = 20) -> List[Dict]:
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=max_turns)
    return list(_sessions[session_id])


def append_session_memory(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=20)
    _sessions[session_id].append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.utcnow().isoformat(),
    })


def clear_session_memory(session_id: str):
    _sessions.pop(session_id, None)


# ── User Memory (persisted JSON per user) ─────────────────────────────
def _user_file(user_id: str) -> Path:
    return MEMORY_DIR / f"{user_id}.json"


def get_user_memory(user_id: str) -> Dict[str, Any]:
    path = _user_file(user_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def set_user_memory(user_id: str, key: str, value: Any):
    mem = get_user_memory(user_id)
    mem[key] = {"value": value, "updated_at": datetime.utcnow().isoformat()}
    with open(_user_file(user_id), "w") as f:
        json.dump(mem, f, indent=2)


def update_user_memory(user_id: str, updates: Dict[str, Any]):
    for k, v in updates.items():
        set_user_memory(user_id, k, v)


# ── Memory Manager (orchestrator) ─────────────────────────────────────
class MemoryManager:
    """
    Aggregates all memory tiers for prompt context construction.
    Pipeline: session + user + semantic → merged context dict
    """

    def build_context(
        self,
        user_id:    str,
        session_id: str,
        query:      str,
        top_k:      int = 5,
    ) -> Dict[str, Any]:
        from retrieval import retrieve_context  # avoid circular import

        session_mem  = get_session_memory(session_id)
        user_mem     = get_user_memory(user_id)
        semantic_mem = retrieve_context(query, top_k=top_k)

        return {
            "session":  session_mem,
            "user":     user_mem,
            "semantic": semantic_mem,
        }

    def build_prompt_context(
        self,
        context: Dict[str, Any],
        query:   str,
    ) -> str:
        """Build a structured prompt string from memory context."""
        parts = []

        # User profile
        user = context.get("user", {})
        if user:
            profile_lines = []
            for k, v in user.items():
                val = v.get("value", v) if isinstance(v, dict) else v
                profile_lines.append(f"- {k}: {val}")
            parts.append("USER PROFILE:\n" + "\n".join(profile_lines))

        # Session history (last 5 turns)
        session = context.get("session", [])[-5:]
        if session:
            history_lines = []
            for turn in session:
                role    = turn.get("role", "")
                content = turn.get("content", "")[:300]
                history_lines.append(f"[{role.upper()}]: {content}")
            parts.append("SESSION CONTEXT:\n" + "\n".join(history_lines))

        # Retrieved document chunks
        semantic = context.get("semantic", [])
        if semantic:
            chunk_lines = []
            for i, chunk in enumerate(semantic, 1):
                src   = chunk.get("source", "")
                score = chunk.get("score", 0)
                text  = chunk.get("text", "")[:600]
                chunk_lines.append(f"[{i}] Source: {src} (relevance: {score})\n{text}")
            parts.append("RETRIEVED KNOWLEDGE:\n" + "\n\n".join(chunk_lines))

        parts.append(f"CURRENT QUESTION:\n{query}")

        return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(parts)


memory_manager = MemoryManager()
