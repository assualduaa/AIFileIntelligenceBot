"""
llm_provider_ollama.py — Local Ollama LLM provider (offline default).

Talks only to the local Ollama REST API — no external network calls, which
is what makes this the primary engine for the "fully offline" objective.

Endpoints used (from Ollama's documented API — verify against your installed
version's `ollama --version` / API docs if a call behaves unexpectedly, since
this was implemented from general knowledge of the Ollama API rather than a
live connection to your machine):
  GET  /api/tags   -> installed models
  POST /api/chat    -> chat-style generation (stream true/false)
"""
import json
import logging
from typing import List, Dict, Any, Optional, Iterator

import httpx

from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS, LLM_TEMPERATURE, LLM_MAX_TOKENS
from llm_provider_base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = OLLAMA_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not reachable at {self.base_url}: {e}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            return [
                {
                    "name": m.get("name") or m.get("model"),
                    "size": m.get("size"),
                    "modified_at": m.get("modified_at"),
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
                }
                for m in data.get("models", [])
            ]
        except Exception as e:
            logger.warning(f"Ollama /api/tags failed ({self.base_url}): {e}")
            return []

    def _messages(self, system_prompt, user_content, chat_history):
        msgs = [{"role": "system", "content": system_prompt}]
        for turn in (chat_history or [])[-6:]:
            role = turn.get("role", "user")
            msgs.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": turn.get("content", "")[:1000],
            })
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def generate(self, system_prompt, user_content, chat_history=None, model=None) -> Dict[str, Any]:
        if not model:
            raise ValueError("OllamaProvider.generate requires an explicit model name.")
        payload = {
            "model": model,
            "messages": self._messages(system_prompt, user_content, chat_history),
            "stream": False,
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
        }
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        answer = (data.get("message") or {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        tokens = int(data.get("eval_count", 0) or 0) + int(data.get("prompt_eval_count", 0) or 0)
        return {"answer": answer, "model": model, "mode": "ollama", "tokens": tokens}

    def generate_stream(self, system_prompt, user_content, chat_history=None, model=None) -> Iterator[str]:
        if not model:
            raise ValueError("OllamaProvider.generate_stream requires an explicit model name.")
        payload = {
            "model": model,
            "messages": self._messages(system_prompt, user_content, chat_history),
            "stream": True,
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
        }
        with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
