"""
llm_provider_external.py — Adapters exposing external LLM APIs (Mistral,
OpenAI, Gemini) behind the common LLMProvider interface.

Kept separate from llm_provider_ollama.py so external providers can be
bolted on without ever touching the offline Ollama path — this is the
"External API Provider" branch of the LLM Service Layer diagram in CR-01.
Claude/other providers can be added the same way later.
"""
import logging
from typing import List, Dict, Any, Optional

import httpx

from config import (
    MISTRAL_API_KEY, OPENAI_API_KEY, MISTRAL_MODEL, LLM_MODEL,
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_BASE_URL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE,
)
from llm_provider_base import LLMProvider

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):
    name = "mistral"

    def is_available(self) -> bool:
        return bool(MISTRAL_API_KEY)

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"name": MISTRAL_MODEL, "size": None}] if MISTRAL_API_KEY else []

    def generate(self, system_prompt, user_content, chat_history=None, model=None) -> Dict[str, Any]:
        from llm import _mistral_response  # reuse existing, already-working integration
        result = _mistral_response(user_content, user_content, chat_history or [])
        if not result:
            raise RuntimeError("Mistral provider returned no result.")
        return result


class OpenAIProvider(LLMProvider):
    name = "openai"

    def is_available(self) -> bool:
        return bool(OPENAI_API_KEY)

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"name": LLM_MODEL, "size": None}] if OPENAI_API_KEY else []

    def generate(self, system_prompt, user_content, chat_history=None, model=None) -> Dict[str, Any]:
        from llm import _openai_response  # reuse existing, already-working integration
        result = _openai_response(user_content, user_content, chat_history or [])
        if not result:
            raise RuntimeError("OpenAI provider returned no result.")
        return result


class GeminiProvider(LLMProvider):
    """
    Google Gemini via the raw REST API (not the Python SDK — see the note
    in config.py). Docs: https://ai.google.dev/api/generate-content
    """
    name = "gemini"

    def is_available(self) -> bool:
        return bool(GOOGLE_API_KEY)

    def list_models(self) -> List[Dict[str, Any]]:
        if not GOOGLE_API_KEY:
            return []
        try:
            r = httpx.get(
                f"{GEMINI_BASE_URL}/models",
                params={"key": GOOGLE_API_KEY},
                timeout=8.0,
            )
            r.raise_for_status()
            data = r.json()
            models = []
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue
                # API returns names like "models/gemini-1.5-flash" — strip the prefix.
                short_name = m.get("name", "").split("/", 1)[-1]
                models.append({"name": short_name, "size": None, "display_name": m.get("displayName")})
            return models
        except Exception as e:
            logger.warning(f"Gemini list_models() failed: {e}")
            return []

    def _contents(self, user_content, chat_history):
        contents = []
        for turn in (chat_history or [])[-6:]:
            role = "model" if turn.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": turn.get("content", "")[:1000]}]})
        contents.append({"role": "user", "parts": [{"text": user_content}]})
        return contents

    def generate(self, system_prompt, user_content, chat_history=None, model=None) -> Dict[str, Any]:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        model_name = model or GEMINI_MODEL
        payload = {
            "contents": self._contents(user_content, chat_history),
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": LLM_TEMPERATURE,
                "maxOutputTokens": LLM_MAX_TOKENS,
            },
        }
        r = httpx.post(
            f"{GEMINI_BASE_URL}/models/{model_name}:generateContent",
            params={"key": GOOGLE_API_KEY},
            json=payload,
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            block_reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason}).")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(p.get("text", "") for p in parts).strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
        usage = data.get("usageMetadata", {})
        tokens = int(usage.get("totalTokenCount", 0) or 0)
        return {"answer": answer, "model": model_name, "mode": "gemini", "tokens": tokens}
