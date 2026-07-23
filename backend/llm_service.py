"""
llm_service.py — LLM Service Layer (provider-agnostic orchestrator).

        LLM Service Layer
                |
        ----------------------
        |                    |
  Ollama Provider       External API Provider
  (Local, default)      (Mistral / OpenAI today, Claude/Gemini pluggable later)

Reads the active model from the `models` DB table (admin-configurable at
runtime — see model_manager.py — no code changes needed to switch models)
and falls back down the provider chain on failure, ending in the existing
offline local synthesizer so the app never hard-fails just because one
provider is down or misconfigured.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from llm_provider_ollama import OllamaProvider
from llm_provider_external import MistralProvider, OpenAIProvider, GeminiProvider

logger = logging.getLogger(__name__)

_RAG_SYSTEM = (
    "You are an AI Document Intelligence Assistant.\n"
    "RULES:\n"
    "- Answer ONLY using the retrieved document context. No outside knowledge.\n"
    "- Be concise and factual. Never hallucinate.\n"
    "- If context is insufficient: 'The answer is not available in the provided document context.'\n"
    "- Always respond in English.\n"
    "- Factual: 1-2 sentences. Explanatory: 3-5 sentences max."
)

_PROVIDERS = {
    "ollama": OllamaProvider(),
    "mistral": MistralProvider(),
    "openai": OpenAIProvider(),
    "gemini": GeminiProvider(),
}

_PROVIDER_ORDER = ["ollama", "mistral", "openai", "gemini"]


def get_active_model_config(db: Session):
    from models_db import ModelConfig
    return db.query(ModelConfig).filter(ModelConfig.active_status == True).first()  # noqa: E712


def _try_providers(
    db: Session,
    system_prompt: str,
    user_content: str,
    chat_history: Optional[List[Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """Walk the provider fallback chain; return the first successful result, or None."""
    chat_history = chat_history or []
    active = get_active_model_config(db)

    order = list(_PROVIDER_ORDER)
    if active and active.provider in _PROVIDERS:
        order = [active.provider] + [p for p in order if p != active.provider]

    for provider_name in order:
        provider = _PROVIDERS[provider_name]
        model_name = active.model_name if (active and active.provider == provider_name) else None

        if provider_name == "ollama":
            if not model_name:
                models = provider.list_models()
                model_name = models[0]["name"] if models else None
            if not model_name:
                continue  # no Ollama models installed — skip to next provider
        elif not provider.is_available():
            continue  # no API key configured — skip to next provider

        try:
            return provider.generate(system_prompt, user_content, chat_history, model_name)
        except Exception as e:
            logger.warning(f"LLM provider '{provider_name}' failed: {e}. Trying next in chain...")
            continue

    return None


def generate_response(
    db: Session,
    prompt_context: str,
    query: str,
    chat_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    result = _try_providers(db, _RAG_SYSTEM, prompt_context, chat_history)
    if result:
        return result
    from llm import _local_response  # offline, always-available final fallback
    return _local_response(prompt_context, query)


def invoke_raw(db: Session, system_prompt: str, user_content: str) -> Optional[str]:
    """Simple text-in/text-out call used by the summary/recommendation chains."""
    result = _try_providers(db, system_prompt, user_content)
    return result["answer"] if result else None


def generate_summary(db: Session, context_chunks: List[Dict]) -> Dict[str, Any]:
    from langchain_pipeline import run_summary_chain
    return run_summary_chain(db, context_chunks)


def generate_recommendations(db: Session, context_chunks: List[Dict]) -> Dict[str, Any]:
    from langchain_pipeline import run_recommendations_chain
    return run_recommendations_chain(db, context_chunks)


def list_all_provider_status() -> Dict[str, bool]:
    """Quick availability snapshot of every registered provider — used by /health."""
    return {name: p.is_available() for name, p in _PROVIDERS.items()}
