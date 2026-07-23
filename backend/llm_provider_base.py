"""
llm_provider_base.py — Abstract LLM provider interface.

        LLM Service Layer
                |
        ----------------------
        |                    |
  Ollama Provider       External API Provider
  (Local)               (Future: OpenAI / Claude / Gemini / other local models)

Every concrete provider implements this contract so llm_service.py can swap
backends at runtime (via the `models` DB table) without any caller needing
to know which engine actually ran.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap connectivity/config check — must NOT perform a full generation call."""
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """Return installed/available models: [{'name': str, 'size': int|None, ...}, ...]"""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_content: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {'answer': str, 'model': str, 'mode': str, 'tokens': int}. Raise on failure."""
        raise NotImplementedError

    def generate_stream(
        self,
        system_prompt: str,
        user_content: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Optional: yield answer chunks as they're generated. Default falls back to non-streaming."""
        result = self.generate(system_prompt, user_content, chat_history, model)
        yield result["answer"]
