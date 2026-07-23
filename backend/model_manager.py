"""
model_manager.py — Ollama model discovery + active-model configuration.

No code changes are needed to switch models: an admin calls
POST /models/refresh to sync installed models from `ollama list` (via the
API), then POST /models/active to flip which one drives the RAG pipeline.
Active model choice is persisted in the `models` DB table.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from llm_provider_ollama import OllamaProvider
from models_db import ModelConfig

logger = logging.getLogger(__name__)
_ollama = OllamaProvider()


def list_ollama_models() -> List[Dict[str, Any]]:
    """Live query against GET /api/tags — does not touch the DB."""
    return _ollama.list_models()


def ollama_is_reachable() -> bool:
    return _ollama.is_available()


def refresh_models(db: Session) -> List[ModelConfig]:
    """Sync the `models` table with whatever is currently installed in Ollama."""
    installed = list_ollama_models()
    existing_names = {
        m.model_name for m in db.query(ModelConfig).filter(ModelConfig.provider == "ollama").all()
    }

    for m in installed:
        if m["name"] not in existing_names:
            db.add(ModelConfig(model_name=m["name"], provider="ollama", active_status=False))
    db.commit()

    has_active = db.query(ModelConfig).filter(ModelConfig.active_status == True).first()  # noqa: E712
    if installed and not has_active:
        first = (
            db.query(ModelConfig)
            .filter(ModelConfig.model_name == installed[0]["name"], ModelConfig.provider == "ollama")
            .first()
        )
        if first:
            first.active_status = True
            db.commit()

    return list_all_models(db)


def get_active_model(db: Session) -> Optional[ModelConfig]:
    return db.query(ModelConfig).filter(ModelConfig.active_status == True).first()  # noqa: E712


def set_active_model(db: Session, model_name: str, provider: str = "ollama") -> ModelConfig:
    db.query(ModelConfig).update({ModelConfig.active_status: False})

    row = (
        db.query(ModelConfig)
        .filter(ModelConfig.model_name == model_name, ModelConfig.provider == provider)
        .first()
    )
    if not row:
        row = ModelConfig(model_name=model_name, provider=provider, active_status=True)
        db.add(row)
    else:
        row.active_status = True

    db.commit()
    db.refresh(row)
    return row


def list_all_models(db: Session) -> List[ModelConfig]:
    return db.query(ModelConfig).order_by(ModelConfig.provider, ModelConfig.model_name).all()
