"""
chat_history.py — Conversation & chat message persistence (per-user).

Every conversation and message is scoped to a user_id; callers (api.py) are
responsible for only ever passing the authenticated current_user's id, which
is what actually enforces "a user must only access their own conversations."
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from models_db import Conversation, ChatMessage


def create_conversation(db: Session, user_id: int, title: str = "New Conversation") -> Conversation:
    conv = Conversation(user_id=user_id, title=(title or "New Conversation")[:300])
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversation(db: Session, user_id: int, conversation_id: int) -> Optional[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )


def list_conversations(db: Session, user_id: int) -> List[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


def delete_conversation(db: Session, user_id: int, conversation_id: int) -> bool:
    conv = get_conversation(db, user_id, conversation_id)
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True


def get_recent_messages(db: Session, conversation_id: int, limit: int = 6) -> List[Dict[str, str]]:
    """Last `limit` turns as [{'role': 'user'|'assistant', 'content': str}, ...] — feeds LLM chat_history."""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(limit)
        .all()
    )
    turns: List[Dict[str, str]] = []
    for m in reversed(rows):
        turns.append({"role": "user", "content": m.question})
        turns.append({"role": "assistant", "content": m.response})
    return turns


def list_messages(db: Session, user_id: int, conversation_id: int) -> List[ChatMessage]:
    conv = get_conversation(db, user_id, conversation_id)
    if not conv:
        return []
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )


def append_message(
    db: Session,
    conversation_id: int,
    user_id: int,
    question: str,
    response: str,
    model_used: str,
    referenced_documents: Optional[List[str]] = None,
) -> ChatMessage:
    msg = ChatMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        question=question,
        response=response,
        model_used=model_used,
        referenced_documents=json.dumps(referenced_documents or []),
    )
    db.add(msg)

    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.updated_at = datetime.utcnow()
        if conv.title == "New Conversation":
            conv.title = question.strip()[:60] or "New Conversation"

    db.commit()
    db.refresh(msg)
    return msg
