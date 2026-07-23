"""
models_db.py — SQLAlchemy ORM models for the offline multi-user platform.

Tables (per CR-01 §DATABASE DESIGN):
  users, documents, conversations, chat_messages, models (ModelConfig)

Note: `User.is_active` is not in the CR's literal users-table field list, but
is required to implement the Admin Panel's "Disable users" requirement — it's
a deliberate, minimal addition, not a hidden scope change.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(80), unique=True, nullable=False, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), nullable=False, default="user")   # "admin" | "user"
    is_active     = Column(Boolean, nullable=False, default=True)        # powers admin "disable user"
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_login    = Column(DateTime, nullable=True)

    documents     = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename      = Column(String(500), nullable=False)
    filepath      = Column(String(1000), nullable=False)
    file_type     = Column(String(20), nullable=True)
    embedding_id  = Column(String(500), nullable=True)   # source key in this user's FAISS metadata
    uploaded_date = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="documents")


class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title      = Column(String(300), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner    = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id                   = Column(Integer, primary_key=True, index=True)
    conversation_id      = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id              = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question             = Column(Text, nullable=False)
    response             = Column(Text, nullable=False)
    model_used           = Column(String(120), nullable=True)
    referenced_documents = Column(Text, nullable=True)   # JSON-encoded list[str] of source filenames
    timestamp            = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class ModelConfig(Base):
    __tablename__ = "models"

    id            = Column(Integer, primary_key=True, index=True)
    model_name    = Column(String(200), nullable=False)
    provider      = Column(String(50), nullable=False, default="ollama")  # "ollama" | "mistral" | "openai"
    active_status = Column(Boolean, nullable=False, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
