"""
database.py — SQLAlchemy engine, session factory, and DB bootstrap.
v3: Offline multi-user platform (SQLite by default — zero external services,
consistent with the "fully offline" objective of CR-01).
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL

logger = logging.getLogger(__name__)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and seed default data. Called once on app startup."""
    import models_db  # noqa: F401 — ensure models are registered on Base.metadata before create_all
    Base.metadata.create_all(bind=engine)
    _seed_defaults()


def _seed_defaults():
    """
    First-run bootstrap:
      - If there are no users at all, create a default admin (role="admin").
      - If there are no model configs, try to auto-detect installed Ollama
        models and activate the first one.
    """
    from models_db import User, ModelConfig
    from auth import hash_password
    from config import DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD

    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username=DEFAULT_ADMIN_USERNAME,
                email=DEFAULT_ADMIN_EMAIL,
                password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
                role="admin",
                is_active=True,
            )
            db.add(admin)
            db.commit()
            logger.warning(
                "No users found — seeded default admin user "
                f"'{DEFAULT_ADMIN_USERNAME}' / '{DEFAULT_ADMIN_EMAIL}'. "
                "This uses a default password from config/.env — change it immediately "
                "via the admin panel or a direct DB update before exposing this on a LAN."
            )

        if db.query(ModelConfig).count() == 0:
            try:
                from model_manager import list_ollama_models
                models = list_ollama_models()
            except Exception as e:
                logger.warning(f"Could not auto-detect Ollama models at startup: {e}")
                models = []

            if models:
                first = models[0]["name"]
                for m in models:
                    db.add(ModelConfig(
                        model_name=m["name"],
                        provider="ollama",
                        active_status=(m["name"] == first),
                    ))
                db.commit()
                logger.info(f"Seeded {len(models)} Ollama model(s) from 'ollama list'; active='{first}'")
            else:
                logger.warning(
                    "No Ollama models detected at startup (is Ollama running at "
                    "OLLAMA_BASE_URL, and have you run 'ollama pull <model>'?). "
                    "The app will fall back to Mistral/OpenAI (if API keys are set) "
                    "or the offline local synthesizer until an admin runs POST /models/refresh."
                )
    finally:
        db.close()
