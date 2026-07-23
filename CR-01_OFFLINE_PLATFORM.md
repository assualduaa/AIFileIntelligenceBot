# CR-01 — Offline Multi-User Platform

Implements: Ollama local LLM integration, dynamic model management, user
authentication, per-user document isolation, and persistent per-user chat
history, on top of the existing PDF/DOCX/OCR/RAG pipeline (unchanged).

## 1. Architecture

```
React Frontend (frontend/index.html — CDN React, single file)
        |
FastAPI Backend (backend/api.py)
        |
Authentication Layer (backend/auth.py — JWT, bcrypt)
        |
RAG Engine (ingestion.py -> embeddings.py -> retrieval.py -> langchain_pipeline.py)
        |
Vector Database (FAISS, one index per user_id)
        |
LLM Service Layer (backend/llm_service.py)
        |         \
   Ollama (local)  External API (Mistral / OpenAI today, Claude/Gemini pluggable later)
```

New/changed backend modules:

| File | Role |
|---|---|
| `database.py` | SQLAlchemy engine/session, `init_db()`, first-run bootstrap (seeds a default admin + Ollama models). |
| `models_db.py` | ORM tables: `users`, `documents`, `conversations`, `chat_messages`, `models`. |
| `auth.py` | Password hashing (bcrypt via passlib), JWT issue/verify, `get_current_user` / `require_admin` FastAPI dependencies. |
| `llm_provider_base.py` | Abstract `LLMProvider` interface every backend implements. |
| `llm_provider_ollama.py` | Local Ollama HTTP client (`/api/tags`, `/api/chat`, streaming). |
| `llm_provider_external.py` | Adapter exposing the existing Mistral/OpenAI code as pluggable providers. |
| `llm_service.py` | Picks the active provider from the `models` table, falls back down the chain, ends at the offline local synthesizer. |
| `model_manager.py` | Ollama model discovery + active-model CRUD (backs the Model Settings UI). |
| `chat_history.py` | Conversation/message CRUD, scoped to `user_id`. |
| `config.py` | Added `OLLAMA_BASE_URL`, `DATABASE_URL`, JWT settings, default-admin seed vars, per-user path helpers. |
| `ingestion.py`, `retrieval.py`, `langchain_pipeline.py` | Every function now takes `user_id` and reads/writes under that user's own `uploads/<id>/` and `vector_store/<id>/` — this is what actually enforces document isolation. |
| `api.py` | All endpoints now require a bearer token (except `/health`); added `/auth/*`, `/admin/*`, `/models*`, `/conversations*`. |

**Existing features preserved as-is:** PDF/DOCX/OCR/Word/Excel-adjacent parsing (`processing.py`), chunking + embeddings (`embeddings.py`), the local regex-based offline answer synthesizer (`llm.py`'s `_smart_synthesize` and friends) — none of that logic was touched, only re-wired to be user-scoped and provider-agnostic.

## 2. Database schema

SQLite by default (`app.db` in the project root) — no external DB service required, consistent with "fully offline." Swap to Postgres by setting `DATABASE_URL` in `.env`; SQLAlchemy handles the rest.

- `users(id, username, email, password_hash, role, is_active, created_at, last_login)` — `is_active` is an addition beyond the CR's literal field list, needed for the Admin Panel's "disable user" feature.
- `documents(id, user_id, filename, filepath, file_type, embedding_id, uploaded_date)`
- `conversations(id, user_id, title, created_at, updated_at)`
- `chat_messages(id, conversation_id, user_id, question, response, model_used, referenced_documents, timestamp)`
- `models(id, model_name, provider, active_status, created_at)`

## 3. Ollama setup (do this on your machine, not in a sandbox)

```bash
ollama --version
ollama list                    # see what's already installed
ollama pull llama3.1:8b        # or any model you prefer
ollama serve                   # if not already running as a service
```

The backend talks to `OLLAMA_BASE_URL` (default `http://localhost:11434`). I could not verify these exact API shapes (`GET /api/tags`, `POST /api/chat`) against a live Ollama instance from this environment — they're implemented from the documented Ollama REST API, but if a call behaves unexpectedly, check `ollama --version` and the API docs for your installed version, since the API has evolved across releases.

On first startup with an empty `models` table, the app auto-detects installed models via `GET /api/tags` and activates the first one. If Ollama isn't reachable yet, it falls back to Mistral/OpenAI (if API keys are set) or the offline local synthesizer, and logs a warning — run `POST /models/refresh` once Ollama is up.

## 4. Running it

```bash
cd backend
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Edit backend/.env — set a real JWT_SECRET_KEY, change DEFAULT_ADMIN_PASSWORD
python main.py
```

First run creates `app.db` and seeds a default admin (`DEFAULT_ADMIN_USERNAME` / `DEFAULT_ADMIN_PASSWORD` from `.env`, defaults `admin` / `ChangeMe123!`). **Change that password immediately** via the Admin Panel or a direct DB update before exposing this on a LAN — the seed exists purely so the app boots usable out of the box.

Open `http://localhost:8000` — the frontend now shows a login screen first.

## 5. API surface (new/changed)

Auth: `POST /auth/register`, `POST /auth/login` (OAuth2 form, `username`+`password`, `username` accepts email too), `POST /auth/logout`, `GET /users/me`.

Admin: `GET /admin/users`, `POST /admin/users`, `PATCH /admin/users/{id}/disable`, `GET /admin/stats`.

Models: `GET /models`, `POST /models/refresh` (admin), `POST /models/active` (admin).

Conversations: `GET /conversations`, `POST /conversations`, `DELETE /conversations/{id}`, `GET /conversations/{id}/messages`.

Existing endpoints (`/upload`, `/chat`, `/query`, `/summary`, `/recommendations`, `/documents`, `/stats`) are unchanged in shape except they now require `Authorization: Bearer <token>` and are scoped to the caller's `user_id`; `/chat` and `/query` take `conversation_id` instead of the old free-text `session_id`/`user_id` fields. Full interactive docs at `/docs` (FastAPI/Swagger — generated from the live code, always accurate).

## 6. What I verified vs. what I could not

Verified by actually running the code in a sandbox (not just reading it):
- Every modified/new backend file compiles and imports cleanly.
- End-to-end via FastAPI's `TestClient`: default-admin bootstrap, login, JWT auth, register, per-user document/conversation isolation (a second user sees empty lists), admin-only routes returning 403 for non-admins, disable-user blocking login, admin self-disable blocked, `/admin/stats`.
- Caught and fixed a real bug this way: `passlib` 1.7.4's bcrypt version-detection breaks on `bcrypt>=4.1` and raises on every `hash()` call — pinned `bcrypt==4.0.1` in `requirements.txt` with a comment explaining why.
- The new frontend JSX transpiles cleanly through Babel (syntax-checked, not executed in a browser).

Not verified (couldn't be, from this sandbox):
- Actual Ollama HTTP calls — no Ollama instance reachable here. Test the Model Settings page and a real chat query on your machine once `ollama serve` is running.
- The frontend running in an actual browser (login flow, history panel, admin panel rendering) — only syntax-checked, not visually/interactively tested.
- FAISS/embedding behavior with real documents end-to-end (langchain_community/torch aren't installed in this lightweight sandbox check) — this code path is unchanged from the original working implementation aside from adding a `user_id` parameter, so risk is low, but it's worth one real upload-and-query test after `pip install -r requirements.txt`.

## 7. Known trade-offs / things you may want to revisit

- Default JWT secret and admin password are insecure placeholders — must be changed before any non-local exposure.
- Per-user FAISS indices (one directory per `user_id`) were chosen over a single shared index with a metadata filter, specifically because the CR requires "no cross-user access should be possible" — a filter bug in a shared index is a much easier way to leak data across users than a wrong directory path.
- `CORS_ORIGINS=*` with `allow_credentials=True` was already the case before this CR; not something introduced here, but worth tightening if you expose this beyond localhost.
