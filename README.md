# RAG-Based AI Bot 🤖

**A multi-user, offline-first document intelligence platform.** Upload your files, chat with them in English, Arabic or Malayalam, and choose which LLM answers — local models by default, or cloud models when you explicitly go online. Every answer is grounded in the user's own documents and cited to the page.

> 🔒 **This is a project showcase.** The source code is in a private repository and is available to hiring teams on request (see [Contact](#-assumay@gmail.com)).

![Architecture](docs/architecture.png)

---

## ⭐ Headline Features

| | Feature | Highlights |
|---|---|---|
| 🔀 | **Multi-LLM with live switching** | Pluggable provider layer: **Ollama** (local), **Mistral**, **OpenAI**, **Gemini**. Switch the active model from the chat UI, even mid-conversation. The model behind every answer is recorded in the chat history. |
| 📂 | **Multi-file upload, many formats** | Upload several files at once: **PDF, DOCX, TXT, PPTX, XLSX, HTML, EML** and **images (OCR)**, with live per-file status tracking. |
| 🌍 | **Multilingual: English, Arabic & Malayalam** | Multilingual embeddings (BAAI/bge-m3), per-question language detection, answers in the user's language, a dedicated Malayalam pipeline (script detection, Unicode normalization, OCR), and Arabic OCR. |
| 🔐 | **Authentication** | JWT sessions, bcrypt password hashing, self-registration and a seeded admin account. |
| 🛡️ | **Role-based access control** | `admin` and `user` roles — admins manage users, platform stats and the active models; users get a private workspace. |
| 💬 | **Persistent chat history** | Conversations are stored per user and restored on reload, with *New Chat* (keep documents) and *New Window* (fresh workspace). |

### Also included
- 🔒 **Per-user & per-conversation data isolation** — separate storage and vector index for every user; documents are scoped to the conversation they were uploaded in.
- 🧭 **Intent routing** — small talk never triggers document retrieval.
- 📍 **Page-level citations** for PDFs, slides and sheets.
- 📝 **On-demand summaries, document overviews and suggested questions**, generated in the document's own language.
- 🧮 **Table-aware PDF extraction**.
- ⚡ **Streaming responses** with a clear "model busy" signal instead of silent hangs.
- 🐳 **Docker-based deployment** (app + local LLM runtime).

---

## 🔒 Offline-first by design

With the default `offline` mode the system hard-gates generation to the local LLM runtime and forces the embedding libraries into offline mode: **no document ever leaves the machine, and there is no silent fallback to a cloud API.** Cloud providers become reachable only when an administrator deliberately enables online mode.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2 |
| Auth & security | PyJWT, bcrypt, role-based route guards |
| LLMs | Ollama (local, e.g. Llama 3.1 8B), Mistral, OpenAI, Gemini |
| Orchestration | LangChain |
| Embeddings | BAAI/bge-m3 (multilingual) via sentence-transformers, PyTorch (CPU) |
| Vector store | FAISS, one index per user |
| Document processing | PyMuPDF, python-docx, python-pptx, openpyxl, BeautifulSoup |
| OCR & language | Tesseract (Malayalam, Arabic, English), EasyOCR, langdetect, indic-nlp-library |
| Frontend | React 18 single-page UI |
| Deployment | Docker & Docker Compose |

---

## ✅ Engineering & Quality

- **Structured UAT:** 30-case user-acceptance register run end-to-end; every defect found was fixed and re-verified against a live backend.
- **Security hardening:** found and fixed a path-traversal vulnerability in file upload during testing.
- **Regression suite:** 20 API- and browser-level checks covering all fixes, passing against a freshly restarted backend.
- **Reproducible builds:** exact-pinned ML dependency stack so Docker builds behave the same as the development environment.
- **Honest performance profiling:** benchmarked multiple local models and settings on CPU-only hardware, and kept the one that produced correct answers rather than the fastest.

---

## 💡 How It Works

1. Sign in — workspace, files and vector index are private to the user.
2. Upload one or many documents; each is extracted (with OCR where needed), chunked, embedded and indexed.
3. Ask a question in English, Arabic or Malayalam; an intent router decides whether it needs the documents at all.
4. Relevant passages are retrieved from the documents in scope for that conversation.
5. The selected LLM answers in the question's language, with page-level citations.
6. The exchange is saved to chat history, tagged with the model that produced it.

---

<!-- Add screenshots here once the images are in docs/screenshots/ :

## 📸 Screenshots

| Chat with citations | Model switching |
|---|---|
| ![Chat](docs/screenshots/chat.png) | ![Models](docs/screenshots/models.png) |

| Malayalam Q&A | Admin & roles |
|---|---|
| ![Malayalam](docs/screenshots/malayalam.png) | ![Admin](docs/screenshots/admin.png) |
-->

## 📬 Contact

Built by **Asna Sherin** — AI Automation Specialist | Agentic AI Engineer | Claude Expert

<!-- Add your LinkedIn / email / portfolio links here -->

Source code and a live walkthrough are available to hiring teams on request.

## 📄 License

© 2026 Asna Sherin. All rights reserved — see [LICENSE](LICENSE).
