# RAG-Based AI Bot 🤖

**A multi-user, offline-first document intelligence platform.** Upload your files, chat with them in English, Arabic or Malayalam, and choose which LLM answers — local models by default, or cloud models when you explicitly go online. Every answer is grounded in the user's own documents and cited to the page.

> 🔒 **This is a project showcase.** The source code is in a private repository and is available to hiring teams on request (see [Contact](#-contact)).

![Architecture](docs/architecture.png)

---

## ⭐ Main Features

| | Feature | Highlights |
|---|---|---|
| 🌍 | **Multilingual** | English, Arabic & Malayalam — ask in any, switch mid-conversation, and the bot follows. |
| 📂 | **Multi-file, multi-format upload** | PDF, DOCX, TXT, PPTX, XLSX, HTML, EML and images (OCR), several at once. |
| 🔀 | **Multi-LLM, interchangeable** | Local and cloud models behind one provider layer; switch live from the chat UI. |
| 🔐 | **User authentication** | Secure token-based sessions and hashed passwords. |
| 🛡️ | **Role-based access control** | Separate admin and user roles. |
| 📍 | **Source citations** | Answers point to the page, slide or sheet they came from. |
| 🎯 | **Scope-based chat** | Chat with one document, several selected documents, or the whole document set. |
| 💬 | **Chat history** | Persistent per-user conversations, restored on reload. |

### 🌍 Multilingual — English, Arabic & Malayalam
- Multilingual embeddings, so a question in one language can retrieve from documents in another.
- Per-question language detection; the answer comes back **in the language of the current question**, even after a mid-conversation switch.
- A dedicated **Malayalam pipeline** (script detection, Unicode normalization, OCR for scanned documents) and **Arabic OCR** support.
- Summaries and suggested questions are written in the document's own language.

### 📂 Multiple file upload, many file types
- Upload several files at once: **PDF, DOCX, TXT, PPTX, XLSX, HTML, EML** and **images (OCR)**.
- **Live per-file status tracking** — a failed file is reported immediately instead of blocking the rest of the batch.
- **Table-aware PDF extraction** keeps the row structure of tables.
- Garbled text is detected and automatically falls back to OCR.

### 🔀 Multi-LLM with live switching
- Pluggable provider layer supporting **local models (Ollama)** and **cloud models (Mistral, OpenAI, Gemini)**.
- Switch the active model from the chat UI, even mid-conversation; the next reply uses it.
- The model behind every answer is **recorded in the chat history**.
- In the default offline mode, cloud providers are blocked — there is no silent cloud fallback.

### 🔐 User authentication
- Token-based login with securely hashed passwords.
- Self-registration and session management.

### 🛡️ Role-based access control
- **Admins** manage users, view platform statistics and control which models are active.
- **Users** get a private workspace: their own files, vector index and conversations.

### 📍 Source citations
- Answers are grounded only in retrieved content and cited to the **page** (PDF), **slide** (PPTX) or **sheet** (XLSX).
- Documents are scoped per conversation, so citations never leak across unrelated chats.

### 🎯 Scope-based chat
- Choose what the bot searches: a **single document**, **multiple selected documents**, or the **whole document set** in the conversation.
- Retrieval and citations stay inside the chosen scope, so answers never mix in unrelated files.
- Follow-up questions such as "how many modules does it have?" stay on the document you were just discussing.

### 💬 Chat history
- Conversations are stored per user and restored on reload.
- *New Chat* clears messages but keeps the documents; *New Window* starts a fresh workspace.

### Also included
- 🔒 **Per-user & per-conversation data isolation** — separate storage and vector index for every user.
- 🧭 **Intent routing** — small talk never triggers document retrieval.
- 📝 **On-demand summaries, document overviews and suggested questions**.
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
| Auth & security | JWT, bcrypt, role-based route guards |
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

- **Structured UAT:** a 30-case user-acceptance register run end-to-end against a live backend; the gaps it found were fixed or mitigated and re-verified.
- **Security hardening:** found and fixed a path-traversal vulnerability in file upload during testing.
- **Regression suite:** 20 API- and browser-level checks covering all fixes, passing against a freshly restarted backend.
- **Reproducible builds:** exact-pinned ML dependency stack so Docker builds behave the same as the development environment.
- **Honest performance profiling:** benchmarked multiple local models and settings on CPU-only hardware, and kept the one that produced correct answers rather than the fastest.

---

## 🧩 Challenges & Solutions

| Challenge | Root cause | What I did |
|---|---|---|
| **Slow responses and slow document upload** | Local LLM and embedding inference runs on CPU-only hardware, and large documents mean a lot of text to embed and index. | Profiled the pipeline, benchmarked several local models, and tuned prompts and generation limits. Kept the model that gave correct answers rather than the fastest, and added a clear "model busy" response instead of silent hangs. **Recommended solution:** integrate a dedicated GPU to speed up both generation and embedding. |
| **Online/offline LLM switching failed silently** | The automatic fallback to the local model hid cloud failures such as exhausted token credits or no internet access, so the system appeared not to switch. | Traced it through the logs, API credentials and token usage, then fixed the switching behaviour and made provider failures visible instead of silently masked. |
| **Arabic and Malayalam text came out wrong** | The OCR and text pipeline lacked proper language support for these scripts. | Researched the problem and integrated Tesseract (via pytesseract) with the Arabic and Malayalam language packs, plus script detection and Unicode normalization. |

---

## 💡 How It Works

1. Sign in — workspace, files and vector index are private to the user.
2. Upload one or many documents; each is extracted (with OCR where needed), chunked, embedded and indexed.
3. Ask a question in English, Arabic or Malayalam; an intent router decides whether it needs the documents at all.
4. Relevant passages are retrieved from the documents in scope for that conversation.
5. The selected LLM answers in the question's language, with page-level citations.
6. The exchange is saved to chat history, tagged with the model that produced it.

---

## 📬 Contact

Built by **Asna Sherin** — AI Automation Specialist | Agentic AI Engineer | Claude Expert
[Email](mailto:assumay@gmail.com) | [LinkedIn](https://www.linkedin.com/in/asnasherinep/) | [Portfolio](https://asna-sherin.vercel.app/)

Source code and a live walkthrough are available to hiring teams on request.

## 📄 License

© 2026 Asna Sherin. All rights reserved — see [LICENSE](LICENSE).
