# Phase 1 — Technical Audit & Gap Analysis Report
### RAG-Based AI Intelligence Bot — Offline/Online Architecture, Arabic+English RAG, Document Format Support

Prepared for: Jameela
Scope: Phase 1 only (audit + gap analysis). No code has been changed. This report is the basis for approving Phase 2 (implementation).
Codebase audited: `C:\Users\user\Documents\Claude\Projects\AI FILE INTELLIGENCE BOT` (backend + frontend + config, as of the current `main` branch, last commit reflected in `.git/refs/heads/main`).

---

## How this audit was done

Every claim below is based on one of three things, and each claim says which:

- **Code-read**: I opened the actual source file and traced the logic.
- **Code-run**: I executed the exact code fragment (not a paraphrase) to confirm behavior.
- **Not verifiable from this session**: requires a check on your machine or a live network fetch I couldn't complete — flagged explicitly, never guessed.

I did not run the application end-to-end (no `pip install`, no live Ollama call, no live upload) — this session's shell for your machine is an isolated sandbox VM without your installed Python packages, Ollama binary, or Tesseract binary. Where the project's own prior documentation (`CR-01_OFFLINE_PLATFORM.md`) already discloses what was and wasn't runtime-tested, I've carried that distinction forward rather than re-asserting it as newly verified.

I also found `suppotring files/review of 3 projects/Offline-RAG-Evaluation-FINAL.docx` — a prior evaluation report addressed to you (dated September 2026) that compared this codebase against two competing builds and recommended it as the foundation, while flagging several of the same gaps independently confirmed below (no offline-only enforcement, open CORS, placeholder JWT secret, a plaintext API key file, missing PPTX/XLSX/HTML/email parsing). I treat that report as corroborating evidence, not as a substitute for reading the code — every finding below was independently verified against the actual source.

---

## A. What already works (code-read + code-run confirmed)

- **FastAPI backend with a real provider-abstraction layer.** `llm_provider_base.py` defines an `LLMProvider` ABC; `llm_provider_ollama.py`, `llm_provider_external.py` (Mistral/OpenAI/Gemini) implement it; `llm_service.py` walks a fallback chain (`ollama → mistral → openai → gemini → local synthesizer`). This is a genuine, working abstraction — not just a diagram. Requirement 1's "provider abstraction" ask is largely already met in shape.
- **Ollama is wired correctly against the documented REST API.** `llm_provider_ollama.py` calls `GET /api/tags` and `POST /api/chat` with a sensible timeout (180s, with a code comment explaining why: first-call model load time on CPU) and a plausible payload shape. I could not call a live Ollama instance from this session to confirm the response parsing end-to-end (see Section on Ollama verification below), but the request/response handling matches Ollama's documented API shape.
- **Per-user multi-tenancy is real, not cosmetic.** Every ingestion, retrieval, and FAISS path takes `user_id` and reads/writes under `uploads/<user_id>/` and `vector_store/<user_id>/` (`config.py`'s `user_upload_dir`/`user_vector_dir`, threaded through `ingestion.py`, `retrieval.py`, `langchain_pipeline.py`). One user's FAISS store is a physically separate directory from another's.
- **JWT auth + bcrypt password hashing** (`auth.py`) is implemented correctly: password verification, token issue/decode, `get_current_user`/`require_admin` FastAPI dependencies, disabled-user checks.
- **SQLite by default, zero external DB service** (`database.py`) — consistent with the offline objective; swappable to Postgres via `DATABASE_URL`.
- **PDF, DOCX, and TXT extraction work and don't call the network.** `processing.py`'s `extract_pdf` (PyMuPDF, with a pytesseract-OCR fallback for scanned pages), `extract_docx` (python-docx, including table cells), and `extract_txt` (multi-encoding: utf-8/utf-16/latin-1) are all local-only.
- **FAISS is the actual active vector store**, confirmed by code-read across `retrieval.py`/`langchain_pipeline.py` — no live code path touches ChromaDB. (`chroma.sqlite3` and `tfidf_vocab.pkl` on disk are leftovers from an earlier version — see Section F.)
- **Local embeddings — no API key needed.** `sentence-transformers` / `HuggingFaceEmbeddings` run on CPU; genuinely offline once the model weights are cached locally.
- **A real offline last-resort fallback exists.** When every LLM provider fails, `llm.py`'s `_local_response` → `_smart_synthesize` produces an answer from the retrieved chunks using regex/keyword heuristics rather than hard-failing. (Its actual capability is narrower than "reasoning" — see Section B.)
- **Ollama is the default-first provider in the fallback order**, and the DB-configurable "active model" (`models` table, admin-settable via `/models/active`) lets you switch models without a code change — this satisfies the ART prompt's "no code changes needed to switch models" requirement.

## B. What partially works (verified, with the specific gap named)

- **Offline capability is a fallback chain, not an enforced mode.** There is no `MODE=offline` / `MODE=online` switch anywhere in `config.py` or `llm_service.py`. If Ollama is unreachable, `_try_providers()` in `llm_service.py` silently falls through to Mistral (whose API key is currently set in your `.env`) and sends your document content to a cloud API — with only a `logger.warning`, no hard stop. For a genuinely air-gapped/offline-guaranteed deployment this is a compliance gap, not a preference; the prior evaluation report flags the same thing independently.
- **The offline "local synthesizer" is not a language model.** `llm.py`'s `_smart_synthesize` and helpers (`_extract_name`, `_extract_employer`, `_extract_role_title`, `_bullet_list`) are English-oriented regex/keyword heuristics, evidently built against resume/CV-shaped documents (the sample files in `uploads/1/` are literally CVs and a GST invoice). It's a real, useful fallback for "no LLM reachable at all," but it is not general-purpose reasoning, and it will not produce coherent Arabic output.
- **OCR exists and has a two-engine fallback (Tesseract → EasyOCR), but is English-only in practice** (see Section C — this is a hard blocker for Requirement 2, not a tuning issue).
- **Document format support is real for 3 formats (PDF/DOCX/TXT) and stubbed/broken for others.** `config.py`'s `SUPPORTED_TYPES` claims `.doc`, `.mp4`, `.mp3`, `.wav` are supported, but: `.doc` is routed to the same `extract_docx()` function as `.docx`, and `python-docx` cannot open the legacy binary `.doc` format — it will raise and the upload will fail with an opaque `RuntimeError`. Audio/video (`.mp4`/`.mp3`/`.wav`) route to `extract_video()`, which does `import whisper` — but `openai-whisper` is commented out in `requirements.txt` ("optional"), so on a fresh install this raises `ImportError` and the file fails ingestion. Both are claimed-supported in code but non-functional out of the box.
- **Source citation is partial.** Retrieved chunks carry `source` (filename) and `chunk_index`, and `/chat` persists `referenced_documents` per message — so file-level citation exists. There is no page number and no click-through, which the prior evaluation report also flags (Requirement 10 in the engineering rules: "maintain source traceability... where the current application supports them" — it does, at file level only).
- **Cross-document retrieval works architecturally** (`retrieve_context` searches the user's whole FAISS index, not one document, and `source` is an optional filter, not a hard scope) — but answer *quality* under cross-document retrieval (e.g., not mixing up entities from two different CVs) is untested from this session; the prior evaluation report flagged this as the single largest open risk on this codebase and recommended a specific entity-confusion test before going further.

## C. What does not work (concrete, code-verified failures)

These are the load-bearing findings for Requirement 2 (Arabic support). I ran the actual code, not a description of it.

1. **A single line of code destroys Arabic text (and any non-Latin script) on *every* ingested document, not just OCR.** `processing.py`, inside `clean_ocr_text()` — which `extract_text()` calls unconditionally for PDF, DOCX, TXT, and image extraction alike:
   ```python
   cleaned = re.sub(r'[^a-zA-Z0-9\s]{3,}', ' ', cleaned)
   ```
   This collapses any run of 3+ characters that are not an ASCII letter, digit, or whitespace into a single space. I ran it:
   ```
   input:  'Hello مرحبا بكم في هذا المستند world 123 test.'
   output: 'Hello     في     world 123 test.'
   ```
   Every Arabic word of 3+ characters is deleted. This means a perfectly clean, digitally-native Arabic PDF or DOCX — no OCR involved at all — loses essentially all of its Arabic content before it's ever chunked or embedded. This single line is the primary reason Arabic RAG cannot work today, and it's a one-line root cause, not a deep architectural problem.

2. **OCR is hardcoded to English**, in both engines:
   - `processing.py`: `pytesseract.image_to_string(img, config="--psm 6")` — no `lang=` argument, so Tesseract uses its default English traineddata even if Arabic (`ara.traineddata`) happens to be installed on the machine.
   - `processing.py`: `reader = easyocr.Reader(["en"], gpu=False, verbose=False)` — hardcoded to English only; EasyOCR does support Arabic (`"ar"`) but it's never requested.
   - There is no mechanism to select OCR language before extraction (a chicken-and-egg problem: language is detected from the *output* of OCR, but OCR itself needs to know the language to read Arabic correctly).

3. **"Always respond in English" is hardcoded into the LLM system prompt — in three separate places**, each an independent copy that would need to be found and fixed individually:
   - `llm.py` line 35 (`_RAG_SYSTEM`, used by `_mistral_response`)
   - `llm_service.py` (`_RAG_SYSTEM`, used by the main provider fallback chain)
   - `langchain_pipeline.py` (`_RAG_SYSTEM`, used by `run_rag_chain`)

   All three contain the literal line `"- Always respond in English.\n"`. Even if retrieval and embeddings supported Arabic perfectly, the model is explicitly instructed not to answer in Arabic.

4. **The embedding model is English-only by design, not just by default.** `EMBEDDING_MODEL=all-MiniLM-L6-v2` (both `.env` and `.env.example`). `sentence-transformers/all-MiniLM-L6-v2` is a monolingual English model — its Arabic embeddings are not semantically meaningful, so Arabic query→Arabic chunk similarity search will not reliably retrieve the right content even before the two bugs above are fixed. There's also no `arabic-reshaper` / `python-bidi` in `requirements.txt` (needed for correct Arabic ligature shaping/RTL display in some rendering contexts) and no RTL handling anywhere in the frontend.

5. **Frontend has zero language/RTL awareness.** `frontend/index.html` line 2 is `<html lang="en">`, hardcoded. No `dir="rtl"` toggle, no i18n strings, no language switcher anywhere in the file (checked by pattern search across the full 61,911-byte file).

6. **A live API key is committed to git in plaintext.** `git ls-files` (run against the actual repo) confirms `suppotring files/mistral key.txt` is tracked in version control — not gitignored, sitting alongside your `MISTRAL_API_KEY` in `.env`. If this repository has ever been pushed to a remote (a `.git/refs/remotes/origin/main` exists, meaning a remote is configured), that key is exposed in the remote history too, not just on disk. **This should be rotated regardless of what happens with Phase 2** — it's independent of the RAG work and worth doing today.

## D. What is missing (present in the requirements, absent from the code)

- **No `MODE=offline`/`MODE=online` configuration switch** (Section B elaborates — currently a fallback chain, not an enforced mode).
- **No PPTX, XLSX, HTML, email (.eml/.msg), CSV, Markdown, JSON, RTF, or OpenDocument support.** `SUPPORTED_TYPES` in `config.py` has exactly 10 entries (`.pdf .docx .doc .txt .png .jpg .jpeg .tiff .mp4 .mp3 .wav`), and `requirements.txt` has no `python-pptx`, `openpyxl`, `beautifulsoup4`/`unstructured`, or email-parsing library. None of these formats are wired anywhere in `processing.py`'s extractor dispatch.
- **No MIME-type / content validation on upload** — `api.py`'s `/upload` handler trusts the file extension only; a renamed file with a mismatched extension would be accepted and fail later, deeper in the pipeline, with a less clear error.
- **No ingestion status tracking.** `/upload` returns `"status": "processing"` immediately and runs ingestion as a `BackgroundTasks` job, but there's no status column on `documents` (`models_db.py`'s `Document` table has no `status` field) — a client has to guess when indexing finished by polling `/documents` and inferring from chunk counts.
- **No page-level citation** (file-level only — see Section B).
- **No dark mode** (flagged in the prior evaluation report too; not independently re-verified against the rendered frontend from this session).

## E. What should be reworked

- **Consolidate the three duplicated `_RAG_SYSTEM` prompt strings into one shared constant** (in `llm_provider_base.py` or a new `prompts.py`) — right now a future language/behavior fix has to be applied three times, and it's already drifted slightly (compare the wording of the three copies).
- **`clean_ocr_text()`'s noise-collapsing regex needs a genuine Unicode-aware readability check**, not an ASCII-only allowlist — this is the single highest-priority code fix in this entire audit (Section C.1).
- **OCR language must be selected before extraction, not inferred after.** The cleanest fix given the existing architecture: run OCR with a fixed multi-language pack (`ara+eng` for Tesseract, `["ar","en"]` for EasyOCR) rather than trying to detect-then-OCR, since detection needs text that doesn't exist yet.
- **Decouple "installed & configured" from "actually reachable" for offline claims.** `OllamaProvider.is_available()` does a real HTTP check, which is good — but nothing in the app refuses to fall through to a cloud provider when the deployment is meant to be offline-only. This needs an explicit gate, not just better logging.
- **The `get_langchain_text_splitter()` function in `langchain_pipeline.py` is dead code** — the docstring/architecture diagram at the top of that file claims "LangChain RecursiveCharacterTextSplitter" is the chunking method, but the actual ingestion path (`ingestion.py` → `embeddings.py`'s `chunk_text()`) uses a hand-rolled regex sentence-splitter that never calls this function. Either wire it in or remove the misleading docstring/diagram.

## F. What should NOT be changed because it already works correctly

- The per-user FAISS isolation design (directory-per-user, not a shared index with a metadata filter) — deliberate, documented trade-off, and the safer choice for "no cross-user access" per `CR-01_OFFLINE_PLATFORM.md`'s own reasoning. Don't touch this for Phase 2.
- JWT/bcrypt auth implementation — correct as implemented; only the *default secret/password values* need changing before non-local exposure, not the mechanism.
- The provider-fallback architecture shape (`LLMProvider` ABC + per-provider adapters + `llm_service.py` orchestrator) — this is exactly the "provider abstraction" pattern the ART prompt asks for. Phase 2 should add a `MODE` gate on top of it, not replace it.
- PDF/DOCX/TXT extraction logic itself (once the cleaning-regex bug is fixed) — PyMuPDF and python-docx usage is correct and doesn't need rework, just the shared post-processing step fixed.
- SQLite-by-default database design — correct for the offline objective, don't introduce Postgres in this phase.

## G. Legacy artifacts (cleanup candidates, not gaps)

- `vector_store/chroma.sqlite3` and `vector_store/tfidf_vocab.pkl` at the project root are leftovers from a pre-CR-01 version. Confirmed by code-read: no current code path (`retrieval.py`, `langchain_pipeline.py`) touches ChromaDB, and `memory.py`'s docstring ("Semantic (ChromaDB via retrieval.py)") is stale — `MemoryManager.build_context()` actually calls the FAISS-backed `retrieve_context()`. Safe to delete once you've confirmed you don't need to migrate anything out of the old Chroma collection.
- `README.md` describes the pre-CR-01 architecture (single global FAISS store, no auth, no multi-user) and is now materially out of date relative to `CR-01_OFFLINE_PLATFORM.md`. Worth a rewrite once Phase 2 lands, not before.

---

## Requirement Validation Table

| Requirement | Status | Evidence | Problem | Required Change |
|---|---|---|---|---|
| Offline Ollama (provider wired) | **PARTIAL** | `llm_provider_ollama.py` correctly implements `/api/tags`, `/api/chat` | Not runtime-verified from this session (no live Ollama reachable in the audit sandbox) | Run `ollama --version && ollama list` on your machine and confirm a real chat round-trip |
| Offline Ollama (enforced offline mode) | **FAIL** | `llm_service.py` `_try_providers()` — no `MODE` flag anywhere in `config.py` | Silently falls through to cloud Mistral if Ollama is down, since `MISTRAL_API_KEY` is set | Add explicit `MODE=offline`/`online` gate that disables non-local providers entirely in offline mode |
| Online/offline provider abstraction | **PASS** | `LLMProvider` ABC + 4 concrete providers + orchestrator, code-read | — | None — this already matches the requested pattern |
| Arabic RAG (end to end) | **FAIL** | `clean_ocr_text()` regex verified by code-run to delete Arabic text; hardcoded English OCR; hardcoded "Always respond in English" ×3; monolingual embedding model | Multiple independent, compounding blockers | See Section C, items 1–5 |
| English RAG | **PASS** | PDF/DOCX/TXT extraction, FAISS retrieval, English OCR, English LLM prompts all functional by code-read | — | None |
| Cross-language retrieval | **FAIL** | Depends on Arabic RAG working at all, plus a multilingual embedding model (currently English-only) | Blocked by the same root causes as Arabic RAG | Multilingual embedding model + Arabic pipeline fixes are prerequisites |
| PDF | **PASS** | `extract_pdf()` — PyMuPDF + pytesseract OCR fallback, code-read | — | None |
| DOC/DOCX | **PARTIAL** | `.docx` works; `.doc` is misrouted to the same extractor, which cannot open legacy binary `.doc` | `.doc` uploads will fail | Route `.doc` to a real legacy-format extractor (e.g., `textract`/LibreOffice headless conversion) or explicitly mark unsupported |
| PPT/PPTX | **FAIL** | Not in `SUPPORTED_TYPES`, no parser in `requirements.txt` or `processing.py` | Not implemented at all | Add `python-pptx`-based extractor |
| XLS/XLSX | **FAIL** | Same as above | Not implemented at all | Add `openpyxl`-based extractor |
| Images | **PARTIAL** | Extraction works (Tesseract→EasyOCR fallback); Arabic OCR is not requested from either engine | English images OCR correctly; Arabic images do not | Pass `lang="ara+eng"` / `["ar","en"]` to both engines |
| Scanned PDFs | **PARTIAL** | OCR fallback path exists in `extract_pdf()` | Same English-only OCR limitation as Images | Same fix as Images |
| OCR Arabic | **FAIL** | Both OCR engines hardcoded to English (code-read, Section C.2) | — | Explicit Arabic language pack on both engines |
| OCR English | **PASS** | Tesseract + EasyOCR both functional for English, code-read | — | None |
| Email (.eml/.msg) | **FAIL** | No parser present | Not implemented at all | Add `.eml` parsing (stdlib `email` module covers most of it) if this format is actually needed |
| HTML | **FAIL** | No parser present | Not implemented at all | Add HTML extraction (e.g., `BeautifulSoup`) if needed |
| Audio/Video transcription | **FAIL** (despite being in `SUPPORTED_TYPES`) | `openai-whisper` commented out in `requirements.txt`; `extract_video()` will raise `ImportError` | Claimed-supported, not functional out of the box | Either install Whisper and enable it, or remove `.mp4/.mp3/.wav` from `SUPPORTED_TYPES` until it is |

---

## Model & Component Recommendations

I want to be direct about confidence levels here, since your instructions ask me to flag anything I can't verify live: **I could not complete a live fetch against `ollama.com/library` or GitHub's Ollama releases page from this session** (the fetch tool's permission prompt wasn't answered in time, and a retry hit the same issue) — so exact current version numbers below are given at the confidence level stated, not asserted as freshly verified. Please treat any specific version number here as something to confirm at `https://ollama.com/library` and `https://github.com/ollama/ollama/releases` before you pull it.

**Ollama version.** I don't have a verified current version number for September 2026 from this session. What I can say with confidence: point releases in the 0.x series have historically been backward-compatible for the `/api/tags` and `/api/chat` endpoints this codebase already uses, so the code shouldn't need changes for a version bump — but please run `ollama --version` on your machine and paste it back to me so I can sanity-check it against the code's assumptions, rather than me guessing a number.

**Generation/reasoning model.** Based on well-established (pre-cutoff, but architecturally stable) knowledge of the current model families, not a live listing check:
- **Qwen2.5/Qwen3** (Alibaba) — strong, broadly-cited multilingual performance including Arabic, wide range of sizes (down to ~7B, which matters for CPU-only or modest-VRAM machines), Apache 2.0 license, well-supported in Ollama's library historically. This is my primary recommendation to evaluate first.
- **Jais** (Inception/G42, UAE-built) — purpose-built for Arabic, worth evaluating specifically because it's designed around Arabic linguistic structure rather than Arabic-as-one-of-many-languages. Given this project's Arabic requirement and UAE context, I'd put this on the shortlist even though general multilingual models often win on raw benchmark tables.
- **Llama 3.x** and **Gemma 2/3** — both have multilingual claims; in my training-era knowledge, Arabic was a secondary/lower-tier language for both compared to Qwen and Jais, but I have not verified this against current benchmarks. I do not have a verified source ranking these for Arabic specifically as of today — treat this as a hypothesis to test, not a conclusion.
- **Recommendation**: pull 2-3 candidates (e.g., a Qwen model, Jais, and one general model you're already comfortable with) and run the same Arabic + English test set against each, since your hardware constraints (CPU vs GPU, RAM) will matter as much as raw language quality — I don't have your machine's specs from this session to narrow this further.

**Embedding model.** This is the one recommendation I'm most confident about, because BGE-M3's multilingual design (100+ languages including Arabic, dense+sparse+multi-vector retrieval, long context) is well-documented and architecturally stable, not a fast-moving claim:
- **BAAI/bge-m3** — purpose-built for multilingual + cross-lingual retrieval, which is exactly what "Arabic document → English question" needs. This directly replaces `all-MiniLM-L6-v2` in `config.py`'s `EMBEDDING_MODEL` and works with the existing `sentence-transformers`/`HuggingFaceEmbeddings` code path with no architecture change — only the model name and re-indexing.
- **intfloat/multilingual-e5-large** — a solid alternative if BGE-M3's resource footprint is too heavy for your hardware; slightly narrower feature set (dense retrieval only) but still strong Arabic+English coverage.
- Either change requires **re-embedding every already-ingested document** — existing FAISS indices were built with `all-MiniLM-L6-v2` and are not compatible with a different embedding model's vector space.

**OCR engine.** Keep the existing Tesseract→EasyOCR fallback architecture — it's sound — but: install the Arabic Tesseract language pack (`ara.traineddata`) alongside English, and pass `lang="ara+eng"` explicitly; and change the EasyOCR reader to `easyocr.Reader(["ar", "en"], ...)`. No new OCR engine is needed; this is a configuration/parameter fix, not a rework.

**Vector database.** Keep FAISS — no changes needed for the offline requirement. BGE-M3's higher dimensionality (1024 vs MiniLM's 384) means existing FAISS index files must be rebuilt, not just re-pointed.

---

## Recommended Architecture

**Offline mode (target — not yet implemented):**
```
Application
    |
Local RAG Pipeline (ingestion.py -> processing.py -> embeddings.py)
    |
Local Embedding Model (BGE-M3 or multilingual-e5, replacing all-MiniLM-L6-v2)
    |
Local Vector Database (FAISS, per-user — unchanged)
    |
Ollama (enforced — no fallthrough to cloud providers when MODE=offline)
    |
Local LLM (Qwen/Jais/other candidate, selected after your own A/B test)
```

**Online mode (already close to this shape today):**
```
Application
    |
RAG Pipeline (unchanged)
    |
Embedding Provider (local model or a cloud embedding API, config-selected)
    |
Vector Database (FAISS, unchanged)
    |
Configurable LLM Provider (Mistral / OpenAI / Gemini — already implemented)
```

The main architectural change Phase 2 needs is not a new layer — it's a `MODE` config flag read by `llm_service.py`'s `_try_providers()` that removes non-Ollama providers from the fallback order entirely when `MODE=offline`, rather than just trying Ollama first.

---

## Supported File Matrix — Current vs. Recommended

| Extension | Current Status | Recommended Classification | Notes |
|---|---|---|---|
| `.pdf` | SUPPORTED | SUPPORTED | Keep as-is |
| `.docx` | SUPPORTED | SUPPORTED | Keep as-is |
| `.doc` | **BROKEN** (claimed, non-functional) | REQUIRES NEW PARSER | Needs a legacy-binary-capable extractor or explicit removal from `SUPPORTED_TYPES` |
| `.txt` | SUPPORTED | SUPPORTED | Keep as-is |
| `.png/.jpg/.jpeg/.tiff` | PARTIALLY SUPPORTED (English OCR only) | SUPPORTED after Arabic OCR fix | See OCR recommendation above |
| `.mp4/.mp3/.wav` | **BROKEN** (claimed, non-functional — Whisper not installed) | REQUIRES NEW PARSER (install Whisper) or NOT RECOMMENDED for Phase 2 (out of scope per ART prompt's Phase 1-3 boundary) | Your ART prompt's Phase 1–3 scope doesn't require audio/video — I'd leave this out of scope rather than add Whisper now |
| `.pptx` | NOT SUPPORTED | REQUIRES NEW PARSER | `python-pptx`, in-scope per your document-format requirement |
| `.xlsx` | NOT SUPPORTED | REQUIRES NEW PARSER | `openpyxl`, in-scope |
| `.html/.htm` | NOT SUPPORTED | REQUIRES NEW PARSER | Straightforward with `BeautifulSoup` |
| `.eml/.msg` | NOT SUPPORTED | REQUIRES NEW PARSER if genuinely needed | Confirm with you whether email ingestion is actually in scope before building it — not mentioned as a current use case in the project's own docs |
| `.csv/.md/.json` | NOT SUPPORTED | REQUIRES NEW PARSER (low effort — plain-text-adjacent) | Cheap to add alongside `.txt` |
| `.rtf/.odt/.ods/.odp` | NOT SUPPORTED | NOT RECOMMENDED for Phase 2 | Low prevalence in typical business use; add only if you have real files in these formats |

---

## Executive Summary

**Does the project currently satisfy Requirements 1–3? PARTIALLY.**

- **Requirement 1 (offline architecture)**: The provider-abstraction pattern is genuinely well-built and Ollama is correctly wired as the default provider. What's missing is an *enforced* offline mode — right now, an unreachable Ollama silently and invisibly routes your document content to a cloud API if one is configured, which defeats the "guaranteed offline" objective even though the code otherwise supports it. This is a config/gating fix, not a rework.
- **Requirement 2 (Arabic + English RAG)**: English works end-to-end. Arabic does not work at all, and the root causes are concrete and fixable: one regex line that destroys non-Latin text on every document (not just OCR), OCR hardcoded to English in two engines, "Always respond in English" hardcoded in three separate prompt strings, and a monolingual embedding model. None of these require an architecture rewrite — they're targeted fixes — but there are five of them and they compound (fixing only the embedding model, for instance, won't help if the text was already destroyed by the cleaning regex before it got there).
- **Requirement 3 (document format support)**: PDF/DOCX/TXT/Images work. PPTX, XLSX, HTML, and email are entirely absent — not partially built, not present at all. `.doc` and audio/video are claimed as supported in config but will fail at runtime.

None of this requires discarding the existing architecture. The per-user isolation, the provider abstraction, the FAISS retrieval pipeline, and the auth/database layer are all sound and should be preserved. The work is concentrated in `processing.py` (text cleaning + OCR language), `config.py`/`llm_service.py` (offline mode gate + embedding model), the three `_RAG_SYSTEM` prompt strings, and adding a handful of new format extractors.

---

## Remaining Issues / Requires Verification

1. **Live Ollama reachability and installed models** — not verified from this session. Please run `ollama --version` and `ollama list` on your actual machine (not this sandbox) and share the output.
2. **Current Ollama stable version number** — I could not complete a live fetch to confirm this; verify at `https://ollama.com` / `https://github.com/ollama/ollama/releases` before pulling a specific version.
3. **Current best Arabic-capable local model, benchmarked** — my recommendation (Qwen/Jais as first candidates) is based on architecturally-stable knowledge, not a live current benchmark; needs your own side-by-side test once Phase 2 lands.
4. **The plaintext `mistral key.txt` in git** — recommend rotating this key regardless of Phase 2 approval status; it's unrelated to the Arabic/offline work and shouldn't wait.
5. **Cross-document entity-confusion risk** — flagged independently by the prior evaluation report and not yet tested by me; worth a dedicated test once Arabic support is fixed (test both languages).
6. **Hardware constraints** — I don't have your machine's CPU/GPU/RAM specs from this session; model size recommendations above are provisional until I know what you're running on.

---

## What Phase 2 Would Change (proposed scope, pending your approval)

1. Fix `clean_ocr_text()`'s regex to be Unicode-aware (the single highest-impact fix).
2. Add explicit `lang="ara+eng"` (Tesseract) / `["ar","en"]` (EasyOCR) to both OCR call sites.
3. Consolidate the three `_RAG_SYSTEM` prompts into one shared constant; make the response-language rule follow the query language instead of hardcoding English.
4. Swap `EMBEDDING_MODEL` to `BAAI/bge-m3` (or `multilingual-e5-large` as a lighter alternative) and re-embed existing documents.
5. Add a `MODE=offline`/`online` config flag that removes cloud providers from `llm_service.py`'s fallback chain entirely when offline.
6. Add PPTX (`python-pptx`) and XLSX (`openpyxl`) extractors; fix or remove the broken `.doc` and audio/video claims.
7. Rotate the exposed Mistral API key (independent of everything else).

**This report stops here, per your phase-control instructions. I have not modified any code.** Let me know if you'd like changes to this gap analysis, or approval to proceed into Phase 2 implementation of items 1–7 above (or a subset of them).
