"""
llm.py - LLM response engine
v3: Ollama (primary, offline) -> Mistral -> OpenAI -> local synthesizer (offline)

Provider selection/fallback now lives in llm_service.py (the LLM Service
Layer). This module still owns the actual Mistral/OpenAI SDK calls (reused
by llm_provider_external.py) and the offline regex-based local synthesizer,
preserved untouched as the final fallback when no provider is reachable.

Public API:
  generate_response(db, prompt_context, query, chat_history) -> Dict
  generate_summary(db, context_chunks)                       -> Dict
  generate_recommendations(db, context_chunks)                -> Dict
  _smart_synthesize(query, prompt_context)               -- langchain_pipeline fallback
  _build_prompt_str(chunks, query)                       -- langchain_pipeline helper
"""
import re
import logging
from typing import List, Dict, Any, Optional

from config import (
    MISTRAL_API_KEY, MISTRAL_MODEL,
    OPENAI_API_KEY, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE,
)

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


# ======================================================================
# PRIMARY ENTRY POINTS
# ======================================================================

def generate_response(db, prompt_context, query, chat_history=None):
    """Delegates to the LLM Service Layer (llm_service.py), which tries the
    admin-configured active provider first (Ollama by default), then falls
    back through the remaining providers, and finally to _local_response()
    below if nothing is reachable."""
    from llm_service import generate_response as _service_generate_response
    return _service_generate_response(db, prompt_context, query, chat_history)


def generate_summary(db, context_chunks):
    from llm_service import generate_summary as _service_generate_summary
    return _service_generate_summary(db, context_chunks)


def generate_recommendations(db, context_chunks):
    from llm_service import generate_recommendations as _service_generate_recommendations
    return _service_generate_recommendations(db, context_chunks)


# ======================================================================
# MISTRAL (Primary LLM)
# ======================================================================

def _mistral_response(prompt_context, query, chat_history):
    # Try LangChain ChatMistralAI first
    try:
        from langchain_mistralai import ChatMistralAI
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        lc_messages = [SystemMessage(content=_RAG_SYSTEM)]
        for turn in chat_history[-6:]:
            role    = turn.get("role", "user")
            content = turn.get("content", "")[:500]
            lc_messages.append(
                AIMessage(content=content) if role == "assistant"
                else HumanMessage(content=content)
            )
        lc_messages.append(HumanMessage(content=prompt_context))

        llm = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model=MISTRAL_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        response = llm.invoke(lc_messages)
        return {"answer": response.content.strip(), "model": MISTRAL_MODEL,
                "mode": "mistral-langchain", "tokens": 0}
    except Exception as e:
        logger.warning(f"LangChain Mistral failed: {e}. Trying direct SDK...")

    # Direct mistralai SDK fallback
    try:
        from mistralai import Mistral
        msgs = [{"role": "system", "content": _RAG_SYSTEM}]
        for turn in chat_history[-6:]:
            msgs.append({"role": turn.get("role","user"), "content": turn.get("content","")[:500]})
        msgs.append({"role": "user", "content": prompt_context})

        client = Mistral(api_key=MISTRAL_API_KEY)
        resp   = client.chat.complete(
            model=MISTRAL_MODEL, messages=msgs,
            max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
        )
        answer = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if hasattr(resp, "usage") and resp.usage else 0
        return {"answer": answer, "model": MISTRAL_MODEL, "mode": "mistral-direct", "tokens": tokens}
    except Exception as e:
        logger.error(f"Mistral direct SDK failed: {e}")
        return None


# ======================================================================
# OPENAI (Legacy fallback)
# ======================================================================

def _openai_response(prompt_context, query, chat_history):
    try:
        from openai import OpenAI
        client   = OpenAI(api_key=OPENAI_API_KEY)
        messages = [{"role": "system", "content": _RAG_SYSTEM}]
        for turn in chat_history[-6:]:
            messages.append({"role": turn.get("role","user"), "content": turn.get("content","")})
        messages.append({"role": "user", "content": prompt_context})
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
        )
        return {"answer": resp.choices[0].message.content.strip(),
                "model": LLM_MODEL, "mode": "openai-fallback",
                "tokens": resp.usage.total_tokens}
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        return None


# ======================================================================
# LOCAL SYNTHESIZER (offline, no API key needed)
# ======================================================================

def _local_response(prompt_context, query):
    answer = _smart_synthesize(query, prompt_context)
    # If synthesizer returned garbled or empty text, give a clear API-unavailable message
    if not answer or not _is_readable(answer, threshold=0.40):
        answer = (
            "The AI model is currently unavailable (no valid API key). "
            "Please add your MISTRAL_API_KEY to the .env file and restart the server."
        )
    return {"answer": answer, "model": "local-synthesizer", "mode": "offline", "tokens": 0}


# ── Helpers used by langchain_pipeline ────────────────────────────────

def _build_prompt_str(chunks, query):
    lines = [
        f"[{i}] Source: {c.get('source','')} (score: {c.get('score',0)})\n{c.get('text','')[:600]}"
        for i, c in enumerate(chunks, 1)
    ]
    return "RETRIEVED KNOWLEDGE:\n" + "\n\n".join(lines) + f"\n\nCURRENT QUESTION:\n{query}"


def _extract_chunks(prompt_context):
    if "RETRIEVED KNOWLEDGE:" not in prompt_context:
        return []
    start = prompt_context.index("RETRIEVED KNOWLEDGE:") + len("RETRIEVED KNOWLEDGE:")
    end   = prompt_context.find("CURRENT QUESTION:", start)
    raw   = prompt_context[start:end].strip() if end != -1 else prompt_context[start:].strip()
    parts = re.split(r"\[\d+\]\s+Source:.*?\n", raw)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _extract_sources(prompt_context):
    """Extract unique source filenames from the prompt context block."""
    return list(dict.fromkeys(re.findall(r"\[\d+\]\s+Source:\s*([^\s(]+)", prompt_context)))


def _is_readable(text: str, threshold: float = 0.50) -> bool:
    """Return True if text has enough alphabetic / digit characters to be meaningful."""
    if not text or len(text) < 8:
        return False
    readable = sum(1 for c in text if c.isalpha() or c.isdigit() or c in " .,!?;:()'\"")
    return readable / len(text) >= threshold


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.strip()) > 15]


def _find(sents, keywords, n=3):
    hits = []
    for s in sents:
        sl = s.lower()
        if any(kw.lower() in sl for kw in keywords):
            hits.append(s)
        if len(hits) >= n:
            break
    return hits


def _trim(text, max_s=3):
    parts  = text.split(". ")
    result = ". ".join(parts[:max_s]).strip()
    return result + "." if result and not result.endswith(".") else result


def _normalize(q):
    q = q.lower().strip()
    q = re.sub(r"[''']", "", q)
    for old, new in [("whos","whose"),("shes","she is"),("hes","he is"),("whats","what is")]:
        q = re.sub(r"\b" + old + r"\b", new, q)
    return re.sub(r"[^\w\s]", " ", q).strip()


_COMPANY_WORDS = {
    "real", "estate", "international", "company", "limited", "ltd", "llc",
    "inc", "corp", "group", "solutions", "services", "technologies", "seeking",
    "hiring", "agency", "institute", "association", "foundation", "enterprise",
}

def _is_company_name(name: str) -> bool:
    """Return True if the name looks like an organisation rather than a person."""
    parts = name.lower().split()
    return any(p in _COMPANY_WORDS for p in parts) or len(parts) > 3


def _extract_name(full_text):
    """
    Extract candidate / person name from resume text.
    Strategy (in priority order):
      1. ALL-CAPS block at the very top of the doc (first 600 chars) — skip company words
      2. Name hinted by LinkedIn URL username (e.g. linkedin.com/in/asnasherin)
      3. Title-Case "First Last" pair — skip any that look like company names
    """
    # ── 1. ALL-CAPS name near the top of the document ────────────────────
    top = full_text[:600]
    for m in re.finditer(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b', top):
        candidate = m.group(1).strip().title()
        if not _is_company_name(candidate):
            return candidate

    # ── 2. LinkedIn URL → username → match ALL-CAPS block in full text ──
    li = re.search(r'linkedin\.com/in/([a-zA-Z0-9\-]+)/?', full_text, re.IGNORECASE)
    if li:
        username = re.sub(r'[\-_]', ' ', li.group(1)).lower()
        # Try to find an ALL-CAPS version of this name in the text
        parts = username.split()
        if len(parts) >= 2:
            pat = r'\b(' + r'\s+'.join(p.upper() for p in parts) + r')\b'
            m = re.search(pat, full_text)
            if m:
                return m.group(1).strip().title()
        # Fallback: capitalise the username
        return ' '.join(p.capitalize() for p in parts)

    # ── 3. Title-Case "First Last" — skip company names ──────────────────
    for m in re.finditer(r'\b([A-Z][a-z]{1,14}\s+[A-Z][a-z]{1,14})\b', full_text):
        candidate = m.group(1)
        if not _is_company_name(candidate):
            return candidate

    return None


def _extract_employer(full_text, sents):
    patterns = [
        r"(EGC\s+Properties[^\n,]{0,60}(?:present|2026|2025|2024)[^\n,]{0,30})",
        r"([A-Z][A-Za-z\s&]+(?:Ltd|LLC|Inc|Corp|Group|Properties|Tech|Solutions)?"
        r"\s*(?:20\d\d)\s*[-]\s*(?:Present|20\d\d))",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:80]
    for s in sents:
        sl = s.lower()
        if any(c in sl for c in ["egc","properties","bayut","real estate"]) and \
           any(d in sl for d in ["present","2026","2025","2024"]):
            return re.split(r"[,;]", s)[0].strip()[:100]
    return None


def _extract_role_title(full_text, sents):
    # 1. Look for "applying for the <Role>" pattern (cover letters)
    m = re.search(r"applying for (?:the )?([A-Z][A-Za-z &/\-]{3,60})\s*(?:role|position|post)?",
                  full_text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip("role position post".split()[0])[:120]

    # 2. Specific known title patterns
    m = re.search(r"(Data\s*[&|]\s*Business\s*Intelligence[^\n+]{0,120})", full_text, re.IGNORECASE)
    if m:
        raw = re.sub(r"\+\d[\d\s]{8,}", "", m.group(1))
        raw = re.sub(r"[|]\s*\S+@\S+", "", raw)
        return re.sub(r"\s+", " ", raw).strip().strip("|").strip()[:120]

    # 3. Sentence scan — skip salutations like "Dear Hiring Manager"
    SALUTATION_SKIP = re.compile(r"^(dear|to whom|hello|hi)\b", re.IGNORECASE)
    for s in sents:
        if SALUTATION_SKIP.match(s.strip()):
            continue
        if any(t in s for t in ["Analyst","Developer","Expert","Manager","Engineer","Specialist","Consultant"]):
            title = re.sub(r"^[A-Z][A-Z ]{3,}\s*", "",
                           s.split("+971")[0].split("*")[0].strip()).strip()
            if len(title) > 5:
                return title[:120]
    return None


def _bullet_list(sents: list, max_items: int = 8) -> str:
    """Format a list of sentences as a readable bullet list."""
    clean = [s.strip().rstrip(".") for s in sents if _is_readable(s)]
    if not clean:
        return ""
    return "\n".join(f"• {s}" for s in clean[:max_items])


def _smart_synthesize(query, prompt_context):
    chunks = _extract_chunks(prompt_context)
    if not chunks:
        return "The document does not contain enough relevant information to answer this query."

    full_text = " ".join(chunks)
    sents     = _sentences(full_text)
    readable  = [s for s in sents if _is_readable(s)]
    q         = _normalize(query)

    # ── File / document name query ────────────────────────────────────────
    if any(p in q for p in ["name of the file","file name","filename","document name",
                              "name of document","name of this file","what file"]):
        sources = _extract_sources(prompt_context)
        if sources:
            names = ", ".join(sources)
            return f"The document{'s' if len(sources) > 1 else ''} in context: {names}"
        return "The file name is not available in the current context."

    # ── Identity ──────────────────────────────────────────────────────────
    if any(p in q for p in ["your name","who are you","what are you"]):
        return "I am the AI File Intelligence Bot — a RAG-powered assistant that answers questions from your uploaded documents."

    # ── Resume: identity (expanded) ───────────────────────────────────────
    if any(p in q for p in ["whose resume","whose cv","this resume","resume belong",
                              "cv belong","who is this","who does this","who wrote",
                              "whose name","name is visible","name visible","name shown",
                              "name appear","name on"]):
        name = _extract_name(full_text)
        if name:
            # Cross-check: does the name appear in the readable context?
            name_in_ctx = any(name.lower() in s.lower() for s in readable)
            qualifier = "" if name_in_ctx else " (based on available context)"
            return f"This is the resume of {name}{qualifier}."
        # Try to find a LinkedIn URL as a hint
        li = re.search(r'linkedin\.com/in/([a-zA-Z0-9\-]+)/?', full_text, re.IGNORECASE)
        if li:
            return f"The candidate's LinkedIn is: linkedin.com/in/{li.group(1)}"
        return "The candidate's name could not be clearly identified from the retrieved context."

    # ── Resume: current employer ──────────────────────────────────────────
    if any(p in q for p in ["current employer","current company","current job","currently work",
                              "last experience","most recent","last company","last job"]):
        employer = _extract_employer(full_text, sents)
        role     = _extract_role_title(full_text, sents)
        if employer and role:
            return (role + " at " + employer + ".") \
                   if not any(t in employer for t in ["Analyst","Developer","Expert"]) \
                   else employer + "."
        return ("Currently works at " + employer + ".") if employer else \
               "The current employer is not explicitly stated in the document."

    # ── Resume: role / title ──────────────────────────────────────────────
    if any(p in q for p in ["position","role","title","designation","job title","what position"]):
        title = _extract_role_title(full_text, sents)
        return title if title else "The position is not clearly identified in the retrieved context."

    # ── Resume: name ──────────────────────────────────────────────────────
    if any(p in q for p in ["her name","his name","person name","full name","candidate name",
                              "applicant name","employee name","what is the name","what name",
                              "name of the person","name of the candidate"]):
        name = _extract_name(full_text)
        return (name + ".") if name else "The name is not clearly extractable from the document."

    # ── Location ──────────────────────────────────────────────────────────
    if any(p in q for p in ["where","location","based","city","country","live"]):
        hits = _find(readable, ["located","based in","address","city","country","state","province",
                                 "abu dhabi","dubai","uae","india","kerala"], n=2)
        if hits:
            return "\n".join(hits)
        loc = re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*,\s*[A-Z][a-z]+", full_text)
        return loc.group(0) if loc else "Location not explicitly stated in the document."

    # ── Contact ──────────────────────────────────────────────────────────
    if any(p in q for p in ["email","phone","contact","number","linkedin","github"]):
        hits = _find(readable, ["@","linkedin","github","phone","tel","mobile","+"], n=2)
        return "\n".join(hits) if hits else "Contact details not found in the retrieved context."

    # ── Experience / duration ─────────────────────────────────────────────
    if any(p in q for p in ["years of experience","how many years","how long","total experience","experience"]):
        hits = _find(readable, ["year","experience","since","from","worked","employed"], n=4)
        result = _bullet_list(hits, max_items=5)
        return result if result else "Experience details not clearly stated in the document."

    # ── Skills / tools / technologies ────────────────────────────────────
    if any(p in q for p in ["skill","tool","technolog","tech stack","expertise","competenc",
                              "profic","capabilit","abilit","what can"]):
        # Broad keyword set — catches both technical and soft skills
        skill_kw = [
            "skill","profic","expert","knowledge","certif","tool","technolog",
            "able","compet","capab","familiar","experienc","develop","analyt",
            "manage","communic","problem","team","leader","solv","design",
            "Power BI","Python","SQL","Excel","Tableau","DAX","ETL","Power Query",
            "Microsoft","programming","database","software","cloud","Azure","AWS",
        ]
        hits = _find(readable, skill_kw, n=12)
        if not hits:
            # Fallback: return all readable sents from context
            hits = readable
        result = _bullet_list(hits, max_items=10)
        return result if result else "Skills information is not clearly captured in the retrieved context."

    # ── Qualifications / education / certificates ─────────────────────────
    if any(p in q for p in ["qualif","certif","degree","education","certified","academ","study","studied"]):
        hits = _find(readable, ["certif","degree","bachelor","master","diploma","university",
                                 "college","school","studied","graduate","CSM","scrum","course"], n=6)
        result = _bullet_list(hits, max_items=6)
        return result if result else "Qualification details not found in the retrieved context."

    # ── User asking to confirm their own statement ────────────────────────
    if any(p in q for p in ["am i right","am i correct","is that right","is that correct",
                              "correct me","right?","correct?","isnt it","isn t it",
                              "na?","na right","isn t that"]):
        # Pull any context-matching evidence from retrieved chunks
        # Extract nouns/entities the user mentioned (words > 3 chars, not stop words)
        stop = {"that","this","right","correct","wrong","about","from","with","have","does"}
        keywords = [w for w in re.sub(r"[^\w\s]","",q).split()
                    if len(w) > 3 and w not in stop]
        hits = _find(readable, keywords, n=3)
        evidence = [s for s in hits if _is_readable(s)]
        if evidence:
            return "Yes, that is correct. " + evidence[0].strip() + "."
        # No specific match — give a general confirmation using name extraction
        name = _extract_name(full_text)
        if name and name.lower() in q.lower():
            return f"Yes, you are right. This is {name}'s document."
        return ("Yes, based on the document that appears to be correct."
                if any(p in q for p in ["right","correct"]) else
                "The document does not clearly confirm or deny this.")

    # ── Yes/No questions ──────────────────────────────────────────────────
    first_word = q.split()[0] if q.split() else ""
    if first_word in ("is","does","did","has","can","are","was","have"):
        keywords = [w for w in q.split() if len(w) > 3]
        hits = _find(readable, keywords, n=3)
        if hits:
            return "Yes. " + hits[0].strip()
        return "The document does not clearly confirm or deny this."

    # ── Summary / overview ────────────────────────────────────────────────
    if any(p in q for p in ["summary","background","about","overview","tell me","describe",
                              "who is","profile","introduction","main"]):
        hits = _find(readable, ["professional","experienc","expert","analyt","develop",
                                 "business","specialist","background","overview"], n=5)
        pool = (hits or readable)[:5]
        return _trim(" ".join(pool), 6)

    # ── "What else" / "more" / follow-up questions ───────────────────────
    if any(p in q for p in ["what else","anything else","more","other","also mention",
                              "besides","additional","further","else is","rest"]):
        result = _bullet_list(readable, max_items=8)
        return result if result else "No additional information found in the retrieved context."

    # ── Generic keyword search ────────────────────────────────────────────
    keywords = [w for w in re.sub(r"[^\w\s]", "", q).split() if len(w) > 3]
    hits     = _find(readable, keywords, n=5)
    readable_hits = [s for s in hits if _is_readable(s)]
    if readable_hits:
        result = _bullet_list(readable_hits, max_items=6)
        return result if result else _trim(" ".join(readable_hits[:3]), 4)

    # Last resort: return first few readable sentences
    if readable:
        return _bullet_list(readable[:4], max_items=4)

    return "The document does not contain enough relevant information to answer this query."
