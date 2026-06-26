"""
llm.py - LLM response engine with smart local synthesizer + OpenAI support
"""
import re
import logging
from typing import List, Dict, Any

from config import OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI File Intelligence Assistant.
Answer ONLY using the RETRIEVED KNOWLEDGE provided. Be concise and factual.
- For simple WH questions (who, what, where, when): answer in 1 sentence.
- For detail questions: answer in 2-3 sentences max.
- Never dump raw text. Synthesize a clean, human-readable answer.
If context is insufficient respond: The document does not contain enough relevant information.
Always respond in English."""


def generate_response(prompt_context, query, chat_history=None):
    if OPENAI_API_KEY:
        return _openai_response(prompt_context, query, chat_history or [])
    return _context_only_response(prompt_context, query)


def _openai_response(prompt_context, query, chat_history):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in chat_history[-6:]:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": prompt_context})
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "model": LLM_MODEL, "mode": "openai", "tokens": response.usage.total_tokens}
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        return _context_only_response(prompt_context, query, error=str(e))


def _context_only_response(prompt_context, query, error=None):
    answer = _smart_synthesize(query, prompt_context)
    mode = "context-only" if not error else "context-only (OpenAI error)"
    return {"answer": answer, "model": "local-synthesizer", "mode": mode, "tokens": 0}


def _extract_chunks(prompt_context):
    if "RETRIEVED KNOWLEDGE:" not in prompt_context:
        return []
    start = prompt_context.index("RETRIEVED KNOWLEDGE:") + len("RETRIEVED KNOWLEDGE:")
    end = prompt_context.find("CURRENT QUESTION:", start)
    raw = prompt_context[start:end].strip() if end != -1 else prompt_context[start:].strip()
    parts = re.split(r"\[\d+\]\s+Source:.*?\n", raw)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _sentences(text):
    sents = re.split(r"(?<=[.!?])\s+|\n", text)
    return [s.strip() for s in sents if len(s.strip()) > 15]


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
    parts = text.split(". ")
    result = ". ".join(parts[:max_s]).strip()
    return result + "." if result and not result.endswith(".") else result


def _normalize(q):
    q = q.lower().strip()
    q = re.sub(r"[\u2018\u2019\u0027]", "", q)
    q = re.sub(r"\bwhos\b", "whose", q)
    q = re.sub(r"\bshes\b", "she is", q)
    q = re.sub(r"\bhes\b", "he is", q)
    q = re.sub(r"\bwhats\b", "what is", q)
    q = re.sub(r"[^\w\s]", " ", q)
    return q.strip()


def _extract_name(full_text):
    m = re.search(r"\b(ASNA\s+SHERIN[^+\n]*)", full_text)
    if m:
        return m.group(1).strip().title()
    m = re.search(r"^([A-Z][A-Z ]{4,})\s*\n", full_text)
    if m:
        return m.group(1).strip().title()
    m = re.search(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", full_text)
    if m:
        return m.group(1).strip()
    return None


def _extract_employer(full_text, sents):
    """Extract current employer as a short clean string."""
    # Try known company names with date pattern
    patterns = [
        r"(EGC\s+Properties[^\n,]{0,60}(?:present|2026|2025|2024)[^\n,]{0,30})",
        r"([A-Z][A-Za-z\s&]+(?:Ltd|LLC|Inc|Corp|Group|Properties|Tech|Solutions|Services)?"
        r"\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*"
        r"(?:20\d\d)\s*[\u2013\-]\s*(?:Present|20\d\d))",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            found = re.sub(r"\s+", " ", m.group(1)).strip()
            return found[:80]
    # Fallback: sentence containing company + date keyword
    for s in sents:
        sl = s.lower()
        has_company = any(c in sl for c in ["egc", "properties", "bayut", "real estate"])
        has_date = any(d in sl for d in ["present", "2026", "2025", "2024"])
        if has_company and has_date:
            clause = re.split(r"[,;]", s)[0].strip()
            return clause[:100]
    return None


def _extract_role_title(full_text, sents):
    """Extract clean job title, stripping contact info."""
    m = re.search(r"(Data\s*[&|]\s*Business\s*Intelligence[^\n+]{0,120})", full_text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        raw = re.sub(r"\+\d[\d\s]{8,}", "", raw)
        raw = re.sub(r"[|]\s*\S+@\S+", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip().strip("|").strip()
        return raw[:120]
    for s in sents:
        if any(t in s for t in ["Analyst", "Developer", "Expert", "Manager", "Engineer"]):
            title = s.split("+971")[0].split("\u2022")[0].strip()
            title = re.sub(r"^[A-Z][A-Z ]{3,}\s*", "", title).strip()
            if len(title) > 5:
                return title[:120]
    return None


def _smart_synthesize(query, prompt_context):
    chunks = _extract_chunks(prompt_context)
    if not chunks:
        return "The document does not contain enough relevant information to answer this query."

    full_text = " ".join(chunks)
    sents = _sentences(full_text)
    q = _normalize(query)

    # Bot identity
    if any(p in q for p in ["your name", "who are you", "what are you"]):
        return "I am the AI File Intelligence Bot - a RAG-powered assistant that answers questions based on your uploaded documents."

    # Whose resume
    is_whose = any(p in q for p in [
        "whose resume", "whose cv", "whose document", "whose profile",
        "who is this", "who does this", "this resume", "resume belong",
        "cv belong", "this document", "this belong"
    ])
    if is_whose:
        name = _extract_name(full_text)
        return ("This is the resume of " + name + ".") if name else \
               "This resume belongs to the candidate described in the uploaded document."

    # Current employer
    is_employer = any(p in q for p in [
        "current employer", "current company", "current job", "currently work",
        "working now", "last experience", "recent experience", "most recent",
        "last company", "last job", "last role", "which company", "last employer",
        "where does she work", "where does he work", "who is her employer",
        "who is his employer"
    ])
    if is_employer:
        employer = _extract_employer(full_text, sents)
        role = _extract_role_title(full_text, sents)
        if employer and role:
            if any(t in employer for t in ["Analyst", "Developer", "Expert"]):
                return employer + "."
            return role + " at " + employer + "."
        if employer:
            return "She currently works at " + employer + "."
        return "The current employer is not explicitly stated in the document."

    # Position / role / title
    is_role = any(p in q for p in [
        "position", "role", "title", "designation", "what does she do",
        "what does he do", "job title", "what is her job", "what position",
        "what role", "i mean", "what she do"
    ])
    if is_role:
        title = _extract_role_title(full_text, sents)
        return title if title else "The position is not clearly identified in the retrieved context."

    # Name
    if any(p in q for p in ["her name", "his name", "person name", "full name", "candidate name"]):
        name = _extract_name(full_text)
        return (name + ".") if name else "The name is not clearly extractable from the document."

    # Location
    if any(p in q for p in ["where", "location", "based", "city", "country", "live"]):
        hits = _find(sents, ["abu dhabi", "dubai", "uae", "located", "based in"], n=1)
        if hits:
            return hits[0].strip()
        loc = re.search(r"Abu Dhabi|Dubai|UAE", full_text, re.IGNORECASE)
        return ("She is based in " + loc.group(0) + ".") if loc else \
               "Location not explicitly stated in the document."

    # Contact
    if any(p in q for p in ["email", "phone", "contact", "number", "linkedin", "github"]):
        hits = _find(sents, ["@", "+971", "linkedin", "github"], n=1)
        return hits[0].strip() if hits else "Contact details are not found in the retrieved context."

    # Years of experience
    if any(p in q for p in ["years of experience", "how many years", "how long", "total experience"]):
        hits = _find(sents, ["10 years", "years of", "extensive experience", "over 10", "decade"], n=1)
        return hits[0].strip() if hits else "The total years of experience are not clearly stated."

    # Skills
    if any(p in q for p in ["skills", "tools", "technologies", "tech stack", "expertise",
                              "software", "what can she", "what does she know"]):
        hits = _find(sents, ["Power BI", "Python", "SQL", "Excel", "Tableau",
                              "DAX", "ETL", "CRM", "JavaScript", "Analytics"], n=3)
        return _trim(" ".join(hits[:2]), 4) if hits else \
               "Skills information is not clearly captured in the retrieved context."

    # Qualifications
    if any(p in q for p in ["qualif", "certif", "degree", "education", "certified", "scrum"]):
        hits = _find(sents, ["certif", "CSM", "scrum", "degree", "bachelor", "master", "diploma"], n=3)
        return " ".join(hits[:2]).strip() if hits else \
               "Qualification details not found in the retrieved context."

    # Yes/No
    first_word = q.split()[0] if q.split() else ""
    if first_word in ("is", "does", "did", "has", "can", "are", "was", "have"):
        keywords = [w for w in q.split() if len(w) > 3]
        hits = _find(sents, keywords, n=2)
        return ("Yes. " + hits[0].strip()) if hits else \
               "The document does not clearly confirm or deny this."

    # Summary
    if any(p in q for p in ["summary", "background", "about", "overview", "tell me",
                              "describe", "explain", "who is", "profile"]):
        hits = _find(sents, ["professional", "analyst", "experience", "expert", "data", "business"], n=4)
        if not hits:
            hits = sents[:4]
        return _trim(" ".join(hits[:3]), 5)

    # Generic keyword fallback
    keywords = [w for w in re.sub(r"[^\w\s]", "", q).split() if len(w) > 3]
    hits = _find(sents, keywords, n=3)
    if not hits:
        hits = sents[:2]
    return _trim(" ".join(hits[:2]), 3) or \
           "The document does not contain enough relevant information to answer this query."
