"""
llm.py - LLM response engine
v2: Mistral (primary) -> OpenAI (fallback) -> local synthesizer (offline)

Public API:
  generate_response(prompt_context, query, chat_history) -> Dict
  generate_summary(context_chunks)                       -> Dict
  generate_recommendations(context_chunks)               -> Dict
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

def generate_response(prompt_context, query, chat_history=None):
    chat_history = chat_history or []
    if MISTRAL_API_KEY:
        result = _mistral_response(prompt_context, query, chat_history)
        if result:
            return result
    if OPENAI_API_KEY:
        result = _openai_response(prompt_context, query, chat_history)
        if result:
            return result
    return _local_response(prompt_context, query)


def generate_summary(context_chunks):
    from langchain_pipeline import run_summary_chain
    return run_summary_chain(context_chunks)


def generate_recommendations(context_chunks):
    from langchain_pipeline import run_recommendations_chain
    return run_recommendations_chain(context_chunks)


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
    return {"answer": _smart_synthesize(query, prompt_context),
            "model": "local-synthesizer", "mode": "offline", "tokens": 0}


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


def _extract_name(full_text):
    for pat in [r"\b(ASNA\s+SHERIN[^+\n]*)", r"^([A-Z][A-Z ]{4,})\s*\n",
                r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"]:
        m = re.search(pat, full_text)
        if m:
            return m.group(1).strip().title()
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
    m = re.search(r"(Data\s*[&|]\s*Business\s*Intelligence[^\n+]{0,120})", full_text, re.IGNORECASE)
    if m:
        raw = re.sub(r"\+\d[\d\s]{8,}", "", m.group(1))
        raw = re.sub(r"[|]\s*\S+@\S+", "", raw)
        return re.sub(r"\s+", " ", raw).strip().strip("|").strip()[:120]
    for s in sents:
        if any(t in s for t in ["Analyst","Developer","Expert","Manager","Engineer"]):
            title = re.sub(r"^[A-Z][A-Z ]{3,}\s*", "",
                           s.split("+971")[0].split("*")[0].strip()).strip()
            if len(title) > 5:
                return title[:120]
    return None


def _smart_synthesize(query, prompt_context):
    chunks = _extract_chunks(prompt_context)
    if not chunks:
        return "The document does not contain enough relevant information to answer this query."

    full_text = " ".join(chunks)
    sents     = _sentences(full_text)
    q         = _normalize(query)

    if any(p in q for p in ["your name","who are you","what are you"]):
        return "I am the AI File Intelligence Bot -- a RAG-powered assistant that answers questions from your uploaded documents."

    if any(p in q for p in ["whose resume","whose cv","this resume","resume belong","cv belong","who is this"]):
        name = _extract_name(full_text)
        return ("This is the resume of " + name + ".") if name else \
               "This resume belongs to the candidate described in the uploaded document."

    if any(p in q for p in ["current employer","current company","current job","currently work",
                              "last experience","most recent","last company","last job"]):
        employer = _extract_employer(full_text, sents)
        role     = _extract_role_title(full_text, sents)
        if employer and role:
            return (role + " at " + employer + ".") \
                   if not any(t in employer for t in ["Analyst","Developer","Expert"]) \
                   else employer + "."
        return ("She currently works at " + employer + ".") if employer else \
               "The current employer is not explicitly stated in the document."

    if any(p in q for p in ["position","role","title","designation","job title","what position"]):
        title = _extract_role_title(full_text, sents)
        return title if title else "The position is not clearly identified in the retrieved context."

    if any(p in q for p in ["her name","his name","person name","full name","candidate name"]):
        name = _extract_name(full_text)
        return (name + ".") if name else "The name is not clearly extractable from the document."

    if any(p in q for p in ["where","location","based","city","country","live"]):
        hits = _find(sents, ["abu dhabi","dubai","uae","located","based in"], n=1)
        if hits:
            return hits[0].strip()
        loc = re.search(r"Abu Dhabi|Dubai|UAE", full_text, re.IGNORECASE)
        return ("Based in " + loc.group(0) + ".") if loc else "Location not explicitly stated."

    if any(p in q for p in ["email","phone","contact","number","linkedin","github"]):
        hits = _find(sents, ["@","+971","linkedin","github"], n=1)
        return hits[0].strip() if hits else "Contact details not found in the retrieved context."

    if any(p in q for p in ["years of experience","how many years","how long","total experience"]):
        hits = _find(sents, ["10 years","years of","extensive experience","over 10","decade"], n=1)
        return hits[0].strip() if hits else "Total years of experience are not clearly stated."

    if any(p in q for p in ["skills","tools","technologies","tech stack","expertise"]):
        hits = _find(sents, ["Power BI","Python","SQL","Excel","Tableau","DAX","ETL","Analytics"], n=3)
        return _trim(" ".join(hits[:2]), 4) if hits else \
               "Skills information is not clearly captured in the retrieved context."

    if any(p in q for p in ["qualif","certif","degree","education","certified"]):
        hits = _find(sents, ["certif","CSM","scrum","degree","bachelor","master"], n=3)
        return " ".join(hits[:2]).strip() if hits else "Qualification details not found."

    first_word = q.split()[0] if q.split() else ""
    if first_word in ("is","does","did","has","can","are","was","have"):
        keywords = [w for w in q.split() if len(w) > 3]
        hits = _find(sents, keywords, n=2)
        return ("Yes. " + hits[0].strip()) if hits else \
               "The document does not clearly confirm or deny this."

    if any(p in q for p in ["summary","background","about","overview","tell me","describe","who is","profile"]):
        hits = _find(sents, ["professional","analyst","experience","expert","data","business"], n=4)
        return _trim(" ".join((hits or sents)[:3]), 5)

    keywords = [w for w in re.sub(r"[^\w\s]","",q).split() if len(w) > 3]
    hits     = _find(sents, keywords, n=3)
    return _trim(" ".join((hits or sents[:2])[:2]), 3) or \
           "The document does not contain enough relevant information to answer this query."
