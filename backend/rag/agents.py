# """
# agents.py
# ---------
# This is the "multi-agent system" piece of the JD. Three small agents,
# each with ONE job, talking to each other through structured (JSON) messages
# instead of free-form text. This is the pattern real agentic frameworks
# (LangGraph, CrewAI, AutoGen) formalize -- here it's built explicitly so you
# can explain every hop in an interview.

# Pipeline:
#   user query
#      -> ROUTER agent   : decide "needs_retrieval" (True/False) + rewritten query
#      -> RETRIEVER       : vector search (not an LLM call, just the vector store)
#      -> ANSWER agent    : generate answer grounded ONLY in retrieved chunks
#      -> GUARDRAIL agent : check the answer for hallucination / unsupported claims
#      -> return final answer (+ metadata: sources, confidence, guardrail verdict)

# Interview talking point:
# "Each agent has a narrow, single-responsibility prompt. This is easier to
# test, debug, and swap models for than one giant prompt trying to do
# routing + retrieval + generation + safety checking all at once. It also
# lets me log where in the pipeline something went wrong."
# """

# import json
# import os
# import time
# from dataclasses import dataclass

# import requests

# # Free Gemini API -- get a key at https://aistudio.google.com/apikey
# # (no credit card required on the free tier)
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# MODEL = "gemini-2.5-flash"
# GEMINI_URL = (
#     f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
#     f"?key={GEMINI_API_KEY}"
# )


# @dataclass
# class AgentTrace:
#     step: str
#     output: dict
#     latency_ms: float


# def _call_llm(system: str, user: str) -> str:
#     if not GEMINI_API_KEY:
#         raise RuntimeError(
#             "GEMINI_API_KEY is not set. Get a free key at "
#             "https://aistudio.google.com/apikey and export it before starting the server."
#         )
#     resp = requests.post(
#         GEMINI_URL,
#         json={
#             "system_instruction": {"parts": [{"text": system}]},
#             "contents": [{"role": "user", "parts": [{"text": user}]}],
#             "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
#         },
#         timeout=60,
#     )
#     if not resp.ok:
#         raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
#     data = resp.json()
#     return data["candidates"][0]["content"]["parts"][0]["text"]


# def router_agent(query: str) -> tuple[dict, AgentTrace]:
#     """Decides whether the query needs document retrieval, and cleans it up."""
#     t0 = time.time()
#     system = (
#         "You are a routing agent. Given a user query, decide if answering it "
#         "requires looking up information in a private document store. "
#         'Respond ONLY with JSON: {"needs_retrieval": true|false, "search_query": "<cleaned up query for search>"}'
#     )
#     raw = _call_llm(system, query)
#     try:
#         parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
#     except Exception:
#         parsed = {"needs_retrieval": True, "search_query": query}
#     trace = AgentTrace("router", parsed, (time.time() - t0) * 1000)
#     return parsed, trace


# def answer_agent(query: str, context_chunks: list[str]) -> tuple[str, AgentTrace]:
#     """Generates an answer grounded strictly in the retrieved context."""
#     t0 = time.time()
#     context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context retrieved."
#     system = (
#         "You are a helpful assistant. Answer the user's question using ONLY the "
#         "context provided below. If the context does not contain the answer, say "
#         "you don't have enough information -- do not make things up.\n\n"
#         f"CONTEXT:\n{context}"
#     )
#     answer = _call_llm(system, query)
#     trace = AgentTrace("answer", {"answer": answer}, (time.time() - t0) * 1000)
#     return answer, trace


# def guardrail_agent(query: str, context_chunks: list[str], answer: str) -> tuple[dict, AgentTrace]:
#     """Validates the answer: is every claim actually supported by the context?"""
#     t0 = time.time()
#     context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context."
#     system = (
#         "You are a strict fact-checking agent. Given a CONTEXT and an ANSWER, "
#         "check whether the answer is fully supported by the context. "
#         'Respond ONLY with JSON: {"supported": true|false, "confidence": 0-1, "reason": "<one sentence>"}'
#     )
#     user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
#     raw = _call_llm(system, user)
#     try:
#         parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
#     except Exception:
#         parsed = {"supported": True, "confidence": 0.5, "reason": "guardrail parse failed"}
#     trace = AgentTrace("guardrail", parsed, (time.time() - t0) * 1000)
#     return parsed, trace


"""
agents.py
---------
This is the "multi-agent system" piece of the JD. Three small agents,
each with ONE job, talking to each other through structured (JSON) messages
instead of free-form text. This is the pattern real agentic frameworks
(LangGraph, CrewAI, AutoGen) formalize -- here it's built explicitly so you
can explain every hop in an interview.

Pipeline:
  user query
     -> ROUTER agent   : decide "needs_retrieval" (True/False) + rewritten query
     -> RETRIEVER       : vector search (not an LLM call, just the vector store)
     -> ANSWER agent    : generate answer grounded ONLY in retrieved chunks
     -> GUARDRAIL agent : check the answer for hallucination / unsupported claims
     -> return final answer (+ metadata: sources, confidence, guardrail verdict)

Interview talking point:
"Each agent has a narrow, single-responsibility prompt. This is easier to
test, debug, and swap models for than one giant prompt trying to do
routing + retrieval + generation + safety checking all at once. It also
lets me log where in the pipeline something went wrong."

LLM provider: Groq (free tier, OpenAI-compatible API) -- get a key at
https://console.groq.com/keys (no credit card required).
"""

import json
import os
import time
from dataclasses import dataclass

import requests

# Free Groq API -- get a key at https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass
class AgentTrace:
    step: str
    output: dict
    latency_ms: float


def _call_llm(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and export it before starting the server."
        )
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        },
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def router_agent(query: str) -> tuple[dict, AgentTrace]:
    """Decides whether the query needs document retrieval, and cleans it up."""
    t0 = time.time()
    system = (
        "You are a routing agent. Given a user query, decide if answering it "
        "requires looking up information in a private document store. "
        'Respond ONLY with JSON: {"needs_retrieval": true|false, "search_query": "<cleaned up query for search>"}'
    )
    raw = _call_llm(system, query)
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        parsed = {"needs_retrieval": True, "search_query": query}
    trace = AgentTrace("router", parsed, (time.time() - t0) * 1000)
    return parsed, trace


def answer_agent(query: str, context_chunks: list[str]) -> tuple[str, AgentTrace]:
    """Generates an answer grounded strictly in the retrieved context."""
    t0 = time.time()
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context retrieved."
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the context does not contain the answer, say "
        "you don't have enough information -- do not make things up.\n\n"
        f"CONTEXT:\n{context}"
    )
    answer = _call_llm(system, query)
    trace = AgentTrace("answer", {"answer": answer}, (time.time() - t0) * 1000)
    return answer, trace


def guardrail_agent(query: str, context_chunks: list[str], answer: str) -> tuple[dict, AgentTrace]:
    """Validates the answer: is every claim actually supported by the context?"""
    t0 = time.time()
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context."
    system = (
        "You are a strict fact-checking agent. Given a CONTEXT and an ANSWER, "
        "check whether the answer is fully supported by the context. "
        'Respond ONLY with JSON: {"supported": true|false, "confidence": 0-1, "reason": "<one sentence>"}'
    )
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    raw = _call_llm(system, user)
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        parsed = {"supported": True, "confidence": 0.5, "reason": "guardrail parse failed"}
    trace = AgentTrace("guardrail", parsed, (time.time() - t0) * 1000)
    return parsed, trace