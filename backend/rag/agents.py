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

LLM provider: OpenRouter (unified OpenAI-compatible API, free-tier models
available) -- get a key at https://openrouter.ai/keys

Multiple free models are configured (AI_MODEL_1..4) and tried in order as a
fallback chain, since free-tier models on OpenRouter can be rate-limited or
temporarily unavailable. If the first model fails, the next one is tried
automatically before giving up.
"""

import json
import os
import time
from dataclasses import dataclass

import requests

# OpenRouter -- get a free key at https://openrouter.ai/keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Fallback chain of free models -- tried in order until one succeeds.
MODELS = [
    os.environ.get("AI_MODEL_1", "nvidia/nemotron-3-ultra-550b-a55b:free").strip(),
    os.environ.get("AI_MODEL_2", "openai/gpt-oss-120b:free").strip(),
    os.environ.get("AI_MODEL_3", "qwen/qwen3-coder:free").strip(),
    os.environ.get("AI_MODEL_4", "openai/gpt-oss-20b:free").strip(),
]
MODELS = [m for m in MODELS if m]  # drop any empty entries


@dataclass
class AgentTrace:
    step: str
    output: dict
    latency_ms: float


def _call_llm(system: str, user: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a free key at "
            "https://openrouter.ai/keys and export it before starting the server."
        )
    if not MODELS:
        raise RuntimeError("No models configured. Set AI_MODEL_1..4 in your environment.")

    last_error = None
    for model in MODELS:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    # Optional but recommended by OpenRouter for routing/analytics.
                    "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:8000"),
                    "X-Title": "DocMind",
                },
                json={
                    "model": model,
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
                last_error = f"{model} -> {resp.status_code}: {resp.text}"
                continue  # try the next model in the fallback chain

            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            last_error = f"{model} -> {e}"
            continue

    raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")


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