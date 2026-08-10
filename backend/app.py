"""
app.py
------
FastAPI server exposing two endpoints:
  POST /ingest  -> upload a .txt or .pdf, it gets chunked + embedded + stored
  POST /chat    -> ask a question, runs the full multi-agent pipeline

Also wires in LangChain's ConversationBufferMemory to demonstrate the
"LangChain Memory" skill from the JD -- it keeps the last few turns of
conversation so follow-up questions like "what about the second one?" work.

Run:
  export ANTHROPIC_API_KEY=sk-ant-...
  uvicorn app:app --reload --port 8000
"""

import os
import shutil
import tempfile
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory

from rag.vectorstore import VectorStore
from rag.ingest import load_text, chunk_text
from rag.agents import router_agent, answer_agent, guardrail_agent

app = FastAPI(title="DocMind API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

store = VectorStore()
memory = ConversationBufferMemory(return_messages=True)


class ChatRequest(BaseModel):
    query: str


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    text = load_text(tmp_path)
    chunks = chunk_text(text)
    store.add(chunks, source=file.filename)
    os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_added": len(chunks)}


@app.post("/chat")
async def chat(req: ChatRequest):
    t0 = time.time()
    trace_log = []

    # 1. Route: does this need retrieval?
    route, router_trace = router_agent(req.query)
    trace_log.append(router_trace.__dict__)

    # 2. Retrieve
    # Retrieval is cheap (no LLM call) and the router agent can misjudge
    # short/vague queries as "no retrieval needed" even when documents are
    # loaded. So: always search if the store has documents, and only use
    # the router's job for rewriting the query, not gating retrieval.
    retrieved_chunks, sources = [], []
    if not store.is_empty():
        search_query = route.get("search_query", req.query)
        results = store.search(search_query, top_k=4)
        retrieved_chunks = [c.text for c, score in results]
        sources = [{"source": c.source, "chunk_id": c.chunk_id, "score": round(score, 3)}
                   for c, score in results]

    # 3. Answer (grounded generation)
    answer, answer_trace = answer_agent(req.query, retrieved_chunks)
    trace_log.append(answer_trace.__dict__)

    # 4. Guardrail check
    verdict, guardrail_trace = guardrail_agent(req.query, retrieved_chunks, answer)
    trace_log.append(guardrail_trace.__dict__)

    # 5. Update LangChain conversation memory
    memory.chat_memory.add_user_message(req.query)
    memory.chat_memory.add_ai_message(answer)

    return {
        "answer": answer,
        "sources": sources,
        "guardrail": verdict,
        "total_latency_ms": round((time.time() - t0) * 1000, 1),
        "trace": trace_log,
    }


@app.get("/history")
async def history():
    return {"messages": [m.content for m in memory.chat_memory.messages]}


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": len(store.chunks)}
