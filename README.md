# DocMind — Multi-Agent RAG Assistant with Guardrails

A small, fully-working project built to map directly onto the Cognizant Ace
Team (Ace Frontier Engineer) skills list: RAG pipelines, vector stores,
LangChain, multi-agent coordination, prompt engineering with structured
outputs, guardrails/AI quality, and a REST API.

## 1. What it does

1. You upload a document (txt/pdf).
2. It gets chunked and embedded (FastEmbed) and stored in a vector store you
   can inspect and search directly.
3. You ask a question in the chat UI.
4. Three agents run in sequence:
   - **Router agent** — decides if the question needs a document lookup, and
     rewrites it into a cleaner search query.
   - **Answer agent** — generates an answer using ONLY the retrieved chunks
     (grounded generation, not free recall).
   - **Guardrail agent** — independently checks whether the answer is
     actually supported by the retrieved context, and returns a confidence
     score and reason.
5. The UI shows the answer, the sources used, the guardrail verdict, and the
   latency of the whole pipeline.

## 2. Why it's built this way (architecture reasoning)

- **Manual cosine-similarity vector store instead of FAISS/Pinecone** — you
  already understand embeddings + cosine similarity from your DSA/RAG
  practice. This project makes that mechanism visible instead of hiding it
  behind a library, which is exactly what you want to be able to explain
  line-by-line in an interview.
- **Three single-purpose agents instead of one big prompt** — easier to
  test, debug, and reason about. Each agent takes structured input and
  returns structured JSON, not paragraphs — this is what "prompt
  engineering & structured outputs" means in the JD.
- **A dedicated guardrail step** — this is the part most student projects
  skip, and it's called out explicitly in the JD ("validate AI-generated
  outputs, implement guardrails, ensure reliable AI operations"). It also
  gives you a natural interview story about hallucination and reliability.
- **LangChain used specifically for conversation memory** — rather than
  bolting LangChain onto everything, it's used where it earns its place:
  managing multi-turn chat history.

## 3. Setup (free — Gemini's free tier, no credit card required)

**Step 1 — Get a free Gemini API key:**
Go to https://aistudio.google.com/apikey, sign in with a Google account, and
click "Create API key." No billing/credit card needed for the free tier
(`gemini-2.0-flash`, which this project uses, has a generous free daily
request limit — plenty for testing and demoing this project).

**Step 2 — Set the key as an environment variable:**

Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your-key-here"
```

Mac/Linux:
```bash
export GEMINI_API_KEY=your-key-here
```

**Step 3 — Install Python deps and start the backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open `frontend/index.html` directly in your browser (double-click it,
or serve it with `python -m http.server` from the frontend folder).

> Note: the `$env:GEMINI_API_KEY` in PowerShell only lasts for that terminal
> session. If you close and reopen the terminal, you'll need to set it
> again before running `uvicorn`.

Try it with a small PDF or txt file first — your own resume works well as a
test document.

## 4. How to extend it (good "next steps" to mention in an interview)

- Swap the manual vector store for FAISS (`langchain_community.vectorstores.FAISS`)
  — same interface, so it's a one-file change. Good way to show you
  understand the abstraction, not just one implementation.
- Add a second retrieval tool (e.g. web search) and let the router agent
  choose between "search my documents" vs "search the web" — this turns it
  into a true multi-tool agent.
- Add OpenTelemetry tracing around each agent call for real observability
  (the JD explicitly lists OpenTelemetry/AgentOps).
- Containerize with a Dockerfile + docker-compose for backend/frontend
  (covers "Containers & cloud-native services").

## 5. Resume bullets you can use

- "Built a multi-agent RAG assistant (Python, FastAPI, LangChain) with a
  dedicated guardrail agent that validates every AI-generated answer
  against retrieved source documents before returning it to the user."
- "Implemented a from-scratch vector similarity search (embeddings +
  cosine similarity) and a 3-agent pipeline (router → retriever → answer →
  guardrail) with structured JSON communication between agents."
- "Designed the system for observability, logging per-agent latency and
  guardrail confidence scores on every request."

## 6. Interview Q&A prep (things you'll likely be asked)

**Q: Walk me through what happens when a user asks a question.**
A: The router agent first decides if retrieval is even needed and rewrites
the query for search. If retrieval is needed, I embed the query, compute
cosine similarity against all stored chunk embeddings, and take the top-4.
Those chunks go into the answer agent's prompt as the *only* source of
truth — the system prompt explicitly forbids answering outside that
context. Then a separate guardrail agent, which never sees the retrieval
step, independently checks whether the answer is actually backed by the
context and returns a confidence score.

**Q: Why did you separate the answer agent and the guardrail agent instead
of asking one model to "answer carefully"?**
A: A model grading its own answer in the same turn is prone to confirming
its own mistake. A separate call, with a separate narrow prompt whose only
job is fact-checking, is a cheap and effective way to catch hallucinations
— similar in spirit to LLM-as-judge evaluation patterns used in production
RAG systems.

**Q: What are the failure modes of this design?**
A: Chunking can split a fact across two chunks so neither ranks highly
enough to be retrieved (I mitigate this with overlap). The guardrail agent
is itself an LLM call, so it can also be wrong — in production I'd want to
sample-audit guardrail decisions, not fully trust them. Cosine similarity
over dense embeddings can also miss purely keyword/exact-match queries
(e.g., an exact product ID) — a production system would combine this with
BM25/keyword search (hybrid retrieval).

**Q: How would you scale this?**
A: Swap the in-memory numpy store for a real vector DB (FAISS index or
managed Pinecone/pgvector) so it doesn't need to hold everything in RAM,
add caching for repeated queries, and move the three sequential agent calls
to run the guardrail check in parallel with returning a "pending" response
that gets validated after the fact for latency-sensitive use cases.

**Q: Why did you use Gemini's free tier instead of a paid API like GPT-4 or
Claude?**
A: To build and iterate without incurring cost during development —
Gemini's free tier is generous enough for a personal project like this. The
important design decision is that the LLM call sits behind one function
(`_call_llm`), so swapping to a different provider (Claude, GPT-4, a local
Ollama model) is a one-function change; nothing else in the pipeline —
routing logic, retrieval, guardrail checks — depends on which model is
behind it.

## 7. Mapping to the Ace Frontier Engineer JD

| JD skill                              | Where it shows up in this project              |
|----------------------------------------|-------------------------------------------------|
| RAG Pipelines                          | `rag/ingest.py`, `rag/vectorstore.py`            |
| Vector DB concepts                     | Cosine-similarity store, with a documented FAISS swap path |
| LangChain                              | `ConversationBufferMemory` in `app.py`           |
| Multi-Agent Coordination                | `rag/agents.py` — router/answer/guardrail agents |
| Prompt Engineering & Structured Outputs | JSON-only agent responses                        |
| Guardrails / Responsible AI             | Dedicated guardrail agent                        |
| REST APIs                              | FastAPI (`/ingest`, `/chat`, `/history`)         |
| AI Quality Metrics / Observability      | Per-agent latency + confidence logging (`trace`) |
