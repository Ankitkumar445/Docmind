# 🧠 DocMind

### Multi-Agent RAG Assistant with AI Guardrails

DocMind is a lightweight **Retrieval-Augmented Generation (RAG) assistant** that lets users upload PDF/TXT documents and ask questions about their content.

It combines **document processing, embeddings, vector similarity search, an LLM, LangChain, multi-agent coordination, structured outputs, and AI guardrails** to generate and validate document-grounded answers.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

- 📄 Upload PDF and TXT documents
- 🧩 Automatic document chunking
- 🧠 Text embeddings using FastEmbed
- 🔎 Semantic search using cosine similarity
- 🤖 Multi-agent RAG workflow
- 🧭 Query routing and rewriting
- 💬 Document-grounded answers using an LLM
- 🛡️ Dedicated AI guardrail for answer validation
- 📊 Confidence score and validation reasoning
- 📚 Retrieved source references
- 🧾 Structured JSON communication between agents
- 💭 Multi-turn conversation support
- ⚡ Pipeline latency tracking
- 🔌 FastAPI REST API
- 🌐 Browser-based chat interface

---

## 🤖 Multi-Agent Workflow

DocMind uses **three specialized agents** working together with a vector retrieval layer:

```
USER QUESTION
      │
      ▼
┌─────────────────┐
│  ROUTER AGENT    │  → decides if retrieval is needed, rewrites query
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ QUERY EMBEDDING  │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  VECTOR SEARCH   │  → cosine similarity, top-k relevant chunks
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  ANSWER AGENT    │  → question + retrieved context → grounded answer
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ GUARDRAIL AGENT  │  → checks support, confidence, reasoning
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ FINAL RESPONSE   │  → answer + sources + verdict + confidence + latency
└─────────────────┘
```

### 1. Router Agent

Receives the user's question and determines whether document retrieval is required. If so, it rewrites the question into a cleaner search query.

```json
{
  "needs_retrieval": true,
  "search_query": "company revenue previous year"
}
```

### 2. Vector Retrieval

The rewritten query is converted into an embedding and compared against document chunk embeddings using cosine similarity to retrieve the most relevant chunks.

```
User Query → Query Embedding → Cosine Similarity → Top-k Relevant Chunks
```

### 3. Answer Agent

Receives the user question and the retrieved document chunks, then generates an answer using the retrieved context as its primary source of information.

```
Question + Retrieved Context → Answer Agent → Grounded Answer
```

### 4. Guardrail Agent

Independently evaluates the generated answer against the retrieved context. It checks whether the answer is supported, whether unsupported claims exist, and produces a confidence score and reasoning.

```json
{
  "supported": true,
  "confidence": 0.91,
  "reason": "The answer is supported by the retrieved document context."
}
```

The final response contains both the generated answer and the guardrail information.

---

## 📚 Document Processing Pipeline

**Ingestion:**

```
PDF / TXT → Text Extraction → Document Chunking → Embedding Generation → Vector Store
```

**Query time:**

```
User Question → Router Agent → Query Rewriting → Query Embedding
→ Cosine Similarity Search → Top-k Relevant Chunks
→ Answer Agent → Guardrail Agent → Final Response
```

---

## 🔎 Vector Search

DocMind currently uses an **in-memory vector store** with cosine similarity. The query is converted into an embedding and compared with the embeddings of stored document chunks, and the most relevant chunks are passed to the Answer Agent.

```
              A · B
cosine(A,B) = ─────────────
              ||A|| ||B||
```

This keeps the retrieval mechanism simple and transparent. A production version could swap the in-memory store for **FAISS**, **pgvector**, **Pinecone**, or another vector database.

---

## 🛡️ AI Guardrails

DocMind does not blindly return the LLM-generated answer. Instead, it performs an additional validation step:

```
Retrieved Context + Generated Answer → Guardrail Agent → Supported / Unsupported + Confidence + Reason
```

This adds a layer for detecting unsupported responses and reducing hallucination risk.

> **Note:** The guardrail itself uses an LLM, so it is not a perfect guarantee against hallucinations. A production system would need additional evaluation, monitoring, and auditing.

---

## 🛠️ Tech Stack

| Category      | Technologies              |
|---------------|----------------------------|
| Language      | Python                     |
| Backend       | FastAPI, Uvicorn           |
| LLM           | Groq API                   |
| Embeddings    | FastEmbed                  |
| AI Framework  | LangChain                  |
| Retrieval     | Cosine Similarity          |
| Vector Store  | In-memory                  |
| Frontend      | HTML, CSS, JavaScript      |
| API           | REST                       |

---

## 📁 Project Structure

```
DocMind/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   │
│   └── rag/
│       ├── agents.py
│       ├── ingest.py
│       └── vectorstore.py
│
├── frontend/
│   └── index.html
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.x
- pip
- A Groq API key

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocMind
```

### 2. Configure your API key

Create an API key and set it as an environment variable.

**Windows PowerShell**
```powershell
$env:GROQ_API_KEY="your-api-key"
```

**macOS / Linux**
```bash
export GROQ_API_KEY="your-api-key"
```

> ⚠️ Never commit your API key to GitHub.

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Start the backend

```bash
uvicorn app:app --reload --port 8000
```

The backend will run at: `http://localhost:8000`

### 5. Start the frontend

Open `frontend/index.html` directly in your browser, **or** serve it locally:

```bash
cd frontend
python -m http.server 5500
```

Then open: `http://localhost:5500`

---

## 🧪 How to Use

1. Start the FastAPI backend.
2. Open the frontend.
3. Upload a PDF or TXT document.
4. Wait for document processing.
5. Ask a question about the uploaded document.
6. The Router Agent analyzes the question.
7. Relevant document chunks are retrieved.
8. The Answer Agent generates a grounded response.
9. The Guardrail Agent validates the response.
10. The UI displays the answer, sources, confidence, and latency.

---

## 🔌 API Endpoints

| Endpoint    | Method | Description                       |
|-------------|--------|------------------------------------|
| `/ingest`   | POST   | Upload and process a document      |
| `/chat`     | POST   | Ask questions using the RAG pipeline |
| `/history`  | GET    | Retrieve conversation history      |

For exact request/response formats, see [`backend/app.py`](backend/app.py).

---

## ⚡ Observability

DocMind tracks basic pipeline information, including:

- Agent execution latency
- Overall request latency
- Guardrail confidence
- Guardrail reasoning

This gives visibility into both system performance and AI response validation.

---

## 🎯 Design Decisions

**Why RAG?**
RAG lets the LLM use information retrieved from the uploaded documents instead of relying only on its pretrained knowledge, producing answers grounded in the user's own data.

**Why multiple agents?**
Instead of one large prompt handling everything, DocMind separates responsibilities:

```
Router → Retrieval → Answer → Guardrail
```

This makes each component easier to understand, test, and debug.

**Why a manual vector store?**
Keeping the embedding and similarity-search mechanism visible (rather than hidden behind a managed vector database) makes it easier to understand how semantic retrieval actually works.

**Why a separate guardrail agent?**
The Answer Agent is responsible for generating a response. The Guardrail Agent has a different job: validating whether that response is actually supported by the retrieved context. Separating these adds an independent validation layer.

---

## ⚠️ Current Limitations

- Vector data is stored in memory only
- Retrieval currently uses cosine similarity only
- No hybrid BM25 + semantic retrieval
- Guardrail decisions rely on an LLM
- No persistent vector database
- No authentication layer
- Not designed for large-scale production workloads

---

## 🔮 Future Improvements

- [ ] Replace in-memory vector store with FAISS or pgvector
- [ ] Add hybrid BM25 + semantic retrieval
- [ ] Add web-search tool
- [ ] Add OpenTelemetry tracing
- [ ] Add AI evaluation metrics
- [ ] Add token and cost tracking
- [ ] Add Docker support
- [ ] Add authentication
- [ ] Add persistent conversation storage
- [ ] Add automated RAG evaluation

---

## 📄 License

This project is licensed under the MIT License.
