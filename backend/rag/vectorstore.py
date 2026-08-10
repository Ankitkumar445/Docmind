"""
vectorstore.py
--------------
A minimal, dependency-light vector store.

Interview talking point:
"I built the retrieval layer from first principles instead of hiding it behind
a black-box vector DB, so I can explain exactly how similarity search works:
embed the chunks, embed the query, rank by cosine similarity, return top-k."

In production you'd swap this class for FAISS / Pinecone / pgvector, but the
*interface* (add, search) stays identical -- that's the point worth making
in an interview: the retrieval abstraction shouldn't care which store backs it.
"""

import numpy as np
from fastembed import TextEmbedding
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    embedding: np.ndarray = field(default=None, repr=False)


class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embedder = TextEmbedding(model_name=model_name)
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        # fastembed returns a generator of np arrays
        return np.array(list(self.embedder.embed(texts)))

    def add(self, texts: list[str], source: str):
        embeddings = self._embed(texts)
        start_id = len(self.chunks)
        for i, (t, e) in enumerate(zip(texts, embeddings)):
            self.chunks.append(Chunk(text=t, source=source, chunk_id=start_id + i, embedding=e))
        self._rebuild_matrix()

    def _rebuild_matrix(self):
        if not self.chunks:
            self._matrix = None
            return
        self._matrix = np.vstack([c.embedding for c in self.chunks])

    @staticmethod
    def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # normalize once -> dot product == cosine similarity
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return m @ q

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None or len(self.chunks) == 0:
            return []
        q_vec = self._embed([query])[0]
        sims = self._cosine_sim(q_vec, self._matrix)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def is_empty(self) -> bool:
        return len(self.chunks) == 0
