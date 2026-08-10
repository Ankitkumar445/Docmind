"""
ingest.py
---------
Turns a raw document (txt/pdf) into overlapping chunks ready for embedding.

Interview talking point:
"I use a sliding-window chunker with overlap so that a fact split across a
chunk boundary is still fully present in at least one chunk. Chunk size and
overlap are the two knobs that most affect RAG answer quality -- too small
and you lose context, too large and irrelevant text dilutes the embedding."
"""

from pypdf import PdfReader


def load_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
