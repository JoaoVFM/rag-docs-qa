from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .chunking import chunk_text
from .store import Chunk, VectorStore

SYSTEM_PROMPT = (
    "Você é um assistente de atendimento. Responda em português usando SOMENTE o contexto "
    "fornecido. Cite as fontes no formato [1], [2]. Se a resposta não estiver no contexto, "
    "diga que não encontrou a informação."
)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def load_documents(docs_dir: str | Path) -> dict[str, str]:
    paths = sorted(Path(docs_dir).glob("*.md")) + sorted(Path(docs_dir).glob("*.txt"))
    return {p.name: p.read_text(encoding="utf-8") for p in paths}


def build_index(docs_dir: str | Path, embedder: Embedder, chunk_size: int = 500, overlap: int = 100) -> VectorStore:
    chunks: list[Chunk] = []
    for name, text in load_documents(docs_dir).items():
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=name, idx=i))
    if not chunks:
        raise ValueError(f"No .md or .txt documents found in {docs_dir}")
    store = VectorStore()
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store


def build_messages(question: str, hits: list[tuple[Chunk, float]]) -> list[dict]:
    context = "\n\n".join(f"[{i}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits, start=1))
    user = f"Contexto:\n{context}\n\nPergunta: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def answer(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    chat_fn: Callable[[list[dict]], str],
    k: int = 3,
) -> tuple[str, list[tuple[Chunk, float]]]:
    qvec = embedder.embed([question])[0]
    hits = store.search(qvec, k=k)
    if not hits:
        return "Índice vazio. Rode o comando 'index' primeiro.", []
    return chat_fn(build_messages(question, hits)), hits
