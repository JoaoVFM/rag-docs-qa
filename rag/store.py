import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    text: str
    source: str
    idx: int


class VectorStore:
    """In-memory vector store using cosine similarity (numpy)."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = self._normalize(np.asarray(vectors, dtype=np.float32))
        self.chunks.extend(chunks)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        q = self._normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        store = cls()
        store.matrix = np.load(directory / "vectors.npy")
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk(**c) for c in raw]
        return store
