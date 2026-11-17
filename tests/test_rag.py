import hashlib
import re

import numpy as np
import pytest

from rag.chunking import chunk_text
from rag.pipeline import answer, build_index
from rag.store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, so tests run offline."""

    DIM = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                col = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
                out[row, col] += 1.0
        return out


def test_chunk_size_and_overlap():
    text = "palavra " * 300
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_store_search_orders_by_similarity():
    store = VectorStore()
    from rag.store import Chunk

    store.add(
        [Chunk("a", "x.md", 0), Chunk("b", "y.md", 0)],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    hits = store.search(np.array([0.9, 0.1]), k=2)
    assert hits[0][0].text == "a"


def test_store_save_and_load(tmp_path):
    from rag.store import Chunk

    store = VectorStore()
    store.add([Chunk("a", "x.md", 0)], np.array([[1.0, 2.0]]))
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.chunks[0].text == "a"
    assert loaded.matrix.shape == (1, 2)


def test_end_to_end_retrieves_right_document():
    embedder = FakeEmbedder()
    store = build_index("data/docs", embedder, chunk_size=300, overlap=50)
    captured = {}

    def fake_chat(messages):
        captured["messages"] = messages
        return "ok"

    _, hits = answer("Como cancelar um pedido?", store, embedder, fake_chat, k=2)
    assert hits[0][0].source == "pedidos.md"
    assert "Contexto:" in captured["messages"][1]["content"]
    assert "[1]" in captured["messages"][1]["content"]
