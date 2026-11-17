"""Tests for rag/models.py using a fake `ollama` module (no server needed)."""
import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def models(monkeypatch):
    calls = []
    fake = types.ModuleType("ollama")

    def embed(model, input):
        calls.append(("embed", model, input))
        return {"embeddings": [[1.0, 2.0, 3.0] for _ in input]}

    def chat(model, messages, options=None):
        calls.append(("chat", model, messages, options))
        return {"message": {"role": "assistant", "content": "olá [1]"}}

    fake.embed, fake.chat = embed, chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    monkeypatch.delitem(sys.modules, "rag.models", raising=False)
    mod = importlib.import_module("rag.models")
    mod.calls = calls
    yield mod
    sys.modules.pop("rag.models", None)


def test_embedder_returns_float32_matrix(models):
    out = models.OllamaEmbedder().embed(["a", "b"])
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    assert models.calls[0] == ("embed", models.EMBED_MODEL, ["a", "b"])


def test_chat_returns_message_content(models):
    msgs = [{"role": "user", "content": "oi"}]
    assert models.ollama_chat(msgs) == "olá [1]"
    kind, model, sent, options = models.calls[0]
    assert (kind, model, sent) == ("chat", models.CHAT_MODEL, msgs)
    assert options["temperature"] <= 0.2
