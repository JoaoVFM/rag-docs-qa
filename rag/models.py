"""Thin wrappers around a local Ollama server (free, no API key)."""
import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL) -> None:
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        response = ollama.embed(model=self.model, input=texts)
        return np.array(response["embeddings"], dtype=np.float32)


def ollama_chat(messages: list[dict], model: str = CHAT_MODEL) -> str:
    response = ollama.chat(model=model, messages=messages, options={"temperature": 0.1})
    return response["message"]["content"]
