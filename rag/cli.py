import argparse

from .models import OllamaEmbedder, ollama_chat
from .pipeline import answer, build_index
from .store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Perguntas e respostas sobre documentos (RAG local)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Indexa os documentos")
    p_index.add_argument("--docs", default="data/docs")
    p_index.add_argument("--out", default="index")

    p_ask = sub.add_parser("ask", help="Faz uma pergunta")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", default="index")
    p_ask.add_argument("-k", type=int, default=3)

    args = parser.parse_args()
    embedder = OllamaEmbedder()

    if args.cmd == "index":
        store = build_index(args.docs, embedder)
        store.save(args.out)
        print(f"Indexados {len(store.chunks)} trechos em '{args.out}/'.")
    else:
        store = VectorStore.load(args.index)
        text, hits = answer(args.question, store, embedder, ollama_chat, k=args.k)
        print(text)
        print("\nFontes:")
        for i, (chunk, score) in enumerate(hits, start=1):
            print(f"  [{i}] {chunk.source} (similaridade {score:.2f})")


if __name__ == "__main__":
    main()
