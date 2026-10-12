"""Build a FAISS vector store (with manifest.json) from the PDF/text files in data/.

Examples (run from the repository root):
    python scripts/build_index.py --provider ollama            # needs a running Ollama + nomic-embed-text
    python scripts/build_index.py --provider openai            # needs OPENAI_API_KEY; makes paid API calls
"""
import _bootstrap  # noqa: F401
import argparse
import sys

from rag_chatbot import config
from rag_chatbot.documents import load_documents, split_documents
from rag_chatbot.indexing import build_index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["ollama", "openai"], required=True)
    ap.add_argument("--source", nargs="+", default=[str(config.DEFAULT_PDF)], help="PDF/TXT files to index")
    ap.add_argument("--out", help="output directory (default: vectorstores/supplements_<provider>)")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    args = ap.parse_args(argv)

    config.load_env()
    out = args.out or str(config.VECTORSTORES_DIR / f"supplements_{args.provider}")
    try:
        if args.provider == "ollama":
            model = config.get_ollama_embed_model()
            config.check_ollama([model])
            embeddings = config.make_ollama_embeddings(model)
        else:
            model = config.get_openai_embedding_model()
            embeddings = config.make_openai_embeddings(model)
        chunks = split_documents(load_documents(args.source), args.chunk_size, args.chunk_overlap)
        print(f"Embedding {len(chunks)} chunks with {args.provider}:{model} ...")
        manifest = build_index(chunks, embeddings, out, provider=args.provider, embedding_model=model,
                               source_files=args.source, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    except (config.MissingConfigError, config.ServiceUnavailableError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {manifest['num_chunks']} vectors (dim {manifest['dimension']}) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
