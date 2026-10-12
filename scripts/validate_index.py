"""Validate a vector store directory: manifest, file hashes, FAISS/docstore consistency.

    python scripts/validate_index.py vectorstores/supplements_ollama
Add --probe-embeddings to also check that the live embedding model's dimension matches (needs the service).
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys

from rag_chatbot import config
from rag_chatbot.indexing import IndexIntegrityError, read_manifest, validate_index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index_dir")
    ap.add_argument("--probe-embeddings", action="store_true")
    args = ap.parse_args(argv)
    config.load_env()
    try:
        manifest = read_manifest(args.index_dir)
        embeddings = None
        if args.probe_embeddings:
            embeddings = (config.make_ollama_embeddings(manifest["embedding_model"])
                          if manifest["provider"] == "ollama"
                          else config.make_openai_embeddings(manifest["embedding_model"]))
        result = validate_index(args.index_dir, embeddings, expected_embedding_model=manifest["embedding_model"])
    except (IndexIntegrityError, config.MissingConfigError, FileNotFoundError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({**result, "embedding_model": manifest["embedding_model"]}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
