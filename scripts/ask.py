"""Interactive command-line chatbot over a vector store built by scripts/build_index.py.

    python scripts/ask.py --provider ollama
    python scripts/ask.py --provider openai      # paid API calls
Type an empty line to quit.
"""
import _bootstrap  # noqa: F401
import argparse
import sys

from rag_chatbot import config
from rag_chatbot.indexing import IndexIntegrityError, load_index, read_manifest
from rag_chatbot.qa import answer_question


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["ollama", "openai"], required=True)
    ap.add_argument("--index", help="vector store directory (default: vectorstores/supplements_<provider>)")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args(argv)
    config.load_env()
    index_dir = args.index or str(config.VECTORSTORES_DIR / f"supplements_{args.provider}")
    try:
        manifest = read_manifest(index_dir)
        if args.provider == "ollama":
            chat = config.get_ollama_chat_model()
            config.check_ollama([manifest["embedding_model"], chat])
            embeddings, llm = config.make_ollama_embeddings(manifest["embedding_model"]), config.make_ollama_llm(chat)
        else:
            embeddings = config.make_openai_embeddings(manifest["embedding_model"])
            llm = config.make_openai_llm()
        store = load_index(index_dir, embeddings, expected_embedding_model=manifest["embedding_model"])
    except (IndexIntegrityError, config.MissingConfigError, config.ServiceUnavailableError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    while True:
        q = input("\nQuestion (blank to quit): ").strip()
        if not q:
            return 0
        try:
            ans = answer_question(q, store, llm, k=args.k)
        except Exception as exc:  # network/model errors should not crash the loop
            print(f"Request failed: {type(exc).__name__}: {exc}")
            continue
        print(f"\n{ans.text}")
        if ans.sources:
            print("Sources: " + "; ".join(ans.sources))


if __name__ == "__main__":
    raise SystemExit(main())
