# Vector stores (generated, not committed)

Everything in this folder except this file is git-ignored. Build stores locally:

```bash
python scripts/build_index.py --provider ollama   # -> vectorstores/supplements_ollama/
python scripts/build_index.py --provider openai   # -> vectorstores/supplements_openai/  (paid API calls)
python scripts/validate_index.py vectorstores/supplements_ollama
```

Each store contains `index.faiss`, `index.pkl` and `manifest.json` (embedding model, dimension, source hashes,
file hashes). **Why they are not committed:** `index.pkl` is a pickle file, and loading a pickle from an
untrusted source can execute arbitrary code. Rebuild locally instead of downloading someone else's store.
An index can only be queried with the embedding model that built it (Ollama `nomic-embed-text` = 768
dimensions; OpenAI `text-embedding-3-small` = 1536 by default); the manifest check enforces this.
