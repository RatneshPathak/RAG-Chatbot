# Test report

Environment: Linux sandbox, Python 3.12.3, fresh virtual environment created from `requirements-dev.txt`
(langchain 0.3.25, langchain-community 0.3.24, langchain-core 0.3.61, langchain-ollama 0.3.3, langchain-openai 0.3.18,
faiss-cpu 1.11.0, numpy 2.2.6, pypdf 5.9.0, pytest 8.3.5). No Ollama server and no OpenAI key were available.

| Command | Result |
|---|---|
| `pip install -r requirements-dev.txt` | succeeded |
| `pip check` | "No broken requirements found." |
| `python -m pytest` | **65 passed**, 0 failed, 0 skipped (4 third-party DeprecationWarnings from faiss/numpy) |
| Mutation checks (deliberately broke the code, re-ran tests, restored) | hash check removed → 2 tests failed; LLM-called-on-empty-context → 2 failed; UTF-16 requirements → 1 failed; local path put in a notebook → 1 failed; restored → 65 passed |
| Real PDF through load → split → build → validate → answer, using test-double embeddings | 17 pages → 86 chunks (max 998 chars), 0 glyph artefacts, validation `ok: true`, citations returned |
| `scripts/validate_index.py` on the original `health_supplements/` | exit 1: "manifest.json is missing. Refusing to unpickle…" (intended) |
| Notebooks executed top-to-bottom with `nbclient` (no services available) | See below |

## Notebook execution (environment limits, not code failures)
| Notebook | Cells that ran | First stop |
|---|---|---|
| 01, 02, 03 | imports/prompt cells OK | `ConnectError: Connection refused` (no Ollama server) |
| Embeddings_generation_withOllama | imports | `ServiceUnavailableError: Cannot reach Ollama…` (intended friendly error) |
| RAG_WITHOLLAMA | – | same friendly Ollama error in the setup cell |
| RAG_withOpenai | optional demo (skips gracefully), imports | `MissingConfigError: OPENAI_API_KEY is not set` (intended) |

All imports in all six notebooks resolve (also asserted by `tests/test_repo_hygiene.py`).

## NOT verified (needs your machine)
* Live Ollama embedding/generation (`nomic-embed-text`, `gemma3:1b`, `llama3.1:8b`, `qwen3:1.7b`) and the resulting answer quality.
* Live OpenAI embedding/chat calls, and whether your chosen `OPENAI_CHAT_MODEL` is available.
* Retrieval quality of the original `health_supplements/` and `myVectorStore/` indexes.
* Notebook cells after the first service call in notebooks 01–03 and the RAG notebooks (they depend on live services).
* Python versions other than 3.12; Windows execution of the pinned requirements (Linux was used here).
* Saved outputs in notebooks 01–03 are from your earlier runs and were not regenerated.
