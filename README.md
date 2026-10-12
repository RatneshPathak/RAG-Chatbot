# RAG-Chatbot

A notebook-based learning project that walks from a first LangChain call to a working
**retrieval-augmented generation (RAG)** chatbot, with two interchangeable back-ends:

* **Local / free:** [Ollama](https://ollama.com) (`nomic-embed-text` embeddings + a local chat model)
* **Cloud / paid:** OpenAI embeddings and chat models

Documents are chunked, embedded, stored in a **FAISS** index, retrieved by similarity, and passed to the LLM, which is
instructed to answer *only* from the retrieved text, cite its sources, and say "I don't know" otherwise.

**Author:** Ratnesh Pathak

> ⚠️ The sample document is a scientific literature review about dietary supplements. It is used purely as demo text.
> Answers from this project are **not medical advice**.

## What is actually implemented

| Piece | File |
|---|---|
| LLM basics, prompts, chains with Ollama | `01-langchain_Basics.ipynb`, `02-langchain_Prompts.ipynb`, `03-langchain_chains.ipynb` |
| Build a FAISS index with Ollama embeddings | `Embeddings_generation_withOllama.ipynb` |
| RAG with a local model (retriever types: similarity, score threshold, MMR; LCEL chain) | `RAG_WITHOLLAMA.ipynb` |
| RAG with OpenAI embeddings + chat | `RAG_withOpenai.ipynb` |
| Reusable helpers: loading/splitting, index build + integrity checks, grounded QA with citations | `rag_chatbot/` |
| CLI: build, validate, chat | `scripts/build_index.py`, `scripts/validate_index.py`, `scripts/ask.py` |
| Automated tests (no network, no paid APIs) | `tests/` |

## How RAG works here

```text
 data/*.pdf|txt ──load (pypdf)──► pages ──split (1000 chars, 150 overlap)──► chunks
                                                                              │ embed (Ollama nomic-embed-text  OR  OpenAI)
                                                                              ▼
 question ──embed (same model)──► FAISS similarity search ──top-k chunks──► prompt ──► chat model ──► answer + citations
                                       ▲                                 (context is delimited and
                                       └──── vectorstores/<name>/          treated as untrusted text)
                                             index.faiss, index.pkl, manifest.json
```

Safeguards in `rag_chatbot/qa.py`: the model is told to use only the supplied context and to refuse otherwise; if retrieval
returns nothing, **the LLM is not called**; citations come from retrieval metadata (`file, p. N`), not from model text;
literal `<context>` tags inside documents are stripped so a document cannot close the context block. These reduce but do
**not** eliminate hallucination or prompt injection — small local models in particular can still misread context.

## Repository structure

```text
RAG-Chatbot/
├── 01-langchain_Basics.ipynb  02-langchain_Prompts.ipynb  03-langchain_chains.ipynb
├── Embeddings_generation_withOllama.ipynb  RAG_WITHOLLAMA.ipynb  RAG_withOpenai.ipynb
├── rag_chatbot/        config.py · documents.py · indexing.py · qa.py
├── scripts/            build_index.py · validate_index.py · ask.py
├── tests/
├── docs/               AUDIT_REPORT.md · TEST_REPORT.md · PUBLISHING_GUIDE.md
├── data/               sample PDF (CC BY 4.0, see data/README.md) + a short text note
├── vectorstores/       generated indexes (git-ignored)
├── requirements.txt  requirements-openai.txt  requirements-dev.txt
├── .env.example  .gitignore  LICENSE  pytest.ini
```

Notebooks stay in the repository root (as in the original project) so relative paths keep working.

## Prerequisites


- **Python 3.11 or 3.12**. The automated test suite passed on Python 3.11.8; Python 3.12 was the original project environment. Live model calls and full notebook execution have not been verified by the automated tests.


## Setup — Windows PowerShell

```powershell
git clone https://github.com/RatneshPathak/RAG-Chatbot.git
cd RAG-Chatbot
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt       # Ollama path
# pip install -r requirements-openai.txt   # add this for the OpenAI notebook
# pip install -r requirements-dev.txt      # tests + Jupyter
```

## Setup — Git Bash / macOS / Linux

```bash
git clone https://github.com/RatneshPathak/RAG-Chatbot.git && cd RAG-Chatbot
python3.12 -m venv .venv
source .venv/Scripts/activate         # Git Bash on Windows. macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama configuration

1. Install Ollama from <https://ollama.com>. On Windows/macOS it runs as a background app; otherwise run `ollama serve`.
2. Download the models the notebooks use:

```powershell
ollama pull nomic-embed-text    # embeddings (used by the index notebook and RAG_WITHOLLAMA)
ollama pull gemma3:1b           # chat model for 01 and RAG_WITHOLLAMA (default)
ollama pull llama3.1:8b         # only for 02-langchain_Prompts.ipynb (large download)
ollama pull qwen3:1.7b          # only for 03-langchain_chains.ipynb
ollama list                     # confirm
```

Override defaults with `OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_CHAT_MODEL` in `.env`. If Ollama is down or a model is
missing, the notebooks/scripts stop with a message that names the missing `ollama pull` command.

## OpenAI configuration

```powershell
Copy-Item .env.example .env       # Git Bash: cp .env.example .env
notepad .env                      # set OPENAI_API_KEY=...  and  OPENAI_CHAT_MODEL=<a model your account can use>
```

* `.env` is git-ignored — **never commit it or paste keys into notebooks.** Restart the Jupyter kernel after editing it.
* `OPENAI_CHAT_MODEL` intentionally has no default because model availability changes. (The original notebook hard-coded
  `gpt-3.5-turbo-0125`, which OpenAI's deprecation page lists for shutdown on 23 Oct 2026.)
* `OPENAI_EMBEDDING_MODEL` defaults to `text-embedding-3-small`.
* The legacy variable name `OPENAI_KEY` (used by the original notebook) is still accepted.

## Running the notebooks (in this order)

Open the folder in VS Code or run `jupyter lab`, and choose the `.venv` kernel. Run each notebook top to bottom.

1. `01-langchain_Basics.ipynb` — `ChatOllama` basics (`gemma3:1b`)
2. `02-langchain_Prompts.ipynb` — message types, prompt templates (`llama3.1:8b`)
3. `03-langchain_chains.ipynb` — `prompt | llm | parser` chains (`qwen3:1.7b`)
4. `Embeddings_generation_withOllama.ipynb` — builds `vectorstores/supplements_ollama/`
5. `RAG_WITHOLLAMA.ipynb` — retrieval + grounded answers with a local model
6. `RAG_withOpenai.ipynb` — same idea with OpenAI (**paid calls**)

## Generate and validate indexes

```powershell
python scripts/build_index.py --provider ollama          # or --provider openai (paid)
python scripts/validate_index.py vectorstores/supplements_ollama
python scripts/validate_index.py vectorstores/supplements_ollama --probe-embeddings   # also checks the live model's dimension
python scripts/ask.py --provider ollama                  # interactive Q&A with citations
```

An index can only be queried with the embedding model that built it. `manifest.json` records the model, dimension,
source-file hashes and index-file hashes; `load_index` refuses to load a store if the manifest is missing, a hash differs,
or the query model differs.

## Testing

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

The tests use deterministic fake embeddings and a fake LLM, so they need **no Ollama server, no API key and no network**.
They cover document loading/splitting, config validation, index build/load/validation (including tamper and model-mismatch
detection), grounded-answer construction, refusal on empty retrieval, citations, prompt-injection delimiting, the scripts'
error paths, and repository hygiene (notebook syntax and imports, no secrets or local paths, UTF-8 requirements, authorship).
**Not covered by automated tests:** live Ollama generation, live OpenAI calls, and full notebook execution — those need the
services above and must be checked manually.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | Start Ollama (`ollama serve`) and check `OLLAMA_BASE_URL`. |
| `these models are not installed: ollama pull …` | Run the printed command. |
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example`; restart the kernel. |
| OpenAI `404 model not found` | Set `OPENAI_CHAT_MODEL` to a model your account can access. |
| `manifest.json is missing … Refusing to unpickle` | The store wasn't built by this project; rebuild with `scripts/build_index.py`. |
| `Index was built with embedding model X` | Query with the same model, or rebuild the index. |
| `ModuleNotFoundError: rag_chatbot` | Start the notebook/script from the repository root. |
| `pip` errors reading `requirements.txt` | The file must be UTF-8 (the original was UTF-16). |
| Windows: script activation blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

## Security and data handling

* No credentials are stored in the repository; `.env`, virtual environments and generated indexes are git-ignored.
* FAISS stores written by LangChain contain a **pickle** file (`index.pkl`). Unpickling untrusted files can execute code, which is why
  indexes are not committed and `load_index` verifies hashes first. Only load indexes you built yourself.
* With the OpenAI path, document text and questions are sent to OpenAI. With Ollama, everything stays on your machine.
* Retrieved text is untrusted input; see "How RAG works here" for the mitigations and their limits.

## Limitations

* Answer quality depends on the model; small local models can still hallucinate or ignore instructions.
* The PDF extractor (`pypdf`) does not do OCR, and tables extract poorly.
* The score-threshold option depends on the embedding model and needs tuning on your own data.
*The original project environment used Python 3.12. The automated test suite also passed on Python 3.11.8. Live model calls and full notebook execution have not been verified by the automated tests.

## Third-party material

`data/1. dietary supplements - for whom.pdf` is © 2021 the author (R. E. Wierzejska), licensee MDPI, distributed under
CC BY 4.0 — see `data/README.md`. The repository's MIT license covers the original code and notebooks only.

## Future improvements

* Add a small web or chat UI on top of `rag_chatbot.qa.answer_question`.
* Evaluate retrieval quality with a labelled question set; add re-ranking.
* Support more document types and OCR; add conversation memory.
* Add continuous integration to run the test suite on each push.

## License

MIT © Ratnesh Pathak — see [LICENSE](LICENSE).
