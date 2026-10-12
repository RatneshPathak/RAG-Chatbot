# Audit report

Scope: the uploaded `RagChatBot.zip` (original project by Ratnesh Pathak). Everything below was observed by inspecting the
files; items marked *inferred* are conclusions, not verified facts.

## Inventory discrepancies
* The archive contained everything in the stated inventory **plus a full Windows virtual environment (`.venv/`, ~273 MB unzipped)** that was not listed.
* No `.env`, `.git`, or other hidden files were present.
* `thakur_college_info.txt` (210 bytes) is not referenced by any notebook.

## Findings

### Critical
| # | Finding | Resolution |
|---|---|---|
| C1 | `.venv/` (Windows binaries, ~273 MB) sits inside the project; pushing it would bloat the repo and leak machine-specific files. | Excluded from the deliverable; `.venv/` added to `.gitignore`. |

### High
| # | Finding | Resolution |
|---|---|---|
| H1 | `requirements.txt` is **UTF-16 encoded**, so `pip install -r` fails on many setups; it is also a 103-line full `pip freeze` (includes `torch`, `sentence-transformers`, Windows-only `pywin32`) and lacks `langchain-openai`. | Replaced by a minimal UTF-8, pinned core list (versions taken from the original freeze), plus `requirements-openai.txt` and `requirements-dev.txt`. Installed in a fresh Python 3.12 venv; `pip check` clean. |
| H2 | `RAG_withOpenai.ipynb` hard-codes `gpt-3.5-turbo-0125`; OpenAI's deprecation page lists that model for shutdown on **23 Oct 2026**. | Model is read from `OPENAI_CHAT_MODEL` (no default). |
| H3 | `health_supplements/` (36 chunks, 768-d) was built from `rag-dataset\regulatory_human_body-generic_manual.pdf` — a human-body manual that is **not in the project** and unrelated to the supplements PDF. The saved output shows "how to gain muscle mass?" answered with kidney facts. | Index not shipped. A new store is built from the included PDF (`vectorstores/supplements_ollama`). |
| H4 | `Embeddings_generation_withOllama.ipynb` never generates an index; its last executed cell failed with `name 'chunks' is not defined`. | Rewritten to load → split → embed → save → reload → query. |
| H5 | Pickle-backed FAISS stores were loaded with `allow_dangerous_deserialization=True` and no provenance or integrity check. | `manifest.json` (model, dimension, SHA-256 of both files) is verified before unpickling; missing/mismatched manifest ⇒ refuse. Indexes are not committed. |
| H6 | Notebook outputs contain another person's Windows profile path (`C:\Users\<name>\AppData\...`) and a different project folder name (`c:\ML\Self-Projects\LU_Rag_YOUTUBE\...`). | All stderr/warning outputs (the only place paths appeared) removed; a test fails if such paths return. |

### Medium
| # | Finding | Resolution |
|---|---|---|
| M1 | OpenAI notebook used deprecated imports (`langchain.embeddings`, `langchain.vectorstores`, community `OpenAIEmbeddings/ChatOpenAI`) and the deprecated `RetrievalQA` / `chain({...})` call. | Uses `langchain-openai` and the new `answer_question` helper. |
| M2 | Key read from non-standard `OPENAI_KEY`, while a comment mentions `OPENAI_API_KEY`; `load_dotenv()` result unused in the Ollama notebook. | `OPENAI_API_KEY` is canonical (`OPENAI_KEY` still accepted); missing/placeholder keys raise a clear error; key never printed. |
| M3 | `myVectorStore/` (392 chunks, 1536-d) has **no metadata** (no source/page), so citations are impossible; chunks are ≤200 characters; PDF glyph artefacts (`/gid00030…`) were embedded. | New indexes use page-level metadata, 1000/150 chunking, and glyph-ID cleanup. |
| M4 | Embedding model of `myVectorStore` cannot be proven: 1536 dimensions fit OpenAI `text-embedding-ada-002` or `text-embedding-3-small` (*inferred; unverifiable without API calls*). | Provenance is recorded in the manifest for every new index; the old one is not reused. |
| M5 | No grounding: no refusal path, no citations, retrieved text unmarked. | `rag_chatbot/qa.py` (see README). Reduces, does not eliminate, hallucination. |
| M6 | `hub.pull("rlm/rag-prompt")` (network call) was immediately overwritten by a custom prompt; `hub` import adds a hidden dependency. | Removed; documented in the notebook. |
| M7 | Undeclared/over-declared dependencies: `sentence-transformers` used in a demo cell, `PyPDF2` (deprecated, superseded by `pypdf`); an embedded diagram mentions PyMuPDF (the `health_supplements` metadata keys look like PyMuPDF output — *inferred*). | Demo cell made optional; `pypdf` used. PyMuPDF (AGPL) deliberately not adopted. |
| M8 | Notebook kernels recorded Python 3.12.2, 3.12.8 and 3.13.2. | Python 3.12 documented as the supported line. |

### Low
* `ChatOllama(..., max_tokens=100)` is not a real option and is silently ignored next to `num_predict=200` → removed.
* MMR retriever used `lambda_mult=1` (no diversity) → 0.5. Diagrams show `llama3.2:1b` while code used `gemma3:1b` → noted.
* Empty trailing cells (01, 03, OpenAI, RAG notebooks) removed; typo-level text left as is. Kernel name `.venv` → `python3`.
* The notebooks' saved outputs (01–03) come from earlier runs and could not be re-executed here; labelled as such.

## FAISS index investigation (read with a restricted unpickler, only `InMemoryDocstore`/`Document`/`dict`/`UUID` allowed)
| | `health_supplements/` | `myVectorStore/` |
|---|---|---|
| Index type / metric | `IndexFlatL2` | `IndexFlatL2` |
| Vectors / dimension | 36 / 768 (fits `nomic-embed-text`, *inferred*) | 392 / 1536 (OpenAI-style, *inferred*) |
| FAISS ↔ docstore ↔ id map | consistent (36/36/36) | consistent (392/392/392) |
| Source content | `rag-dataset\regulatory_human_body-generic_manual.pdf` (not provided) | text of the supplements PDF (no metadata) |
| Retrieval quality | **not testable** (needs the original embedding service); saved outputs show unrelated answers | **not testable** (needs OpenAI key) |
| Interchangeable? | **No** — different dimensions, models and documents | |

## Privacy / licensing
* **No API keys, tokens or private keys found** (pattern scan of the full upload; two hits were false positives inside third-party library files in `.venv`).
* The PDF is a published paper under **CC BY 4.0** (stated in the PDF). Redistribution is allowed with attribution; included unmodified with attribution in `data/README.md`. Its author/editor names are legitimate third-party attribution, not project authors.
* `myVectorStore/index.pkl` contains text of that paper including the paper author's institutional e-mail address; another reason not to commit the pickle.
* Project authorship: LICENSE, README, notebook metadata and package docstring all name **Ratnesh Pathak**.

## Remaining risks
* Live Ollama and OpenAI behaviour is untested (see TEST_REPORT.md).
* Small local models can still hallucinate or follow injected instructions despite the safeguards.
* The remote repository's current contents and history were not visible to me; check them before pushing (PUBLISHING_GUIDE.md).
