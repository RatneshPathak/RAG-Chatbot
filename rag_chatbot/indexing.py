"""Build, load and validate FAISS vector stores with a provenance manifest.

LangChain stores the FAISS document map (``index.pkl``) with pickle. Unpickling untrusted data can run
arbitrary code. To reduce that risk, every store built here gets a ``manifest.json`` that records the
embedding model, vector dimension, source files and SHA-256 hashes of both index files. ``load_index``
refuses to unpickle a store whose manifest is missing or whose hashes do not match.
This detects accidental corruption and tampering *after* the build; it is not a defence against a
store that was malicious when you built it, so only build from files you trust.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

MANIFEST = "manifest.json"
_FILES = ("index.faiss", "index.pkl")


class IndexIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(
    chunks: list[Document],
    embeddings,
    out_dir: str | Path,
    *,
    provider: str,
    embedding_model: str,
    source_files: list[str | Path] = (),
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Embed ``chunks``, save a FAISS store to ``out_dir`` and write ``manifest.json``."""
    if not chunks:
        raise ValueError("No chunks to index.")
    out_dir = Path(out_dir)
    store = FAISS.from_documents(chunks, embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))
    manifest = {
        "provider": provider,
        "embedding_model": embedding_model,
        "dimension": int(store.index.d),
        "num_chunks": int(store.index.ntotal),
        "index_type": type(store.index).__name__,
        "sources": {Path(p).name: _sha256(Path(p)) for p in source_files},
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_sha256": {f: _sha256(out_dir / f) for f in _FILES},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(index_dir: str | Path) -> dict:
    p = Path(index_dir) / MANIFEST
    if not p.is_file():
        raise IndexIntegrityError(
            f"{p} is missing. Refusing to unpickle a vector store of unknown origin. "
            "Rebuild it with scripts/build_index.py."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def verify_files(index_dir: str | Path) -> dict:
    index_dir = Path(index_dir)
    manifest = read_manifest(index_dir)
    for name in _FILES:
        f = index_dir / name
        if not f.is_file():
            raise IndexIntegrityError(f"{f} is missing.")
        if _sha256(f) != manifest.get("file_sha256", {}).get(name):
            raise IndexIntegrityError(f"{name} does not match the hash recorded in {MANIFEST}; refusing to load.")
    return manifest


class _NullEmbeddings:
    """Placeholder so a store can be structurally validated without any embedding service."""

    def embed_documents(self, texts):  # pragma: no cover
        raise RuntimeError("no embeddings configured")

    def embed_query(self, text):  # pragma: no cover
        raise RuntimeError("no embeddings configured")


def load_index(index_dir: str | Path, embeddings, *, expected_embedding_model: str | None = None) -> FAISS:
    """Load a store built by :func:`build_index`, after integrity and embedding-model checks."""
    manifest = verify_files(index_dir)
    if expected_embedding_model and manifest["embedding_model"] != expected_embedding_model:
        raise IndexIntegrityError(
            f"Index was built with embedding model '{manifest['embedding_model']}' but you are querying with "
            f"'{expected_embedding_model}'. Vectors from different models are not comparable; rebuild the index "
            "or use the matching model."
        )
    store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    if store.index.d != manifest["dimension"] or store.index.ntotal != manifest["num_chunks"]:
        raise IndexIntegrityError("FAISS index shape does not match the manifest.")
    return store


def validate_index(index_dir: str | Path, embeddings=None, *, expected_embedding_model: str | None = None) -> dict:
    """Structural checks. If ``embeddings`` is given, also confirms its dimension matches the index."""
    store = load_index(index_dir, embeddings or _NullEmbeddings(), expected_embedding_model=expected_embedding_model)
    ids = store.index_to_docstore_id
    docs = store.docstore._dict
    problems = []
    if not (store.index.ntotal == len(ids) == len(docs)):
        problems.append("vector count, id map and docstore sizes differ")
    if any(i not in docs for i in ids.values()):
        problems.append("id map points at missing documents")
    if any(not getattr(d, "page_content", "").strip() for d in docs.values()):
        problems.append("empty documents present")
    if any("source" not in d.metadata for d in docs.values()):
        problems.append("some chunks have no 'source' metadata (citations impossible)")
    if embeddings is not None:
        dim = len(embeddings.embed_query("dimension probe"))
        if dim != store.index.d:
            problems.append(f"embedding model returns {dim} dims but index has {store.index.d}")
    return {"ok": not problems, "problems": problems, "num_chunks": store.index.ntotal, "dimension": store.index.d}
