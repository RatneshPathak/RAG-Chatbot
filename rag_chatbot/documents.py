"""Load PDFs / text files into LangChain Documents (with source metadata) and split them."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Some PDF fonts extract as runs like "/gid00030/gid00035"; they are glyph IDs, not text.
_GID_RE = re.compile(r"(?:/gid\d{3,6})+")
_WS_RE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    text = _GID_RE.sub(" ", text or "")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str | Path) -> list[Document]:
    """One Document per non-empty page; metadata: ``source`` (file name) and ``page`` (1-based)."""
    from pypdf import PdfReader

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    docs = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
    if not docs:
        raise ValueError(f"No extractable text in {path.name} (scanned PDF? OCR is not supported).")
    return docs


def load_text_file(path: str | Path) -> list[Document]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    text = clean_text(path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"{path.name} is empty.")
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]


def load_documents(paths: list[str | Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        suffix = Path(p).suffix.lower()
        if suffix == ".pdf":
            docs += load_pdf(p)
        elif suffix in {".txt", ".md"}:
            docs += load_text_file(p)
        else:
            raise ValueError(f"Unsupported file type: {p}")
    return docs


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError("Splitting produced no chunks.")
    return chunks
