import hashlib
import re
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (L2-normalised). Test double only; never used in production code."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str):
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Records the messages it receives and returns a canned reply."""

    def __init__(self, reply="Vitamin D is discussed [1]."):
        self.reply, self.calls = reply, []

    def invoke(self, messages):
        self.calls.append(messages)

        class R:
            content = self.reply

        return R()


@pytest.fixture
def emb():
    return FakeEmbeddings()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def sample_docs():
    from langchain_core.documents import Document

    return [
        Document(page_content="Vitamin D supports bone health and calcium absorption.", metadata={"source": "paper.pdf", "page": 3}),
        Document(page_content="Omega-3 fatty acids are found in fish oil supplements.", metadata={"source": "paper.pdf", "page": 5}),
        Document(page_content="Thakur College is located in Kandivali East, Mumbai.", metadata={"source": "note.txt", "page": 1}),
    ]


@pytest.fixture
def store(tmp_path, emb, sample_docs):
    from rag_chatbot.indexing import build_index, load_index

    build_index(sample_docs, emb, tmp_path / "idx", provider="fake", embedding_model="fake-64")
    return load_index(tmp_path / "idx", emb, expected_embedding_model="fake-64")
