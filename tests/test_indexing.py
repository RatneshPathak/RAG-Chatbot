import json

import pytest

from rag_chatbot.indexing import IndexIntegrityError, build_index, load_index, read_manifest, validate_index


def test_build_creates_manifest_and_roundtrip(store, tmp_path):
    m = read_manifest(tmp_path / "idx")
    assert m["embedding_model"] == "fake-64" and m["dimension"] == 64 and m["num_chunks"] == 3
    assert set(m["file_sha256"]) == {"index.faiss", "index.pkl"}
    assert store.index.ntotal == 3


def test_retrieval_returns_relevant_doc(store):
    top = store.similarity_search("calcium and bone health with vitamin D", k=1)[0]
    assert "Vitamin D" in top.page_content and top.metadata["page"] == 3


def test_validate_ok(store, tmp_path, emb):
    r = validate_index(tmp_path / "idx", emb, expected_embedding_model="fake-64")
    assert r["ok"] and r["num_chunks"] == 3 and r["dimension"] == 64


def test_missing_manifest_refuses_to_unpickle(store, tmp_path, emb):
    (tmp_path / "idx" / "manifest.json").unlink()
    with pytest.raises(IndexIntegrityError, match="manifest"):
        load_index(tmp_path / "idx", emb)


def test_tampered_pickle_detected(store, tmp_path, emb):
    p = tmp_path / "idx" / "index.pkl"
    p.write_bytes(p.read_bytes() + b"\x00")
    with pytest.raises(IndexIntegrityError, match="hash"):
        load_index(tmp_path / "idx", emb)


def test_embedding_model_mismatch_rejected(store, tmp_path, emb):
    with pytest.raises(IndexIntegrityError, match="embedding model"):
        load_index(tmp_path / "idx", emb, expected_embedding_model="other-model")


def test_shape_mismatch_with_manifest_rejected(store, tmp_path, emb):
    mp = tmp_path / "idx" / "manifest.json"
    m = json.loads(mp.read_text())
    m["dimension"] = 999
    mp.write_text(json.dumps(m))
    with pytest.raises(IndexIntegrityError, match="shape"):
        load_index(tmp_path / "idx", emb)


def test_dimension_probe_mismatch_reported(store, tmp_path):
    from conftest import FakeEmbeddings

    r = validate_index(tmp_path / "idx", FakeEmbeddings(dim=32))
    assert not r["ok"] and any("dims" in p for p in r["problems"])


def test_build_rejects_empty(emb, tmp_path):
    with pytest.raises(ValueError):
        build_index([], emb, tmp_path / "x", provider="fake", embedding_model="fake")
