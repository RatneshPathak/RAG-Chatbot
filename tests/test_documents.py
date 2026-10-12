import pytest

from rag_chatbot import config
from rag_chatbot.documents import clean_text, load_documents, load_pdf, split_documents


def test_clean_text_removes_glyph_ids():
    assert "gid" not in clean_text("Hello /gid00030/gid00035 world")
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_load_real_pdf_has_metadata_and_no_gid():
    docs = load_pdf(config.DEFAULT_PDF)
    assert len(docs) >= 10
    assert all(d.metadata["source"] == config.DEFAULT_PDF.name and d.metadata["page"] >= 1 for d in docs)
    assert not any("/gid" in d.page_content for d in docs)


def test_split_and_text_file():
    chunks = split_documents(load_documents([config.DATA_DIR / "thakur_college_info.txt"]), 100, 10)
    assert chunks and all(len(c.page_content) <= 100 for c in chunks)
    assert chunks[0].metadata["source"] == "thakur_college_info.txt"


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")
    with pytest.raises(ValueError):
        load_documents([tmp_path / "x.docx"])
    empty = tmp_path / "e.txt"
    empty.write_text("  ")
    with pytest.raises(ValueError):
        load_documents([empty])
    with pytest.raises(ValueError):
        split_documents([], 100, 10)
    with pytest.raises(ValueError):
        split_documents([], 10, 10)
