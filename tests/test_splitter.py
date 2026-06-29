"""文本切分测试（不依赖外部 API）。"""
from __future__ import annotations

from app.services.document_loader import PageContent
from app.services.splitter import clean_text, split_document


def test_clean_text() -> None:
    raw = "你好\u3000\u3000世界\n\n\n\n第二段"
    out = clean_text(raw)
    assert "你" in out
    assert "\n\n\n" not in out


def test_split_document(sample_text: str) -> None:
    pages = [PageContent(page=1, text=sample_text)]
    chunks = split_document(
        pages,
        doc_id="testdoc",
        filename="demo.txt",
        chunk_size=80,
        chunk_overlap=20,
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_id
        assert c.metadata["doc_id"] == "testdoc"
        assert c.metadata["filename"] == "demo.txt"
        assert c.metadata["page"] == 1
        assert c.metadata["chunk_id"] == c.chunk_id
