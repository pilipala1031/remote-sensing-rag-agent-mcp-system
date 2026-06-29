"""文档解析测试。"""
from __future__ import annotations

from pathlib import Path

from app.services.document_loader import load_document


def test_load_txt(tmp_path: Path) -> None:
    f = tmp_path / "demo.txt"
    f.write_text("遥感是采集地表信息的手段。", encoding="utf-8")
    pages = load_document(f)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert "遥感" in pages[0].text


def test_load_markdown(tmp_path: Path) -> None:
    f = tmp_path / "demo.md"
    f.write_text("# 标题\n正文内容。", encoding="utf-8")
    pages = load_document(f)
    assert len(pages) == 1
    assert "正文" in pages[0].text
