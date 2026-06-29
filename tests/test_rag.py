"""RAG 查询测试。

需要真实 API Key 与已入库的文档。
通过 mock Retriever 验证拒答与正常回答两条路径。
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import SourceItem
from app.services.rag_service import RAGAnswer, RAGService


def test_rag_refuse_when_empty() -> None:
    svc = RAGService(retriever=MagicMock(), llm=MagicMock())
    svc.retriever.retrieve.return_value = []
    result = svc.answer("任意问题")
    assert result.refused is True
    assert result.sources == []


def test_rag_with_hits() -> None:
    svc = RAGService(retriever=MagicMock(), llm=MagicMock())
    svc.retriever.retrieve.return_value = [
        {
            "chunk_id": "abc123",
            "score": 0.85,
            "content": "Landsat 8 TIRS Band 10 中心波长 10.9 μm。",
            "filename": "landsat.pdf",
            "page": 3,
            "doc_id": "d1",
        }
    ]
    svc.llm.chat.return_value = "Band 10 中心波长约 10.9 μm。"
    result = svc.answer("Landsat 8 Band 10 波长?")
    assert result.refused is False
    assert "10.9" in result.answer
    assert len(result.sources) == 1
    assert isinstance(result.sources[0], SourceItem)
