"""RAG (/api/chat/query) 的 Work Unit 候选对象测试。

通过 mock get_rag_service，验证响应中附带 work_unit_candidate，
且不会自动落盘到 data/work_units/。
不真实调用 LLM。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import work_units as work_units_api
from app.main import app
from app.schemas import SourceItem
from app.services.rag_service import RAGAnswer

client = TestClient(app)


def _mock_rag_answer() -> RAGAnswer:
    return RAGAnswer(
        answer="Band 10 中心波长约 10.9 μm。",
        sources=[
            SourceItem(
                filename="landsat.pdf",
                page=3,
                chunk_id="abc123",
                score=0.85,
                content_preview="Landsat 8 TIRS Band 10 中心波长 10.9 μm。",
            )
        ],
        refused=False,
    )


@patch("app.api.chat.get_rag_service")
def test_rag_response_has_work_unit_candidate(mock_get_svc: MagicMock) -> None:
    """1/2/3. RAG response 存在 work_unit_candidate，entry=rag，endpoint 正确。"""
    mock_svc = MagicMock()
    mock_svc.answer.return_value = _mock_rag_answer()
    mock_get_svc.return_value = mock_svc

    resp = client.post(
        "/api/chat/query",
        json={"question": "Landsat 8 Band 10 波长?", "top_k": 5, "use_rerank": False},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # 原有字段仍存在且未改变
    assert data["answer"] and data["sources"] and data["refused"] is False

    # 1. candidate 存在
    assert "work_unit_candidate" in data
    cand = data["work_unit_candidate"]
    assert cand is not None
    # 2. entry == rag
    assert cand["entry"] == "rag"
    # 3. replay_payload.endpoint == /api/chat/query
    assert cand["replay_payload"]["endpoint"] == "/api/chat/query"
    assert cand["replay_payload"]["body"]["question"] == "Landsat 8 Band 10 波长?"
    assert cand["replay_payload"]["body"]["top_k"] == 5
    assert cand["replay_payload"]["body"]["use_rerank"] is False

    # sources 复用响应已有结果（不额外检索），且已序列化为 dict
    assert cand["sources"][0]["filename"] == "landsat.pdf"


@patch("app.api.chat.get_rag_service")
def test_rag_does_not_persist_work_unit(
    mock_get_svc: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """7. RAG 查询不自动落盘（查询后 work_units 目录应仍为空）。"""
    # 把 store 目录重定向到临时目录，确认查询不写入
    monkeypatch.setattr(work_units_api, "_get_work_unit_dir", lambda: tmp_path)

    mock_svc = MagicMock()
    mock_svc.answer.return_value = _mock_rag_answer()
    mock_get_svc.return_value = mock_svc

    resp = client.post("/api/chat/query", json={"question": "Landsat 8 Band 10 波长?"})
    assert resp.status_code == 200

    # 临时目录下不应出现任何 Work Unit 文件
    assert list(tmp_path.glob("*.json")) == []
