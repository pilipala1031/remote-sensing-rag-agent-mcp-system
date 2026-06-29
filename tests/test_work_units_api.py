"""Work Unit API 集成测试（POST/GET list/GET detail/DELETE）。

通过 monkeypatch 替换 _get_work_unit_dir 为 pytest 临时目录，
避免污染真实 data/work_units/。
不真实调用 LLM / RAGService / AgentService。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import work_units as work_units_api
from app.main import app

client = TestClient(app)


@pytest.fixture
def isolated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 API 读取的 Work Unit 目录重定向到临时目录。"""
    monkeypatch.setattr(work_units_api, "_get_work_unit_dir", lambda: tmp_path)
    return tmp_path


def _save_payload(entry: str = "agent", question: str = "U-Net 和 DeepLabV3+ 哪个更好？") -> dict:
    return {
        "entry": entry,
        "question": question,
        "answer": "U-Net 适合小样本，DeepLabV3+ 适合多尺度场景。",
        "sources": [{"filename": "02_models.md", "page": 1, "chunk_id": "abc", "score": 0.84}],
        "refused": False,
        "tool_calls": [{"name": "knowledge_base_search", "elapsed": 1.2}],
        "trace_events": [{"step": 1, "event": "agent_started"}],
        "timing": {"total_elapsed": 3.5},
        "verification": {"verified": True},
        "errors": [],
        "replay_payload": {"question": question, "include_trace": True},
    }


def test_post_save_success(isolated_dir: Path) -> None:
    """1. POST 保存成功。"""
    resp = client.post("/api/work_units", json=_save_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["work_unit_id"].startswith("wu_")
    assert body["message"] == "Work Unit 已保存"


def test_get_list_success(isolated_dir: Path) -> None:
    """2. GET list 成功。"""
    client.post("/api/work_units", json=_save_payload(entry="rag", question="Q1"))
    client.post("/api/work_units", json=_save_payload(entry="agent", question="Q2"))

    resp = client.get("/api/work_units")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert len(body["work_units"]) == 2
    assert all("work_unit_id" in w for w in body["work_units"])


def test_get_detail_success(isolated_dir: Path) -> None:
    """3. GET detail 成功。"""
    saved = client.post("/api/work_units", json=_save_payload()).json()

    resp = client.get(f"/api/work_units/{saved['work_unit_id']}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["work_unit_id"] == saved["work_unit_id"]
    assert body["entry"] == "agent"
    # replay_payload 完整保留（v1 只保存不执行）
    assert body["replay_payload"]["include_trace"] is True


def test_delete_success(isolated_dir: Path) -> None:
    """4. DELETE 成功。"""
    saved = client.post("/api/work_units", json=_save_payload()).json()

    resp = client.delete(f"/api/work_units/{saved['work_unit_id']}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"] == "Work Unit 已删除"
    assert body["work_unit_id"] == saved["work_unit_id"]

    # 删除后 GET 应 404
    assert client.get(f"/api/work_units/{saved['work_unit_id']}").status_code == 404


def test_get_nonexistent_returns_404(isolated_dir: Path) -> None:
    """5. 不存在 id 返回 404。"""
    resp = client.get("/api/work_units/wu_not_exist")
    assert resp.status_code == 404
    assert "Work Unit 不存在" in resp.json()["detail"]


def test_delete_nonexistent_returns_404(isolated_dir: Path) -> None:
    """DELETE 不存在 id 也返回 404。"""
    resp = client.delete("/api/work_units/wu_not_exist")
    assert resp.status_code == 404
    assert "Work Unit 不存在" in resp.json()["detail"]


def test_entry_filter(isolated_dir: Path) -> None:
    """6. entry 过滤可用。"""
    client.post("/api/work_units", json=_save_payload(entry="rag", question="Q1"))
    client.post("/api/work_units", json=_save_payload(entry="agent", question="Q2"))
    client.post("/api/work_units", json=_save_payload(entry="agent", question="Q3"))

    rag = client.get("/api/work_units?entry=rag").json()
    agent = client.get("/api/work_units?entry=agent").json()
    assert rag["total"] == 1 and rag["work_units"][0]["entry"] == "rag"
    assert agent["total"] == 2 and all(w["entry"] == "agent" for w in agent["work_units"])


def test_invalid_entry_returns_400(isolated_dir: Path) -> None:
    """附加：非法 entry 返回 400。"""
    resp = client.get("/api/work_units?entry=xxx")
    assert resp.status_code == 400


def test_replay_endpoint_not_exposed(isolated_dir: Path) -> None:
    """7. v1 不存在 replay 端点（POST /api/work_units/{id}/replay 应 404 / 405）。"""
    saved = client.post("/api/work_units", json=_save_payload()).json()
    resp = client.post(f"/api/work_units/{saved['work_unit_id']}/replay", json={})
    assert resp.status_code in (404, 405)
