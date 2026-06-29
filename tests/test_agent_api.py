"""Agent API + Service 测试。

两部分：
1. Service 层 — RemoteSensingAgentService.query() 单元测试（mock run_langchain_agent）
2. API 层    — POST /api/agent/query 集成测试（TestClient + mock AgentService）

同时验证 /api/chat/query 未受影响。
不真实调用 LLM。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents.agent_service import (
    EMPTY_QUESTION_ANSWER,
    ERROR_ANSWER,
    RemoteSensingAgentService,
    get_agent_service,
)
from app.main import app


# ============================================================================ #
#  辅助：构造 mock 返回                                                         #
# ============================================================================ #

def _mock_normal_result() -> dict:
    """构造 Agent 正常返回。"""
    return {
        "answer": "DeepLabV3+ 采用 ASPP 模块，适合多尺度特征提取。",
        "sources": [
            {
                "filename": "02_models.md",
                "page": 1,
                "chunk_id": "abc123",
                "score": 0.88,
                "content_preview": "DeepLabV3+ 采用 ASPP...",
            }
        ],
        "refused": False,
        "tool_calls": [
            {
                "tool": "knowledge_base_search",
                "input": "DeepLabV3+",
                "status": "success",
                "output_summary": "检索到 1 个相关片段",
                "elapsed": 0.123,
                "error": None,
            }
        ],
        "agent_trace": [
            "agent_started",
            "tool_called:knowledge_base_search",
            "tool_result_parsed",
            "agent_finished",
        ],
        "trace_events": [
            {"step": 1, "event": "agent_started", "timestamp": 0.0, "detail": None},
            {"step": 2, "event": "tool_called", "timestamp": 0.0, "detail": "knowledge_base_search"},
            {"step": 3, "event": "tool_result_parsed", "timestamp": 0.123, "detail": "knowledge_base_search"},
            {"step": 4, "event": "agent_finished", "timestamp": 1.0, "detail": None},
        ],
        "errors": [],
        "timing": {
            "total_elapsed": 1.23,
            "agent_invoke_elapsed": 1.0,
            "tool_search_elapsed_total": 0.123,
        },
    }


def _mock_refused_result() -> dict:
    """构造 Agent 拒答返回。"""
    return {
        "answer": "根据当前知识库内容，无法确定该问题的答案。",
        "sources": [],
        "refused": True,
        "tool_calls": [
            {
                "tool": "knowledge_base_search",
                "input": "天气",
                "status": "success",
                "output_summary": "未检索到相关知识库内容",
                "elapsed": 0.05,
                "error": None,
            }
        ],
        "agent_trace": [
            "agent_started",
            "tool_called:knowledge_base_search",
            "tool_result_parsed",
            "agent_finished",
        ],
        "trace_events": [
            {"step": 1, "event": "agent_started", "timestamp": 0.0, "detail": None},
            {"step": 2, "event": "tool_called", "timestamp": 0.0, "detail": "knowledge_base_search"},
            {"step": 3, "event": "tool_result_parsed", "timestamp": 0.05, "detail": "knowledge_base_search"},
            {"step": 4, "event": "agent_finished", "timestamp": 0.45, "detail": None},
        ],
        "errors": [],
        "timing": {
            "total_elapsed": 0.5,
            "agent_invoke_elapsed": 0.45,
            "tool_search_elapsed_total": 0.05,
        },
    }


# ============================================================================ #
#  Service 层：RemoteSensingAgentService.query()                               #
# ============================================================================ #

@patch("app.agents.agent_service.run_langchain_agent")
def test_service_query_normal(mock_run: pytest.mock.Mock) -> None:
    """正常问题 → 返回 run_langchain_agent 的结果。"""
    mock_run.return_value = _mock_normal_result()

    svc = RemoteSensingAgentService()
    result = svc.query("DeepLabV3+ 有什么特点")

    assert result["answer"] == _mock_normal_result()["answer"]
    assert result["refused"] is False
    assert len(result["sources"]) == 1
    mock_run.assert_called_once_with("DeepLabV3+ 有什么特点")


@patch("app.agents.agent_service.run_langchain_agent")
def test_service_query_passes_stripped_question(mock_run: pytest.mock.Mock) -> None:
    """query strip 前后空白。"""
    mock_run.return_value = _mock_normal_result()

    svc = RemoteSensingAgentService()
    svc.query("  DeepLabV3+  ")

    mock_run.assert_called_once_with("DeepLabV3+")


def test_service_query_empty_string() -> None:
    """空字符串 → refused=True + "请输入有效问题。" + errors=["empty question"]。"""
    svc = RemoteSensingAgentService()
    result = svc.query("")

    assert result["answer"] == EMPTY_QUESTION_ANSWER
    assert result["refused"] is True
    assert "empty question" in result["errors"]


def test_service_query_whitespace_only() -> None:
    """纯空白 → refused=True。"""
    svc = RemoteSensingAgentService()
    result = svc.query("   \n\t  ")

    assert result["refused"] is True
    assert "empty question" in result["errors"]


@patch("app.agents.agent_service.run_langchain_agent")
def test_service_query_empty_does_not_call_agent(mock_run: pytest.mock.Mock) -> None:
    """空问题不调用 agent。"""
    svc = RemoteSensingAgentService()
    svc.query("")

    mock_run.assert_not_called()


@patch("app.agents.agent_service.run_langchain_agent")
def test_service_query_exception_fallback(mock_run: pytest.mock.Mock) -> None:
    """run_langchain_agent 抛异常 → 异常兜底。"""
    mock_run.side_effect = RuntimeError("LLM 连接失败")

    svc = RemoteSensingAgentService()
    result = svc.query("正常问题")

    assert result["answer"] == ERROR_ANSWER
    assert result["refused"] is True
    assert result["agent_trace"] == ["agent_service_error"]
    assert "LLM 连接失败" in result["errors"][0]


@patch("app.agents.agent_service.run_langchain_agent")
def test_service_query_refused_result(mock_run: pytest.mock.Mock) -> None:
    """拒答结果正确透传。"""
    mock_run.return_value = _mock_refused_result()

    svc = RemoteSensingAgentService()
    result = svc.query("不相关的问题")

    assert result["refused"] is True
    assert "无法确定" in result["answer"]


# ============================================================================ #
#  API 层：POST /api/agent/query                                               #
# ============================================================================ #

@patch("app.api.agent.get_agent_service")
def test_api_query_normal(mock_get_svc: MagicMock) -> None:
    """POST /api/agent/query 正常返回所有字段。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_normal_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post(
        "/api/agent/query",
        json={"question": "请比较 U-Net 和 DeepLabV3+"},
    )

    assert resp.status_code == 200
    data = resp.json()

    # 所有必需字段存在
    required_keys = {"answer", "sources", "refused", "tool_calls", "agent_trace", "errors"}
    assert required_keys.issubset(data.keys())

    # 字段内容正确
    assert "DeepLabV3+" in data["answer"]
    assert data["refused"] is False
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "02_models.md"
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool"] == "knowledge_base_search"
    assert "agent_started" in data["agent_trace"]
    assert data["errors"] == []

    # 调用了 service（默认 include_trace=True）
    mock_svc.query.assert_called_once_with("请比较 U-Net 和 DeepLabV3+", include_trace=True, use_rerank=None, enable_cache=None)

    # Work Unit 候选对象已附带（不自动落盘）
    assert "work_unit_candidate" in data
    cand = data["work_unit_candidate"]
    assert cand is not None
    assert cand["entry"] == "agent"
    assert cand["replay_payload"]["endpoint"] == "/api/agent/query"
    assert cand["replay_payload"]["body"]["question"] == "请比较 U-Net 和 DeepLabV3+"


@patch("app.api.agent.get_agent_service")
def test_api_query_refused(mock_get_svc: MagicMock) -> None:
    """POST /api/agent/query 拒答场景。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_refused_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post(
        "/api/agent/query",
        json={"question": "今天天气"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert "无法确定" in data["answer"]
    assert data["sources"] == []


@patch("app.api.agent.get_agent_service")
def test_api_query_tool_calls_format(mock_get_svc: MagicMock) -> None:
    """tool_calls 元素包含 tool / input / status / output_summary / error / elapsed。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_normal_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    data = resp.json()
    tc = data["tool_calls"][0]
    assert "tool" in tc
    assert "input" in tc
    assert "status" in tc
    assert "output_summary" in tc
    assert "error" in tc
    assert "elapsed" in tc
    assert tc["elapsed"] == 0.123


@patch("app.api.agent.get_agent_service")
def test_api_query_sources_format(mock_get_svc: MagicMock) -> None:
    """sources 元素包含 filename / page / chunk_id / score / content_preview。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_normal_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    data = resp.json()
    src = data["sources"][0]
    assert "filename" in src
    assert "page" in src
    assert "chunk_id" in src
    assert "score" in src
    assert "content_preview" in src


def test_api_query_empty_question() -> None:
    """空问题返回 HTTP 400。"""
    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": ""})

    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]


def test_api_query_whitespace_question() -> None:
    """纯空白问题返回 HTTP 400。"""
    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "   "})

    assert resp.status_code == 400


def test_api_query_missing_question_field() -> None:
    """缺少 question 字段返回 HTTP 422（Pydantic 校验）。"""
    client = TestClient(app)
    resp = client.post("/api/agent/query", json={})

    assert resp.status_code == 422


@patch("app.api.agent.get_agent_service")
def test_api_query_service_exception_returns_500(mock_get_svc: MagicMock) -> None:
    """AgentService.query 抛异常 → HTTP 500。"""
    mock_svc = MagicMock()
    mock_svc.query.side_effect = RuntimeError("内部错误")
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    assert resp.status_code == 500
    assert "Agent 查询失败" in resp.json()["detail"]


# ============================================================================ #
#  /api/agent/query — verification 模式                                         #
# ============================================================================ #

def _mock_deferred_result() -> dict:
    """deferred 模式的 Agent 返回（verification.pending=True）。"""
    r = _mock_normal_result()
    r["verification"] = {
        "enabled": True,
        "mode": "deferred",
        "level": "lightweight",
        "pending": True,
        "verified": None,
        "confidence": None,
        "ungrounded_claims": [],
        "reason": "Evidence Verification 将在独立请求中执行。",
        "timing": {"verification_elapsed": 0.0},
    }
    return r


def _mock_off_result() -> dict:
    """off 模式的 Agent 返回（verification.enabled=False）。"""
    r = _mock_normal_result()
    r["verification"] = {
        "enabled": False,
        "mode": "off",
        "level": None,
        "pending": False,
        "verified": None,
        "confidence": None,
        "ungrounded_claims": [],
        "reason": "Evidence Verification 未启用。",
        "timing": {"verification_elapsed": 0.0},
    }
    return r


def _mock_sync_result() -> dict:
    """sync 模式的 Agent 返回（verification 包含完整结果）。"""
    r = _mock_normal_result()
    r["verification"] = {
        "enabled": True,
        "mode": "sync",
        "level": "lightweight",
        "pending": False,
        "verified": True,
        "confidence": "high",
        "ungrounded_claims": [],
        "reason": "回答有据。",
        "timing": {"verification_elapsed": 0.42},
    }
    return r


@patch("app.api.agent.get_agent_service")
def test_api_query_deferred_returns_pending(mock_get_svc: MagicMock) -> None:
    """deferred 模式下 /api/agent/query 返回 pending=True。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_deferred_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verification"]["pending"] is True
    assert data["verification"]["mode"] == "deferred"
    assert data["verification"]["enabled"] is True
    assert data["verification"]["verified"] is None
    # answer 仍然正常返回
    assert "DeepLabV3+" in data["answer"]


@patch("app.api.agent.get_agent_service")
def test_api_query_off_returns_disabled(mock_get_svc: MagicMock) -> None:
    """off 模式下 /api/agent/query 返回 enabled=False。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_off_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verification"]["enabled"] is False
    assert data["verification"]["mode"] == "off"
    assert data["verification"]["verified"] is None


@patch("app.api.agent.get_agent_service")
def test_api_query_sync_returns_full_verification(mock_get_svc: MagicMock) -> None:
    """sync 模式下 /api/agent/query 返回完整 verification 结果。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_sync_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verification"]["enabled"] is True
    assert data["verification"]["pending"] is False
    assert data["verification"]["verified"] is True
    assert data["verification"]["confidence"] == "high"


# ============================================================================ #
#  POST /api/agent/verify                                                       #
# ============================================================================ #

@patch("app.api.agent.verify_answer")
def test_api_verify_returns_verification(mock_verify: MagicMock) -> None:
    """/api/agent/verify 能返回 verification 结果。"""
    mock_verify.return_value = {
        "enabled": True,
        "mode": "sync",
        "level": "lightweight",
        "pending": False,
        "verified": True,
        "confidence": "high",
        "ungrounded_claims": [],
        "reason": "回答有据。",
        "timing": {"verification_elapsed": 0.42},
    }

    client = TestClient(app)
    resp = client.post(
        "/api/agent/verify",
        json={
            "question": "DeepLabV3+ 有什么特点",
            "answer": "DeepLabV3+ 采用 ASPP 模块。",
            "sources": [
                {
                    "filename": "02_models.md",
                    "page": 1,
                    "chunk_id": "abc123",
                    "score": 0.88,
                    "content_preview": "DeepLabV3+ 采用 ASPP...",
                }
            ],
            "tool_calls": [],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "verification" in data
    assert data["verification"]["verified"] is True
    assert data["verification"]["confidence"] == "high"

    # verify_answer 被调用
    mock_verify.assert_called_once()


@patch("app.api.agent.verify_answer")
def test_api_verify_empty_question_returns_400(mock_verify: MagicMock) -> None:
    """/api/agent/verify 空问题返回 HTTP 400。"""
    client = TestClient(app)
    resp = client.post(
        "/api/agent/verify",
        json={"question": "", "answer": "test"},
    )

    assert resp.status_code == 400
    mock_verify.assert_not_called()


@patch("app.api.agent.verify_answer")
def test_api_verify_empty_answer_returns_400(mock_verify: MagicMock) -> None:
    """/api/agent/verify 空 answer 返回 HTTP 400。"""
    client = TestClient(app)
    resp = client.post(
        "/api/agent/verify",
        json={"question": "test", "answer": ""},
    )

    assert resp.status_code == 400
    mock_verify.assert_not_called()


def test_api_verify_missing_fields_returns_422() -> None:
    """/api/agent/verify 缺少必需字段返回 HTTP 422。"""
    client = TestClient(app)
    resp = client.post("/api/agent/verify", json={})

    assert resp.status_code == 422


@patch("app.api.agent.verify_answer")
def test_api_verify_exception_returns_500(mock_verify: MagicMock) -> None:
    """verify_answer 抛异常 → HTTP 500。"""
    mock_verify.side_effect = RuntimeError("内部错误")

    client = TestClient(app)
    resp = client.post(
        "/api/agent/verify",
        json={"question": "test", "answer": "test answer"},
    )

    assert resp.status_code == 500
    assert "verification 失败" in resp.json()["detail"]


# ============================================================================ #
#  /api/chat/query 未受影响                                                    #
# ============================================================================ #

def test_chat_query_endpoint_still_exists() -> None:
    """/api/chat/query 路由仍然存在（未被删除或改名）。"""
    client = TestClient(app)

    # 空问题应返回 400（证明路由存在且正常工作）
    resp = client.post("/api/chat/query", json={"question": ""})
    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]


def test_agent_query_and_chat_query_coexist() -> None:
    """两个查询接口共存于 OpenAPI schema。"""
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    assert "/api/chat/query" in paths
    assert "/api/agent/query" in paths
    assert "/api/agent/verify" in paths


def test_health_still_works() -> None:
    """/health 端点正常。"""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ============================================================================ #
#  get_agent_service 单例                                                       #
# ============================================================================ #

def test_get_agent_service_returns_instance() -> None:
    """get_agent_service 返回 RemoteSensingAgentService 实例。"""
    svc = get_agent_service()
    assert isinstance(svc, RemoteSensingAgentService)


def test_get_agent_service_singleton() -> None:
    """get_agent_service 返回同一实例。"""
    svc1 = get_agent_service()
    svc2 = get_agent_service()
    assert svc1 is svc2


# ============================================================================ #
#  timing 字段                                                                  #
# ============================================================================ #

@patch("app.api.agent.get_agent_service")
def test_api_query_timing_present(mock_get_svc: MagicMock) -> None:
    """POST /api/agent/query 正常返回包含 timing 字段。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_normal_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    data = resp.json()
    assert "timing" in data
    timing = data["timing"]
    assert "total_elapsed" in timing
    assert "agent_invoke_elapsed" in timing
    assert "tool_search_elapsed_total" in timing
    assert timing["total_elapsed"] == 1.23
    assert timing["agent_invoke_elapsed"] == 1.0
    assert timing["tool_search_elapsed_total"] == 0.123


@patch("app.api.agent.get_agent_service")
def test_api_query_timing_on_refused(mock_get_svc: MagicMock) -> None:
    """拒答场景 timing 字段仍然透传。"""
    mock_svc = MagicMock()
    mock_svc.query.return_value = _mock_refused_result()
    mock_get_svc.return_value = mock_svc

    client = TestClient(app)
    resp = client.post("/api/agent/query", json={"question": "test"})

    data = resp.json()
    assert "timing" in data
    assert data["timing"]["total_elapsed"] == 0.5


def test_service_query_empty_returns_timing() -> None:
    """空问题时返回的 timing 字段含 3 个零值 key。"""
    svc = RemoteSensingAgentService()
    result = svc.query("")

    assert "timing" in result
    assert result["timing"]["total_elapsed"] == 0.0
    assert result["timing"]["agent_invoke_elapsed"] == 0.0
    assert result["timing"]["tool_search_elapsed_total"] == 0.0


@patch("app.agents.agent_service.run_langchain_agent")
def test_service_query_exception_returns_timing(mock_run: pytest.mock.Mock) -> None:
    """run_langchain_agent 抛异常时异常兜底也返回 timing。"""
    mock_run.side_effect = RuntimeError("LLM 连接失败")

    svc = RemoteSensingAgentService()
    result = svc.query("正常问题")

    assert "timing" in result
    assert result["timing"]["total_elapsed"] == 0.0
    assert result["timing"]["agent_invoke_elapsed"] == 0.0
    assert result["timing"]["tool_search_elapsed_total"] == 0.0
