"""Agent 类型定义测试。

验证 AgentToolCall / AgentSource / AgentRunResult / build_refusal_result。
不调用 LLM，不调用向量数据库，仅测试 Pydantic 模型行为。
"""
from __future__ import annotations

from app.agents.types import (
    DEFAULT_REFUSAL_ANSWER,
    AgentRunResult,
    AgentSource,
    AgentToolCall,
    build_refusal_result,
)


# ---------- AgentToolCall ----------

def test_tool_call_with_string_input() -> None:
    """AgentToolCall 可用字符串 input 创建。"""
    tc = AgentToolCall(
        tool="rag_search",
        input="Landsat 8 波段",
        status="success",
        output_summary="[1] 来源：landsat.pdf，第3页",
    )
    assert tc.tool == "rag_search"
    assert tc.input == "Landsat 8 波段"
    assert tc.status == "success"
    assert tc.output_summary is not None
    assert "landsat.pdf" in tc.output_summary
    assert tc.error is None


def test_tool_call_with_dict_input() -> None:
    """AgentToolCall 可用字典 input 创建。"""
    tc = AgentToolCall(
        tool="rag_search",
        input={"query": "NDVI 公式", "top_k": 5},
        status="success",
    )
    assert isinstance(tc.input, dict)
    assert tc.input["query"] == "NDVI 公式"


def test_tool_call_error_status() -> None:
    """AgentToolCall status=error 时 error 字段有值。"""
    tc = AgentToolCall(
        tool="rag_search",
        input="查询",
        status="error",
        error="连接超时",
    )
    assert tc.status == "error"
    assert tc.error == "连接超时"
    assert tc.output_summary is None


def test_tool_call_input_defaults_none() -> None:
    """AgentToolCall input 默认为 None。"""
    tc = AgentToolCall(tool="rag_search", status="success")
    assert tc.input is None
    assert tc.output_summary is None
    assert tc.error is None


# ---------- AgentSource ----------

def test_agent_source_full_fields() -> None:
    """AgentSource 所有字段正确赋值。"""
    src = AgentSource(
        filename="landsat.pdf",
        page=3,
        chunk_id="abc123def456",
        score=0.92,
        content_preview="Band 10 中心波长 10.9 μm",
    )
    assert src.filename == "landsat.pdf"
    assert src.page == 3
    assert src.chunk_id == "abc123def456"
    assert src.score == 0.92
    assert "10.9" in src.content_preview


def test_agent_source_nullable_page_score() -> None:
    """AgentSource page 和 score 允许 None。"""
    src = AgentSource(
        filename="unknown.txt",
        chunk_id="xyz",
        content_preview="内容",
    )
    assert src.page is None
    assert src.score is None


# ---------- AgentRunResult ----------

def test_run_result_all_fields() -> None:
    """AgentRunResult 所有字段均可正确赋值。"""
    sources = [
        AgentSource(filename="a.pdf", page=1, chunk_id="c1", score=0.8, content_preview="..."),
    ]
    tool_calls = [
        AgentToolCall(tool="rag_search", input="问题", status="success", output_summary="结果"),
    ]
    trace = ["收到问题", "调用 rag_search", "生成回答"]
    errors = ["某步骤警告"]

    result = AgentRunResult(
        answer="最终回答",
        sources=sources,
        refused=False,
        tool_calls=tool_calls,
        agent_trace=trace,
        errors=errors,
    )

    assert result.answer == "最终回答"
    assert len(result.sources) == 1
    assert result.sources[0].filename == "a.pdf"
    assert result.refused is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "rag_search"
    assert len(result.agent_trace) == 3
    assert len(result.errors) == 1


def test_run_result_defaults() -> None:
    """AgentRunResult 默认值：所有列表为空，refused=False。"""
    result = AgentRunResult(answer="回答")
    assert result.answer == "回答"
    assert result.sources == []
    assert result.refused is False
    assert result.tool_calls == []
    assert result.agent_trace == []
    assert result.errors == []


# ---------- build_refusal_result ----------

def test_build_refusal_result_default() -> None:
    """build_refusal_result 默认拒答文案正确，refused=True。"""
    result = build_refusal_result()
    assert result.refused is True
    assert result.answer == DEFAULT_REFUSAL_ANSWER
    assert "无法确定" in result.answer


def test_build_refusal_result_custom_reason() -> None:
    """build_refusal_result 接受自定义拒答文案。"""
    custom = "知识库中没有相关遥感卫星数据。"
    result = build_refusal_result(reason=custom)
    assert result.refused is True
    assert result.answer == custom


def test_build_refusal_result_empty_lists() -> None:
    """build_refusal_result 所有列表字段为空。"""
    result = build_refusal_result()
    assert result.sources == []
    assert result.tool_calls == []
    assert result.agent_trace == []
    assert result.errors == []


def test_default_refusal_answer_constant() -> None:
    """DEFAULT_REFUSAL_ANSWER 常量值正确。"""
    assert DEFAULT_REFUSAL_ANSWER == "根据当前知识库内容，无法确定该问题的答案。"
