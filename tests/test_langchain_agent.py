"""LangChain Agent 测试。

覆盖：
1. build_chat_model 配置读取 + 异常处理（原有测试保留）
2. build_remote_sensing_agent / build_agent 组装逻辑
3. run_langchain_agent 正常回答解析
4. run_langchain_agent tool_calls 解析
5. run_langchain_agent sources 提取
6. run_langchain_agent 异常兜底
7. _parse_agent_result 拒答 / 无工具调用场景

不真实调用外部 LLM，通过 mock agent + 真实 LangChain 消息类验证解析逻辑。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ============================================================================ #
#  autouse fixture：mock verify_answer，避免测试中真实调用 LLM API               #
# ============================================================================ #

@pytest.fixture(autouse=True)
def _mock_verification():
    """自动 mock verify_answer，避免测试中真实调用 LLM API。

    Evidence Verification 的逻辑在 test_verification.py 中独立、完整地测试，
    此处仅需验证 Agent 解析 / 执行 / timing 逻辑，不需要真实 LLM 调用。
    """
    with patch("app.agents.langchain_agent.verify_answer") as mock_verify:
        mock_verify.return_value = {
            "verified": True,
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "mock: 测试环境不调用真实 LLM",
            "timing": {"verification_elapsed": 0.001},
        }
        yield


# ============================================================================ #
#  辅助：构造 mock agent 返回的 messages                                        #
# ============================================================================ #

def _make_tool_result_json(
    success: bool = True,
    sources: list[dict] | None = None,
    summary: str = "检索到 2 个相关片段",
    error: str | None = None,
    search_elapsed: float = 0.123,
) -> str:
    """构造 knowledge_base_search 工具返回的 JSON 字符串。"""
    data: dict = {
        "success": success,
        "query": "test",
        "contexts": [],
        "sources": sources or [],
        "summary": summary,
        "timing": {"search_elapsed": search_elapsed},
    }
    if error:
        data["error"] = error
    return json.dumps(data, ensure_ascii=False)


def _make_sample_sources() -> list[dict]:
    """构造示例 sources。"""
    return [
        {
            "filename": "02_models.md",
            "page": 1,
            "chunk_id": "abc123def456",
            "score": 0.85,
            "content_preview": "DeepLabV3+ 采用 ASPP 模块...",
        },
        {
            "filename": "02_models.md",
            "page": 2,
            "chunk_id": "xyz789abc012",
            "score": 0.78,
            "content_preview": "U-Net 采用编码器-解码器结构...",
        },
    ]


def _make_mock_agent_response_normal() -> dict:
    """构造正常回答的 agent 返回（含工具调用）。"""
    sources = _make_sample_sources()
    tool_json = _make_tool_result_json(success=True, sources=sources)
    return {
        "messages": [
            HumanMessage(content="DeepLabV3+ 和 U-Net 在遥感分割中有什么区别？"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "knowledge_base_search",
                        "args": {"query": "DeepLabV3+ U-Net 遥感分割"},
                        "id": "call_001",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=tool_json,
                tool_call_id="call_001",
                name="knowledge_base_search",
            ),
            AIMessage(content="DeepLabV3+ 和 U-Net 的主要区别如下：\n1. 结构设计...\n[来源：02_models.md，第1页，abc123def456]"),
        ]
    }


def _make_mock_agent_response_refused() -> dict:
    """构造拒答场景的 agent 返回（工具返回空结果）。"""
    tool_json = _make_tool_result_json(
        success=False, sources=[], summary="未检索到相关知识库内容"
    )
    return {
        "messages": [
            HumanMessage(content="今天天气怎么样？"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "knowledge_base_search",
                        "args": {"query": "今天天气"},
                        "id": "call_002",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=tool_json,
                tool_call_id="call_002",
                name="knowledge_base_search",
            ),
            AIMessage(content="根据当前知识库内容，无法确定该问题的答案。"),
        ]
    }


def _make_mock_agent_response_no_tool() -> dict:
    """构造无工具调用的 agent 返回。"""
    return {
        "messages": [
            HumanMessage(content="你好"),
            AIMessage(content="你好！我是遥感语义分割领域研究助手，请问有什么可以帮您？"),
        ]
    }


def _make_mock_agent_response_error_tool() -> dict:
    """构造工具调用失败的 agent 返回。"""
    tool_json = _make_tool_result_json(
        success=False, sources=[], summary="检索失败", error="连接超时"
    )
    return {
        "messages": [
            HumanMessage(content="查询数据集信息"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "knowledge_base_search",
                        "args": {"query": "数据集"},
                        "id": "call_003",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=tool_json,
                tool_call_id="call_003",
                name="knowledge_base_search",
            ),
            AIMessage(content="检索过程中出现问题，请稍后重试。"),
        ]
    }


# ============================================================================ #
#  build_chat_model 配置读取（原有测试）                                         #
# ============================================================================ #

def test_build_chat_model_reads_config_from_settings() -> None:
    """build_chat_model 从 get_settings() 读取全部配置，不硬编码。"""
    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.ChatOpenAI") as mock_chat_cls,
    ):
        mock_s = MagicMock()
        mock_s.llm_api_key = "test-key-123"
        mock_s.llm_base_url = "https://api.deepseek.com/v1"
        mock_s.llm_model = "deepseek-chat"
        mock_s.agent_max_tokens = 1000
        mock_settings.return_value = mock_s

        from app.agents.langchain_agent import build_chat_model

        llm = build_chat_model()

        mock_chat_cls.assert_called_once()
        _, kwargs = mock_chat_cls.call_args
        assert kwargs["api_key"] == "test-key-123"
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"
        assert kwargs["model"] == "deepseek-chat"
        assert llm is mock_chat_cls.return_value


def test_build_chat_model_temperature_is_zero() -> None:
    """build_chat_model 使用 temperature=0。"""
    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.ChatOpenAI") as mock_chat_cls,
    ):
        mock_s = MagicMock()
        mock_s.llm_api_key = "key"
        mock_s.llm_base_url = "https://api.test.com/v1"
        mock_s.llm_model = "model"
        mock_s.agent_max_tokens = 1000
        mock_settings.return_value = mock_s

        from app.agents.langchain_agent import build_chat_model

        build_chat_model()

        _, kwargs = mock_chat_cls.call_args
        assert kwargs["temperature"] == 0


def test_build_chat_model_raises_without_api_key() -> None:
    """缺少 LLM_API_KEY 时抛出 ValueError。"""
    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.ChatOpenAI"),
    ):
        mock_s = MagicMock()
        mock_s.llm_api_key = ""
        mock_s.llm_base_url = "https://api.test.com/v1"
        mock_s.llm_model = "test-model"
        mock_settings.return_value = mock_s

        from app.agents.langchain_agent import build_chat_model

        with pytest.raises(ValueError, match="LLM_API_KEY"):
            build_chat_model()


def test_build_chat_model_raises_without_base_url() -> None:
    """缺少 LLM_BASE_URL 时抛出 ValueError。"""
    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.ChatOpenAI"),
    ):
        mock_s = MagicMock()
        mock_s.llm_api_key = "test-key"
        mock_s.llm_base_url = ""
        mock_s.llm_model = "test-model"
        mock_settings.return_value = mock_s

        from app.agents.langchain_agent import build_chat_model

        with pytest.raises(ValueError, match="LLM_BASE_URL"):
            build_chat_model()


def test_build_chat_model_raises_without_model() -> None:
    """缺少 LLM_MODEL 时抛出 ValueError。"""
    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.ChatOpenAI"),
    ):
        mock_s = MagicMock()
        mock_s.llm_api_key = "test-key"
        mock_s.llm_base_url = "https://api.test.com/v1"
        mock_s.llm_model = ""
        mock_settings.return_value = mock_s

        from app.agents.langchain_agent import build_chat_model

        with pytest.raises(ValueError, match="LLM_MODEL"):
            build_chat_model()


def test_build_llm_is_alias_of_build_chat_model() -> None:
    """build_llm 是 build_chat_model 的别名。"""
    from app.agents.langchain_agent import build_chat_model, build_llm

    assert build_llm is build_chat_model


# ============================================================================ #
#  build_remote_sensing_agent / build_agent 组装逻辑                           #
# ============================================================================ #

def test_build_remote_sensing_agent_calls_create_agent() -> None:
    """build_remote_sensing_agent 调用 create_agent 并传入 model + tools + prompt。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model") as mock_build,
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_llm = MagicMock(name="chat_model")
        mock_build.return_value = mock_llm
        mock_create.return_value = MagicMock(name="compiled_agent")

        from app.agents.langchain_agent import build_remote_sensing_agent

        agent = build_remote_sensing_agent()

        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["model"] is mock_llm
        assert len(kwargs["tools"]) >= 1
        assert "遥感语义分割" in kwargs["system_prompt"]
        assert agent is mock_create.return_value


def test_build_agent_is_alias() -> None:
    """build_agent 委托给 build_remote_sensing_agent。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model"),
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_create.return_value = MagicMock()

        from app.agents.langchain_agent import build_agent, build_remote_sensing_agent

        a1 = build_agent()
        assert mock_create.call_count == 1

        a2 = build_remote_sensing_agent()
        assert mock_create.call_count == 2


def test_build_remote_sensing_agent_uses_default_tools() -> None:
    """默认使用 DEFAULT_TOOLS（含 knowledge_base_search）。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model"),
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_create.return_value = MagicMock()

        from app.agents.langchain_agent import DEFAULT_TOOLS, build_remote_sensing_agent

        build_remote_sensing_agent()

        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == DEFAULT_TOOLS


def test_build_remote_sensing_agent_accepts_custom_tools() -> None:
    """支持传入自定义工具列表。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model"),
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        custom_tool = MagicMock(name="custom")
        mock_create.return_value = MagicMock()

        from app.agents.langchain_agent import build_remote_sensing_agent

        build_remote_sensing_agent(tools=[custom_tool])

        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == [custom_tool]


# ============================================================================ #
#  run_langchain_agent — 正常回答解析                                           #
# ============================================================================ #

def test_run_langchain_agent_normal_answer() -> None:
    """正常回答场景：answer 正确提取，refused=False。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("DeepLabV3+ 和 U-Net 区别", agent=mock_agent)

    assert isinstance(result, dict)
    assert "DeepLabV3+" in result["answer"]
    assert result["refused"] is False
    assert result["errors"] == []


def test_run_langchain_agent_answer_has_source_citation() -> None:
    """正常回答的 answer 末尾包含来源引用。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    assert "[来源：" in result["answer"]


# ============================================================================ #
#  run_langchain_agent — tool_calls 解析                                        #
# ============================================================================ #

def test_run_langchain_agent_tool_calls_extracted() -> None:
    """tool_calls 正确解析，包含 tool / input / status / output_summary。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["tool"] == "knowledge_base_search"
    assert "DeepLabV3+" in tc["input"]
    assert tc["status"] == "success"
    assert tc["output_summary"] is not None
    assert "检索到" in tc["output_summary"]
    assert not tc["error"]  # trim_tool_calls 将 None 转为空字符串


def test_run_langchain_agent_tool_calls_error_status() -> None:
    """工具调用失败时 tool_calls status=error，error 字段有值。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_error_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("数据集", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["status"] == "error"
    assert "连接超时" in tc["error"]


# ============================================================================ #
#  run_langchain_agent — sources 提取                                           #
# ============================================================================ #

def test_run_langchain_agent_sources_extracted() -> None:
    """sources 从工具返回 JSON 中正确提取。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    assert len(result["sources"]) == 2
    src = result["sources"][0]
    assert src["filename"] == "02_models.md"
    assert src["page"] == 1
    assert src["chunk_id"] == "abc123def456"
    assert src["score"] == 0.85
    assert "DeepLabV3+" in src["content_preview"]


def test_run_langchain_agent_sources_empty_when_refused() -> None:
    """拒答场景 sources 为空。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_refused()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("天气", agent=mock_agent)

    assert result["sources"] == []


# ============================================================================ #
#  run_langchain_agent — agent_trace                                            #
# ============================================================================ #

def test_run_langchain_agent_trace_with_tool() -> None:
    """工具被调用时 agent_trace 包含完整执行轨迹。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    trace = result["agent_trace"]
    assert "agent_started" in trace
    assert "tool_called:knowledge_base_search" in trace
    assert "tool_result_parsed" in trace
    assert "agent_finished" in trace
    # 顺序正确
    assert trace.index("agent_started") < trace.index("tool_called:knowledge_base_search")
    assert trace.index("tool_called:knowledge_base_search") < trace.index("tool_result_parsed")
    assert trace.index("tool_result_parsed") < trace.index("agent_finished")


def test_run_langchain_agent_trace_no_tool() -> None:
    """无工具调用时 agent_trace 为 ["no_tool_called"]。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_no_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("你好", agent=mock_agent)

    assert result["agent_trace"] == ["no_tool_called"]


# ============================================================================ #
#  run_langchain_agent — 拒答场景                                               #
# ============================================================================ #

def test_run_langchain_agent_refused_detection() -> None:
    """answer 包含拒答文案时 refused=True。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_refused()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("天气", agent=mock_agent)

    assert result["refused"] is True
    assert "无法确定" in result["answer"]


# ============================================================================ #
#  run_langchain_agent — 异常兜底                                               #
# ============================================================================ #

def test_run_langchain_agent_exception_fallback() -> None:
    """agent.invoke 抛异常时返回异常兜底结构。"""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("LLM 连接超时")

    from app.agents.langchain_agent import ERROR_ANSWER, run_langchain_agent

    result = run_langchain_agent("任意问题", agent=mock_agent)

    assert result["answer"] == ERROR_ANSWER
    assert result["refused"] is True
    assert result["sources"] == []
    assert result["tool_calls"] == []
    assert result["agent_trace"] == ["agent_started", "agent_error"]
    assert len(result["errors"]) == 1
    assert "LLM 连接超时" in result["errors"][0]


def test_run_langchain_agent_exception_with_different_errors() -> None:
    """不同异常类型都能被兜底捕获。"""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = ValueError("配置错误")

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert result["refused"] is True
    assert "配置错误" in result["errors"][0]


# ============================================================================ #
#  _parse_agent_result — 直接单元测试                                           #
# ============================================================================ #

def test_parse_result_empty_messages() -> None:
    """空 messages 返回空结果。"""
    from app.agents.langchain_agent import _parse_agent_result

    result = _parse_agent_result({"messages": []})

    assert result["answer"] == ""
    assert result["sources"] == []
    assert result["tool_calls"] == []
    assert result["agent_trace"] == ["no_tool_called"]
    assert result["refused"] is False


def test_parse_result_dict_output_keys() -> None:
    """_parse_agent_result 返回 dict 包含所有必需 key。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    required_keys = {
        "answer",
        "sources",
        "refused",
        "tool_calls",
        "agent_trace",
        "errors",
        "timing",
    }
    assert required_keys.issubset(result.keys())


# ============================================================================ #
#  get_remote_sensing_agent — @lru_cache 单例                                   #
# ============================================================================ #

def test_get_remote_sensing_agent_uses_lru_cache() -> None:
    """get_remote_sensing_agent 使用 @lru_cache，连续调用返回同一实例。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model") as mock_build,
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_build.return_value = MagicMock(name="chat_model")
        mock_create.return_value = MagicMock(name="compiled_agent")

        from app.agents.langchain_agent import get_remote_sensing_agent

        get_remote_sensing_agent.cache_clear()

        a1 = get_remote_sensing_agent()
        a2 = get_remote_sensing_agent()

        assert a1 is a2
        assert mock_create.call_count == 1
        assert mock_build.call_count == 1


def test_get_remote_sensing_agent_cache_clear_rebuilds() -> None:
    """cache_clear 后再次调用会重新构建 Agent。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model") as mock_build,
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_build.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        from app.agents.langchain_agent import get_remote_sensing_agent

        get_remote_sensing_agent.cache_clear()
        get_remote_sensing_agent()
        assert mock_create.call_count == 1

        get_remote_sensing_agent.cache_clear()
        get_remote_sensing_agent()
        assert mock_create.call_count == 2


# ============================================================================ #
#  run_langchain_agent — timing 字段                                            #
# ============================================================================ #

def test_run_langchain_agent_returns_timing() -> None:
    """run_langchain_agent 返回结果包含 timing 字段，含 3 个 float key。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert "timing" in result
    timing = result["timing"]
    assert isinstance(timing, dict)
    assert "total_elapsed" in timing
    assert "agent_invoke_elapsed" in timing
    assert "tool_search_elapsed_total" in timing
    assert isinstance(timing["total_elapsed"], float)
    assert isinstance(timing["agent_invoke_elapsed"], float)
    assert isinstance(timing["tool_search_elapsed_total"], float)


def test_run_langchain_agent_timing_on_exception() -> None:
    """agent.invoke 抛异常时 timing 字段仍然存在。"""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("超时")

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert "timing" in result
    timing = result["timing"]
    assert timing["agent_invoke_elapsed"] == 0.0
    assert timing["tool_search_elapsed_total"] == 0.0
    assert isinstance(timing["total_elapsed"], float)


def test_run_langchain_agent_tool_search_elapsed_total_aggregated() -> None:
    """多个工具调用的 elapsed 被汇总到 timing.tool_search_elapsed_total。"""
    sources = _make_sample_sources()
    tool_json = _make_tool_result_json(success=True, sources=sources, search_elapsed=0.5)

    # 构造含两次工具调用的 agent 返回
    return_val = {
        "messages": [
            HumanMessage(content="test"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "knowledge_base_search",
                        "args": {"query": "query1"},
                        "id": "call_a",
                        "type": "tool_call",
                    },
                    {
                        "name": "knowledge_base_search",
                        "args": {"query": "query2"},
                        "id": "call_b",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(content=tool_json, tool_call_id="call_a", name="knowledge_base_search"),
            ToolMessage(content=tool_json, tool_call_id="call_b", name="knowledge_base_search"),
            AIMessage(content="两次检索后的回答"),
        ]
    }

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = return_val

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    # search_elapsed=0.5 x 2 = 1.0
    assert result["timing"]["tool_search_elapsed_total"] == 0.5 + 0.5


# ============================================================================ #
#  run_langchain_agent — tool_calls 包含 elapsed                                #
# ============================================================================ #

def test_run_langchain_agent_tool_calls_contain_elapsed() -> None:
    """tool_calls 元素包含 elapsed 字段（从 timing.search_elapsed 提取）。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert "elapsed" in tc
    # _make_tool_result_json 中 search_elapsed=0.123
    assert tc["elapsed"] == 0.123


def test_run_langchain_agent_tool_calls_elapsed_none_on_error() -> None:
    """工具调用失败时 elapsed 仍有值（来自 timing.search_elapsed）。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_error_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert "elapsed" in tc


# ============================================================================ #
#  _parse_agent_result — timing 相关                                            #
# ============================================================================ #

def test_parse_result_has_timing_key() -> None:
    """_parse_agent_result 返回的 dict 中不包含 timing（由 run_langchain_agent 添加）。"""
    from app.agents.langchain_agent import _parse_agent_result

    result = _parse_agent_result({"messages": []})

    # _parse_agent_result 内部不含 timing，run_langchain_agent 负责添加
    assert "timing" not in result
    # 但包含 _tool_search_elapsed_total
    assert "_tool_search_elapsed_total" in result


def test_parse_result_tool_search_elapsed_total_zero_on_no_tool() -> None:
    """无工具调用时 _tool_search_elapsed_total=0.0。"""
    from app.agents.langchain_agent import _parse_agent_result

    result = _parse_agent_result(_make_mock_agent_response_no_tool())

    assert result["_tool_search_elapsed_total"] == 0.0


# ============================================================================ #
#  Multi-Tool 架构测试（Block 3 新增）                                           #
# ============================================================================ #
#                                                                              #
#  验证 DEFAULT_TOOLS 包含 6 个工具、_parse_agent_result 能正确处理             #
#  多个不同工具的调用、结构化工具无 sources 不报错、                             #
#  _extract_tool_input 对不同工具提取正确参数。                                  #
# ============================================================================ #

def _make_domain_tool_json(
    tool_name: str,
    success: bool = True,
    summary: str = "",
    data: dict | None = None,
    error: str | None = None,
) -> str:
    """构造结构化工具（非 knowledge_base_search）返回的 JSON 字符串。"""
    payload: dict = {
        "success": success,
        "tool": tool_name,
        "query": "test",
        "data": data,
        "summary": summary,
    }
    if error:
        payload["error"] = error
    return json.dumps(payload, ensure_ascii=False)


def _make_calculator_json(
    metric: str = "IoU",
    result: float = 0.7273,
    inputs: dict | None = None,
    summary: str = "IoU = 80 / (80 + 10 + 20) = 0.7273",
) -> str:
    """构造 metrics_calculator 返回的 JSON 字符串。"""
    return json.dumps({
        "success": True,
        "tool": "metrics_calculator",
        "metric": metric,
        "inputs": inputs or {"TP": 80, "FP": 10, "FN": 20},
        "result": result,
        "formula": f"{metric} = TP / (TP + FP + FN)",
        "summary": summary,
    }, ensure_ascii=False)


def _make_mock_agent_response_multi_tool() -> dict:
    """构造多工具调用的 agent 返回（dataset_spec_lookup + knowledge_base_search）。"""
    kb_json = _make_tool_result_json(success=True, sources=_make_sample_sources())
    dataset_json = _make_domain_tool_json(
        tool_name="dataset_spec_lookup",
        success=True,
        summary="找到 LoveDA 数据集的结构化信息。",
        data={"name": "LoveDA", "classes": 7, "resolution": "0.3 m"},
    )
    return {
        "messages": [
            HumanMessage(content="LoveDA 数据集有什么特点？"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dataset_spec_lookup",
                        "args": {"dataset_name": "LoveDA"},
                        "id": "call_m1",
                        "type": "tool_call",
                    },
                    {
                        "name": "knowledge_base_search",
                        "args": {"query": "LoveDA 数据集特点"},
                        "id": "call_m2",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(content=dataset_json, tool_call_id="call_m1", name="dataset_spec_lookup"),
            ToolMessage(content=kb_json, tool_call_id="call_m2", name="knowledge_base_search"),
            AIMessage(content=(
                "LoveDA 是一个遥感语义分割数据集，包含 7 个类别...\n"
                "（使用工具：dataset_spec_lookup, knowledge_base_search）"
            )),
        ]
    }


def _make_mock_agent_response_calculator_only() -> dict:
    """构造仅调用 metrics_calculator 的 agent 返回（无 knowledge_base_search）。"""
    calc_json = _make_calculator_json()
    return {
        "messages": [
            HumanMessage(content="IoU, TP=80, FP=10, FN=20"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "metrics_calculator",
                        "args": {"metric_name": "IoU", "values": "TP=80, FP=10, FN=20"},
                        "id": "call_c1",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(content=calc_json, tool_call_id="call_c1", name="metrics_calculator"),
            AIMessage(content="IoU = 0.7273\n（使用工具：metrics_calculator）"),
        ]
    }


def _make_mock_agent_response_model_comparison() -> dict:
    """构造 model_comparison_table 调用的 agent 返回。"""
    comp_json = json.dumps({
        "success": True,
        "tool": "model_comparison_table",
        "query": "U-Net, DeepLabV3+",
        "models_found": ["U-Net", "DeepLabV3+"],
        "models_not_found": [],
        "comparison": [
            {"name": "U-Net", "architecture_type": "Encoder-Decoder"},
            {"name": "DeepLabV3+", "architecture_type": "Dilated Convolution"},
        ],
        "summary": "找到 2 个模型进行对比。",
    }, ensure_ascii=False)
    return {
        "messages": [
            HumanMessage(content="比较 U-Net 和 DeepLabV3+"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "model_comparison_table",
                        "args": {"models": "U-Net, DeepLabV3+"},
                        "id": "call_cmp1",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(content=comp_json, tool_call_id="call_cmp1", name="model_comparison_table"),
            AIMessage(content="U-Net 和 DeepLabV3+ 的对比如下...\n（使用工具：model_comparison_table）"),
        ]
    }


# ============================================================================ #
#  DEFAULT_TOOLS 包含 6 个工具                                                   #
# ============================================================================ #

def test_default_tools_has_seven_tools() -> None:
    """DEFAULT_TOOLS 包含 7 个工具。"""
    from app.agents.langchain_agent import DEFAULT_TOOLS

    assert len(DEFAULT_TOOLS) == 7


def test_default_tools_names() -> None:
    """DEFAULT_TOOLS 中包含正确的 7 个工具名。"""
    from app.agents.langchain_agent import DEFAULT_TOOLS

    names = {t.name for t in DEFAULT_TOOLS}
    expected = {
        "knowledge_base_search",
        "plan_and_search",
        "dataset_overview",
        "dataset_spec_lookup",
        "model_comparison_table",
        "metric_formula_lookup",
        "metrics_calculator",
    }
    assert names == expected


def test_default_tools_all_have_invoke() -> None:
    """每个工具都有 invoke 方法。"""
    from app.agents.langchain_agent import DEFAULT_TOOLS

    for t in DEFAULT_TOOLS:
        assert hasattr(t, "invoke")


def test_build_agent_uses_seven_tools() -> None:
    """build_remote_sensing_agent 传入 7 个工具给 create_agent。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model"),
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_create.return_value = MagicMock()

        from app.agents.langchain_agent import build_remote_sensing_agent

        build_remote_sensing_agent()

        _, kwargs = mock_create.call_args
        assert len(kwargs["tools"]) == 7


def test_build_remote_sensing_agent_log_shows_seven() -> None:
    """构建日志显示工具数=7。"""
    with (
        patch("app.agents.langchain_agent.build_chat_model"),
        patch("app.agents.langchain_agent.create_agent") as mock_create,
    ):
        mock_create.return_value = MagicMock()

        from app.agents.langchain_agent import build_remote_sensing_agent

        build_remote_sensing_agent()

        # 日志中的工具数应为 7
        _, kwargs = mock_create.call_args
        tools = kwargs["tools"]
        assert len(tools) == 7


# ============================================================================ #
#  多工具调用解析                                                                #
# ============================================================================ #

def test_multi_tool_two_different_tools_parsed() -> None:
    """Agent 调用 dataset_spec_lookup + knowledge_base_search，两个工具都被解析。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_multi_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("LoveDA 特点", agent=mock_agent)

    assert len(result["tool_calls"]) == 2
    tool_names = [tc["tool"] for tc in result["tool_calls"]]
    assert "dataset_spec_lookup" in tool_names
    assert "knowledge_base_search" in tool_names


def test_multi_tool_trace_records_all() -> None:
    """agent_trace 记录所有工具调用。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_multi_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    trace = result["agent_trace"]
    assert "tool_called:dataset_spec_lookup" in trace
    assert "tool_called:knowledge_base_search" in trace
    assert trace.count("tool_result_parsed") == 2
    assert "agent_finished" in trace


def test_multi_tool_sources_only_from_kb() -> None:
    """sources 只来自 knowledge_base_search，不来自结构化工具。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_multi_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    # _make_sample_sources 返回 2 条 sources，全部来自 knowledge_base_search
    assert len(result["sources"]) == 2
    assert result["sources"][0]["filename"] == "02_models.md"


def test_domain_tool_no_sources_no_error() -> None:
    """结构化工具（metrics_calculator）无 sources，解析正常不报错。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_calculator_only()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("IoU TP=80 FP=10 FN=20", agent=mock_agent)

    assert result["sources"] == []
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["tool"] == "metrics_calculator"
    assert tc["status"] == "success"
    assert "0.7273" in tc["output_summary"]


def test_model_comparison_tool_parsed() -> None:
    """model_comparison_table 工具结果正确解析。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_model_comparison()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("比较 U-Net 和 DeepLabV3+", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["tool"] == "model_comparison_table"
    assert tc["status"] == "success"
    assert "2 个模型" in tc["output_summary"]
    # model_comparison_table 不返回 sources
    assert result["sources"] == []


def test_domain_tool_elapsed_is_none() -> None:
    """结构化工具的 elapsed 为 None（无 timing.search_elapsed）。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_calculator_only()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["elapsed"] is None


def test_multi_tool_tool_search_elapsed_only_from_kb() -> None:
    """timing.tool_search_elapsed_total 只累计 knowledge_base_search 的耗时。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_multi_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    # 只有 knowledge_base_search 贡献 timing，_make_tool_result_json 默认 0.123
    assert result["timing"]["tool_search_elapsed_total"] == 0.123


# ============================================================================ #
#  _extract_tool_input 辅助函数                                                  #
# ============================================================================ #

def test_extract_tool_input_knowledge_base_search() -> None:
    """knowledge_base_search 提取 query 参数。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("knowledge_base_search", {"query": "DeepLabV3+"})
    assert result == "DeepLabV3+"


def test_extract_tool_input_plan_and_search() -> None:
    """plan_and_search 提取 query 参数。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("plan_and_search", {"query": "复杂遥感分割问题"})
    assert result == "复杂遥感分割问题"


def test_extract_tool_input_dataset_spec_lookup() -> None:
    """dataset_spec_lookup 提取 dataset_name 参数。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("dataset_spec_lookup", {"dataset_name": "LoveDA"})
    assert result == "LoveDA"


def test_extract_tool_input_model_comparison_table() -> None:
    """model_comparison_table 提取 models 参数。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("model_comparison_table", {"models": "U-Net, DeepLabV3+"})
    assert result == "U-Net, DeepLabV3+"


def test_extract_tool_input_metric_formula_lookup() -> None:
    """metric_formula_lookup 提取 metric_name 参数。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("metric_formula_lookup", {"metric_name": "mIoU"})
    assert result == "mIoU"


def test_extract_tool_input_metrics_calculator() -> None:
    """metrics_calculator 提取 metric_name + values。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input(
        "metrics_calculator",
        {"metric_name": "IoU", "values": "TP=80, FP=10, FN=20"},
    )
    assert "IoU" in result
    assert "TP=80" in result


def test_extract_tool_input_unknown_tool_fallback() -> None:
    """未知工具名回退到 str(args)。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("unknown_tool", {"foo": "bar"})
    assert "foo" in result
    assert "bar" in result


def test_extract_tool_input_non_dict_args() -> None:
    """非 dict 参数直接 str()。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("any_tool", "raw_string")
    assert result == "raw_string"


# ============================================================================ #
#  Multi-Tool 场景下 tool_calls input 提取                                       #
# ============================================================================ #

def test_multi_tool_input_extraction() -> None:
    """多工具场景下每个 tool_call 的 input 正确提取。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_multi_tool()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    tc_dict = {tc["tool"]: tc for tc in result["tool_calls"]}
    assert tc_dict["dataset_spec_lookup"]["input"] == "LoveDA"
    assert "LoveDA" in tc_dict["knowledge_base_search"]["input"]


def test_calculator_input_extraction() -> None:
    """metrics_calculator 的 input 包含 metric_name 和 values。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_calculator_only()

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    tc = result["tool_calls"][0]
    assert "IoU" in tc["input"]
    assert "TP=80" in tc["input"]


# ============================================================================ #
#  Multi-Tool 异常兜底                                                          #
# ============================================================================ #

def test_multi_tool_exception_fallback_still_works() -> None:
    """多工具架构下异常兜底仍然正常。"""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("LLM 不可用")

    from app.agents.langchain_agent import ERROR_ANSWER, run_langchain_agent

    result = run_langchain_agent("test", agent=mock_agent)

    assert result["answer"] == ERROR_ANSWER
    assert result["refused"] is True
    assert result["sources"] == []
    assert result["tool_calls"] == []
    assert "agent_error" in result["agent_trace"]


# ============================================================================ #
#  Prompt 更新验证                                                               #
# ============================================================================ #

def test_system_prompt_mentions_seven_tools() -> None:
    """系统提示词提及全部 7 个工具。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    assert "knowledge_base_search" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "plan_and_search" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "dataset_overview" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "dataset_spec_lookup" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "model_comparison_table" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "metric_formula_lookup" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "metrics_calculator" in REMOTE_SENSING_AGENT_SYSTEM_PROMPT


def test_system_prompt_has_tool_selection_guide() -> None:
    """系统提示词包含工具选择指南。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "选择指南" in prompt or "优先使用" in prompt
    assert "不要编造" in prompt
    assert "使用工具" in prompt


def test_system_prompt_has_plan_and_search_boundary() -> None:
    """系统提示词包含 plan_and_search 使用边界说明。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    # 应包含不适合 plan_and_search 的问题类型说明
    assert "不适合" in prompt or "不要" in prompt
    # 应包含 success=false 的处理指引
    assert "success=false" in prompt or "门控拦截" in prompt


def test_system_prompt_no_hard_tool_limit() -> None:
    """系统提示词不包含硬限制（如"最多只能调用一次工具"）。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "最多只能调用一次" not in prompt
    assert "最多调用一次" not in prompt
    assert "只能调用一次工具" not in prompt
    assert "最多两次" not in prompt
    assert "必须调用" not in prompt
    assert "禁止调用" not in prompt


# ============================================================================ #
#  Block 5: Agent 工具选择 Prompt 优化验证                                      #
# ============================================================================ #

def test_system_prompt_has_tool_selection_principles() -> None:
    """系统提示词包含工具选择原则章节。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "工具选择原则" in prompt


def test_system_prompt_prefers_minimal_tools() -> None:
    """系统提示词强调优先选择最少数量的工具。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "最少数量" in prompt or "最少" in prompt


def test_system_prompt_avoids_redundant_structured_tool_calls() -> None:
    """系统提示词包含避免重复调用结构化工具的指引。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    # 应包含不要重复调用同一结构化工具的说明
    assert "重复调用" in prompt or "连续调用" in prompt


def test_system_prompt_has_decision_table() -> None:
    """系统提示词包含工具决策表。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "工具决策表" in prompt or "决策表" in prompt


def test_system_prompt_has_answer_length_requirements() -> None:
    """系统提示词包含默认回答长度要求。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    # 默认 600-900 字
    assert "600" in prompt
    assert "900" in prompt
    # 简单问题 300-600 字
    assert "300" in prompt


def test_system_prompt_has_no_expand_unrelated_background() -> None:
    """系统提示词要求不扩写无关背景。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "不扩写" in prompt or "无关背景" in prompt


def test_system_prompt_success_false_guidance() -> None:
    """系统提示词包含 success=false 时换用其他工具的指引。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    assert "success=false" in prompt


def test_system_prompt_does_not_mandate_specific_tool() -> None:
    """系统提示词不强制要求必须调用某个特定工具。"""
    from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT

    prompt = REMOTE_SENSING_AGENT_SYSTEM_PROMPT
    # 不应出现 "必须调用 dataset_overview" / "必须使用 plan_and_search" 等
    for tool in [
        "必须调用 dataset_overview",
        "必须调用 dataset_spec_lookup",
        "必须调用 plan_and_search",
        "必须调用 knowledge_base_search",
        "必须使用 plan_and_search",
    ]:
        assert tool not in prompt


# ============================================================================ #
#  Verification 模式测试（off / sync / deferred）                                #
# ============================================================================ #
#                                                                              #
#  验证 run_langchain_agent 根据 agent_verification_mode 选择是否调用           #
#  verify_answer，同时 Agent 工具调用解析不受影响。                              #
# ============================================================================ #


def test_run_langchain_agent_deferred_skips_verification() -> None:
    """deferred 模式下 run_langchain_agent 不调用 verify_answer，返回 pending=True。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.verify_answer") as mock_verify,
    ):
        mock_settings.return_value.agent_verification_mode = "deferred"
        mock_settings.return_value.agent_verification_level = "lightweight"
        mock_settings.return_value.enable_agent_verification = True
        mock_settings.return_value.llm_api_key = "fake"
        mock_settings.return_value.llm_base_url = "fake"
        mock_settings.return_value.llm_model = "fake"

        from app.agents.langchain_agent import run_langchain_agent

        result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    # verify_answer 不应被调用
    mock_verify.assert_not_called()

    # verification 结构正确
    assert result["verification"]["enabled"] is True
    assert result["verification"]["mode"] == "deferred"
    assert result["verification"]["pending"] is True
    assert result["verification"]["verified"] is None

    # Agent 工具调用解析不受影响
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "knowledge_base_search"
    assert "DeepLabV3+" in result["answer"]


def test_run_langchain_agent_sync_calls_verification() -> None:
    """sync 模式下 run_langchain_agent 调用 verify_answer。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.verify_answer") as mock_verify,
    ):
        mock_settings.return_value.agent_verification_mode = "sync"
        mock_settings.return_value.agent_verification_level = "lightweight"
        mock_settings.return_value.enable_agent_verification = True
        mock_settings.return_value.llm_api_key = "fake"
        mock_settings.return_value.llm_base_url = "fake"
        mock_settings.return_value.llm_model = "fake"

        mock_verify.return_value = {
            "enabled": True,
            "mode": "sync",
            "level": "lightweight",
            "pending": False,
            "verified": True,
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "mock verify",
            "timing": {"verification_elapsed": 0.01},
        }

        from app.agents.langchain_agent import run_langchain_agent

        result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    # verify_answer 被调用
    mock_verify.assert_called_once()
    assert result["verification"]["pending"] is False
    assert result["verification"]["verified"] is True

    # Agent 工具调用解析不受影响
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "knowledge_base_search"


def test_run_langchain_agent_off_skips_verification() -> None:
    """off 模式下 run_langchain_agent 不调用 verify_answer，返回 enabled=False。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = _make_mock_agent_response_normal()

    with (
        patch("app.agents.langchain_agent.get_settings") as mock_settings,
        patch("app.agents.langchain_agent.verify_answer") as mock_verify,
    ):
        mock_settings.return_value.agent_verification_mode = "off"
        mock_settings.return_value.agent_verification_level = "lightweight"
        mock_settings.return_value.enable_agent_verification = True
        mock_settings.return_value.llm_api_key = "fake"
        mock_settings.return_value.llm_base_url = "fake"
        mock_settings.return_value.llm_model = "fake"

        from app.agents.langchain_agent import run_langchain_agent

        result = run_langchain_agent("DeepLabV3+", agent=mock_agent)

    mock_verify.assert_not_called()
    assert result["verification"]["enabled"] is False
    assert result["verification"]["mode"] == "off"

    # Agent 工具调用解析不受影响
    assert len(result["tool_calls"]) == 1
    assert "DeepLabV3+" in result["answer"]


# ============================================================================ #
#  Block 2: deduplicate_and_trim_sources / trim_tool_calls                     #
# ============================================================================ #

from app.agents.langchain_agent import (  # noqa: E402
    deduplicate_and_trim_sources,
    trim_tool_calls,
)


# ---------- deduplicate_and_trim_sources ----------

def test_dedup_sources_empty_returns_empty_list() -> None:
    """空 sources 返回空列表。"""
    assert deduplicate_and_trim_sources([]) == []
    assert deduplicate_and_trim_sources(None) == []  # type: ignore[arg-type]


def test_dedup_sources_by_chunk_id() -> None:
    """按 chunk_id 去重，保留最高分条目。"""
    sources = [
        {"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.7, "content_preview": "低分"},
        {"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "高分"},
        {"filename": "b.md", "page": 2, "chunk_id": "c2", "score": 0.8, "content_preview": "不同chunk"},
    ]
    result = deduplicate_and_trim_sources(sources)

    assert len(result) == 2
    # c1 去重后保留高分条目
    c1 = [s for s in result if s["chunk_id"] == "c1"][0]
    assert c1["score"] == 0.9
    assert c1["content_preview"] == "高分"


def test_dedup_sources_max_five() -> None:
    """最多保留 5 条 sources。"""
    sources = [
        {"filename": f"f{i}.md", "page": i, "chunk_id": f"c{i}", "score": 0.9 - i * 0.05, "content_preview": "..."}
        for i in range(10)
    ]
    result = deduplicate_and_trim_sources(sources)

    assert len(result) == 5
    # 按分数降序，保留前 5
    assert result[0]["score"] >= result[-1]["score"]


def test_dedup_sources_content_preview_truncated() -> None:
    """content_preview 被截断到 preview_max_chars。"""
    long_preview = "X" * 500
    sources = [
        {"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": long_preview},
    ]
    result = deduplicate_and_trim_sources(sources, preview_max_chars=150)

    assert len(result) == 1
    preview = result[0]["content_preview"]
    assert len(preview) == 153  # 150 + "..."
    assert preview.endswith("...")


def test_dedup_sources_sorted_by_score_desc() -> None:
    """sources 按 score 降序排列。"""
    sources = [
        {"filename": "low.md", "page": 1, "chunk_id": "low", "score": 0.3, "content_preview": "..."},
        {"filename": "high.md", "page": 2, "chunk_id": "high", "score": 0.95, "content_preview": "..."},
        {"filename": "mid.md", "page": 3, "chunk_id": "mid", "score": 0.6, "content_preview": "..."},
    ]
    result = deduplicate_and_trim_sources(sources)

    assert len(result) == 3
    assert result[0]["score"] == 0.95
    assert result[1]["score"] == 0.6
    assert result[2]["score"] == 0.3


def test_dedup_sources_no_chunk_id_fallback_dedup() -> None:
    """无 chunk_id 时按 filename + page + content_preview 去重。"""
    sources = [
        {"filename": "a.md", "page": 1, "score": 0.9, "content_preview": "相同内容"},
        {"filename": "a.md", "page": 1, "score": 0.8, "content_preview": "相同内容"},
        {"filename": "a.md", "page": 2, "score": 0.7, "content_preview": "不同页面"},
    ]
    result = deduplicate_and_trim_sources(sources)

    assert len(result) == 2


def test_dedup_sources_no_score_does_not_crash() -> None:
    """无 score 字段时不崩溃。"""
    sources = [
        {"filename": "a.md", "page": 1, "chunk_id": "c1", "content_preview": "..."},
        {"filename": "b.md", "page": 2, "chunk_id": "c2", "content_preview": "..."},
    ]
    result = deduplicate_and_trim_sources(sources)

    assert len(result) == 2


def test_dedup_sources_non_dict_elements_skipped() -> None:
    """非 dict 元素被跳过。"""
    sources = [
        {"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."},
        "not a dict",
        None,
        42,
    ]
    result = deduplicate_and_trim_sources(sources)

    assert len(result) == 1


def test_dedup_sources_custom_max_sources() -> None:
    """自定义 max_sources 参数。"""
    sources = [
        {"filename": f"f{i}.md", "page": i, "chunk_id": f"c{i}", "score": 0.9 - i * 0.1, "content_preview": "..."}
        for i in range(5)
    ]
    result = deduplicate_and_trim_sources(sources, max_sources=3)

    assert len(result) == 3


# ---------- trim_tool_calls ----------

def test_trim_tool_calls_empty_returns_empty_list() -> None:
    """空 tool_calls 返回空列表。"""
    assert trim_tool_calls([]) == []
    assert trim_tool_calls(None) == []  # type: ignore[arg-type]


def test_trim_tool_calls_output_summary_truncated() -> None:
    """output_summary 被截断到 200 字符。"""
    long_summary = "X" * 500
    tool_calls = [
        {"tool": "knowledge_base_search", "input": "test", "status": "success", "output_summary": long_summary, "elapsed": 0.1, "error": None},
    ]
    result = trim_tool_calls(tool_calls)

    assert len(result) == 1
    summary = result[0]["output_summary"]
    assert len(summary) == 203  # 200 + "..."
    assert summary.endswith("...")


def test_trim_tool_calls_error_truncated() -> None:
    """error 被截断到 200 字符。"""
    long_error = "E" * 500
    tool_calls = [
        {"tool": "knowledge_base_search", "input": "test", "status": "error", "output_summary": "fail", "elapsed": None, "error": long_error},
    ]
    result = trim_tool_calls(tool_calls)

    assert len(result) == 1
    err = result[0]["error"]
    assert len(err) == 203  # 200 + "..."
    assert err.endswith("...")


def test_trim_tool_calls_only_keeps_specified_fields() -> None:
    """只保留 tool / input / status / output_summary / elapsed / error。"""
    tool_calls = [
        {
            "tool": "dataset_spec_lookup",
            "input": "LoveDA",
            "status": "success",
            "output_summary": "找到 LoveDA",
            "elapsed": 0.05,
            "error": None,
            "raw_output": "huge blob that should be removed",
            "messages": ["should", "not", "leak"],
        },
    ]
    result = trim_tool_calls(tool_calls)

    assert len(result) == 1
    item = result[0]
    keep_keys = {"tool", "input", "status", "output_summary", "elapsed", "error"}
    assert set(item.keys()) == keep_keys
    assert "raw_output" not in item
    assert "messages" not in item


def test_trim_tool_calls_missing_fields_does_not_crash() -> None:
    """字段缺失时不崩溃，补默认值。"""
    tool_calls = [
        {"tool": "test_tool"},
        {},
    ]
    result = trim_tool_calls(tool_calls)

    assert len(result) == 2
    assert result[0]["tool"] == "test_tool"
    assert result[0]["status"] == "unknown"
    assert result[1]["tool"] == "unknown"
    assert result[1]["status"] == "unknown"


def test_trim_tool_calls_non_dict_elements_skipped() -> None:
    """非 dict 元素被跳过。"""
    tool_calls = [
        {"tool": "test", "status": "success"},
        "not a dict",
        None,
    ]
    result = trim_tool_calls(tool_calls)

    assert len(result) == 1


# ---------- _parse_agent_result applies dedup + trim ----------

def test_parse_agent_result_applies_source_dedup() -> None:
    """_parse_agent_result 对 sources 去重。"""
    from app.agents.langchain_agent import _parse_agent_result

    dup_sources = [
        {"filename": "a.md", "page": 1, "chunk_id": "dup", "score": 0.7, "content_preview": "低分"},
        {"filename": "a.md", "page": 1, "chunk_id": "dup", "score": 0.95, "content_preview": "高分"},
    ]
    tool_json = _make_tool_result_json(sources=dup_sources)
    messages = [
        HumanMessage(content="test"),
        AIMessage(content="", tool_calls=[{"name": "knowledge_base_search", "args": {"query": "test"}, "id": "tc1"}]),
        ToolMessage(content=tool_json, tool_call_id="tc1", name="knowledge_base_search"),
        AIMessage(content="回答完毕"),
    ]

    result = _parse_agent_result({"messages": messages})
    assert len(result["sources"]) == 1
    assert result["sources"][0]["score"] == 0.95


def test_parse_agent_result_trims_tool_output_summary() -> None:
    """_parse_agent_result 对 tool_calls output_summary 截断。"""
    from app.agents.langchain_agent import _parse_agent_result

    long_summary = "S" * 500
    tool_json = _make_tool_result_json(summary=long_summary)
    messages = [
        HumanMessage(content="test"),
        AIMessage(content="", tool_calls=[{"name": "knowledge_base_search", "args": {"query": "test"}, "id": "tc1"}]),
        ToolMessage(content=tool_json, tool_call_id="tc1", name="knowledge_base_search"),
        AIMessage(content="done"),
    ]

    result = _parse_agent_result({"messages": messages})
    output_summary = result["tool_calls"][0]["output_summary"]
    assert len(output_summary) <= 203  # 200 + "..."


def test_parse_agent_result_sources_max_five() -> None:
    """_parse_agent_result 最多保留 5 条 sources。"""
    from app.agents.langchain_agent import _parse_agent_result

    many_sources = [
        {"filename": f"f{i}.md", "page": i, "chunk_id": f"c{i}", "score": 0.9 - i * 0.05, "content_preview": "..."}
        for i in range(10)
    ]
    tool_json = _make_tool_result_json(sources=many_sources)
    messages = [
        HumanMessage(content="test"),
        AIMessage(content="", tool_calls=[{"name": "knowledge_base_search", "args": {"query": "test"}, "id": "tc1"}]),
        ToolMessage(content=tool_json, tool_call_id="tc1", name="knowledge_base_search"),
        AIMessage(content="done"),
    ]

    result = _parse_agent_result({"messages": messages})
    assert len(result["sources"]) <= 5


# ============================================================================ #
#  Block 3: dataset_overview 工具集成测试                                        #
# ============================================================================ #

def test_default_tools_includes_dataset_overview() -> None:
    """DEFAULT_TOOLS 包含 dataset_overview。"""
    from app.agents.langchain_agent import DEFAULT_TOOLS

    names = {t.name for t in DEFAULT_TOOLS}
    assert "dataset_overview" in names


def test_parse_result_dataset_overview_tool_called() -> None:
    """_parse_agent_result 能解析 tool_called:dataset_overview。"""
    from app.agents.langchain_agent import _parse_agent_result

    # dataset_overview 返回的 JSON（无 sources / 无 timing）
    overview_json = json.dumps({
        "success": True,
        "tool": "dataset_overview",
        "query": "数据集特点",
        "summary": "遥感语义分割数据集通常具有高空间分辨率等特点。",
        "common_features": ["高空间分辨率", "类别不平衡"],
        "common_challenges": ["城乡场景差异"],
        "related_datasets": ["LoveDA", "iSAID"],
        "summary_short": "已总结共同特点。",
    }, ensure_ascii=False)

    messages = [
        HumanMessage(content="数据集有什么特点"),
        AIMessage(content="", tool_calls=[{"name": "dataset_overview", "args": {"query": "数据集特点"}, "id": "tc1"}]),
        ToolMessage(content=overview_json, tool_call_id="tc1", name="dataset_overview"),
        AIMessage(content="遥感语义分割数据集有以下特点..."),
    ]

    result = _parse_agent_result({"messages": messages})

    # tool_calls 被正确解析
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "dataset_overview"
    assert result["tool_calls"][0]["status"] == "success"

    # agent_trace 包含 tool_called:dataset_overview
    assert any("dataset_overview" in t for t in result["agent_trace"])

    # answer 被提取
    assert "遥感语义分割数据集" in result["answer"]


def test_parse_result_dataset_overview_no_sources_no_crash() -> None:
    """dataset_overview 没有 sources 时不崩溃，sources 为空列表。"""
    from app.agents.langchain_agent import _parse_agent_result

    overview_json = json.dumps({
        "success": True,
        "tool": "dataset_overview",
        "query": "",
        "summary": "总结。",
        "common_features": ["高分辨率"],
        "common_challenges": ["小目标"],
        "related_datasets": ["LoveDA"],
        "summary_short": "总结。",
    }, ensure_ascii=False)

    messages = [
        HumanMessage(content="test"),
        AIMessage(content="", tool_calls=[{"name": "dataset_overview", "args": {"query": ""}, "id": "tc1"}]),
        ToolMessage(content=overview_json, tool_call_id="tc1", name="dataset_overview"),
        AIMessage(content="回答"),
    ]

    result = _parse_agent_result({"messages": messages})

    # sources 应为空列表，不崩溃
    assert result["sources"] == []
    assert len(result["tool_calls"]) == 1


def test_extract_tool_input_dataset_overview() -> None:
    """_extract_tool_input 能提取 dataset_overview 的 query 参数。"""
    from app.agents.langchain_agent import _extract_tool_input

    result = _extract_tool_input("dataset_overview", {"query": "数据集共性"})
    assert "数据集共性" in result


def test_run_langchain_agent_dataset_overview_no_sources() -> None:
    """run_langchain_agent 在 Agent 调用 dataset_overview（无 sources）时正常返回。"""
    overview_json = json.dumps({
        "success": True,
        "tool": "dataset_overview",
        "query": "数据集特点",
        "summary": "遥感数据集具有高分辨率特点。",
        "common_features": ["高分辨率"],
        "common_challenges": ["小目标"],
        "related_datasets": ["LoveDA"],
        "summary_short": "总结。",
    }, ensure_ascii=False)

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [
            HumanMessage(content="数据集有什么特点"),
            AIMessage(content="", tool_calls=[{"name": "dataset_overview", "args": {"query": "数据集特点"}, "id": "tc1"}]),
            ToolMessage(content=overview_json, tool_call_id="tc1", name="dataset_overview"),
            AIMessage(content="遥感数据集通常具有高空间分辨率等特点。"),
        ]
    }

    from app.agents.langchain_agent import run_langchain_agent

    result = run_langchain_agent("数据集有什么特点", agent=mock_agent)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "dataset_overview"
    assert result["sources"] == []
    assert "遥感数据集" in result["answer"]
