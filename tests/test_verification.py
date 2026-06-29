"""Evidence Verification 模块测试。

测试覆盖：
1. 拒答时 verified=True（不调用 LLM）
2. LLM 返回 verified=true 时解析正常
3. LLM 返回未证实论断时解析正常
4. LLM 返回非法 JSON 时 fallback
5. verification off 模式不调用 LLM
6. 无 sources 且无 tool_calls 时 verified=False
7. 无 sources 但有结构化工具调用时正常验证
8. lightweight 模式裁剪 answer/sources/tool_calls
9. deferred pending 结构正确
10. off 模式 make_off_result 结构正确

所有测试不真实调用外部 LLM，通过注入 mock llm_client 实现。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.verification import (
    make_deferred_pending_result,
    make_off_result,
    verify_answer,
)


# ============================================================================ #
#  辅助：构造测试数据                                                           #
# ============================================================================ #

def _make_sources(n: int = 1) -> list[dict]:
    """构造来源列表。"""
    return [
        {
            "filename": f"doc_{i}.md",
            "page": i + 1,
            "chunk_id": f"abc{i:03d}",
            "score": 0.88 - i * 0.01,
            "content_preview": f"DeepLabV3+ 采用 ASPP 模块，适合多尺度特征提取。片段 {i}。" * 3,
        }
        for i in range(n)
    ]


def _make_tool_calls(n: int = 1) -> list[dict]:
    """构造工具调用列表。"""
    return [
        {
            "tool": f"tool_{i}",
            "input": f"query_{i}",
            "status": "success",
            "output_summary": f"检索到 {i + 1} 个相关片段。" * 5,
            "elapsed": 0.123,
            "error": None,
        }
        for i in range(n)
    ]


def _make_mock_llm(response_text: str) -> MagicMock:
    """构造 mock LLM 客户端，.chat() 返回指定文本。"""
    mock = MagicMock()
    mock.chat.return_value = response_text
    return mock


# 拒答文案
REFUSAL_ANSWER = "根据当前知识库内容，无法确定该问题的答案。"


# ============================================================================ #
#  测试 1：拒答时 verified=True（不调用 LLM）                                   #
# ============================================================================ #

def test_refusal_answer_returns_verified_true_without_llm() -> None:
    """拒答回答直接返回 verified=True，不调用 LLM。"""
    mock_llm = _make_mock_llm("should not be called")

    result = verify_answer(
        question="不相关的问题",
        answer=REFUSAL_ANSWER,
        sources=[],
        tool_calls=_make_tool_calls(),
        llm_client=mock_llm,
    )

    assert result["verified"] is True
    assert result["confidence"] == "high"
    assert result["ungrounded_claims"] == []
    assert "拒答" in result["reason"]
    assert "verification_elapsed" in result["timing"]

    # LLM 不应被调用
    mock_llm.chat.assert_not_called()


# ============================================================================ #
#  测试 2：LLM 返回 verified=true 时解析正常                                    #
# ============================================================================ #

def test_llm_returns_verified_true() -> None:
    """LLM 返回 verified=true 的 JSON，解析正常。"""
    llm_response = json.dumps(
        {
            "verified": True,
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "回答中的主要论断可以在 sources 或工具结果中找到依据。",
        }
    )
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="DeepLabV3+ 有什么特点",
        answer="DeepLabV3+ 采用 ASPP 模块，适合多尺度特征提取。",
        sources=_make_sources(),
        tool_calls=_make_tool_calls(),
        llm_client=mock_llm,
    )

    assert result["verified"] is True
    assert result["confidence"] == "high"
    assert result["ungrounded_claims"] == []
    assert "主要论断" in result["reason"]
    assert "verification_elapsed" in result["timing"]

    # LLM 被调用了一次
    mock_llm.chat.assert_called_once()


# ============================================================================ #
#  测试 3：LLM 返回未证实论断时解析正常                                          #
# ============================================================================ #

def test_llm_returns_ungrounded_claims() -> None:
    """LLM 返回 verified=false 并指出未证实论断。"""
    llm_response = json.dumps(
        {
            "verified": False,
            "confidence": "medium",
            "ungrounded_claims": ["DeepLabV3+ 在 LoveDA 上达到 80% mIoU"],
            "reason": "sources 中没有找到该具体数值。",
        }
    )
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="DeepLabV3+ 在 LoveDA 上的性能",
        answer="DeepLabV3+ 在 LoveDA 上达到 80% mIoU，采用 ASPP 模块。",
        sources=_make_sources(),
        tool_calls=_make_tool_calls(),
        llm_client=mock_llm,
    )

    assert result["verified"] is False
    assert result["confidence"] == "medium"
    assert len(result["ungrounded_claims"]) == 1
    assert "80% mIoU" in result["ungrounded_claims"][0]
    assert "没有找到" in result["reason"]
    assert "verification_elapsed" in result["timing"]


# ============================================================================ #
#  测试 4：LLM 返回非法 JSON 时 fallback                                        #
# ============================================================================ #

def test_llm_returns_invalid_json_fallback() -> None:
    """LLM 返回非法 JSON，fallback 为 verified=False / confidence=low。"""
    mock_llm = _make_mock_llm("这不是一个有效的 JSON 格式回复。")

    result = verify_answer(
        question="DeepLabV3+ 有什么特点",
        answer="DeepLabV3+ 采用 ASPP 模块。",
        sources=_make_sources(),
        tool_calls=_make_tool_calls(),
        llm_client=mock_llm,
    )

    assert result["verified"] is False
    assert result["confidence"] == "low"
    assert result["ungrounded_claims"] == []
    assert "解析失败" in result["reason"]
    assert "verification_elapsed" in result["timing"]


def test_llm_returns_empty_string_fallback() -> None:
    """LLM 返回空字符串，fallback 正常。"""
    mock_llm = _make_mock_llm("")

    result = verify_answer(
        question="test",
        answer="some answer",
        sources=_make_sources(),
        llm_client=mock_llm,
    )

    assert result["verified"] is False
    assert result["confidence"] == "low"


def test_llm_returns_markdown_code_block_json() -> None:
    """LLM 返回 markdown 代码块包裹的 JSON，应正确解析。"""
    llm_response = """```json
{"verified": true, "confidence": "high", "ungrounded_claims": [], "reason": "全部有据。"}
```"""
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="test",
        answer="some answer",
        sources=_make_sources(),
        llm_client=mock_llm,
    )

    assert result["verified"] is True
    assert result["confidence"] == "high"


# ============================================================================ #
#  测试 5：verification off 模式不调用 LLM（run_langchain_agent 层面）             #
# ============================================================================ #

@patch("app.agents.langchain_agent.get_settings")
@patch("app.agents.langchain_agent.verify_answer")
@patch("app.agents.langchain_agent._parse_agent_result")
@patch("app.agents.langchain_agent.get_remote_sensing_agent")
def test_verification_off_mode_does_not_call_verify(
    mock_get_agent: MagicMock,
    mock_parse: MagicMock,
    mock_verify: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """AGENT_VERIFICATION_MODE=off 时，不调用 verify_answer。"""
    from app.agents.langchain_agent import run_langchain_agent

    # 构造 mock settings，verification off
    mock_settings.return_value.enable_agent_verification = True
    mock_settings.return_value.agent_verification_mode = "off"
    mock_settings.return_value.agent_verification_level = "lightweight"
    mock_settings.return_value.llm_api_key = "fake"
    mock_settings.return_value.llm_base_url = "fake"
    mock_settings.return_value.llm_model = "fake"

    # 构造 mock agent 结果
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": []}
    mock_get_agent.return_value = mock_agent

    # 构造 mock parse 结果
    mock_parse.return_value = {
        "answer": "test answer",
        "sources": [],
        "refused": False,
        "tool_calls": [],
        "agent_trace": ["no_tool_called"],
        "errors": [],
        "_tool_search_elapsed_total": 0.0,
    }

    result = run_langchain_agent("test question")

    # verify_answer 不应被调用
    mock_verify.assert_not_called()

    # verification 字段标记为未启用
    assert result["verification"]["enabled"] is False
    assert result["verification"]["mode"] == "off"
    assert result["verification"]["verified"] is None
    assert "未启用" in result["verification"]["reason"]


@patch("app.agents.langchain_agent.get_settings")
@patch("app.agents.langchain_agent.verify_answer")
@patch("app.agents.langchain_agent._parse_agent_result")
@patch("app.agents.langchain_agent.get_remote_sensing_agent")
def test_verification_sync_mode_calls_verify_answer(
    mock_get_agent: MagicMock,
    mock_parse: MagicMock,
    mock_verify: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """AGENT_VERIFICATION_MODE=sync 时，调用 verify_answer。"""
    from app.agents.langchain_agent import run_langchain_agent

    # 构造 mock settings，verification sync
    mock_settings.return_value.enable_agent_verification = True
    mock_settings.return_value.agent_verification_mode = "sync"
    mock_settings.return_value.agent_verification_level = "lightweight"
    mock_settings.return_value.llm_api_key = "fake"
    mock_settings.return_value.llm_base_url = "fake"
    mock_settings.return_value.llm_model = "fake"

    # 构造 mock agent 结果
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": []}
    mock_get_agent.return_value = mock_agent

    # 构造 mock parse 结果
    mock_parse.return_value = {
        "answer": "test answer",
        "sources": [{"content_preview": "test", "filename": "test.md"}],
        "refused": False,
        "tool_calls": [],
        "agent_trace": [],
        "errors": [],
        "_tool_search_elapsed_total": 0.0,
    }

    # mock verify_answer 返回
    mock_verify.return_value = {
        "enabled": True,
        "mode": "sync",
        "level": "lightweight",
        "pending": False,
        "verified": True,
        "confidence": "high",
        "ungrounded_claims": [],
        "reason": "OK",
        "timing": {"verification_elapsed": 0.01},
    }

    result = run_langchain_agent("test question")

    # verify_answer 被调用
    mock_verify.assert_called_once()
    assert result["verification"]["enabled"] is True
    assert result["verification"]["verified"] is True


@patch("app.agents.langchain_agent.get_settings")
@patch("app.agents.langchain_agent.verify_answer")
@patch("app.agents.langchain_agent._parse_agent_result")
@patch("app.agents.langchain_agent.get_remote_sensing_agent")
def test_verification_deferred_mode_returns_pending(
    mock_get_agent: MagicMock,
    mock_parse: MagicMock,
    mock_verify: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """AGENT_VERIFICATION_MODE=deferred 时，不调用 verify_answer，返回 pending=True。"""
    from app.agents.langchain_agent import run_langchain_agent

    # 构造 mock settings，verification deferred
    mock_settings.return_value.enable_agent_verification = True
    mock_settings.return_value.agent_verification_mode = "deferred"
    mock_settings.return_value.agent_verification_level = "lightweight"
    mock_settings.return_value.llm_api_key = "fake"
    mock_settings.return_value.llm_base_url = "fake"
    mock_settings.return_value.llm_model = "fake"

    # 构造 mock agent 结果
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": []}
    mock_get_agent.return_value = mock_agent

    # 构造 mock parse 结果
    mock_parse.return_value = {
        "answer": "test answer",
        "sources": [{"content_preview": "test", "filename": "test.md"}],
        "refused": False,
        "tool_calls": [],
        "agent_trace": [],
        "errors": [],
        "_tool_search_elapsed_total": 0.0,
    }

    result = run_langchain_agent("test question")

    # deferred 模式下 verify_answer 不应被调用
    mock_verify.assert_not_called()

    # 返回 pending=True
    assert result["verification"]["enabled"] is True
    assert result["verification"]["mode"] == "deferred"
    assert result["verification"]["pending"] is True
    assert result["verification"]["verified"] is None
    assert "独立请求" in result["verification"]["reason"]

    # answer/sources/tool_calls 正常返回
    assert result["answer"] == "test answer"
    assert len(result["sources"]) == 1


# ============================================================================ #
#  测试 6：无 sources 且无 tool_calls 时 verified=False                          #
# ============================================================================ #

def test_no_sources_no_tool_calls_returns_false() -> None:
    """既无 sources 也无有效 tool_calls → verified=False。"""
    mock_llm = _make_mock_llm("should not be called")

    result = verify_answer(
        question="test",
        answer="some answer",
        sources=[],
        tool_calls=None,
        llm_client=mock_llm,
    )

    assert result["verified"] is False
    assert result["confidence"] == "low"
    assert "无法验证" in result["reason"]

    # LLM 不应被调用
    mock_llm.chat.assert_not_called()


def test_no_sources_but_has_tool_output_calls_llm() -> None:
    """无 sources 但有结构化工具调用（output_summary）→ 正常调用 LLM 验证。"""
    llm_response = json.dumps(
        {
            "verified": True,
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "工具输出支撑了回答。",
        }
    )
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="LoveDA 有多少类别",
        answer="LoveDA 数据集包含 7 个类别。",
        sources=[],
        tool_calls=[
            {
                "tool": "dataset_spec_lookup",
                "input": "LoveDA",
                "status": "success",
                "output_summary": "LoveDA: 7 classes",
                "elapsed": 0.05,
                "error": None,
            }
        ],
        llm_client=mock_llm,
    )

    assert result["verified"] is True
    mock_llm.chat.assert_called_once()


# ============================================================================ #
#  测试 7：LLM 调用异常时 fallback                                               #
# ============================================================================ #

def test_llm_call_exception_fallback() -> None:
    """LLM 调用抛异常 → fallback verified=False。"""
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = RuntimeError("LLM 连接失败")

    result = verify_answer(
        question="test",
        answer="some answer",
        sources=_make_sources(),
        llm_client=mock_llm,
    )

    assert result["verified"] is False
    assert result["confidence"] == "low"
    assert "调用失败" in result["reason"]
    assert "verification_elapsed" in result["timing"]


# ============================================================================ #
#  测试 8：返回格式完整性                                                        #
# ============================================================================ #

def test_result_format_completeness() -> None:
    """验证返回的 dict 包含所有必需字段。"""
    llm_response = json.dumps(
        {
            "verified": True,
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "test",
        }
    )
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="test",
        answer="test answer",
        sources=_make_sources(),
        llm_client=mock_llm,
    )

    required_keys = {
        "enabled", "mode", "level", "pending",
        "verified", "confidence", "ungrounded_claims", "reason", "timing",
    }
    assert required_keys.issubset(result.keys())
    assert isinstance(result["verified"], bool)
    assert isinstance(result["confidence"], str)
    assert isinstance(result["ungrounded_claims"], list)
    assert isinstance(result["reason"], str)
    assert isinstance(result["timing"], dict)
    assert "verification_elapsed" in result["timing"]


def test_string_verified_field_coerced() -> None:
    """LLM 返回 verified 为字符串 "true" 时，正确转换为 bool。"""
    llm_response = json.dumps(
        {
            "verified": "true",
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "test",
        }
    )
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="test",
        answer="test answer",
        sources=_make_sources(),
        llm_client=mock_llm,
    )

    assert result["verified"] is True


# ============================================================================ #
#  测试 9：lightweight 模式裁剪                                                  #
# ============================================================================ #

def test_lightweight_trims_answer() -> None:
    """lightweight 模式裁剪 answer 到 800 字。"""
    # 在末尾加一个唯一标记，确保裁剪后该标记不在 prompt 中
    long_answer = "这是一段很长的回答。" * 200 + "UNIQUE_END_MARKER_XYZ"
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer=long_answer,
        sources=_make_sources(),
        llm_client=mock_llm,
        level="lightweight",
    )

    # 验证 LLM 收到的 prompt 中 answer 被裁剪
    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    # 裁剪后不应包含末尾的唯一标记
    assert "UNIQUE_END_MARKER_XYZ" not in prompt
    # 确认裁剪确实发生了（prompt 不应包含完整的 ~2000 字）
    assert len(long_answer) > 1600


def test_lightweight_trims_sources() -> None:
    """lightweight 模式裁剪 sources 到最多 5 条。"""
    sources = _make_sources(20)  # 20 条 sources
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer="test answer",
        sources=sources,
        llm_client=mock_llm,
        level="lightweight",
    )

    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    # 只应包含前 5 个来源
    assert "doc_0" in prompt  # 第 1 个
    assert "doc_4" in prompt  # 第 5 个
    assert "doc_5" not in prompt  # 第 6 个不应出现


def test_lightweight_trims_tool_calls() -> None:
    """lightweight 模式裁剪 tool_calls 到最多 6 条。"""
    tool_calls = _make_tool_calls(20)  # 20 条 tool_calls
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer="test answer",
        sources=_make_sources(),
        tool_calls=tool_calls,
        llm_client=mock_llm,
        level="lightweight",
    )

    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    # 只应包含前 6 个工具
    assert "tool_0" in prompt  # 第 1 个
    assert "tool_5" in prompt  # 第 6 个
    assert "tool_6" not in prompt  # 第 7 个不应出现


def test_lightweight_trims_content_preview() -> None:
    """lightweight 模式裁剪每条 source 的 content_preview 到 150 字。"""
    long_preview = "A" * 500
    sources = [{"filename": "test.md", "page": 1, "chunk_id": "abc", "score": 0.9, "content_preview": long_preview}]
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer="test answer",
        sources=sources,
        llm_client=mock_llm,
        level="lightweight",
    )

    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    # 150 字 + "..." = 153，不应包含完整的 500 字
    assert "A" * 200 not in prompt


def test_lightweight_trims_output_summary() -> None:
    """lightweight 模式裁剪每条 tool_call 的 output_summary 到 200 字。"""
    long_summary = "B" * 500
    tool_calls = [{"tool": "test_tool", "status": "success", "output_summary": long_summary}]
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer="test answer",
        sources=_make_sources(),
        tool_calls=tool_calls,
        llm_client=mock_llm,
        level="lightweight",
    )

    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    # 200 字 + "..." = 203，不应包含完整的 500 字
    assert "B" * 250 not in prompt


# ============================================================================ #
#  测试 10：full 模式裁剪                                                        #
# ============================================================================ #

def test_full_mode_trims_answer() -> None:
    """full 模式裁剪 answer 到 1500 字（比 lightweight 更宽松）。"""
    # 在末尾加一个唯一标记，确保裁剪后该标记不在 prompt 中
    long_answer = "这是一段很长的回答。" * 300 + "UNIQUE_END_MARKER_XYZ"
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer=long_answer,
        sources=_make_sources(),
        llm_client=mock_llm,
        level="full",
    )

    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    # full 模式裁剪到 1500 字，不应包含末尾标记
    assert "UNIQUE_END_MARKER_XYZ" not in prompt


def test_full_mode_allows_more_sources() -> None:
    """full 模式允许最多 8 条 sources。"""
    sources = _make_sources(20)
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    verify_answer(
        question="test",
        answer="test answer",
        sources=sources,
        llm_client=mock_llm,
        level="full",
    )

    call_args = mock_llm.chat.call_args
    prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    assert "doc_0" in prompt
    assert "doc_7" in prompt  # 第 8 个应出现
    assert "doc_8" not in prompt  # 第 9 个不应出现


# ============================================================================ #
#  测试 11：make_off_result / make_deferred_pending_result                        #
# ============================================================================ #

def test_make_off_result_structure() -> None:
    """make_off_result 返回正确的 disabled 结构。"""
    result = make_off_result()

    assert result["enabled"] is False
    assert result["mode"] == "off"
    assert result["level"] is None
    assert result["pending"] is False
    assert result["verified"] is None
    assert result["confidence"] is None
    assert result["ungrounded_claims"] == []
    assert "未启用" in result["reason"]
    assert result["timing"]["verification_elapsed"] == 0.0


def test_make_deferred_pending_result_structure() -> None:
    """make_deferred_pending_result 返回正确的 pending 结构。"""
    result = make_deferred_pending_result("lightweight")

    assert result["enabled"] is True
    assert result["mode"] == "deferred"
    assert result["level"] == "lightweight"
    assert result["pending"] is True
    assert result["verified"] is None
    assert result["confidence"] is None
    assert result["ungrounded_claims"] == []
    assert "独立请求" in result["reason"]
    assert result["timing"]["verification_elapsed"] == 0.0


# ============================================================================ #
#  测试 12：level 参数覆盖 Settings 默认值                                        #
# ============================================================================ #

def test_level_param_overrides_settings() -> None:
    """显式传入 level 参数时，覆盖 Settings 中的默认值。"""
    long_answer = "这是一段很长的回答。" * 200  # ~2000 字
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    # 显式传入 full，即使 Settings 可能默认 lightweight
    result = verify_answer(
        question="test",
        answer=long_answer,
        sources=_make_sources(),
        llm_client=mock_llm,
        level="full",
    )

    assert result["level"] == "full"


def test_invalid_level_falls_back_to_lightweight() -> None:
    """无效 level 值回退到 lightweight。"""
    llm_response = json.dumps(
        {"verified": True, "confidence": "high", "ungrounded_claims": [], "reason": "ok"}
    )
    mock_llm = _make_mock_llm(llm_response)

    result = verify_answer(
        question="test",
        answer="test answer",
        sources=_make_sources(),
        llm_client=mock_llm,
        level="invalid_level",
    )

    assert result["level"] == "lightweight"
