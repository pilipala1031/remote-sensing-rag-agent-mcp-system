"""Plan-and-Search 查询分解工具测试。

覆盖：
- 工具元信息（@tool 装饰、名称、英文 description）
- _parse_decomposition 多格式解析（纯 JSON / markdown 代码块 / 嵌入文本 / 异常回退）
- _decompose_query 成功 / 失败回退
- _merge_search_results 去重 / 排序 / 耗时累加
- plan_and_search 端到端（mock 分解 + mock 检索）
- 空 query / 异常兜底 / JSON 合法性
- timing 结构兼容 parse_tool_result

不调用真实 LLM 和向量数据库，全部通过 mock 验证逻辑。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.planning_tools import (
    _merge_search_results,
    _parse_decomposition,
    plan_and_search,
    should_use_plan_and_search,
)


# ============================================================================ #
#  辅助：构造 mock 数据                                                          #
# ============================================================================ #

def _make_search_json(
    success: bool = True,
    sources: list[dict] | None = None,
    contexts: list[dict] | None = None,
    summary: str = "检索到 2 个相关片段",
    search_elapsed: float = 0.123,
) -> str:
    """构造 _cached_search 返回的 JSON 字符串。"""
    if sources is None:
        sources = [
            {
                "filename": "02_models.md",
                "page": 1,
                "chunk_id": "chunk_001",
                "score": 0.85,
                "content_preview": "DeepLabV3+ 采用 ASPP 模块...",
            },
            {
                "filename": "02_models.md",
                "page": 2,
                "chunk_id": "chunk_002",
                "score": 0.78,
                "content_preview": "U-Net 采用编码器-解码器结构...",
            },
        ]
    if contexts is None:
        contexts = [
            {
                "source_id": "source_1",
                "content": "DeepLabV3+ 采用 ASPP 模块...",
                "source": "02_models.md，第1页，chunk_id=chunk_001",
                "score": 0.85,
            },
            {
                "source_id": "source_2",
                "content": "U-Net 采用编码器-解码器结构...",
                "source": "02_models.md，第2页，chunk_id=chunk_002",
                "score": 0.78,
            },
        ]
    data: dict = {
        "success": success,
        "query": "test",
        "contexts": contexts,
        "sources": sources,
        "summary": summary,
        "timing": {"search_elapsed": search_elapsed},
    }
    return json.dumps(data, ensure_ascii=False)


def _make_empty_search_json(search_elapsed: float = 0.05) -> str:
    """构造检索结果为空的 JSON。"""
    return json.dumps({
        "success": False,
        "query": "test",
        "contexts": [],
        "sources": [],
        "summary": "未检索到相关知识库内容",
        "timing": {"search_elapsed": search_elapsed},
        "error": None,
    }, ensure_ascii=False)


# ============================================================================ #
#  工具元信息                                                                    #
# ============================================================================ #

def test_plan_and_search_has_correct_name() -> None:
    """plan_and_search 具有正确的 name 属性。"""
    assert hasattr(plan_and_search, "name")
    assert plan_and_search.name == "plan_and_search"


def test_plan_and_search_has_invoke() -> None:
    """plan_and_search 具有 invoke 方法。"""
    assert hasattr(plan_and_search, "invoke")
    assert callable(plan_and_search.invoke)


def test_plan_and_search_description_in_english() -> None:
    """description 为英文，帮助 LLM 判断何时调用。"""
    desc = plan_and_search.description
    assert isinstance(desc, str)
    assert len(desc) > 20
    desc_lower = desc.lower()
    assert "decompose" in desc_lower or "sub-quer" in desc_lower
    assert "remote sensing" in desc_lower or "knowledge base" in desc_lower


# ============================================================================ #
#  _parse_decomposition                                                          #
# ============================================================================ #

def test_parse_decomposition_clean_json() -> None:
    """纯 JSON 格式正确解析。"""
    content = '{"sub_queries": ["子查询1", "子查询2", "子查询3"]}'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 3
    assert result[0] == "子查询1"
    assert result[2] == "子查询3"


def test_parse_decomposition_markdown_fenced() -> None:
    """Markdown 代码块包裹的 JSON 正确解析。"""
    content = '```json\n{"sub_queries": ["查询A", "查询B"]}\n```'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 2
    assert result[0] == "查询A"


def test_parse_decomposition_markdown_no_lang() -> None:
    """不带 json 标记的代码块也能解析。"""
    content = '```\n{"sub_queries": ["查询A"]}\n```'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 1
    assert result[0] == "查询A"


def test_parse_decomposition_embedded_in_text() -> None:
    """JSON 嵌在自然语言中也能提取。"""
    content = '好的，我来分解：\n{"sub_queries": ["方面1", "方面2"]}\n以上就是分解结果。'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 2
    assert result[0] == "方面1"


def test_parse_decomposition_empty_content() -> None:
    """空内容回退到 fallback_query。"""
    result = _parse_decomposition("", fallback_query="原始查询")
    assert result == ["原始查询"]


def test_parse_decomposition_none_content() -> None:
    """None 内容回退到 fallback_query。"""
    result = _parse_decomposition(None, fallback_query="原始查询")
    assert result == ["原始查询"]


def test_parse_decomposition_invalid_json() -> None:
    """非法 JSON 回退到 fallback_query。"""
    result = _parse_decomposition("这不是JSON", fallback_query="原始查询")
    assert result == ["原始查询"]


def test_parse_decomposition_missing_key() -> None:
    """缺少 sub_queries 键回退到 fallback_query。"""
    content = '{"queries": ["q1", "q2"]}'
    result = _parse_decomposition(content, fallback_query="原始查询")
    assert result == ["原始查询"]


def test_parse_decomposition_max_four_queries() -> None:
    """超过 4 个子查询时截断为 4 个。"""
    content = '{"sub_queries": ["q1", "q2", "q3", "q4", "q5", "q6"]}'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 4


def test_parse_decomposition_dedup() -> None:
    """重复子查询去重（大小写不敏感）。"""
    content = '{"sub_queries": ["查询A", "查询a", "查询B", "查询b"]}'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 2
    assert result[0] == "查询A"
    assert result[1] == "查询B"


def test_parse_decomposition_filters_empty_strings() -> None:
    """空字符串子查询被过滤。"""
    content = '{"sub_queries": ["查询A", "", "  ", "查询B"]}'
    result = _parse_decomposition(content, fallback_query="原始查询")

    assert len(result) == 2


def test_parse_decomposition_no_fallback_query() -> None:
    """无 fallback_query 且解析失败返回空列表。"""
    result = _parse_decomposition("garbage", fallback_query="")
    assert result == []


def test_parse_decomposition_sub_queries_not_list() -> None:
    """sub_queries 不是列表时回退。"""
    content = '{"sub_queries": "not a list"}'
    result = _parse_decomposition(content, fallback_query="原始查询")
    assert result == ["原始查询"]


# ============================================================================ #
#  _decompose_query                                                              #
# ============================================================================ #

@patch("app.agents.planning_tools._decompose_query")
def test_decompose_query_success_via_mock(mock_decompose: MagicMock) -> None:
    """通过 mock 验证 _decompose_query 返回值结构。"""
    mock_decompose.return_value = (["子查询1", "子查询2"], 0.5)

    # 直接调用 mock 验证
    result = mock_decompose("复杂问题")
    assert len(result[0]) == 2
    assert result[1] == 0.5


def test_decompose_query_real_success() -> None:
    """_decompose_query 成功调用 LLM 并解析子查询。"""
    mock_response = MagicMock()
    mock_response.content = '{"sub_queries": ["遥感分割挑战", "常用分割模型", "评价指标"]}'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("app.agents.langchain_agent.get_agent_llm", return_value=mock_llm):
        from app.agents.planning_tools import _decompose_query

        sub_queries, elapsed = _decompose_query("遥感分割的挑战和模型")

    assert isinstance(sub_queries, list)
    assert len(sub_queries) == 3
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
    mock_llm.invoke.assert_called_once()


def test_decompose_query_llm_failure_fallback() -> None:
    """LLM 调用失败时回退为 [原始查询]。"""
    with patch(
        "app.agents.langchain_agent.get_agent_llm",
        side_effect=RuntimeError("LLM 不可用"),
    ):
        from app.agents.planning_tools import _decompose_query

        sub_queries, elapsed = _decompose_query("复杂问题")

    assert sub_queries == ["复杂问题"]
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_decompose_query_returns_timing() -> None:
    """_decompose_query 返回的 elapsed 为 float 类型。"""
    mock_response = MagicMock()
    mock_response.content = '{"sub_queries": ["查询1"]}'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("app.agents.langchain_agent.get_agent_llm", return_value=mock_llm):
        from app.agents.planning_tools import _decompose_query

        _, elapsed = _decompose_query("test query")

    assert isinstance(elapsed, float)


# ============================================================================ #
#  _merge_search_results                                                         #
# ============================================================================ #

def test_merge_no_duplicates() -> None:
    """两个子查询结果无重复 chunk_id，全部保留。"""
    json1 = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "内容1", "source": "a.md，第1页，chunk_id=c1", "score": 0.9}],
        search_elapsed=0.1,
    )
    json2 = _make_search_json(
        sources=[{"filename": "b.md", "page": 2, "chunk_id": "c2", "score": 0.8, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "内容2", "source": "b.md，第2页，chunk_id=c2", "score": 0.8}],
        search_elapsed=0.2,
    )

    contexts, sources, elapsed = _merge_search_results([json1, json2])

    assert len(contexts) == 2
    assert len(sources) == 2
    # 按分数降序排列
    assert contexts[0]["score"] == 0.9
    assert contexts[1]["score"] == 0.8
    # source_id 重新编号
    assert contexts[0]["source_id"] == "source_1"
    assert contexts[1]["source_id"] == "source_2"
    # 耗时累加（浮点精度容差）
    assert elapsed == pytest.approx(0.3)


def test_merge_with_duplicates_keeps_highest_score() -> None:
    """相同 chunk_id 出现在多个子查询结果中，保留最高分。"""
    json1 = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "dup", "score": 0.7, "content_preview": "低分"}],
        contexts=[{"source_id": "source_1", "content": "低分内容", "source": "a.md，第1页，chunk_id=dup", "score": 0.7}],
        search_elapsed=0.1,
    )
    json2 = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "dup", "score": 0.95, "content_preview": "高分"}],
        contexts=[{"source_id": "source_1", "content": "高分内容", "source": "a.md，第1页，chunk_id=dup", "score": 0.95}],
        search_elapsed=0.2,
    )

    contexts, sources, elapsed = _merge_search_results([json1, json2])

    assert len(contexts) == 1
    assert contexts[0]["score"] == 0.95
    assert sources[0]["score"] == 0.95
    assert "高分" in sources[0]["content_preview"]


def test_merge_all_failed_returns_empty() -> None:
    """所有子查询都检索失败，返回空列表。"""
    empty_json = _make_empty_search_json(search_elapsed=0.05)
    contexts, sources, elapsed = _merge_search_results([empty_json, empty_json])

    assert contexts == []
    assert sources == []
    assert elapsed == 0.1


def test_merge_empty_input() -> None:
    """空输入返回空列表。"""
    contexts, sources, elapsed = _merge_search_results([])

    assert contexts == []
    assert sources == []
    assert elapsed == 0.0


def test_merge_invalid_json_skipped() -> None:
    """非法 JSON 条目被跳过，不报错。"""
    json1 = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "内容1", "source": "a.md，第1页，chunk_id=c1", "score": 0.9}],
        search_elapsed=0.1,
    )

    contexts, sources, elapsed = _merge_search_results(["NOT JSON", json1])

    assert len(contexts) == 1
    assert sources[0]["chunk_id"] == "c1"


def test_merge_sorted_by_score_desc() -> None:
    """合并结果按 score 降序排列。"""
    json1 = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "low", "score": 0.3, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "低分", "source": "a.md，第1页，chunk_id=low", "score": 0.3}],
        search_elapsed=0.1,
    )
    json2 = _make_search_json(
        sources=[{"filename": "b.md", "page": 2, "chunk_id": "high", "score": 0.95, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "高分", "source": "b.md，第2页，chunk_id=high", "score": 0.95}],
        search_elapsed=0.1,
    )
    json3 = _make_search_json(
        sources=[{"filename": "c.md", "page": 3, "chunk_id": "mid", "score": 0.6, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "中分", "source": "c.md，第3页，chunk_id=mid", "score": 0.6}],
        search_elapsed=0.1,
    )

    contexts, sources, elapsed = _merge_search_results([json1, json2, json3])

    assert len(contexts) == 3
    assert contexts[0]["score"] == 0.95
    assert contexts[1]["score"] == 0.6
    assert contexts[2]["score"] == 0.3


def test_merge_elapsed_aggregation() -> None:
    """耗时正确累加。"""
    json1 = _make_search_json(search_elapsed=0.5)
    json2 = _make_search_json(search_elapsed=0.3)

    _, _, elapsed = _merge_search_results([json1, json2])

    assert elapsed == 0.8


# ============================================================================ #
#  plan_and_search 端到端（mock 分解 + mock 检索）                                #
# ============================================================================ #

@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_success(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """端到端成功：分解 → 检索 → 合并。"""
    mock_decompose.return_value = (["子查询1", "子查询2"], 0.5)
    mock_cached.side_effect = [
        _make_search_json(
            sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."}],
            contexts=[{"source_id": "source_1", "content": "内容1", "source": "a.md，第1页，chunk_id=c1", "score": 0.9}],
            search_elapsed=0.1,
        ),
        _make_search_json(
            sources=[{"filename": "b.md", "page": 2, "chunk_id": "c2", "score": 0.8, "content_preview": "..."}],
            contexts=[{"source_id": "source_1", "content": "内容2", "source": "b.md，第2页，chunk_id=c2", "score": 0.8}],
            search_elapsed=0.2,
        ),
    ]

    raw = plan_and_search.invoke({"query": "对比复杂遥感问题"})

    assert isinstance(raw, str)
    data = json.loads(raw)

    assert data["success"] is True
    assert data["tool"] == "plan_and_search"
    assert data["query"] == "对比复杂遥感问题"
    assert len(data["sub_queries"]) == 2
    assert len(data["contexts"]) == 2
    assert len(data["sources"]) == 2
    assert "分解为 2 个子查询" in data["summary"]
    # timing 结构
    assert "timing" in data
    assert data["timing"]["planning_elapsed"] == 0.5
    assert data["timing"]["search_elapsed"] == 0.3
    assert isinstance(data["timing"]["total_elapsed"], float)
    # mock 验证
    assert mock_decompose.call_count == 1
    assert mock_cached.call_count == 2


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_dedup(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """两个子查询返回相同 chunk_id，合并后去重。"""
    dup_json = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "same", "score": 0.85, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "内容", "source": "a.md，第1页，chunk_id=same", "score": 0.85}],
        search_elapsed=0.1,
    )
    mock_decompose.return_value = (["查询1", "查询2"], 0.3)
    mock_cached.side_effect = [dup_json, dup_json]

    raw = plan_and_search.invoke({"query": "对比复杂问题"})

    data = json.loads(raw)
    assert data["success"] is True
    assert len(data["contexts"]) == 1
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "same"


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_all_empty_results(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """所有子查询都返回空 → success=False。"""
    empty_json = _make_empty_search_json()
    mock_decompose.return_value = (["查询1", "查询2"], 0.2)
    mock_cached.side_effect = [empty_json, empty_json]

    raw = plan_and_search.invoke({"query": "对比不相关的问题"})

    data = json.loads(raw)
    assert data["success"] is False
    assert data["contexts"] == []
    assert data["sources"] == []
    assert "未检索到" in data["summary"]
    assert len(data["sub_queries"]) == 2


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_partial_empty(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """部分子查询返回空，其余有结果 → 只保留有结果的。"""
    mock_decompose.return_value = (["有结果", "无结果"], 0.2)
    mock_cached.side_effect = [
        _make_search_json(
            sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."}],
            contexts=[{"source_id": "source_1", "content": "内容", "source": "a.md，第1页，chunk_id=c1", "score": 0.9}],
            search_elapsed=0.1,
        ),
        _make_empty_search_json(search_elapsed=0.05),
    ]

    raw = plan_and_search.invoke({"query": "对比部分匹配问题"})

    data = json.loads(raw)
    assert data["success"] is True
    assert len(data["contexts"]) == 1


def test_plan_and_search_empty_query() -> None:
    """空 query 返回 success=False。"""
    raw = plan_and_search.invoke({"query": ""})

    data = json.loads(raw)
    assert data["success"] is False
    assert data["tool"] == "plan_and_search"
    assert data["contexts"] == []
    assert data["sources"] == []


def test_plan_and_search_whitespace_query() -> None:
    """纯空白 query 返回 success=False。"""
    raw = plan_and_search.invoke({"query": "   "})

    data = json.loads(raw)
    assert data["success"] is False


@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_exception_fallback(mock_decompose: MagicMock) -> None:
    """_decompose_query 抛异常时 plan_and_search 返回 error JSON。"""
    mock_decompose.side_effect = RuntimeError("意外错误")

    raw = plan_and_search.invoke({"query": "对比触发异常"})

    data = json.loads(raw)
    assert data["success"] is False
    assert data["tool"] == "plan_and_search"
    assert "意外错误" in data["error"]
    assert data["contexts"] == []
    assert data["sources"] == []


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_returns_valid_json(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search 返回合法 JSON（正常路径）。"""
    mock_decompose.return_value = (["q1"], 0.1)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比测试"})

    assert isinstance(raw, str)
    data = json.loads(raw)  # 不抛异常即为合法 JSON
    assert "success" in data
    assert "tool" in data
    assert "timing" in data


def test_plan_and_search_error_returns_valid_json() -> None:
    """plan_and_search 异常时也返回合法 JSON。"""
    with patch(
        "app.agents.planning_tools._decompose_query",
        side_effect=ValueError("测试异常"),
    ):
        raw = plan_and_search.invoke({"query": "对比 test"})

    assert isinstance(raw, str)
    data = json.loads(raw)
    assert "success" in data
    assert data["success"] is False
    assert "error" in data


# ============================================================================ #
#  timing 结构兼容性                                                             #
# ============================================================================ #

@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_timing_has_search_elapsed(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """timing.search_elapsed 存在，兼容 parse_tool_result。"""
    mock_decompose.return_value = (["q1", "q2"], 0.5)
    mock_cached.side_effect = [
        _make_search_json(search_elapsed=0.3),
        _make_search_json(search_elapsed=0.2),
    ]

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    assert "search_elapsed" in data["timing"]
    assert data["timing"]["search_elapsed"] == 0.5  # 0.3 + 0.2


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_timing_has_planning_elapsed(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """timing.planning_elapsed 记录 LLM 分解耗时。"""
    mock_decompose.return_value = (["q1"], 1.234)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    assert data["timing"]["planning_elapsed"] == 1.234


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_timing_has_total_elapsed(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """timing.total_elapsed 存在且为 float。"""
    mock_decompose.return_value = (["q1"], 0.1)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    assert "total_elapsed" in data["timing"]
    assert isinstance(data["timing"]["total_elapsed"], float)


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_compatible_with_parse_tool_result(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search 返回可被 parse_tool_result 正确解析。"""
    from app.agents.tools import parse_tool_result

    mock_decompose.return_value = (["q1"], 0.1)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比 test"})
    parsed = parse_tool_result(raw)

    assert parsed["success"] is True
    assert len(parsed["sources"]) == 2
    assert parsed["elapsed"] == 0.123
    assert isinstance(parsed["timing"], dict)


# ============================================================================ #
#  子查询数量约束                                                                 #
# ============================================================================ #

@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_four_sub_queries(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """4 个子查询都被检索。"""
    mock_decompose.return_value = (
        ["查询1", "查询2", "查询3", "查询4"],
        0.5,
    )
    mock_cached.return_value = _make_search_json(
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.8, "content_preview": "..."}],
        contexts=[{"source_id": "source_1", "content": "...", "source": "a.md，第1页，chunk_id=c1", "score": 0.8}],
        search_elapsed=0.1,
    )

    raw = plan_and_search.invoke({"query": "对比复杂问题"})
    data = json.loads(raw)

    assert data["success"] is True
    assert len(data["sub_queries"]) == 4
    assert mock_cached.call_count == 4


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_single_sub_query(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """分解为 1 个子查询也能正常工作。"""
    mock_decompose.return_value = (["单一查询"], 0.1)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比简单问题"})
    data = json.loads(raw)

    assert data["success"] is True
    assert len(data["sub_queries"]) == 1
    assert mock_cached.call_count == 1


# ============================================================================ #
#  查询归一化传递                                                                #
# ============================================================================ #

@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_normalizes_sub_queries(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """子查询在传给 _cached_search 前经过 normalize_query 归一化。"""
    mock_decompose.return_value = (["  Query  With  Spaces  "], 0.1)
    mock_cached.return_value = _make_search_json()

    plan_and_search.invoke({"query": "对比 test"})

    # _cached_search 收到的参数应为归一化后的字符串
    call_args = mock_cached.call_args
    normalized_arg = call_args[0][0]  # 第一个位置参数
    assert normalized_arg == "query with spaces"


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_skips_empty_sub_queries(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """空子查询被跳过，不调用 _cached_search。"""
    mock_decompose.return_value = (["有效查询", "", "  "], 0.1)
    mock_cached.return_value = _make_search_json()

    plan_and_search.invoke({"query": "对比 test"})

    # 只有 "有效查询" 会触发检索
    assert mock_cached.call_count == 1


# ============================================================================ #
#  source_id 重新编号                                                            #
# ============================================================================ #

@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_source_ids_renumbered(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """合并后的 contexts source_id 从 source_1 开始重新编号。"""
    mock_decompose.return_value = (["q1", "q2"], 0.1)
    mock_cached.side_effect = [
        _make_search_json(
            sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."}],
            contexts=[{"source_id": "source_99", "content": "...", "source": "a.md，第1页，chunk_id=c1", "score": 0.9}],
            search_elapsed=0.1,
        ),
        _make_search_json(
            sources=[{"filename": "b.md", "page": 2, "chunk_id": "c2", "score": 0.8, "content_preview": "..."}],
            contexts=[{"source_id": "source_99", "content": "...", "source": "b.md，第2页，chunk_id=c2", "score": 0.8}],
            search_elapsed=0.1,
        ),
    ]

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    assert data["contexts"][0]["source_id"] == "source_1"
    assert data["contexts"][1]["source_id"] == "source_2"


# ============================================================================ #
#  Block 2: 工具输出长度限制验证                                                 #
# ============================================================================ #

@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_summary_within_limit(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search summary 不超过 200 字符。"""
    mock_decompose.return_value = (["q1", "q2"], 0.1)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    summary = data.get("summary", "")
    assert len(summary) <= 200, f"summary 过长: {len(summary)}"


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_contexts_content_within_limit(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search 透传 contexts.content，不增大原始长度。

    _cached_search 在真实环境中已将 content 截断到 ≤500 字符，
    plan_and_search 的 _merge_search_results 不应增大它。
    """
    mock_decompose.return_value = (["q1"], 0.1)
    # mock 数据模拟 _cached_search 已截断后的输出（≤500）
    mock_cached.return_value = _make_search_json(
        contexts=[{
            "source_id": "source_1",
            "content": "C" * 400,
            "source": "a.md，第1页，chunk_id=c1",
            "score": 0.9,
        }],
        sources=[{"filename": "a.md", "page": 1, "chunk_id": "c1", "score": 0.9, "content_preview": "..."}],
        search_elapsed=0.1,
    )

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    for ctx in data.get("contexts", []):
        content = ctx.get("content", "")
        # 透传后 content 不应增大
        assert len(content) <= 500, f"content 过长: {len(content)}"


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_sources_preview_within_limit(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search 透传 sources.content_preview，不增大原始长度。

    _cached_search 在真实环境中已将 content_preview 截断到 ≤150 字符，
    plan_and_search 的 _merge_search_results 不应增大它。
    """
    mock_decompose.return_value = (["q1"], 0.1)
    # mock 数据模拟 _cached_search 已截断后的输出（≤150）
    mock_cached.return_value = _make_search_json(
        sources=[{
            "filename": "a.md",
            "page": 1,
            "chunk_id": "c1",
            "score": 0.9,
            "content_preview": "P" * 100,
        }],
        contexts=[{
            "source_id": "source_1",
            "content": "...",
            "source": "a.md，第1页，chunk_id=c1",
            "score": 0.9,
        }],
        search_elapsed=0.1,
    )

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    for src in data.get("sources", []):
        preview = src.get("content_preview", "")
        # 透传后 preview 不应增大
        assert len(preview) <= 150, f"preview 过长: {len(preview)}"


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_empty_summary_within_limit(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search 无结果时 summary 不超过 200 字符。"""
    mock_decompose.return_value = (["q1", "q2"], 0.2)
    mock_cached.return_value = _make_empty_search_json()

    raw = plan_and_search.invoke({"query": "对比无结果查询"})
    data = json.loads(raw)

    summary = data.get("summary", "")
    assert len(summary) <= 200


@patch("app.agents.planning_tools._cached_search")
@patch("app.agents.planning_tools._decompose_query")
def test_plan_and_search_output_is_valid_json_and_compact(
    mock_decompose: MagicMock,
    mock_cached: MagicMock,
) -> None:
    """plan_and_search 输出为合法 JSON 且 summary 不超长。"""
    mock_decompose.return_value = (["q1"], 0.1)
    mock_cached.return_value = _make_search_json()

    raw = plan_and_search.invoke({"query": "对比 test"})
    data = json.loads(raw)

    assert "success" in data
    assert "tool" in data
    assert len(data.get("summary", "")) <= 200


# ============================================================================ #
#  Block 4: plan_and_search 准入门控测试                                        #
# ============================================================================ #

# ---------- should_use_plan_and_search 直接测试 ----------

def test_gate_dataset_overview_question_returns_false() -> None:
    """'告诉我语义分割数据集有什么特点' 应返回 false。"""
    suitable, reason = should_use_plan_and_search("告诉我语义分割数据集有什么特点")
    assert suitable is False
    assert isinstance(reason, str)


def test_gate_what_is_miou_returns_false() -> None:
    """'什么是 mIoU' 应返回 false。"""
    suitable, reason = should_use_plan_and_search("什么是 mIoU")
    assert suitable is False


def test_gate_loveda_classes_returns_false() -> None:
    """'LoveDA 有哪些类别' 应返回 false（只出现一个实体）。"""
    suitable, reason = should_use_plan_and_search("LoveDA 有哪些类别")
    assert suitable is False


def test_gate_what_is_unet_returns_false() -> None:
    """'U-Net 是什么' 应返回 false（只出现一个实体）。"""
    suitable, reason = should_use_plan_and_search("U-Net 是什么")
    assert suitable is False


def test_gate_calculate_iou_returns_false() -> None:
    """'帮我计算 IoU' 应返回 false。"""
    suitable, reason = should_use_plan_and_search("帮我计算 IoU")
    assert suitable is False


def test_gate_pixel_accuracy_returns_false() -> None:
    """'请解释 Pixel Accuracy' 应返回 false。"""
    suitable, reason = should_use_plan_and_search("请解释 Pixel Accuracy")
    assert suitable is False


def test_gate_empty_query_returns_false() -> None:
    """空查询应返回 false。"""
    suitable, _ = should_use_plan_and_search("")
    assert suitable is False


def test_gate_comparison_keyword_returns_true() -> None:
    """包含比较关键词 '对比' 应返回 true。"""
    suitable, _ = should_use_plan_and_search("对比 U-Net 和 DeepLabV3+")
    assert suitable is True


def test_gate_two_entities_returns_true() -> None:
    """包含两个已知实体应返回 true。"""
    suitable, _ = should_use_plan_and_search("DeepLabV3+ 在 LoveDA 上的表现")
    assert suitable is True


def test_gate_multi_aspect_returns_true() -> None:
    """多方面分析模式应返回 true。"""
    suitable, _ = should_use_plan_and_search("对比 DeepLabV3+ 和 SegFormer 在 LoveDA 上的架构差异和表现")
    assert suitable is True


def test_gate_reason_is_meaningful_string() -> None:
    """返回的 reason 是有意义的字符串。"""
    _, reason_false = should_use_plan_and_search("什么是 mIoU")
    assert len(reason_false) > 0

    _, reason_true = should_use_plan_and_search("对比两个模型")
    assert len(reason_true) > 0


# ---------- plan_and_search 门控拦截端到端测试 ----------

def test_gate_blocked_does_not_call_llm() -> None:
    """should_use=false 时不调用 LLM（_decompose_query 不被调用）。"""
    with patch("app.agents.planning_tools._decompose_query") as mock_decompose:
        raw = plan_and_search.invoke({"query": "告诉我语义分割数据集有什么特点"})
        data = json.loads(raw)

        mock_decompose.assert_not_called()
    assert data["success"] is False


def test_gate_blocked_does_not_call_vector_db() -> None:
    """should_use=false 时不调用向量库（_cached_search 不被调用）。"""
    with patch("app.agents.planning_tools._cached_search") as mock_cached:
        raw = plan_and_search.invoke({"query": "什么是 mIoU"})
        data = json.loads(raw)

        mock_cached.assert_not_called()
    assert data["success"] is False


def test_gate_blocked_returns_valid_json() -> None:
    """should_use=false 时返回合法 JSON。"""
    raw = plan_and_search.invoke({"query": "LoveDA 有哪些类别"})
    data = json.loads(raw)

    assert data["success"] is False
    assert data["tool"] == "plan_and_search"
    assert data["contexts"] == []
    assert data["sources"] == []
    assert "reason" in data
    assert "timing" in data
    assert data["timing"]["planning_elapsed"] == 0.0
    assert data["timing"]["search_elapsed"] == 0.0


def test_gate_blocked_summary_within_limit() -> None:
    """门控拦截时 summary 不超过 200 字符。"""
    raw = plan_and_search.invoke({"query": "什么是 mIoU"})
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_gate_passed_still_invokes_decompose() -> None:
    """通过门控后仍然正常调用 _decompose_query。"""
    with patch("app.agents.planning_tools._decompose_query") as mock_decompose, \
         patch("app.agents.planning_tools._cached_search") as mock_cached:
        mock_decompose.return_value = (["q1"], 0.1)
        mock_cached.return_value = _make_search_json()

        raw = plan_and_search.invoke({"query": "对比 DeepLabV3+ 和 SegFormer"})
        data = json.loads(raw)

        mock_decompose.assert_called_once()
    assert data["success"] is True
