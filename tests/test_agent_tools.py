"""Agent 工具测试：验证 knowledge_base_search / parse_tool_result / 缓存 / 压缩。

使用 mock Retriever 避免依赖真实 Embedding API 和 Chroma 数据库。
不调用 LLM。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.tools import (
    clear_agent_search_cache,
    knowledge_base_search,
    normalize_query,
    parse_tool_result,
    truncate_text,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个测试前后清空 LRU 缓存，避免测试间互相干扰。"""
    clear_agent_search_cache()
    yield
    clear_agent_search_cache()


# ============================================================================ #
#  truncate_text                                                               #
# ============================================================================ #

def test_truncate_text_short() -> None:
    """短文本不截断。"""
    assert truncate_text("hello", 100) == "hello"


def test_truncate_text_exact_length() -> None:
    """刚好等于 max_chars 时不截断。"""
    text = "a" * 50
    assert truncate_text(text, 50) == text


def test_truncate_text_long_adds_ellipsis() -> None:
    """超长文本截断并加省略号。"""
    text = "a" * 600
    result = truncate_text(text, 500)
    assert len(result) == 503  # 500 + "..."
    assert result.endswith("...")


def test_truncate_text_none_returns_empty() -> None:
    """None 返回空字符串。"""
    assert truncate_text(None, 100) == ""


def test_truncate_text_empty_returns_empty() -> None:
    """空字符串返回空字符串。"""
    assert truncate_text("", 100) == ""


def test_truncate_text_special_chars() -> None:
    """特殊字符不报错。"""
    assert truncate_text("🎉🎊💙", 2) == "🎉🎊..."


# ============================================================================ #
#  normalize_query                                                             #
# ============================================================================ #

def test_normalize_query_basic() -> None:
    """strip + lower。"""
    assert normalize_query("  Hello World  ") == "hello world"


def test_normalize_query_collapses_whitespace() -> None:
    """连续空白归一化为单个空格。"""
    assert normalize_query("a   b\n\tc") == "a b c"


def test_normalize_query_empty() -> None:
    """空字符串返回空字符串。"""
    assert normalize_query("") == ""


def test_normalize_query_none() -> None:
    """None 返回空字符串。"""
    assert normalize_query(None) == ""


def test_normalize_query_whitespace_only() -> None:
    """纯空白返回空字符串。"""
    assert normalize_query("   \n\t  ") == ""


# ============================================================================ #
#  工具元信息                                                                   #
# ============================================================================ #

def test_knowledge_base_search_is_tool() -> None:
    """knowledge_base_search 被 @tool 装饰，具有 LangChain Tool 接口。"""
    assert hasattr(knowledge_base_search, "name")
    assert hasattr(knowledge_base_search, "description")
    assert hasattr(knowledge_base_search, "invoke")
    assert knowledge_base_search.name == "knowledge_base_search"


def test_tool_description_in_english_for_llm() -> None:
    """工具描述为英文，帮助 LLM 判断何时调用。"""
    desc = knowledge_base_search.description
    assert "remote sensing" in desc.lower()
    assert "knowledge base" in desc.lower()


# ============================================================================ #
#  正常检索                                                                     #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_search_returns_valid_json_with_hits(mock_retrieve: MagicMock) -> None:
    """有检索结果时返回可解析的 JSON，success=true。"""
    mock_retrieve.return_value = [
        {
            "chunk_id": "abc123def456",
            "score": 0.85,
            "content": "Landsat 8 TIRS Band 10 中心波长 10.9 μm。",
            "filename": "landsat.pdf",
            "page": 3,
            "doc_id": "d1",
        }
    ]

    result = knowledge_base_search.invoke({"query": "Landsat 8 Band 10"})

    assert isinstance(result, str)
    data = json.loads(result)
    assert data["success"] is True
    assert data["query"] == "landsat 8 band 10"
    assert len(data["contexts"]) == 1
    assert len(data["sources"]) == 1
    assert "检索到" in data["summary"]


@patch("app.agents.tools._retrieve")
def test_search_sources_have_required_fields(mock_retrieve: MagicMock) -> None:
    """sources 元素包含 filename / page / chunk_id / score / content_preview。"""
    mock_retrieve.return_value = [
        {
            "chunk_id": "c1",
            "score": 0.92,
            "content": "NDVI = (NIR - Red) / (NIR + Red)",
            "filename": "metrics.pdf",
            "page": 5,
            "doc_id": "d2",
        }
    ]

    result = knowledge_base_search.invoke({"query": "NDVI"})
    data = json.loads(result)
    src = data["sources"][0]

    assert src["filename"] == "metrics.pdf"
    assert src["page"] == 5
    assert src["chunk_id"] == "c1"
    assert src["score"] == 0.92
    assert "NDVI" in src["content_preview"]


@patch("app.agents.tools._retrieve")
def test_search_contexts_have_required_fields(mock_retrieve: MagicMock) -> None:
    """contexts 元素包含 source_id / content / source / score。"""
    mock_retrieve.return_value = [
        {
            "chunk_id": "c1",
            "score": 0.88,
            "content": "DeepLabV3+ 采用 ASPP 模块。",
            "filename": "models.md",
            "page": 2,
            "doc_id": "d3",
        }
    ]

    result = knowledge_base_search.invoke({"query": "DeepLabV3+"})
    data = json.loads(result)
    ctx = data["contexts"][0]

    assert ctx["source_id"] == "source_1"
    assert "DeepLabV3+" in ctx["content"]
    assert "models.md" in ctx["source"]
    assert ctx["score"] == 0.88


@patch("app.agents.tools._retrieve")
def test_search_multiple_hits(mock_retrieve: MagicMock) -> None:
    """多条检索结果全部出现在 JSON 中。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "片段一", "filename": "a.md", "page": 1, "doc_id": "d1"},
        {"chunk_id": "c2", "score": 0.8, "content": "片段二", "filename": "b.md", "page": 2, "doc_id": "d2"},
        {"chunk_id": "c3", "score": 0.7, "content": "片段三", "filename": "c.md", "page": 3, "doc_id": "d3"},
    ]

    result = knowledge_base_search.invoke({"query": "测试"})
    data = json.loads(result)

    assert data["success"] is True
    assert len(data["contexts"]) == 3
    assert len(data["sources"]) == 3
    assert "3" in data["summary"]


# ============================================================================ #
#  内容压缩                                                                     #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_contexts_content_truncated_to_500(mock_retrieve: MagicMock) -> None:
    """contexts.content 不超过 500 字符（不含省略号）。"""
    long_content = "X" * 1000
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": long_content, "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    result = knowledge_base_search.invoke({"query": "long"})
    data = json.loads(result)
    content = data["contexts"][0]["content"]

    # 截断后应为 500 + "..."
    assert len(content) == 503
    assert content.endswith("...")


@patch("app.agents.tools._retrieve")
def test_sources_preview_truncated_to_150(mock_retrieve: MagicMock) -> None:
    """sources.content_preview 不超过 150 字符（不含省略号）。"""
    long_content = "Y" * 300
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": long_content, "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    result = knowledge_base_search.invoke({"query": "preview"})
    data = json.loads(result)
    preview = data["sources"][0]["content_preview"]

    assert len(preview) == 153  # 150 + "..."
    assert preview.endswith("...")


# ============================================================================ #
#  timing                                                                       #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_timing_search_elapsed_present(mock_retrieve: MagicMock) -> None:
    """返回 JSON 中包含 timing.search_elapsed。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "test", "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    result = knowledge_base_search.invoke({"query": "timing"})
    data = json.loads(result)

    assert "timing" in data
    assert "search_elapsed" in data["timing"]
    assert isinstance(data["timing"]["search_elapsed"], (int, float))


@patch("app.agents.tools._retrieve")
def test_timing_search_elapsed_on_empty(mock_retrieve: MagicMock) -> None:
    """空结果时 timing.search_elapsed 存在。"""
    mock_retrieve.return_value = []

    result = knowledge_base_search.invoke({"query": "empty"})
    data = json.loads(result)

    assert "timing" in data
    assert "search_elapsed" in data["timing"]


# ============================================================================ #
#  空结果                                                                       #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_search_empty_returns_success_false(mock_retrieve: MagicMock) -> None:
    """检索结果为空时 success=false，summary 提示无内容。"""
    mock_retrieve.return_value = []

    result = knowledge_base_search.invoke({"query": "不相关的问题"})

    data = json.loads(result)
    assert data["success"] is False
    assert data["contexts"] == []
    assert data["sources"] == []
    assert "未检索到" in data["summary"]


@patch("app.agents.tools._retrieve")
def test_search_empty_no_error_key(mock_retrieve: MagicMock) -> None:
    """空结果时 error 字段为 None。"""
    mock_retrieve.return_value = []

    result = knowledge_base_search.invoke({"query": "空"})
    data = json.loads(result)

    assert data["success"] is False
    assert data.get("error") is None


# ============================================================================ #
#  异常情况                                                                     #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_search_exception_returns_error_json(mock_retrieve: MagicMock) -> None:
    """检索抛异常时返回 success=false + error 字段。"""
    mock_retrieve.side_effect = RuntimeError("Chroma 连接失败")

    result = knowledge_base_search.invoke({"query": "测试"})

    data = json.loads(result)
    assert data["success"] is False
    assert data["contexts"] == []
    assert data["sources"] == []
    assert data["summary"] == "检索失败"
    assert "Chroma 连接失败" in data["error"]


@patch("app.agents.tools._retrieve")
def test_search_exception_preserves_query(mock_retrieve: MagicMock) -> None:
    """异常时 JSON 仍保留原始 query（归一化后）。"""
    mock_retrieve.side_effect = Exception("超时")

    result = knowledge_base_search.invoke({"query": "DeepLabV3+ U-Net 对比"})
    data = json.loads(result)

    assert data["query"] == "deeplabv3+ u-net 对比"


# ============================================================================ #
#  空 query 不调用向量库                                                         #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_empty_query_does_not_call_retrieve(mock_retrieve: MagicMock) -> None:
    """空 query 直接返回 success=false，不调用向量库。"""
    result = knowledge_base_search.invoke({"query": ""})

    data = json.loads(result)
    assert data["success"] is False
    mock_retrieve.assert_not_called()


@patch("app.agents.tools._retrieve")
def test_whitespace_query_does_not_call_retrieve(mock_retrieve: MagicMock) -> None:
    """纯空白 query 直接返回 success=false，不调用向量库。"""
    result = knowledge_base_search.invoke({"query": "   \n\t  "})

    data = json.loads(result)
    assert data["success"] is False
    mock_retrieve.assert_not_called()


# ============================================================================ #
#  LRU 缓存                                                                     #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_cache_hit_on_same_query(mock_retrieve: MagicMock) -> None:
    """相同 query 第二次命中缓存，_retrieve 只调用一次。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "test", "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    knowledge_base_search.invoke({"query": "cache test"})
    knowledge_base_search.invoke({"query": "cache test"})

    assert mock_retrieve.call_count == 1


@patch("app.agents.tools._retrieve")
def test_cache_miss_on_different_query(mock_retrieve: MagicMock) -> None:
    """不同 query 各自检索。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "test", "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    knowledge_base_search.invoke({"query": "query one"})
    knowledge_base_search.invoke({"query": "query two"})

    assert mock_retrieve.call_count == 2


@patch("app.agents.tools._retrieve")
def test_cache_normalized_query(mock_retrieve: MagicMock) -> None:
    """归一化后相同的 query 命中缓存（大小写/空白差异）。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "test", "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    knowledge_base_search.invoke({"query": "DeepLabV3+"})
    knowledge_base_search.invoke({"query": "  deeplabv3+  "})

    assert mock_retrieve.call_count == 1


@patch("app.agents.tools._retrieve")
def test_clear_agent_search_cache(mock_retrieve: MagicMock) -> None:
    """clear_agent_search_cache 后相同 query 重新检索。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "test", "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    knowledge_base_search.invoke({"query": "clear test"})
    assert mock_retrieve.call_count == 1

    clear_agent_search_cache()

    knowledge_base_search.invoke({"query": "clear test"})
    assert mock_retrieve.call_count == 2


# ============================================================================ #
#  parse_tool_result                                                            #
# ============================================================================ #

def test_parse_tool_result_success() -> None:
    """parse_tool_result 正常解析成功 JSON。"""
    raw = json.dumps({
        "success": True,
        "query": "NDVI",
        "contexts": [{"source_id": "source_1", "content": "...", "source": "...", "score": 0.8}],
        "sources": [{"filename": "a.pdf", "page": 1, "chunk_id": "c1", "score": 0.8, "content_preview": "..."}],
        "summary": "检索到 1 个相关片段",
        "timing": {"search_elapsed": 0.123},
    })

    parsed = parse_tool_result(raw)

    assert parsed["success"] is True
    assert len(parsed["sources"]) == 1
    assert parsed["sources"][0]["filename"] == "a.pdf"
    assert parsed["summary"] == "检索到 1 个相关片段"
    assert parsed["elapsed"] == 0.123


def test_parse_tool_result_empty_result() -> None:
    """parse_tool_result 解析空结果 JSON。"""
    raw = json.dumps({
        "success": False,
        "query": "空",
        "contexts": [],
        "sources": [],
        "summary": "未检索到相关知识库内容",
        "timing": {"search_elapsed": 0.0},
    })

    parsed = parse_tool_result(raw)

    assert parsed["success"] is False
    assert parsed["sources"] == []
    assert "未检索到" in parsed["summary"]


def test_parse_tool_result_error_json() -> None:
    """parse_tool_result 解析错误 JSON。"""
    raw = json.dumps({
        "success": False,
        "query": "测试",
        "contexts": [],
        "sources": [],
        "summary": "检索失败",
        "error": "连接超时",
        "timing": {"search_elapsed": 0.0},
    })

    parsed = parse_tool_result(raw)

    assert parsed["success"] is False
    assert parsed["sources"] == []
    assert parsed["summary"] == "检索失败"


def test_parse_tool_result_invalid_json_fallback() -> None:
    """parse_tool_result 对非法 JSON 有 fallback。"""
    parsed = parse_tool_result("这不是 JSON {{{")

    assert parsed["success"] is False
    assert parsed["sources"] == []
    assert "解析失败" in parsed["summary"]
    assert "error" in parsed


def test_parse_tool_result_none_fallback() -> None:
    """parse_tool_result 对 None 输入有 fallback。"""
    parsed = parse_tool_result(None)  # type: ignore[arg-type]

    assert parsed["success"] is False
    assert parsed["sources"] == []


def test_parse_tool_result_missing_keys() -> None:
    """parse_tool_result 对缺少 key 的 JSON 有默认值。"""
    raw = json.dumps({"success": True})

    parsed = parse_tool_result(raw)

    assert parsed["success"] is True
    assert parsed["sources"] == []
    assert parsed["summary"] == ""


def test_parse_tool_result_no_timing() -> None:
    """没有 timing 字段时 elapsed=None。"""
    raw = json.dumps({
        "success": True,
        "query": "test",
        "sources": [],
        "summary": "ok",
    })

    parsed = parse_tool_result(raw)

    assert parsed["elapsed"] is None
    assert parsed["timing"] is None


# ============================================================================ #
#  Block 2: 工具输出长度限制验证                                                 #
# ============================================================================ #

@patch("app.agents.tools._retrieve")
def test_summary_length_within_limit(mock_retrieve: MagicMock) -> None:
    """knowledge_base_search summary 长度不超过 200 字符。"""
    mock_retrieve.return_value = [
        {"chunk_id": "c1", "score": 0.9, "content": "test", "filename": "a.md", "page": 1, "doc_id": "d1"}
    ]

    result = knowledge_base_search.invoke({"query": "summary test"})
    data = json.loads(result)

    summary = data.get("summary", "")
    assert len(summary) <= 200, f"summary 过长: {len(summary)} > 200"


@patch("app.agents.tools._retrieve")
def test_contexts_content_max_500_with_many_hits(mock_retrieve: MagicMock) -> None:
    """多条命中时每个 context content 仍不超过 500 字符。"""
    mock_retrieve.return_value = [
        {"chunk_id": f"c{i}", "score": 0.9 - i * 0.01, "content": "Y" * 800, "filename": f"f{i}.md", "page": i, "doc_id": f"d{i}"}
        for i in range(5)
    ]

    result = knowledge_base_search.invoke({"query": "multi"})
    data = json.loads(result)

    for ctx in data["contexts"]:
        content = ctx["content"]
        assert len(content) <= 503, f"content 过长: {len(content)}"  # 500 + "..."


@patch("app.agents.tools._retrieve")
def test_sources_preview_max_150_with_many_hits(mock_retrieve: MagicMock) -> None:
    """多条命中时每个 source content_preview 仍不超过 150 字符。"""
    mock_retrieve.return_value = [
        {"chunk_id": f"c{i}", "score": 0.9 - i * 0.01, "content": "Z" * 400, "filename": f"f{i}.md", "page": i, "doc_id": f"d{i}"}
        for i in range(5)
    ]

    result = knowledge_base_search.invoke({"query": "preview multi"})
    data = json.loads(result)

    for src in data["sources"]:
        preview = src["content_preview"]
        assert len(preview) <= 153, f"preview 过长: {len(preview)}"  # 150 + "..."
