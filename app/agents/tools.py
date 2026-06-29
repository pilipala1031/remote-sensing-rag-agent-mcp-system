"""Agent 工具定义：将现有 RAG 检索能力封装为 LangChain Tool。

工具通过 Retriever 复用 VectorStore -> Chroma 全链路，
不修改任何现有检索逻辑，不在 Tool 内调用 LLM。

核心工具：
    knowledge_base_search  ——  遥感知识库检索，返回 JSON 字符串

性能优化：
    - 对真实检索函数使用 functools.lru_cache 缓存（maxsize=128）
    - 工具返回内容压缩：contexts.content ≤ 500 字符，sources.content_preview ≤ 150 字符
    - 返回 timing.search_elapsed 用于耗时追踪
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

from app.config import get_settings
from app.services.retriever import Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------------- #
#  文本压缩 / 归一化辅助函数                                                  #
# -------------------------------------------------------------------------- #

# contexts 中每个 content 最大字符数
_MAX_CONTEXT_CHARS = 500

# sources 中每个 content_preview 最大字符数
_MAX_PREVIEW_CHARS = 150


def truncate_text(text: str | None, max_chars: int) -> str:
    """安全截断文本，超出长度时追加省略号。

    Args:
        text: 原始文本，None 视为空字符串。
        max_chars: 最大保留字符数（不含省略号）。

    Returns:
        截断后的字符串。超长时末尾追加 "..."。
        None 返回空字符串，不因特殊字符报错。
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def normalize_query(query: str | None) -> str:
    """归一化查询字符串，用作 LRU 缓存 key 的一部分。

    处理规则：
        - strip 前后空白
        - 连续空白归一化为单个空格
        - 转小写
        - None / 空字符串返回空字符串
    """
    if not query:
        return ""
    return re.sub(r"\s+", " ", query.strip()).lower()


# -------------------------------------------------------------------------- #
#  Rerank 覆盖标志（由 AgentService 设置，用于前端开关控制）                    #
# -------------------------------------------------------------------------- #

#: 前端传入的 rerank 覆盖值。None = 使用 .env 配置，True/False = 覆盖。
_rerank_override: bool | None = None


def set_rerank_override(value: bool | None) -> None:
    """设置 rerank 覆盖标志，供 AgentService 在调用 Agent 前设置。

    Args:
        value: None = 使用 .env 配置；True = 强制开启；False = 强制关闭。
    """
    global _rerank_override
    _rerank_override = value


# -------------------------------------------------------------------------- #
#  内部检索函数（供 AgentService / 测试直接调用，返回原始 hit 列表）          #
# -------------------------------------------------------------------------- #

def _retrieve(
    query: str, top_k: int | None = None
) -> list[dict[str, Any]]:
    """调用 Retriever 检索知识库，返回原始 hit 字典列表。

    每个 dict 含 chunk_id / score / content / filename / page / doc_id。

    Args:
        query: 检索语句
        top_k: 返回条数，None 则使用 settings.top_k
    """
    settings = get_settings()
    actual_top_k = top_k if top_k is not None else settings.top_k
    retriever = Retriever()
    hits = retriever.retrieve(
        query,
        top_k=actual_top_k,
        similarity_threshold=settings.similarity_threshold,
        use_rerank=_rerank_override,
    )
    logger.info("_retrieve query=%r top_k=%d 命中 %d 条", query, actual_top_k, len(hits))
    return hits


# -------------------------------------------------------------------------- #
#  hit → JSON 结构转换（压缩后）                                              #
# -------------------------------------------------------------------------- #

def _hit_to_context(h: dict[str, Any], idx: int) -> dict[str, Any]:
    """将单个 hit 转为 JSON 中 contexts 数组的元素（压缩后，供 Agent 阅读）。"""
    content = h.get("content", "") or ""
    filename = h.get("filename", "unknown")
    page = h.get("page", 1)
    chunk_id = h.get("chunk_id", "")
    return {
        "source_id": f"source_{idx + 1}",
        "content": truncate_text(content.replace("\n", " "), _MAX_CONTEXT_CHARS),
        "source": f"{filename}，第{page}页，chunk_id={chunk_id}",
        "score": round(float(h.get("score", 0.0)), 4),
    }


def _hit_to_source(h: dict[str, Any]) -> dict[str, Any]:
    """将单个 hit 转为 JSON 中 sources 数组的元素（含压缩后的 content_preview）。"""
    content = h.get("content", "") or ""
    return {
        "filename": h.get("filename", "unknown"),
        "page": h.get("page", 1),
        "chunk_id": h.get("chunk_id", ""),
        "score": round(float(h.get("score", 0.0)), 4),
        "content_preview": truncate_text(
            content.replace("\n", " "), _MAX_PREVIEW_CHARS
        ),
    }


# -------------------------------------------------------------------------- #
#  JSON 构建函数（含 timing）                                                 #
# -------------------------------------------------------------------------- #

def _round_elapsed(elapsed: float) -> float:
    """保留 4 位小数，非数字返回 0.0。"""
    try:
        return round(float(elapsed), 4)
    except (TypeError, ValueError):
        return 0.0


def _build_success_json(
    query: str, hits: list[dict[str, Any]], search_elapsed: float
) -> str:
    """构建检索成功的 JSON 字符串（压缩上下文 + timing）。"""
    data = {
        "success": True,
        "query": query,
        "summary": f"检索到 {len(hits)} 个相关片段",
        "contexts": [_hit_to_context(h, i) for i, h in enumerate(hits)],
        "sources": [_hit_to_source(h) for h in hits],
        "timing": {"search_elapsed": _round_elapsed(search_elapsed)},
    }
    return json.dumps(data, ensure_ascii=False)


def _build_empty_json(query: str, search_elapsed: float = 0.0) -> str:
    """构建检索结果为空的 JSON 字符串。"""
    data = {
        "success": False,
        "query": query,
        "summary": "未检索到相关知识库内容",
        "contexts": [],
        "sources": [],
        "timing": {"search_elapsed": _round_elapsed(search_elapsed)},
        "error": None,
    }
    return json.dumps(data, ensure_ascii=False)


def _build_error_json(
    query: str, error: str, search_elapsed: float = 0.0
) -> str:
    """构建检索异常的 JSON 字符串。"""
    data = {
        "success": False,
        "query": query,
        "summary": "检索失败",
        "contexts": [],
        "sources": [],
        "timing": {"search_elapsed": _round_elapsed(search_elapsed)},
        "error": error,
    }
    return json.dumps(data, ensure_ascii=False)


# -------------------------------------------------------------------------- #
#  LRU 缓存的检索函数（缓存 JSON 字符串，不缓存 LangChain Tool wrapper）      #
# -------------------------------------------------------------------------- #
#
# 注意（第一版限制）：
#   - 如果知识库发生新增、删除或重新入库，LRU 缓存不会自动失效。
#     后续可以在 documents ingest/delete 后调用 clear_agent_search_cache()。
#   - 当前已在 app/api/documents.py 的 ingest / delete 接口中调用清理。
#   - 如果 .env 中 TOP_K 变化，因 lru_cache 的 get_settings 是 @lru_cache 的，
#     需要重启服务才能生效。
#


@lru_cache(maxsize=128)
def _cached_search(normalized_query: str, top_k: int) -> str:
    """执行真实向量检索并返回压缩后的 JSON 字符串。

    被 @lru_cache 缓存，相同 (normalized_query, top_k) 第二次直接返回缓存结果。
    异常不会被缓存（lru_cache 不缓存异常），下次调用会重新检索。

    Args:
        normalized_query: 经 normalize_query 归一化的查询字符串
        top_k: 检索返回条数
    """
    start = time.time()
    hits = _retrieve(normalized_query, top_k=top_k)
    elapsed = time.time() - start

    if not hits:
        logger.info("_cached_search 结果为空: query=%r", normalized_query)
        return _build_empty_json(normalized_query, elapsed)

    logger.info(
        "_cached_search 成功: query=%r, 命中 %d 条, 耗时 %.4fs",
        normalized_query,
        len(hits),
        elapsed,
    )
    return _build_success_json(normalized_query, hits, elapsed)


def clear_agent_search_cache() -> None:
    """手动清空 Agent 检索缓存。

    在文档入库 / 删除后调用，确保新内容可被检索到。
    """
    _cached_search.cache_clear()
    logger.info("Agent 检索缓存已清空")


# -------------------------------------------------------------------------- #
#  LangChain Tool                                                            #
# -------------------------------------------------------------------------- #

@tool
def knowledge_base_search(query: str) -> str:
    """Search the local remote sensing semantic segmentation knowledge base.

    Use this tool when the user asks about remote sensing datasets,
    segmentation models, evaluation metrics, challenges, or methods.
    The tool returns relevant context chunks and source metadata.

    Args:
        query: The search query in Chinese or English.

    Returns:
        A JSON string with keys: success, query, summary, contexts, sources, timing.
        On error, an additional "error" key is present.
    """
    normalized = normalize_query(query)

    # 空 query 直接返回，不调用向量库
    if not normalized:
        return _build_empty_json(query or "", 0.0)

    settings = get_settings()
    try:
        return _cached_search(normalized, settings.top_k)
    except Exception as e:
        logger.error("knowledge_base_search 检索失败: %s", e)
        return _build_error_json(normalized, str(e))


# -------------------------------------------------------------------------- #
#  解析辅助函数                                                              #
# -------------------------------------------------------------------------- #

def parse_tool_result(tool_output: str) -> dict[str, Any]:
    """从工具返回的 JSON 字符串中提取结构化信息。

    用于 AgentService / langchain_agent 从 tool message 中提取
    sources / summary / timing / elapsed / error。

    Args:
        tool_output: knowledge_base_search 返回的 JSON 字符串。

    Returns:
        dict，固定包含以下 key：
        - success: bool
        - sources: list[dict]
        - summary: str
        - error: str | None
        - timing: dict | None  （原始 timing 对象）
        - elapsed: float | None（从 timing.search_elapsed 提取，便于直接使用）
    """
    try:
        data = json.loads(tool_output)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("parse_tool_result JSON 解析失败: %s", e)
        return {
            "success": False,
            "sources": [],
            "summary": "工具输出解析失败",
            "error": f"无法解析工具输出: {str(tool_output)[:200]}",
            "timing": None,
            "elapsed": None,
        }

    timing = data.get("timing")
    elapsed = None
    if isinstance(timing, dict):
        try:
            elapsed = float(timing.get("search_elapsed", 0))
        except (TypeError, ValueError):
            elapsed = None

    return {
        "success": data.get("success", False),
        "sources": data.get("sources", []),
        "summary": data.get("summary", ""),
        "error": data.get("error"),
        "timing": timing,
        "elapsed": elapsed,
    }
