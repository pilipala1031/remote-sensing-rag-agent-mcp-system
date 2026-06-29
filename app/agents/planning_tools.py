"""Plan-and-Search 查询分解工具。

将复杂问题分解为 2-4 个子查询，分别检索后合并去重，
返回更全面的知识库检索结果。

使用 LLM 进行查询分解（lazy import get_agent_llm 复用 Agent 单例的 ChatOpenAI），
复用 _cached_search 进行实际检索（自动享受 LRU 缓存）。

工具列表：
    plan_and_search  ——  复杂问题查询分解 + 多次检索 + 合并去重

设计要点：
    - _decompose_query 使用 lazy import 引入 get_agent_llm，
      复用 Agent 单例的 ChatOpenAI（而非 build_chat_model 创建新实例），
      确保 LLM 缓存控制对此调用也生效。
    - 每个子查询通过 _cached_search 检索，命中 LRU 缓存时零开销。
    - 合并阶段按 chunk_id 去重，保留最高分条目，按分数降序排列。
    - timing 结构兼容 parse_tool_result：包含 search_elapsed 字段。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

from app.agents.tools import _cached_search, normalize_query
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------------- #
#  常量                                                                        #
# -------------------------------------------------------------------------- #

#: 子查询最大数量
_MAX_SUB_QUERIES = 4

#: plan_and_search 准入门控 — 已知模型实体
_KNOWN_MODEL_ENTITIES: List[str] = [
    "deeplabv3+", "deeplabv3", "segformer", "u-net", "unet",
    "pspnet", "fcn", "swin-transformer", "swin transformer", "swin",
]

#: plan_and_search 准入门控 — 已知数据集实体
_KNOWN_DATASET_ENTITIES: List[str] = [
    "loveda", "isaid", "deepglobe", "potsdam", "vaihingen",
]

#: plan_and_search 准入门控 — 比较关键词
_COMPARISON_KEYWORDS: List[str] = [
    "对比", "比较", "差异", "优缺点", "分别",
    "哪个更适合", "表现差异", "从.*角度",
]

#: plan_and_search 准入门控 — 多方面分析模式
_MULTI_ASPECT_PATTERNS: List[Tuple[str, str]] = [
    (r"架构.*指标|指标.*架构", "同时涉及架构和指标"),
    (r"数据集.*模型.*表现|模型.*数据集.*表现", "同时涉及数据集、模型和表现"),
    (r"方法.*适用场景.*局限|适用场景.*方法.*局限", "同时涉及方法、场景和局限"),
]


# -------------------------------------------------------------------------- #
#  准入门控                                                                    #
# -------------------------------------------------------------------------- #

def should_use_plan_and_search(query: str) -> Tuple[bool, str]:
    """判断一个查询是否适合使用 plan_and_search 进行复杂分解。

    准入条件（满足任一即返回 True）：
    A. 出现比较关键词（对比 / 比较 / 差异 / 优缺点 / 分别 / 哪个更适合 / 表现差异 / 从…角度）
    B. 出现两个及以上已知模型或数据集实体
    C. 明确要求多方面分析（架构+指标 / 数据集+模型+表现 / 方法+适用场景+局限）

    Args:
        query: 用户原始查询

    Returns:
        (是否适合复杂分解, 原因说明)
    """
    if not query or not query.strip():
        return False, "查询为空"

    q = query.strip().lower()

    # A. 比较关键词
    for kw in _COMPARISON_KEYWORDS:
        if re.search(kw, q):
            return True, f"检测到比较关键词：{kw}"

    # B. 两个及以上已知实体
    entity_count = 0
    matched_entities: List[str] = []
    for entity in _KNOWN_MODEL_ENTITIES + _KNOWN_DATASET_ENTITIES:
        if entity in q:
            entity_count += 1
            matched_entities.append(entity)
            # 避免同一实体的不同写法重复计数（如 deeplabv3+ / deeplabv3）
            if entity_count >= 2:
                return True, f"检测到多个已知实体：{', '.join(matched_entities[:3])}"

    # C. 多方面分析模式
    for pattern, desc in _MULTI_ASPECT_PATTERNS:
        if re.search(pattern, q):
            return True, f"检测到多方面分析需求：{desc}"

    return False, "该问题不需要复杂分解，建议使用更专注的工具"

#: 查询分解系统提示词
_DECOMPOSE_SYSTEM_PROMPT = (
    "你是一个遥感语义分割领域的查询分解专家。\n"
    "请将用户的复杂问题分解为 2-4 个独立的子查询，"
    "每个子查询聚焦问题的不同方面。\n\n"
    "要求：\n"
    "1. 每个子查询应该是一个完整、可独立检索的查询语句\n"
    "2. 子查询之间应覆盖问题的不同方面，尽量互补\n"
    "3. 使用中文编写子查询\n"
    "4. 严格返回 JSON 格式，不要输出任何其他内容\n\n"
    "返回格式：\n"
    '{"sub_queries": ["子查询1", "子查询2", "子查询3"]}'
)


# -------------------------------------------------------------------------- #
#  查询分解                                                                    #
# -------------------------------------------------------------------------- #

def _parse_decomposition(content: str, fallback_query: str) -> List[str]:
    """从 LLM 返回内容中解析子查询列表。

    支持以下格式：
    - 纯 JSON：{"sub_queries": [...]}
    - Markdown 代码块包裹的 JSON
    - JSON 嵌在自然语言文本中

    解析失败时回退为 [fallback_query]。
    子查询去重（大小写不敏感）后限制在 _MAX_SUB_QUERIES 个以内。

    Args:
        content: LLM 返回的原始文本
        fallback_query: 解析失败时的回退查询

    Returns:
        子查询字符串列表，至少包含 1 个元素
    """
    if not content or not content.strip():
        return [fallback_query] if fallback_query else []

    text = content.strip()

    # 尝试从 markdown 代码块中提取 JSON
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 尝试直接提取 JSON 对象
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        data = json.loads(text)
        raw_queries = data.get("sub_queries", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("查询分解 JSON 解析失败，回退到原始查询")
        return [fallback_query] if fallback_query else []

    if not isinstance(raw_queries, list):
        return [fallback_query] if fallback_query else []

    # 过滤空字符串、去重（大小写不敏感）、限制数量
    seen: set[str] = set()
    queries: List[str] = []
    for q in raw_queries:
        q_str = str(q).strip()
        if not q_str:
            continue
        q_lower = q_str.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            queries.append(q_str)
        if len(queries) >= _MAX_SUB_QUERIES:
            break

    if not queries:
        return [fallback_query] if fallback_query else []

    return queries


def _decompose_query(query: str) -> Tuple[List[str], float]:
    """使用 LLM 将复杂查询分解为 2-4 个子查询。

    Lazy import build_chat_model 以避免循环依赖
    （planning_tools → langchain_agent → planning_tools）。

    Args:
        query: 用户原始查询

    Returns:
        (子查询列表, 分解耗时秒)。分解失败时返回 ([query], elapsed)。
    """
    start = time.time()

    # Lazy import 避免循环依赖
    # 使用 get_agent_llm() 复用 Agent 单例的 ChatOpenAI，
    # 确保 LLM 缓存控制（set_agent_llm_cache）对此调用也生效。
    from app.agents.langchain_agent import get_agent_llm

    try:
        llm = get_agent_llm()
        response = llm.invoke([
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ])
        content = response.content if hasattr(response, "content") else str(response)

        sub_queries = _parse_decomposition(content, fallback_query=query)
        elapsed = time.time() - start

        logger.info(
            "查询分解完成: query=%r → %d 个子查询, 耗时 %.4fs",
            query,
            len(sub_queries),
            elapsed,
        )
        return sub_queries, elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error("查询分解失败，回退到原始查询: %s", e)
        return [query], elapsed


# -------------------------------------------------------------------------- #
#  结果合并 / 去重                                                             #
# -------------------------------------------------------------------------- #

def _merge_search_results(
    sub_query_results: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """合并多个子查询的检索 JSON 结果，按 chunk_id 去重。

    遍历每个子查询的 _cached_search 返回 JSON：
    - 提取 contexts 和 sources（索引一一对应）
    - 以 chunk_id 为 key 去重，保留最高分条目
    - 累加各子查询的 search_elapsed

    Args:
        sub_query_results: 每个子查询的 _cached_search 返回 JSON 字符串列表

    Returns:
        (合并后的 contexts, 合并后的 sources, 总检索耗时)
        contexts 按 score 降序排列，source_id 重新编号。
    """
    # chunk_id → {context, source, score}
    seen: Dict[str, Dict[str, Any]] = {}
    total_elapsed = 0.0

    for sub_json in sub_query_results:
        try:
            data = json.loads(sub_json)
        except (json.JSONDecodeError, TypeError):
            continue

        # 累加检索耗时（无论成功或失败）
        timing = data.get("timing", {})
        if isinstance(timing, dict):
            try:
                total_elapsed += float(timing.get("search_elapsed", 0))
            except (TypeError, ValueError):
                pass

        if not data.get("success", False):
            continue

        contexts = data.get("contexts", [])
        sources = data.get("sources", [])

        for ctx, src in zip(contexts, sources):
            chunk_id = src.get("chunk_id", "")
            score = float(src.get("score", 0.0))

            if chunk_id in seen:
                # 保留更高分的条目
                if score > seen[chunk_id]["score"]:
                    seen[chunk_id] = {
                        "context": ctx,
                        "source": src,
                        "score": score,
                    }
            else:
                seen[chunk_id] = {
                    "context": ctx,
                    "source": src,
                    "score": score,
                }

    # 按分数降序排列
    sorted_entries = sorted(
        seen.values(), key=lambda x: x["score"], reverse=True
    )

    merged_contexts: List[Dict[str, Any]] = []
    merged_sources: List[Dict[str, Any]] = []

    for idx, entry in enumerate(sorted_entries):
        ctx = dict(entry["context"])
        ctx["source_id"] = f"source_{idx + 1}"  # 重新编号
        merged_contexts.append(ctx)
        merged_sources.append(entry["source"])

    return merged_contexts, merged_sources, total_elapsed


# -------------------------------------------------------------------------- #
#  @tool 工具                                                                  #
# -------------------------------------------------------------------------- #

@tool
def plan_and_search(query: str) -> str:
    """Use this tool only for complex multi-entity, multi-aspect comparison questions. Do not use it for simple factual questions, metric calculation, single dataset lookup, or general dataset overview.

    This tool uses an LLM to decompose the question into 2-4 focused sub-queries,
    searches the knowledge base for each, then merges and deduplicates results
    by chunk_id for comprehensive coverage.

    Args:
        query: The complex question to decompose and search.

    Returns:
        A JSON string with merged contexts, sources, sub_queries, and timing.
        Keys: success, tool, query, sub_queries, summary, contexts, sources, timing.
    """
    total_start = time.time()

    normalized = normalize_query(query)

    # 空 query 直接返回
    if not normalized:
        result: Dict[str, Any] = {
            "success": False,
            "tool": "plan_and_search",
            "query": query or "",
            "sub_queries": [],
            "summary": "查询为空",
            "contexts": [],
            "sources": [],
            "timing": {
                "planning_elapsed": 0.0,
                "search_elapsed": 0.0,
                "total_elapsed": 0.0,
            },
            "error": None,
        }
        return json.dumps(result, ensure_ascii=False)

    # ---------- 准入门控 ----------
    # 不适合复杂分解的问题直接返回，不调用 LLM / 不调用向量库
    suitable, gate_reason = should_use_plan_and_search(query)
    if not suitable:
        gate_result: Dict[str, Any] = {
            "success": False,
            "tool": "plan_and_search",
            "query": query,
            "summary": (
                "该问题不需要复杂分解，建议使用 dataset_overview、"
                "dataset_spec_lookup、metric_formula_lookup、"
                "metrics_calculator 或 knowledge_base_search。"
            ),
            "reason": gate_reason,
            "contexts": [],
            "sources": [],
            "timing": {
                "planning_elapsed": 0.0,
                "search_elapsed": 0.0,
                "total_elapsed": 0.0,
            },
        }
        logger.info("plan_and_search 准入门控拦截: query=%r, reason=%s", query, gate_reason)
        return json.dumps(gate_result, ensure_ascii=False)

    settings = get_settings()
    top_k = settings.top_k

    try:
        # Step 1: LLM 查询分解
        sub_queries, planning_elapsed = _decompose_query(query)

        # Step 2: 对每个子查询执行检索（复用 _cached_search LRU 缓存）
        sub_results: List[str] = []
        for sq in sub_queries:
            sq_normalized = normalize_query(sq)
            if not sq_normalized:
                continue
            result_json = _cached_search(sq_normalized, top_k)
            sub_results.append(result_json)

        # Step 3: 合并去重
        merged_contexts, merged_sources, sub_search_total = _merge_search_results(
            sub_results
        )

        total_elapsed = time.time() - total_start

        if not merged_contexts:
            result = {
                "success": False,
                "tool": "plan_and_search",
                "query": query,
                "sub_queries": sub_queries,
                "summary": "查询分解后未检索到相关知识库内容",
                "contexts": [],
                "sources": [],
                "timing": {
                    "planning_elapsed": round(planning_elapsed, 4),
                    "search_elapsed": round(sub_search_total, 4),
                    "total_elapsed": round(total_elapsed, 4),
                },
                "error": None,
            }
            logger.info(
                "plan_and_search 无结果: query=%r, sub_queries=%d",
                query,
                len(sub_queries),
            )
            return json.dumps(result, ensure_ascii=False)

        result = {
            "success": True,
            "tool": "plan_and_search",
            "query": query,
            "sub_queries": sub_queries,
            "summary": (
                f"将查询分解为 {len(sub_queries)} 个子查询，"
                f"合并后去重得到 {len(merged_contexts)} 个相关片段"
            ),
            "contexts": merged_contexts,
            "sources": merged_sources,
            "timing": {
                "planning_elapsed": round(planning_elapsed, 4),
                "search_elapsed": round(sub_search_total, 4),
                "total_elapsed": round(total_elapsed, 4),
            },
        }
        logger.info(
            "plan_and_search 成功: query=%r, sub_queries=%d, 合并 %d 条, "
            "planning=%.4fs, search=%.4fs, total=%.4fs",
            query,
            len(sub_queries),
            len(merged_contexts),
            planning_elapsed,
            sub_search_total,
            total_elapsed,
        )
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        total_elapsed = time.time() - total_start
        logger.error("plan_and_search 异常: %s", e, exc_info=True)
        error_result: Dict[str, Any] = {
            "success": False,
            "tool": "plan_and_search",
            "query": query,
            "sub_queries": [],
            "summary": "查询分解检索失败",
            "contexts": [],
            "sources": [],
            "timing": {
                "planning_elapsed": 0.0,
                "search_elapsed": 0.0,
                "total_elapsed": round(total_elapsed, 4),
            },
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False)
