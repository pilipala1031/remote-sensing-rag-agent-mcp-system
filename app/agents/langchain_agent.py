"""LangChain create_agent 工厂：构建 Agent 编译图并执行问答。

使用 langchain.agents.create_agent 创建 Agent，
模型通过 langchain_openai.ChatOpenAI 接入 OpenAI 兼容接口（GLM / DeepSeek / Qwen），
配置全部从 .env 读取，不硬编码。

性能优化：
- get_remote_sensing_agent() 使用 @lru_cache(maxsize=1) 单例化，
  避免每次请求重复 create_agent / build_chat_model。
  注意：如果 .env 模型配置（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）变化，
  需要重启服务才能生效（lru_cache 不会自动刷新）。
- run_langchain_agent 返回 timing 字段，记录总耗时 / agent.invoke 耗时 / 工具检索总耗时。

注意：
- 不直接使用 langgraph.StateGraph，
  create_agent 内部返回的 CompiledStateGraph 由 LangChain 管理。
- 不修改 app/core/llm.py 中的 OpenAICompatibleLLMClient，
  RAG 问答仍使用原有 LLM Client，Agent 层单独使用 ChatOpenAI。
- 不限制 Agent 的工具调用次数或 LLM 调用轮次。
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from functools import lru_cache
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain_core.caches import InMemoryCache, RETURN_VAL_TYPE
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.agents.domain_tools import (
    dataset_overview,
    dataset_spec_lookup,
    metric_formula_lookup,
    metrics_calculator,
    model_comparison_table,
)
from app.agents.planning_tools import plan_and_search
from app.agents.prompts import REMOTE_SENSING_AGENT_SYSTEM_PROMPT
from app.agents.tools import knowledge_base_search, parse_tool_result
from app.agents.verification import (
    make_deferred_pending_result,
    make_off_result,
    verify_answer,
)
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Agent 可用工具列表（Multi-Tool 架构：7 个工具）
DEFAULT_TOOLS: List[BaseTool] = [
    knowledge_base_search,
    plan_and_search,
    dataset_overview,
    dataset_spec_lookup,
    model_comparison_table,
    metric_formula_lookup,
    metrics_calculator,
]

# 拒答关键词（与 prompts.py 中拒答文案一致）
REFUSAL_MARKER = "根据当前知识库内容，无法确定该问题的答案。"

# 异常兜底文案
ERROR_ANSWER = "Agent 执行过程中出现异常，无法完成回答。"


# -------------------------------------------------------------------------- #
#  模型构建                                                                  #
# -------------------------------------------------------------------------- #

def build_chat_model() -> ChatOpenAI:
    """从 .env 配置构建 ChatOpenAI 模型实例。

    create_agent 要求模型支持 bind_tools()，
    因此使用 ChatOpenAI 而非 app/core/llm.py 中的 OpenAICompatibleLLMClient
    （后者继承 BaseChatModel 但未实现 bind_tools）。

    配置来源（app/config.py Settings）：
        - settings.llm_api_key   ← .env LLM_API_KEY
        - settings.llm_base_url  ← .env LLM_BASE_URL
        - settings.llm_model     ← .env LLM_MODEL

    Returns:
        ChatOpenAI: 可用于 create_agent 的 LangChain chat model。

    Raises:
        ValueError: 当 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 任一缺失时。
    """
    settings = get_settings()

    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY 未配置，请在 .env 中设置 LLM_API_KEY")
    if not settings.llm_base_url:
        raise ValueError("LLM_BASE_URL 未配置，请在 .env 中设置 LLM_BASE_URL")
    if not settings.llm_model:
        raise ValueError("LLM_MODEL 未配置，请在 .env 中设置 LLM_MODEL")

    llm_kwargs: dict[str, Any] = dict(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
    )
    # 仅当 AGENT_MAX_TOKENS > 0 时传 max_tokens，避免不支持该参数的模型报错
    if settings.agent_max_tokens and settings.agent_max_tokens > 0:
        llm_kwargs["max_tokens"] = settings.agent_max_tokens

    llm = ChatOpenAI(**llm_kwargs)
    logger.info(
        "Agent ChatOpenAI 已构建: model=%s, base_url=%s, max_tokens=%s",
        settings.llm_model,
        settings.llm_base_url,
        llm_kwargs.get("max_tokens", "N/A"),
    )
    return llm


# 向后兼容别名：早期代码使用 build_llm 名称
build_llm = build_chat_model


# -------------------------------------------------------------------------- #
#  LLM 响应缓存（仅 Agent 路径 ChatOpenAI 生效）                              #
# -------------------------------------------------------------------------- #
#
# 设计要点：
# - InMemoryCache 子类增加命中 / 未命中计数，便于前端展示缓存命中情况。
# - 缓存实例是模块级共享的：多次请求之间共享缓存条目（进程级生命周期）。
# - 通过修改 ChatOpenAI 的 cache 属性（True / False）实现逐请求开关。
# - 注意：这是全局可变状态，非线程安全；当前部署为单进程同步模型，可接受。
#
# tool_call_id 归一化：
# - LangChain _generate_with_cache 已将 message.id 设为 None，
#   但 tool_calls[].id 和 ToolMessage.tool_call_id 未归一化。
# - 这些 ID 每次调用不同（由 LLM API / LangGraph 生成），
#   导致 Agent 第二轮 LLM 调用（含 tool 结果）永远 cache miss。
# - 在 lookup / update 中统一归一化，使缓存 key 只依赖语义内容。
#


#: 匹配 tool_calls 数组中的 "id": "call_xxx" （字符串值，非 LC 类型标识符或 null）
_TOOL_CALL_ID_RE = re.compile(r'("id":\s*)"[^"]*"')
#: 匹配 ToolMessage 中的 "tool_call_id": "call_xxx"
_TOOL_CALL_REF_RE = re.compile(r'("tool_call_id":\s*)"[^"]*"')


def _normalize_cache_key(key: str) -> str:
    """归一化缓存 key，移除 tool_call_id 中的非确定性 ID。

    安全性说明：
    - message.id 已被 _generate_with_cache 设为 None（"id": null），不被匹配
    - LC 类型标识符 "id": ["langchain", ...] 是列表，不被匹配
    - 仅匹配字符串值的 "id": "..."（即 tool_call ID）和 "tool_call_id": "..."
    """
    key = _TOOL_CALL_ID_RE.sub('\\1""', key)
    key = _TOOL_CALL_REF_RE.sub('\\1""', key)
    return key


class _TrackingInMemoryCache(InMemoryCache):
    """带命中 / 未命中计数 + tool_call_id 归一化的 InMemoryCache。

    在 InMemoryCache.lookup / update 基础上：
    1. 归一化 prompt 中的 tool_call_id，使缓存 key 只依赖语义内容
    2. 统计 hits / misses，供前端展示本次请求的缓存命中情况
    3. miss 时记录 prompt 的 md5 哈希，便于诊断 key 不匹配的根因
    """

    def __init__(self, *, maxsize: int | None = None) -> None:
        super().__init__(maxsize=maxsize)
        self._hits = 0
        self._misses = 0
        #: 记录所有 lookup/update 的 key md5，用于诊断
        self._key_log: list[dict] = []

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        normalized_prompt = _normalize_cache_key(prompt)
        result = super().lookup(normalized_prompt, llm_string)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
            if logger.isEnabledFor(logging.DEBUG):
                key_hash = hashlib.md5(normalized_prompt.encode()).hexdigest()[:12]
                self._key_log.append({"action": "lookup_miss", "key": key_hash})
        return result

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        normalized_prompt = _normalize_cache_key(prompt)
        super().update(normalized_prompt, llm_string, return_val)

    def get_stats(self) -> dict:
        """返回当前命中统计。"""
        return {"hits": self._hits, "misses": self._misses}

    def get_key_log(self) -> list[dict]:
        """返回 key 日志，用于诊断缓存不匹配根因。"""
        return self._key_log

    def reset_stats(self) -> None:
        """清零命中统计（每次请求前调用）。"""
        self._hits = 0
        self._misses = 0
        self._key_log = []


#: 模块级共享缓存实例（进程级生命周期，跨请求共享）
_agent_cache = _TrackingInMemoryCache()

#: 模块级 LLM 引用（由 get_remote_sensing_agent 首次构建时设置）
_agent_llm: ChatOpenAI | None = None


def set_agent_llm_cache(enabled: bool | None) -> bool:
    """切换 Agent ChatOpenAI 的 LLM 缓存。

    通过修改 ChatOpenAI.cache 属性实现逐请求开关：
    - enabled=True  → 挂载共享 _TrackingInMemoryCache
    - enabled=False → 关闭缓存
    - enabled=None  → 使用 .env 中 ENABLE_AGENT_CACHE 配置

    首次调用时若 _agent_llm 尚未初始化，会触发 Agent 单例构建。

    Args:
        enabled: 是否启用缓存；None 则读取配置默认值。

    Returns:
        实际是否启用了缓存。
    """
    global _agent_llm
    if _agent_llm is None:
        # 触发单例构建（会设置 _agent_llm）
        get_remote_sensing_agent()

    settings = get_settings()
    actual = enabled if enabled is not None else settings.enable_agent_cache

    if actual:
        _agent_llm.cache = _agent_cache
        logger.info("Agent LLM 缓存已启用（InMemoryCache）")
    else:
        _agent_llm.cache = False
        logger.info("Agent LLM 缓存已关闭")

    return actual


def reset_agent_cache_stats() -> None:
    """清零缓存命中统计（每次请求前调用）。"""
    _agent_cache.reset_stats()


def get_agent_cache_stats() -> dict:
    """返回当前缓存命中统计。"""
    return _agent_cache.get_stats()


def get_agent_cache_key_log() -> list[dict]:
    """返回缓存 key 日志，用于诊断缓存不匹配根因。"""
    return _agent_cache.get_key_log()


def clear_agent_llm_cache() -> None:
    """清空缓存中的所有条目（用于文档变更后避免过期回答）。"""
    _agent_cache.clear()
    logger.info("Agent LLM 缓存已清空（所有条目移除）")


def get_agent_llm() -> ChatOpenAI:
    """获取 Agent 的 ChatOpenAI 实例（与 _agent_llm 同一对象）。

    供 planning_tools 等内部模块复用，避免创建独立的 ChatOpenAI
    导致缓存控制失效（新实例 cache=False，不受 set_agent_llm_cache 影响）。

    Returns:
        Agent 单例使用的 ChatOpenAI 实例。
    """
    global _agent_llm
    if _agent_llm is None:
        get_remote_sensing_agent()
    return _agent_llm


# -------------------------------------------------------------------------- #
#  Agent 构建                                                                #
# -------------------------------------------------------------------------- #

def build_remote_sensing_agent(tools: Optional[List[BaseTool]] = None) -> Any:
    """构建遥感领域研究助手 Agent。

    使用 create_agent 创建 Agent 编译图，
    默认工具为 DEFAULT_TOOLS（5 个工具：knowledge_base_search +
    dataset_spec_lookup + model_comparison_table +
    metric_formula_lookup + metrics_calculator），
    系统提示词为 REMOTE_SENSING_AGENT_SYSTEM_PROMPT。

    Args:
        tools: 工具列表，为 None 则使用 DEFAULT_TOOLS

    Returns:
        CompiledStateGraph: 可通过 .invoke({"messages": [...]}) 调用。
    """
    model = build_chat_model()
    agent_tools = tools if tools is not None else DEFAULT_TOOLS

    logger.info("正在构建遥感 Agent，工具数=%d", len(agent_tools))
    agent = create_agent(
        model=model,
        tools=agent_tools,
        system_prompt=REMOTE_SENSING_AGENT_SYSTEM_PROMPT,
    )
    logger.info("遥感 Agent 构建完成")
    return agent


# 向后兼容别名
def build_agent(tools: Optional[List[BaseTool]] = None) -> Any:
    """build_remote_sensing_agent 的别名（向后兼容）。"""
    return build_remote_sensing_agent(tools)


# -------------------------------------------------------------------------- #
#  Agent 单例（@lru_cache 避免每次请求重复 create_agent）                     #
# -------------------------------------------------------------------------- #
#
# 注意：lru_cache 会缓存第一次构建的 Agent 实例。
# 如果 .env 中 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 发生变化，
# 需要重启服务才能生效。
# 可通过 get_remote_sensing_agent.cache_clear() 手动清除缓存（主要用于测试）。
#


@lru_cache(maxsize=1)
def get_remote_sensing_agent() -> Any:
    """获取遥感 Agent 单例。

    首次调用时构建 Agent（build_chat_model + create_agent），
    后续调用直接返回缓存的实例。

    重要：Agent 保持完整的自主工具调用能力，
    不限制工具调用次数或 LLM 调用轮次。
    create_agent 内部的 agent loop 由 LangChain 管理，
    模型可以自主决定调用工具 0 次、1 次或多次。

    Returns:
        CompiledStateGraph: 可通过 .invoke({"messages": [...]}) 调用。
    """
    global _agent_llm
    logger.info("首次构建遥感 Agent 单例")
    _agent_llm = build_chat_model()
    agent = create_agent(
        model=_agent_llm,
        tools=DEFAULT_TOOLS,
        system_prompt=REMOTE_SENSING_AGENT_SYSTEM_PROMPT,
    )
    logger.info("遥感 Agent 单例构建完成，后续请求将复用此实例")
    return agent


# -------------------------------------------------------------------------- #
#  sources 去重裁剪 & tool_calls 裁剪                                          #
# -------------------------------------------------------------------------- #

#: sources 去重后最大保留条数
_MAX_SOURCES = 5

#: sources content_preview 截断长度
_SOURCE_PREVIEW_MAX = 150

#: tool_calls output_summary 截断长度
_TOOL_OUTPUT_SUMMARY_MAX = 200

#: tool_calls error 截断长度
_TOOL_ERROR_MAX = 200


def _safe_truncate(text: Any, max_chars: int) -> str:
    """安全截断文本，None 或空值返回空字符串。"""
    if not text:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."


def deduplicate_and_trim_sources(
    sources: list,
    max_sources: int = _MAX_SOURCES,
    preview_max_chars: int = _SOURCE_PREVIEW_MAX,
) -> list:
    """对 sources 按 chunk_id 去重并裁剪。

    规则：
    - 按 chunk_id 去重；若无 chunk_id 则按 filename + page + content_preview 去重。
    - 若 score 是数值，按 score 降序排列。
    - 最多保留 max_sources 条。
    - content_preview 截断到 preview_max_chars。
    - 字段缺失时不崩溃。
    - 空 sources 返回 []。

    Args:
        sources: 原始 source 字典列表
        max_sources: 最大保留条数
        preview_max_chars: content_preview 截断长度

    Returns:
        去重裁剪后的 source 字典列表
    """
    if not sources or not isinstance(sources, list):
        return []

    seen: dict[str, dict] = {}

    for src in sources:
        if not isinstance(src, dict):
            continue

        chunk_id = src.get("chunk_id", "")
        if chunk_id:
            dedup_key = chunk_id
        else:
            filename = src.get("filename", "")
            page = src.get("page", "")
            preview = src.get("content_preview", "")
            dedup_key = f"{filename}:{page}:{preview}"

        score = src.get("score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        if dedup_key in seen:
            if score > seen[dedup_key]["score"]:
                seen[dedup_key] = {**src, "score": score}
        else:
            seen[dedup_key] = {**src, "score": score}

    sorted_sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    trimmed = sorted_sources[:max_sources]

    result = []
    for src in trimmed:
        item = dict(src)
        if "content_preview" in item:
            item["content_preview"] = _safe_truncate(
                item.get("content_preview"), preview_max_chars
            )
        result.append(item)

    return result


def trim_tool_calls(
    tool_calls: list,
    output_summary_max_chars: int = _TOOL_OUTPUT_SUMMARY_MAX,
    error_max_chars: int = _TOOL_ERROR_MAX,
) -> list:
    """裁剪 tool_calls，只保留必需字段并截断长文本。

    规则：
    - 只保留 tool / input / status / output_summary / elapsed / error。
    - output_summary 截断到 output_summary_max_chars。
    - error 截断到 error_max_chars。
    - 不返回完整 tool raw output。
    - 字段缺失时不崩溃。

    Args:
        tool_calls: 原始 tool_call 字典列表
        output_summary_max_chars: output_summary 截断长度
        error_max_chars: error 截断长度

    Returns:
        裁剪后的 tool_call 字典列表
    """
    if not tool_calls or not isinstance(tool_calls, list):
        return []

    keep_keys = {"tool", "input", "status", "output_summary", "elapsed", "error"}
    result = []

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        item = {}
        for k in keep_keys:
            if k in tc:
                val = tc[k]
                if k == "output_summary":
                    item[k] = _safe_truncate(val, output_summary_max_chars)
                elif k == "error":
                    item[k] = _safe_truncate(val, error_max_chars)
                else:
                    item[k] = val
        # 确保基本字段存在
        item.setdefault("tool", "unknown")
        item.setdefault("status", "unknown")
        result.append(item)

    return result


# -------------------------------------------------------------------------- #
#  Agent 执行 + 结果解析                                                     #
# -------------------------------------------------------------------------- #

_TOOL_PRIMARY_ARG: dict[str, str] = {
    "knowledge_base_search": "query",
    "plan_and_search": "query",
    "dataset_overview": "query",
    "dataset_spec_lookup": "dataset_name",
    "model_comparison_table": "models",
    "metric_formula_lookup": "metric_name",
}


def _extract_tool_input(tool_name: str, tool_args: Any) -> str:
    """从工具调用参数中提取人类可读的输入摘要。

    不同工具使用不同的参数名，此函数统一提取主要输入。
    metrics_calculator 特殊处理：拼接 metric_name + values。
    """
    if not isinstance(tool_args, dict):
        return str(tool_args)

    if tool_name == "metrics_calculator":
        parts = []
        for k in ("metric_name", "values"):
            if k in tool_args:
                parts.append(str(tool_args[k]))
        return ", ".join(parts) if parts else str(tool_args)

    arg_key = _TOOL_PRIMARY_ARG.get(tool_name)
    if arg_key:
        return str(tool_args.get(arg_key, tool_args))
    return str(tool_args)


def _parse_agent_result(result: dict, invoke_elapsed: float = 0.0) -> dict:
    """从 agent.invoke() 返回结果中提取 answer / sources / tool_calls / trace。

    遍历 messages 列表，按消息类型提取信息：
    - AIMessage: 提取 tool_calls 请求（支持多个不同工具）+ 最终回答文本
    - ToolMessage: 解析工具返回的 JSON，提取 sources + elapsed

    多工具支持：
    - 所有工具的返回 JSON 都通过 parse_tool_result 解析（具有缺省 key 的 fallback）
    - 只有 knowledge_base_search 返回 sources；其他结构化工具无 sources 是正常行为
    - 只有 knowledge_base_search 返回 timing.search_elapsed；其他工具的 elapsed 为 None
    - agent_trace 中以 tool_called:<name> 记录每个工具调用

    trace_events：
    - 与 agent_trace 互补的结构化事件列表
    - 每条事件携带 step / event / timestamp / detail
    - timestamp 基于 tool elapsed 累积估算（非真实 wall-clock）

    Args:
        result: agent.invoke() 返回的 dict。
        invoke_elapsed: agent.invoke 总耗时（秒），用于 agent_finished 事件时间戳。

    Returns:
        dict，包含 answer / sources / refused / tool_calls / agent_trace /
        trace_events / errors 以及 _tool_search_elapsed_total（内部用）
    """
    messages = result.get("messages", [])

    answer = ""
    sources: list[dict] = []
    tool_calls: list[dict] = []
    agent_trace: list[str] = ["agent_started"]
    trace_events: list[dict] = []
    tool_was_called = False
    tool_search_elapsed_total = 0.0

    # 结构化轨迹事件辅助函数
    _event_step = 0

    def _add_trace_event(event: str, detail: str | None = None, timestamp: float = 0.0) -> None:
        nonlocal _event_step
        _event_step += 1
        trace_events.append({
            "step": _event_step,
            "event": event,
            "timestamp": round(timestamp, 4),
            "detail": detail,
        })

    _add_trace_event("agent_started", timestamp=0.0)

    # 暂存 tool_call 请求，按 id 匹配后续 ToolMessage
    pending_calls: dict[str, dict] = {}
    # 累积工具耗时，用于估算各事件时间戳
    cumulative_tool_time = 0.0

    for msg in messages:
        msg_type = getattr(msg, "type", "")

        # ---------- AIMessage ----------
        if msg_type == "ai":
            # 提取 tool_calls（工具调用请求）
            tc_list = getattr(msg, "tool_calls", None) or []
            for tc in tc_list:
                tool_was_called = True
                tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

                # 提取 input（适应不同工具的参数名）
                tc_input = _extract_tool_input(tc_name, tc_args)

                agent_trace.append(f"tool_called:{tc_name}")
                _add_trace_event("tool_called", detail=tc_name, timestamp=cumulative_tool_time)

                call_record: dict[str, Any] = {
                    "tool": tc_name,
                    "input": tc_input,
                    "status": "success",
                    "output_summary": None,
                    "elapsed": None,
                    "error": None,
                }
                tool_calls.append(call_record)
                pending_calls[tc_id] = call_record

            # 提取最终回答（最后一条非空 content 的 AIMessage）
            content = getattr(msg, "content", "")
            if content:
                answer = content

        # ---------- ToolMessage ----------
        elif msg_type == "tool":
            content = getattr(msg, "content", "")
            tool_call_id = getattr(msg, "tool_call_id", "")
            # ToolMessage 携带的工具名（用于日志区分）
            msg_tool_name = getattr(msg, "name", "unknown")

            # 用 parse_tool_result 解析 JSON（对所有工具统一解析，
            # 缺省 key 有 fallback：非 knowledge_base_search 工具
            # 的 sources 为 []、elapsed 为 None，不会报错）
            parsed = parse_tool_result(content)
            tool_sources = parsed.get("sources", [])
            sources.extend(tool_sources)

            # 累加工具检索耗时（仅 knowledge_base_search 返回 timing）
            elapsed = parsed.get("elapsed")
            if isinstance(elapsed, (int, float)):
                tool_search_elapsed_total += float(elapsed)

            # 更新对应的 tool_call 记录
            call_record = pending_calls.get(tool_call_id)
            if call_record is not None:
                call_record["output_summary"] = parsed.get("summary", "")
                call_record["elapsed"] = elapsed
                if not parsed.get("success", False):
                    call_record["status"] = "error"
                    err = parsed.get("error")
                    call_record["error"] = err

            agent_trace.append("tool_result_parsed")

            # 累积工具耗时用于后续事件时间戳估算
            if isinstance(elapsed, (int, float)):
                cumulative_tool_time += float(elapsed)
            _add_trace_event("tool_result_parsed", timestamp=cumulative_tool_time, detail=msg_tool_name)

    # ---------- 收尾 ----------
    if tool_was_called:
        agent_trace.append("agent_finished")
        _add_trace_event("agent_finished", timestamp=invoke_elapsed)
    else:
        agent_trace = ["no_tool_called"]
        _add_trace_event("no_tool_called", timestamp=invoke_elapsed)

    # 判断拒答
    refused = REFUSAL_MARKER in answer

    # ---------- sources 去重裁剪 & tool_calls 裁剪 ----------
    sources = deduplicate_and_trim_sources(sources)
    tool_calls = trim_tool_calls(tool_calls)

    logger.info(
        "Agent 结果解析完成: answer_len=%d, sources=%d, tool_calls=%d, trace_events=%d, "
        "refused=%s, tool_search_elapsed=%.4fs",
        len(answer),
        len(sources),
        len(tool_calls),
        len(trace_events),
        refused,
        tool_search_elapsed_total,
    )

    return {
        "answer": answer,
        "sources": sources,
        "refused": refused,
        "tool_calls": tool_calls,
        "agent_trace": agent_trace,
        "trace_events": trace_events,
        "errors": [],
        "_tool_search_elapsed_total": tool_search_elapsed_total,
    }


def run_langchain_agent(question: str, agent: Any = None) -> dict:
    """执行遥感领域研究助手 Agent，返回结构化结果。

    性能优化：
    - 当 agent=None 时，使用 get_remote_sensing_agent() 单例，避免重复 create_agent。
    - 返回 timing 字段记录各阶段耗时。

    Args:
        question: 用户问题
        agent: 可选的预构建 Agent 实例（便于测试注入 mock）。
               为 None 则使用 get_remote_sensing_agent() 单例。

    Returns:
        dict，统一格式：
        {
            "answer": str,
            "sources": list,
            "refused": bool,
            "tool_calls": list,
            "agent_trace": list,
            "errors": list,
            "timing": {
                "total_elapsed": float,
                "agent_invoke_elapsed": float,
                "tool_search_elapsed_total": float
            }
        }
    """
    logger.info("Agent 开始处理问题: %r", question)
    total_start = time.time()

    try:
        # 使用单例 Agent（或测试注入的 mock）
        actual_agent = agent if agent is not None else get_remote_sensing_agent()

        invoke_start = time.time()
        result = actual_agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question},
                ]
            }
        )
        agent_invoke_elapsed = time.time() - invoke_start

        parsed = _parse_agent_result(result, invoke_elapsed=agent_invoke_elapsed)

        # 从 _parse_agent_result 中提取工具检索总耗时
        tool_search_elapsed_total = parsed.pop("_tool_search_elapsed_total", 0.0)

        # ---------- Evidence Verification（证据校验） ----------
        # 支持三种模式：off / sync / deferred
        # off: 不执行校验，返回 enabled=false
        # sync: 同步执行校验
        # deferred: 跳过校验，返回 pending=true，前端再调 /api/agent/verify
        settings = get_settings()
        v_mode = settings.agent_verification_mode

        if v_mode == "off" or not settings.enable_agent_verification:
            parsed["verification"] = make_off_result()
        elif v_mode == "sync":
            verification_result = verify_answer(
                question=question,
                answer=parsed["answer"],
                sources=parsed["sources"],
                tool_calls=parsed.get("tool_calls"),
            )
            parsed["verification"] = verification_result
        else:
            # deferred（默认）
            parsed["verification"] = make_deferred_pending_result(
                settings.agent_verification_level
            )

        total_elapsed = time.time() - total_start

        parsed["timing"] = {
            "total_elapsed": round(total_elapsed, 4),
            "agent_invoke_elapsed": round(agent_invoke_elapsed, 4),
            "tool_search_elapsed_total": round(tool_search_elapsed_total, 4),
        }

        return parsed

    except Exception as e:
        logger.error("Agent 执行异常: %s", e, exc_info=True)
        total_elapsed = time.time() - total_start
        s = get_settings()
        return {
            "answer": ERROR_ANSWER,
            "sources": [],
            "refused": True,
            "tool_calls": [],
            "agent_trace": ["agent_started", "agent_error"],
            "trace_events": [
                {"step": 1, "event": "agent_started", "timestamp": 0.0, "detail": None},
                {"step": 2, "event": "agent_error", "timestamp": round(total_elapsed, 4), "detail": str(e)[:200]},
            ],
            "errors": [str(e)],
            "verification": {
                "enabled": s.enable_agent_verification and s.agent_verification_mode != "off",
                "mode": s.agent_verification_mode,
                "level": s.agent_verification_level,
                "pending": False,
                "verified": None,
                "confidence": None,
                "ungrounded_claims": [],
                "reason": "Agent 执行异常，未进行证据校验。",
                "timing": {"verification_elapsed": 0.0},
            },
            "timing": {
                "total_elapsed": round(total_elapsed, 4),
                "agent_invoke_elapsed": 0.0,
                "tool_search_elapsed_total": 0.0,
            },
        }
