"""Agent 对外服务层：封装 run_langchain_agent，提供统一入口。

职责：
1. 输入校验（空问题拦截）
2. 响应级缓存检查（ENABLE_AGENT_RESPONSE_CACHE 控制开关）
3. 委托 run_langchain_agent 执行 Agent 问答
4. 异常兜底，确保调用方始终拿到结构化 dict

与 RAGService 的关系：
- RAGService（/api/chat/query）流程不变
- RemoteSensingAgentService（/api/agent/query）作为新增能力并存
- Agent 层通过 Tool 复用 Retriever / VectorStore / Embedding / Chroma 底层

缓存层次：
- L1: 响应级缓存（response_cache）— 缓存完整 Agent 响应，命中后零开销
- L2: LangChain LLM Cache — 缓存单次 LLM 调用，Agent 多轮中仅 Round 1 命中
"""
from __future__ import annotations

import logging

from app.agents.langchain_agent import (
    get_agent_cache_key_log,
    get_agent_cache_stats,
    reset_agent_cache_stats,
    run_langchain_agent,
    set_agent_llm_cache,
)
from app.agents.response_cache import (
    build_agent_response_cache_key,
    get_agent_response_cache,
)
from app.agents.tools import set_rerank_override
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 异常兜底文案
ERROR_ANSWER = "Agent 执行过程中出现异常，无法完成回答。"

# 空问题文案
EMPTY_QUESTION_ANSWER = "请输入有效问题。"


class RemoteSensingAgentService:
    """遥感领域 Agent 问答服务。

    封装 run_langchain_agent，提供输入校验和异常兜底。
    所有调用方通过 get_agent_service() 获取单例实例。
    """

    def query(
        self,
        question: str,
        include_trace: bool = True,
        use_rerank: bool | None = None,
        enable_cache: bool | None = None,
    ) -> dict:
        """处理用户问题，返回统一格式的 Agent 回答。

        Args:
            question: 用户问题
            include_trace: 是否在返回中包含 trace/tool_calls 等调试信息。
                为 False 时返回的 agent_trace / trace_events / tool_calls 为空列表，
                可减少响应体积。
            use_rerank: 是否启用 rerank 重排序。None 则使用 .env 配置。
            enable_cache: 是否启用 LLM 响应缓存。None 则使用 .env 中
                ENABLE_AGENT_CACHE 配置。仅对 Agent 路径的 ChatOpenAI 生效。

        Returns:
            dict，固定包含以下 key：
            {
                "answer": str,
                "sources": list,
                "refused": bool,
                "tool_calls": list,
                "agent_trace": list,
                "trace_events": list,
                "errors": list
            }
        """
        # ---------- 输入校验 ----------
        if not question or not question.strip():
            logger.warning("Agent 收到空问题，返回拒答")
            return {
                "answer": EMPTY_QUESTION_ANSWER,
                "sources": [],
                "refused": True,
                "tool_calls": [],
                "agent_trace": [],
                "trace_events": [],
                "errors": ["empty question"],
                "verification": {
                    "enabled": True,
                    "mode": "off",
                    "level": None,
                    "pending": False,
                    "verified": None,
                    "confidence": None,
                    "ungrounded_claims": [],
                    "reason": "空问题，未进行证据校验。",
                    "timing": {"verification_elapsed": 0.0},
                },
                "timing": {
                    "total_elapsed": 0.0,
                    "agent_invoke_elapsed": 0.0,
                    "tool_search_elapsed_total": 0.0,
                },
            }

        question = question.strip()
        logger.info("AgentService 开始处理问题: %r", question)

        # ---------- 响应级缓存检查（L1） ----------
        settings = get_settings()
        response_cache_enabled = settings.enable_agent_response_cache

        if response_cache_enabled:
            cache = get_agent_response_cache()
            cache_key = build_agent_response_cache_key(
                question=question,
                use_rerank=use_rerank,
                include_trace=include_trace,
            )
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info("Agent 响应缓存命中，直接返回缓存结果")
                cached_result.setdefault("timing", {})
                cached_result["timing"]["response_cache_hit"] = True
                return cached_result
            logger.info("Agent 响应缓存未命中，执行 Agent")

        # ---------- 执行 Agent ----------
        try:
            set_rerank_override(use_rerank)

            # 设置 LLM 缓存开关（仅 Agent 路径 ChatOpenAI 生效）
            cache_enabled = set_agent_llm_cache(enable_cache)
            if cache_enabled:
                reset_agent_cache_stats()

            result = run_langchain_agent(question)

            # 执行完毕后重置标志
            set_rerank_override(None)
            # 每次请求结束后关闭缓存，下次请求由 enable_cache 重新决定
            set_agent_llm_cache(False)

            # 注入缓存命中统计到 timing
            if cache_enabled:
                stats = get_agent_cache_stats()
                result.setdefault("timing", {})
                result["timing"]["cache_enabled"] = True
                result["timing"]["cache_hits"] = stats["hits"]
                result["timing"]["cache_misses"] = stats["misses"]
                logger.info(
                    "Agent LLM 缓存统计: hits=%d, misses=%d "
                    "(misses=首次调用写入缓存的次数，hits=直接返回缓存的次数)",
                    stats["hits"],
                    stats["misses"],
                )
                # 诊断 key hash 仅在 DEBUG 级别输出
                if logger.isEnabledFor(logging.DEBUG):
                    key_log = get_agent_cache_key_log()
                    logger.debug(
                        "Agent LLM 缓存 key 日志: %s",
                        key_log,
                    )
            else:
                result.setdefault("timing", {})
                result["timing"]["cache_enabled"] = False

            # include_trace=False 时清除调试信息（不改变 answer/sources/refused/timing/verification）
            if not include_trace:
                result["tool_calls"] = []
                result["agent_trace"] = []
                result["trace_events"] = []

            # ---------- 写入响应级缓存（L1） ----------
            if response_cache_enabled:
                result.setdefault("timing", {})
                result["timing"]["response_cache_hit"] = False
                try:
                    cache.put(cache_key, result)
                    logger.info("Agent 响应已写入缓存")
                except Exception as cache_err:
                    logger.warning("写入响应缓存失败: %s", cache_err)

            logger.info(
                "AgentService 完成: refused=%s, sources=%d, include_trace=%s",
                result.get("refused"),
                len(result.get("sources", [])),
                include_trace,
            )
            return result

        except Exception as e:
            logger.error("AgentService 执行异常: %s", e, exc_info=True)
            return {
                "answer": ERROR_ANSWER,
                "sources": [],
                "refused": True,
                "tool_calls": [],
                "agent_trace": ["agent_service_error"],
                "trace_events": [],
                "errors": [str(e)],
                "verification": {
                    "enabled": True,
                    "mode": "off",
                    "level": None,
                    "pending": False,
                    "verified": None,
                    "confidence": None,
                    "ungrounded_claims": [],
                    "reason": "Agent 服务层异常，未进行证据校验。",
                    "timing": {"verification_elapsed": 0.0},
                },
                "timing": {
                    "total_elapsed": 0.0,
                    "agent_invoke_elapsed": 0.0,
                    "tool_search_elapsed_total": 0.0,
                },
            }


# 向后兼容别名
AgentService = RemoteSensingAgentService


# -------------------------------------------------------------------------- #
#  单例工厂（与 chat.py get_rag_service 风格一致）                            #
# -------------------------------------------------------------------------- #

_agent_service: RemoteSensingAgentService | None = None


def get_agent_service() -> RemoteSensingAgentService:
    """获取 Agent 服务单例。

    与 app/api/chat.py 中 get_rag_service() 风格一致，
    避免每次请求重复创建实例。
    """
    global _agent_service
    if _agent_service is None:
        _agent_service = RemoteSensingAgentService()
    return _agent_service
