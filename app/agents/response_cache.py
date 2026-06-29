"""Agent 响应级缓存（Response Cache）。

在 AgentService.query 层缓存完整 Agent 响应，
相同问题 + 相同配置第二次直接返回缓存结果，零 LLM / 工具调用。

设计要点：
1. 缓存 key 包含：归一化问题文本 + 检索/模型/校验配置 + 语料库版本 + 领域数据哈希
   + Prompt 文本哈希，确保配置变更、知识库更新或 prompt 调整后不会返回过期回答。
2. TTL 过期自动失效；max_size 满时淘汰最旧条目。
3. 文档入库 / 删除时调用 clear_agent_response_cache() 主动清空。
4. 可通过 ENABLE_AGENT_RESPONSE_CACHE 配置开关（默认 True）。

与 LangChain LLM Cache（enable_agent_cache）的区别：
- LLM Cache 缓存单次 LLM 调用（Agent 多轮循环中仅 Round 1 能命中）
- Response Cache 缓存整个 Agent 响应（命中后零开销直接返回）
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, Optional

from app.agents.tools import normalize_query
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------------- #
#  语料库版本 & 领域数据哈希                                                   #
# -------------------------------------------------------------------------- #

_corpus_version_cache: Dict[str, Any] = {"hash": "", "ts": 0.0}
#: 语料库版本缓存有效期（秒），避免每次 cache key 计算都查询 Chroma
_CORPUS_VERSION_TTL = 30.0


def _compute_corpus_version() -> str:
    """计算当前知识库语料的版本哈希。

    基于 Chroma 中所有文档的 doc_id + chunk_count 生成 md5 哈希。
    文档入库 / 删除 / 修改后，哈希值会变化，使缓存 key 自然失效。

    带 30 秒短期缓存，避免每次 Agent 请求都查询 Chroma。

    Returns:
        语料库版本的 md5 哈希（12 字符）。
    """
    now = time.time()
    if now - _corpus_version_cache["ts"] < _CORPUS_VERSION_TTL:
        return _corpus_version_cache["hash"]

    try:
        from app.services.vector_store import VectorStore

        store = VectorStore()
        docs = store.list_documents()
        # 按 doc_id 排序确保稳定性
        sorted_docs = sorted(docs, key=lambda d: d.get("doc_id", ""))
        version_str = json.dumps(
            [
                {"doc_id": d.get("doc_id", ""), "chunks": d.get("chunk_count", 0)}
                for d in sorted_docs
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        version_hash = hashlib.md5(version_str.encode()).hexdigest()[:12]
    except Exception as e:
        logger.warning("计算语料库版本失败，使用时间戳兜底: %s", e)
        version_hash = f"err_{int(now)}"

    _corpus_version_cache["hash"] = version_hash
    _corpus_version_cache["ts"] = now
    return version_hash


def _compute_domain_data_hash() -> str:
    """计算领域结构化数据（datasets/models/metrics JSON）的哈希。

    确保 domain_data/*.json 内容变更后缓存 key 自然失效。

    Returns:
        领域数据 md5 哈希（12 字符）。
    """
    try:
        from app.agents.domain_data_loader import _DOMAIN_DATA_DIR

        parts = []
        for json_file in sorted(_DOMAIN_DATA_DIR.glob("*.json")):
            content = json_file.read_text(encoding="utf-8")
            parts.append(f"{json_file.name}:{hashlib.md5(content.encode()).hexdigest()[:8]}")
        combined = "|".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    except Exception as e:
        logger.warning("计算领域数据哈希失败: %s", e)
        return "unknown"


@lru_cache(maxsize=1)
def _compute_prompt_hash() -> str:
    """计算所有影响 Agent / RAG 输出的 prompt 文本哈希。

    涵盖以下会进入 LLM 输入的 prompt 常量：
    - RAG 系统提示词 + 拒答文案（app/core/prompts.py）
    - Agent 系统提示词 + 拒答文案（app/agents/prompts.py）
    - 查询分解 prompt（app/agents/planning_tools.py::_DECOMPOSE_SYSTEM_PROMPT）
    - 证据校验 prompt + 用户模板（app/agents/verification.py）

    任一 prompt 文案变化都会导致哈希变化，进而使 Response Cache
    全部 key 自然失效，避免"改了 prompt 但缓存仍返回旧答案"的陷阱。

    使用 lru_cache(1) 确保整个进程生命周期只计算一次：
    - prompt 在运行时是不可变的字符串常量
    - 改 prompt 需要重启服务，重启后 lru_cache 自然清空
    - 单次计算开销 < 1ms（仅哈希几 KB 字符串）

    Returns:
        prompt 文本组合的 sha256 前 12 字符；失败时返回 "unknown"。
    """
    try:
        from app.core.prompts import RAG_SYSTEM_PROMPT, REFUSAL_ANSWER
        from app.agents.prompts import (
            REMOTE_SENSING_AGENT_SYSTEM_PROMPT,
            AGENT_REFUSAL_ANSWER,
        )
        from app.agents.planning_tools import _DECOMPOSE_SYSTEM_PROMPT
        from app.agents.verification import (
            VERIFICATION_SYSTEM_PROMPT,
            VERIFICATION_USER_TEMPLATE,
        )

        # AGENT_SYSTEM_PROMPT 是 REMOTE_SENSING_AGENT_SYSTEM_PROMPT 的别名，
        # 无需重复纳入；其余每个常量都是独立的 prompt 文本。
        parts = [
            RAG_SYSTEM_PROMPT,
            REFUSAL_ANSWER,
            REMOTE_SENSING_AGENT_SYSTEM_PROMPT,
            AGENT_REFUSAL_ANSWER,
            _DECOMPOSE_SYSTEM_PROMPT,
            VERIFICATION_SYSTEM_PROMPT,
            VERIFICATION_USER_TEMPLATE,
        ]
        combined = "\n---\n".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:12]
    except Exception as e:
        logger.warning("计算 prompt 哈希失败: %s", e)
        return "unknown"


# -------------------------------------------------------------------------- #
#  缓存 key 构建                                                              #
# -------------------------------------------------------------------------- #

def build_agent_response_cache_key(
    question: str,
    use_rerank: Optional[bool] = None,
    include_trace: bool = True,
) -> str:
    """构建 Agent 响应缓存的 key。

    key 包含以下因素，确保任一变化时缓存不会错误命中：
    - normalized_question（strip + lowercase + whitespace collapse）
    - use_rerank（None 则解析为 .env 配置值）
    - top_k、similarity_threshold、rerank_candidate_k（检索参数）
    - llm_model、agent_max_tokens（模型配置）
    - verification_mode、verification_level（校验配置）
    - corpus_version（知识库文档列表哈希，带 30s 短期缓存）
    - domain_data_hash（领域 JSON 文件哈希）
    - prompt_hash（所有影响输出的 prompt 文本哈希，进程级缓存）
    - include_trace（是否包含调试信息，影响返回内容）

    Args:
        question: 用户原始问题
        use_rerank: 是否启用 rerank。None 则使用 .env 配置。
        include_trace: 是否包含 trace 信息。

    Returns:
        缓存 key 字符串（sha256 前 32 字符）。
    """
    settings = get_settings()

    # 解析 use_rerank：None → 使用配置默认值
    actual_rerank = use_rerank if use_rerank is not None else settings.use_rerank

    # 归一化问题文本
    normalized_q = normalize_query(question)

    # 语料库版本 & 领域数据哈希 & prompt 哈希
    corpus_ver = _compute_corpus_version()
    domain_hash = _compute_domain_data_hash()
    prompt_hash = _compute_prompt_hash()

    key_parts = [
        f"q={normalized_q}",
        f"rerank={actual_rerank}",
        f"top_k={settings.top_k}",
        f"thresh={settings.similarity_threshold}",
        f"cand_k={settings.rerank_candidate_k}",
        f"model={settings.llm_model}",
        f"max_tok={settings.agent_max_tokens}",
        f"v_mode={settings.agent_verification_mode}",
        f"v_level={settings.agent_verification_level}",
        f"corpus={corpus_ver}",
        f"domain={domain_hash}",
        f"prompt={prompt_hash}",
        f"trace={include_trace}",
    ]

    raw_key = "|".join(key_parts)
    return hashlib.sha256(raw_key.encode()).hexdigest()[:32]


# -------------------------------------------------------------------------- #
#  TTL + MaxSize 有序缓存                                                     #
# -------------------------------------------------------------------------- #

class AgentResponseCache:
    """带 TTL 过期和 max_size 淘汰的有序字典缓存。

    - get(key): 查找缓存，TTL 过期则视为未命中并删除条目。
    - put(key, value): 写入缓存，超出 max_size 时淘汰最旧条目。
    - clear(): 清空所有缓存。
    - stats(): 返回命中 / 未命中 / 条目数统计。

    线程安全说明：当前部署为单进程同步模型，不加锁。
    如需多线程部署，应在外部加锁或改用 threading.Lock。
    """

    def __init__(self, ttl_seconds: int = 600, max_size: int = 100) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[dict]:
        """查找缓存条目。

        TTL 过期的条目会被删除并视为未命中。

        Args:
            key: build_agent_response_cache_key 生成的 key。

        Returns:
            缓存的 Agent 响应 dict，或 None（未命中 / 已过期）。
        """
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        cached_at, value = entry
        now = time.time()

        # TTL 检查（ttl=0 表示永不过期）
        if self._ttl > 0 and (now - cached_at) > self._ttl:
            del self._store[key]
            self._misses += 1
            logger.debug("响应缓存 TTL 过期: key=%s", key)
            return None

        # LRU：移到末尾（最近使用）
        self._store.move_to_end(key)
        self._hits += 1
        logger.debug("响应缓存命中: key=%s", key)
        return value

    def put(self, key: str, value: dict) -> None:
        """写入缓存条目。

        仅缓存非异常结果（refused=False 且 errors 为空，或 refused=True 但
        不是异常拒答时也缓存，避免重复触发慢查询）。
        超出 max_size 时淘汰最旧条目。

        Args:
            key: build_agent_response_cache_key 生成的 key。
            value: Agent 响应 dict。
        """
        # 不缓存异常结果
        if value.get("errors") and any(
            "异常" in str(e) for e in value.get("errors", [])
        ):
            logger.debug("跳过缓存异常结果: key=%s", key)
            return

        now = time.time()

        # 如果 key 已存在，更新值并移到末尾
        if key in self._store:
            self._store[key] = (now, value)
            self._store.move_to_end(key)
            return

        # max_size 淘汰
        while len(self._store) >= self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("响应缓存淘汰最旧条目: key=%s", evicted_key)

        self._store[key] = (now, value)
        logger.debug("响应缓存写入: key=%s, 当前条目数=%d", key, len(self._store))

    def clear(self) -> int:
        """清空所有缓存条目。

        Returns:
            被清除的条目数。
        """
        count = len(self._store)
        self._store.clear()
        logger.info("响应缓存已清空（%d 条）", count)
        return count

    def stats(self) -> dict:
        """返回缓存统计。"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }

    def reset_stats(self) -> None:
        """清零命中统计。"""
        self._hits = 0
        self._misses = 0


# -------------------------------------------------------------------------- #
#  模块级单例                                                                  #
# -------------------------------------------------------------------------- #

#: 模块级缓存单例（进程级生命周期）
_agent_response_cache: AgentResponseCache | None = None


def get_agent_response_cache() -> AgentResponseCache:
    """获取 Agent 响应缓存单例。

    首次调用时根据 .env 配置（TTL、max_size）初始化。
    """
    global _agent_response_cache
    if _agent_response_cache is None:
        settings = get_settings()
        _agent_response_cache = AgentResponseCache(
            ttl_seconds=settings.agent_response_cache_ttl_seconds,
            max_size=settings.agent_response_cache_max_size,
        )
        logger.info(
            "Agent 响应缓存初始化: ttl=%ds, max_size=%d",
            settings.agent_response_cache_ttl_seconds,
            settings.agent_response_cache_max_size,
        )
    return _agent_response_cache


def clear_agent_response_cache() -> int:
    """清空 Agent 响应缓存。

    在文档入库 / 删除时调用，确保知识库更新后不会返回过期缓存。

    Returns:
        被清除的条目数。
    """
    if _agent_response_cache is None:
        return 0
    return _agent_response_cache.clear()


def invalidate_corpus_version() -> None:
    """强制刷新语料库版本缓存。

    在文档入库 / 删除后调用，使下一次 build_agent_response_cache_key
    重新计算语料库哈希。
    """
    global _corpus_version_cache
    _corpus_version_cache["hash"] = ""
    _corpus_version_cache["ts"] = 0.0
