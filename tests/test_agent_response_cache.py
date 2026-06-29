"""Agent 响应级缓存（Response Cache）单元测试。

覆盖场景：
1. 相同问题命中 — 第二次调用直接返回缓存
2. 不同问题不命中 — 不同问题不会错误命中
3. TTL 过期后不命中 — 超时后缓存自动失效
4. max_size 淘汰 — 超出容量后淘汰最旧条目
5. 文档入库后缓存清空 — clear_agent_response_cache() 清空全部
6. 文档删除后缓存清空 — 同上
7. use_rerank 变化导致 key 不同 — 配置不同时缓存隔离
8. 异常结果不被缓存 — errors 包含异常信息时跳过缓存写入
9. prompt 文本变化导致 key 不同 — 改 prompt 后自动失效，避免返回过期答案
10. prompt 未变化时 key 稳定 — 不破坏现有缓存命中率
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from app.agents.response_cache import (
    AgentResponseCache,
    build_agent_response_cache_key,
    clear_agent_response_cache,
    get_agent_response_cache,
    invalidate_corpus_version,
)


# -------------------------------------------------------------------------- #
#  公共 fixture                                                               #
# -------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _reset_response_cache():
    """每个测试前重置响应缓存单例和语料库版本缓存。"""
    # 重置模块级单例
    import app.agents.response_cache as rc_module
    rc_module._agent_response_cache = None
    rc_module._corpus_version_cache = {"hash": "", "ts": 0.0}
    # 清空 prompt 哈希的 lru_cache，避免跨测试污染
    rc_module._compute_prompt_hash.cache_clear()
    yield
    # 测试后清理
    rc_module._agent_response_cache = None
    rc_module._corpus_version_cache = {"hash": "", "ts": 0.0}
    rc_module._compute_prompt_hash.cache_clear()


def _make_mock_settings(
    enable_response_cache=True,
    ttl_seconds=600,
    max_size=100,
    use_rerank=False,
    top_k=5,
    similarity_threshold=0.3,
    rerank_candidate_k=10,
    llm_model="GLM-5.1",
    agent_max_tokens=1000,
    agent_verification_mode="deferred",
    agent_verification_level="lightweight",
):
    """创建模拟 Settings 对象。"""
    mock = MagicMock()
    mock.enable_agent_response_cache = enable_response_cache
    mock.agent_response_cache_ttl_seconds = ttl_seconds
    mock.agent_response_cache_max_size = max_size
    mock.use_rerank = use_rerank
    mock.top_k = top_k
    mock.similarity_threshold = similarity_threshold
    mock.rerank_candidate_k = rerank_candidate_k
    mock.llm_model = llm_model
    mock.agent_max_tokens = agent_max_tokens
    mock.agent_verification_mode = agent_verification_mode
    mock.agent_verification_level = agent_verification_level
    return mock


def _mock_corpus_version():
    """Mock 语料库版本计算，返回固定哈希。"""
    with patch(
        "app.agents.response_cache._compute_corpus_version",
        return_value="fixed_corpus_hash",
    ):
        yield


def _mock_domain_data_hash():
    """Mock 领域数据哈希计算，返回固定哈希。"""
    with patch(
        "app.agents.response_cache._compute_domain_data_hash",
        return_value="fixed_domain_hash",
    ):
        yield


# -------------------------------------------------------------------------- #
#  测试 1: 相同问题命中                                                       #
# -------------------------------------------------------------------------- #

def test_same_question_cache_hit():
    """相同问题第二次调用应命中缓存，不执行 Agent。"""
    cache = AgentResponseCache(ttl_seconds=600, max_size=100)

    result1 = {
        "answer": "DeepLabV3+ 是一种语义分割模型",
        "sources": [{"chunk_id": "abc", "score": 0.9}],
        "refused": False,
        "errors": [],
    }

    key = "test_key_001"
    cache.put(key, result1)

    cached = cache.get(key)
    assert cached is not None
    assert cached["answer"] == result1["answer"]
    assert cached["sources"] == result1["sources"]

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


# -------------------------------------------------------------------------- #
#  测试 2: 不同问题不命中                                                     #
# -------------------------------------------------------------------------- #

def test_different_question_cache_miss():
    """不同问题的缓存 key 不同，不会错误命中。"""
    with patch("app.agents.response_cache.get_settings", return_value=_make_mock_settings()):
        with patch(
            "app.agents.response_cache._compute_corpus_version",
            return_value="corpus_v1",
        ):
            with patch(
                "app.agents.response_cache._compute_domain_data_hash",
                return_value="domain_v1",
            ):
                key1 = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=False
                )
                key2 = build_agent_response_cache_key(
                    "什么是 SegFormer？", use_rerank=False
                )

    assert key1 != key2, "不同问题应产生不同缓存 key"


# -------------------------------------------------------------------------- #
#  测试 3: TTL 过期后不命中                                                   #
# -------------------------------------------------------------------------- #

def test_ttl_expiry():
    """TTL 过期后缓存条目应自动失效。"""
    cache = AgentResponseCache(ttl_seconds=1, max_size=100)

    result = {"answer": "测试回答", "sources": [], "refused": False, "errors": []}
    cache.put("ttl_key", result)

    # 立即查询应命中
    assert cache.get("ttl_key") is not None

    # 等待 TTL 过期
    time.sleep(1.1)

    # 过期后查询应未命中
    assert cache.get("ttl_key") is None

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


# -------------------------------------------------------------------------- #
#  测试 4: max_size 淘汰                                                      #
# -------------------------------------------------------------------------- #

def test_max_size_eviction():
    """超出 max_size 时应淘汰最旧条目。"""
    cache = AgentResponseCache(ttl_seconds=0, max_size=3)

    for i in range(3):
        cache.put(f"key_{i}", {"answer": f"answer_{i}", "errors": []})

    assert cache.stats()["size"] == 3

    # 写入第 4 条，应淘汰 key_0（最旧）
    cache.put("key_3", {"answer": "answer_3", "errors": []})

    assert cache.stats()["size"] == 3
    assert cache.get("key_0") is None, "最旧条目应被淘汰"
    assert cache.get("key_3") is not None, "最新条目应存在"


# -------------------------------------------------------------------------- #
#  测试 5: 文档入库后缓存清空                                                 #
# -------------------------------------------------------------------------- #

def test_clear_after_ingest():
    """clear_agent_response_cache() 应清空所有缓存条目。"""
    # 初始化缓存单例
    with patch("app.agents.response_cache.get_settings", return_value=_make_mock_settings()):
        cache = get_agent_response_cache()

    cache.put("key_a", {"answer": "a", "errors": []})
    cache.put("key_b", {"answer": "b", "errors": []})
    assert cache.stats()["size"] == 2

    # 模拟文档入库后调用清空
    cleared = clear_agent_response_cache()
    assert cleared == 2
    assert cache.stats()["size"] == 0


# -------------------------------------------------------------------------- #
#  测试 6: 文档删除后缓存清空                                                 #
# -------------------------------------------------------------------------- #

def test_clear_after_delete():
    """文档删除后 clear_agent_response_cache() 同样清空所有缓存。"""
    with patch("app.agents.response_cache.get_settings", return_value=_make_mock_settings()):
        cache = get_agent_response_cache()

    cache.put("key_x", {"answer": "x", "errors": []})
    assert cache.stats()["size"] == 1

    cleared = clear_agent_response_cache()
    assert cleared == 1
    assert cache.stats()["size"] == 0


# -------------------------------------------------------------------------- #
#  测试 7: use_rerank 变化导致 key 不同                                      #
# -------------------------------------------------------------------------- #

def test_different_rerank_different_key():
    """use_rerank=True 和 use_rerank=False 应产生不同缓存 key。"""
    with patch("app.agents.response_cache.get_settings", return_value=_make_mock_settings()):
        with patch(
            "app.agents.response_cache._compute_corpus_version",
            return_value="corpus_v1",
        ):
            with patch(
                "app.agents.response_cache._compute_domain_data_hash",
                return_value="domain_v1",
            ):
                key_no_rerank = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=False
                )
                key_with_rerank = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=True
                )

    assert key_no_rerank != key_with_rerank, (
        "use_rerank 不同应产生不同缓存 key"
    )


# -------------------------------------------------------------------------- #
#  测试 8: 异常结果不被缓存                                                   #
# -------------------------------------------------------------------------- #

def test_error_result_not_cached():
    """包含异常信息的 Agent 结果不应被写入缓存。"""
    cache = AgentResponseCache(ttl_seconds=600, max_size=100)

    error_result = {
        "answer": "Agent 执行过程中出现异常，无法完成回答。",
        "sources": [],
        "refused": True,
        "errors": ["Agent 执行异常: ConnectionError"],
    }

    cache.put("error_key", error_result)

    # 异常结果不应被缓存
    assert cache.get("error_key") is None
    assert cache.stats()["size"] == 0

    # 正常结果应被缓存
    normal_result = {
        "answer": "DeepLabV3+ 使用空洞空间金字塔池化模块",
        "sources": [],
        "refused": False,
        "errors": [],
    }
    cache.put("normal_key", normal_result)
    assert cache.get("normal_key") is not None


# -------------------------------------------------------------------------- #
#  额外测试: AgentService 集成 — 缓存命中跳过 Agent 执行                      #
# -------------------------------------------------------------------------- #

def test_agent_service_response_cache_integration():
    """AgentService.query 在缓存命中时应跳过 run_langchain_agent。"""
    from app.agents.agent_service import RemoteSensingAgentService

    mock_settings = _make_mock_settings(enable_response_cache=True)

    with patch("app.agents.agent_service.get_settings", return_value=mock_settings):
        with patch("app.agents.response_cache.get_settings", return_value=mock_settings):
            with patch(
                "app.agents.response_cache._compute_corpus_version",
                return_value="corpus_v1",
            ):
                with patch(
                    "app.agents.response_cache._compute_domain_data_hash",
                    return_value="domain_v1",
                ):
                    svc = RemoteSensingAgentService()

                    # Mock run_langchain_agent 以跟踪是否被调用
                    with patch(
                        "app.agents.agent_service.run_langchain_agent"
                    ) as mock_run:
                        mock_run.return_value = {
                            "answer": "DeepLabV3+ 是编码器-解码器架构",
                            "sources": [{"chunk_id": "c1", "score": 0.85}],
                            "refused": False,
                            "tool_calls": [],
                            "agent_trace": ["agent_started", "agent_finished"],
                            "trace_events": [],
                            "errors": [],
                            "verification": {"enabled": True, "mode": "off"},
                            "timing": {"total_elapsed": 5.0},
                        }

                        # 也需要 mock LLM 缓存相关
                        with patch(
                            "app.agents.langchain_agent.set_agent_llm_cache",
                            return_value=False,
                        ):
                            with patch(
                                "app.agents.tools.set_rerank_override"
                            ):
                                # 第一次调用：缓存未命中，执行 Agent
                                result1 = svc.query(
                                    "什么是 DeepLabV3+？", include_trace=True
                                )
                                assert mock_run.call_count == 1
                                assert result1["timing"]["response_cache_hit"] is False

                                # 第二次调用：缓存命中，跳过 Agent
                                result2 = svc.query(
                                    "什么是 DeepLabV3+？", include_trace=True
                                )
                                assert mock_run.call_count == 1, (
                                    "缓存命中时不应再次调用 run_langchain_agent"
                                )
                                assert result2["timing"]["response_cache_hit"] is True
                                assert result2["answer"] == result1["answer"]


# -------------------------------------------------------------------------- #
#  测试 9: prompt 文本变化导致 key 不同                                       #
# -------------------------------------------------------------------------- #

def test_prompt_change_invalidates_key(monkeypatch):
    """prompt 文本变化后，缓存 key 必须变化，避免返回基于旧 prompt 的过期答案。

    场景：开发者修改了 RAG_SYSTEM_PROMPT（例如调整长度约束、防幻觉条款），
    重启服务后老缓存应全部自然失效，所有请求重新走 LLM。
    """
    import app.core.prompts as core_prompts
    import app.agents.response_cache as rc_module

    with patch("app.agents.response_cache.get_settings", return_value=_make_mock_settings()):
        with patch(
            "app.agents.response_cache._compute_corpus_version",
            return_value="corpus_v1",
        ):
            with patch(
                "app.agents.response_cache._compute_domain_data_hash",
                return_value="domain_v1",
            ):
                # 第一组：原始 prompt 下构建 key
                rc_module._compute_prompt_hash.cache_clear()
                key_before = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=False
                )

                # 模拟修改 RAG_SYSTEM_PROMPT 文案（追加一条新约束）
                original_prompt = core_prompts.RAG_SYSTEM_PROMPT
                monkeypatch.setattr(
                    core_prompts,
                    "RAG_SYSTEM_PROMPT",
                    original_prompt + "\n5. 新增约束：回答必须控制在 500 字以内。",
                )

                # 清空 lru_cache，模拟"服务重启后重新计算 prompt 哈希"
                rc_module._compute_prompt_hash.cache_clear()
                key_after = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=False
                )

    assert key_before != key_after, (
        "prompt 文本变化后缓存 key 必须变化，否则会返回基于旧 prompt 的过期答案"
    )


# -------------------------------------------------------------------------- #
#  测试 10: prompt 未变化时 key 稳定（不破坏现有命中率）                      #
# -------------------------------------------------------------------------- #

def test_same_prompt_produces_same_key():
    """prompt 未变化时，相同问题 + 相同配置应产生相同 key，保持原有命中率。

    这条测试保障本次改动对生产环境命中率零影响：日常运行（prompt 不变）
    下，缓存命中行为与改动前完全一致。
    """
    with patch("app.agents.response_cache.get_settings", return_value=_make_mock_settings()):
        with patch(
            "app.agents.response_cache._compute_corpus_version",
            return_value="corpus_v1",
        ):
            with patch(
                "app.agents.response_cache._compute_domain_data_hash",
                return_value="domain_v1",
            ):
                key1 = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=False
                )
                key2 = build_agent_response_cache_key(
                    "什么是 DeepLabV3+？", use_rerank=False
                )

    assert key1 == key2, (
        "prompt 未变化时缓存 key 必须稳定，否则会破坏现有缓存命中率"
    )
