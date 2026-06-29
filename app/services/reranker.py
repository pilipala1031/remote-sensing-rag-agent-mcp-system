"""SiliconFlow Rerank API 客户端（生产版本）。

使用 BAAI/bge-reranker-v2-m3 cross-encoder 对向量检索结果进行重排序。
复用 SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL，无需额外配置密钥。
支持优雅降级：当 API 调用失败时，返回原始向量检索顺序。

配置（通过 .env 控制）：
    USE_RERANK=true              开启 rerank（默认 false）
    RERANK_CANDIDATE_K=10        向量检索候选数量（默认 10）
    RERANK_MODEL=BAAI/bge-reranker-v2-m3   rerank 模型名称
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import requests

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 默认 rerank 模型
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_rerank_config() -> Tuple[str, str, str]:
    """从 Settings 读取 rerank 配置。

    复用 SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL，
    模型名从 settings.rerank_model 读取（默认 BAAI/bge-reranker-v2-m3）。

    Returns:
        (api_key, base_url, model)
    """
    settings = get_settings()
    api_key = settings.siliconflow_api_key
    base_url = settings.siliconflow_base_url.rstrip("/")
    model = settings.rerank_model or DEFAULT_RERANK_MODEL
    return api_key, base_url, model


def rerank(
    query: str,
    documents: List[str],
    top_n: int | None = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """调用 SiliconFlow rerank API 对文档进行重排序。

    Args:
        query: 用户查询文本。
        documents: 待排序的文档文本列表。
        top_n: 返回前 N 个结果，None 表示返回全部。
        timeout: 请求超时秒数。

    Returns:
        重排序结果列表，每项包含:
        - index: 原始文档索引（int）
        - relevance_score: 相关性分数（float，越高越相关）

    Raises:
        RuntimeError: API 调用失败或配置缺失。
    """
    if not documents:
        return []

    api_key, base_url, model = _get_rerank_config()

    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY 未配置，无法调用 rerank API")

    url = f"{base_url}/rerank"
    payload: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False,
        "max_chunks_per_doc": 512,
    }
    if top_n is not None:
        payload["top_n"] = min(top_n, len(documents))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        raise RuntimeError(
            f"Rerank API 返回 {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    results = data.get("results", [])

    logger.info(
        "Rerank 完成: %d 篇文档, top_n=%s, 耗时 %.3fs, 模型=%s",
        len(documents), top_n, elapsed, model,
    )

    return results


def rerank_search_results(
    query: str,
    search_results: List[dict],
    final_top_k: int,
) -> tuple[List[dict], float, bool]:
    """对向量检索结果进行 rerank 并返回重排序后的 top_k 结果。

    原始向量相似度 score 保留不变，新增 rerank_score 字段。
    API 调用失败时优雅降级，返回原始向量顺序。

    Args:
        query: 用户查询文本。
        search_results: 向量检索结果列表（已按 score 降序）。
        final_top_k: rerank 后保留的结果数量。

    Returns:
        (reranked_results, rerank_elapsed, used_fallback)
        - reranked_results: 重排序后的 top_k 结果列表
        - rerank_elapsed: rerank 调用耗时（秒）
        - used_fallback: 是否因 API 失败而回退到原始向量顺序
    """
    if not search_results:
        return [], 0.0, False

    # 提取文档文本用于 rerank
    documents = [r.get("content", "") for r in search_results]

    try:
        t0 = time.perf_counter()
        rerank_results = rerank(
            query=query,
            documents=documents,
            top_n=min(final_top_k, len(documents)),
        )
        elapsed = time.perf_counter() - t0

        # 按 rerank 结果重排序
        reranked: List[dict] = []
        for item in rerank_results:
            idx = item["index"]
            score = item["relevance_score"]
            if 0 <= idx < len(search_results):
                result = dict(search_results[idx])  # 浅拷贝，保留原始字段
                result["rerank_score"] = score
                reranked.append(result)

        if not reranked:
            logger.warning("Rerank 返回空结果，回退到原始向量顺序")
            return search_results[:final_top_k], elapsed, True

        logger.info(
            "Rerank 重排序完成: %d → %d 结果, 耗时 %.3fs",
            len(search_results), len(reranked), elapsed,
        )
        return reranked, elapsed, False

    except Exception as e:
        logger.warning("Rerank 调用失败，回退到原始向量顺序: %s", e)
        return search_results[:final_top_k], 0.0, True
