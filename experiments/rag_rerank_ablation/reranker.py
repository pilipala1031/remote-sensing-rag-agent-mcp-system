"""SiliconFlow Rerank API 客户端。

使用 BAAI/bge-reranker-v2-m3 模型对向量检索结果进行重排序。
支持优雅降级：当 API 调用失败时，返回原始向量检索顺序。

环境变量（从 .env 读取，不在正式 config.py 中声明，仅供实验使用）：
    RERANK_API_KEY: rerank API 密钥（留空则复用 SILICONFLOW_API_KEY）
    RERANK_BASE_URL: rerank API 基础 URL（留空则复用 SILICONFLOW_BASE_URL）
    RERANK_MODEL: rerank 模型名称（默认 BAAI/bge-reranker-v2-m3）

CLI 用法：
    python -m experiments.rag_rerank_ablation.reranker  # 自检
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import requests

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 默认 rerank 模型
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_rerank_config() -> Tuple[str, str, str]:
    """从环境变量读取 rerank 配置。

    优先级：RERANK_* > SILICONFLOW_* (via Settings) > 默认值。

    通过 python-dotenv 加载 .env 文件，再通过 get_settings() 获取
    SILICONFLOW 配置作为 fallback。RERANK_* 变量不在正式 config.py 中声明，
    从 os.getenv() 读取。

    Returns:
        (api_key, base_url, model)
    """
    # 加载 .env 到 os.environ（幂等，不覆盖已有环境变量）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # RERANK_* 优先从 os.getenv 读取
    api_key = os.getenv("RERANK_API_KEY", "")
    base_url = os.getenv("RERANK_BASE_URL", "")
    model = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)

    # 留空则 fallback 到 SILICONFLOW 配置（通过 Settings 单例读取 .env）
    if not api_key or not base_url:
        from app.config import get_settings
        settings = get_settings()
        if not api_key:
            api_key = settings.siliconflow_api_key
        if not base_url:
            base_url = settings.siliconflow_base_url

    # 确保 base_url 不以 / 结尾
    base_url = base_url.rstrip("/")
    return api_key, base_url, model


def rerank(
    query: str,
    documents: List[str],
    top_n: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """调用 SiliconFlow rerank API 对文档进行重排序。

    Args:
        query: 用户查询文本。
        documents: 待排序的文档文本列表。
        top_n: 返回前 N 个结果，None 表示返回全部。
        api_key: API 密钥，None 则从环境变量读取。
        base_url: API 基础 URL，None 则从环境变量读取。
        model: 模型名称，None 则从环境变量读取。
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

    if api_key is None or base_url is None or model is None:
        env_key, env_url, env_model = _get_rerank_config()
        api_key = api_key or env_key
        base_url = base_url or env_url
        model = model or env_model

    if not api_key:
        raise RuntimeError("RERANK_API_KEY 未配置，无法调用 rerank API")

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
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[List[dict], float, bool]:
    """对向量检索结果进行 rerank 并返回重排序后的 top_k 结果。

    在 rerank_search_results 中，rerank 仅改变排序顺序，
    原始向量相似度 score 保留不变，新增 rerank_score 字段。

    Args:
        query: 用户查询文本。
        search_results: 向量检索结果列表（已按 score 降序）。
        final_top_k: rerank 后保留的结果数量。
        api_key / base_url / model: 同 rerank()。

    Returns:
        (reranked_results, rerank_elapsed, used_fallback)
        - reranked_results: 重排序后的 top_k 结果列表
          每项在原始字段基础上新增 rerank_score
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
            api_key=api_key,
            base_url=base_url,
            model=model,
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


def self_test() -> None:
    """简单自检：验证 rerank API 是否可用。"""
    print("Rerank 自检...")
    api_key, base_url, model = _get_rerank_config()
    print(f"  API Key: {'已配置' if api_key else '未配置'}")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")

    if not api_key:
        print("  ❌ 未配置 API Key，跳过 API 调用测试")
        return

    try:
        results = rerank(
            query="遥感语义分割评价指标",
            documents=[
                "mIoU 是遥感语义分割的首选评价指标",
                "今天天气很好",
                "U-Net 采用编码器-解码器结构",
                "Pixel Accuracy 在类别不平衡时有局限性",
            ],
            top_n=2,
        )
        print(f"  ✅ API 调用成功，返回 {len(results)} 条结果:")
        for r in results:
            print(f"    index={r['index']}, score={r['relevance_score']:.4f}")
    except Exception as e:
        print(f"  ❌ API 调用失败: {e}")


if __name__ == "__main__":
    self_test()
