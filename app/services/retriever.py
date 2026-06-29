"""检索器：cosine 召回 → 可选 rerank 精排 → 截断 top-K。

VectorStore 按 cosine 相似度检索，按相似度降序返回 top_k 条结果。
当 settings.use_rerank=True 时，先检索 candidate_k 条候选，
经 cross-encoder rerank 后保留 top_k 条。
"""
from __future__ import annotations

from typing import List, Optional

from app.config import get_settings
from app.services.reranker import rerank_search_results
from app.services.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.store = store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[dict]:
        settings = get_settings()
        final_k = top_k or settings.top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        # 前端传入的 use_rerank 优先于 .env 配置
        effective_rerank = use_rerank if use_rerank is not None else settings.use_rerank

        if effective_rerank:
            # 两阶段检索：先取 candidate_k 条候选，rerank 后保留 final_k 条
            candidate_k = settings.rerank_candidate_k
            hits = self.store.search(
                query=query,
                top_k=candidate_k,
                similarity_threshold=threshold,
            )

            if not hits:
                logger.info("检索 query=%r cosine 命中 0 条（rerank 模式）", query)
                return []

            reranked, rerank_elapsed, used_fallback = rerank_search_results(
                query=query,
                search_results=hits,
                final_top_k=final_k,
            )
            logger.info(
                "检索 query=%r cosine 命中 %d 条, rerank 后 %d 条 "
                "(rerank %.3fs, fallback=%s)",
                query, len(hits), len(reranked), rerank_elapsed, used_fallback,
            )
            return reranked
        else:
            # 原始流程：纯向量检索
            hits = self.store.search(
                query=query,
                top_k=final_k,
                similarity_threshold=threshold,
            )
            logger.info("检索 query=%r cosine 命中 %d 条", query, len(hits))
            return hits
