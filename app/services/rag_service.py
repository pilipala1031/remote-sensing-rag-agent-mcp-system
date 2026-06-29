"""RAG 主流程编排：检索 -> Prompt 拼接 -> LLM 回答 / 拒答 / 来源引用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.config import get_settings
from app.core.llm import OpenAICompatibleLLMClient
from app.core.prompts import RAG_SYSTEM_PROMPT, REFUSAL_ANSWER
from app.services.retriever import Retriever
from app.utils.logger import get_logger
from app.schemas import SourceItem

logger = get_logger(__name__)


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SourceItem]
    refused: bool


class RAGService:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[OpenAICompatibleLLMClient] = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or OpenAICompatibleLLMClient()

    # ---------- 主入口 ----------
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_rerank: Optional[bool] = None,
    ) -> RAGAnswer:
        settings = get_settings()
        top_k = top_k or settings.top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

        # 1. 检索
        hits = self.retriever.retrieve(
            question, top_k=top_k, similarity_threshold=threshold,
            use_rerank=use_rerank,
        )

        # 2. 拒答判断
        if not hits:
            logger.info("检索结果为空或相似度过低，拒答")
            return RAGAnswer(
                answer=REFUSAL_ANSWER,
                sources=[],
                refused=True,
            )

        # 3. 拼接上下文
        context = self._build_context(hits)

        # 4. 调用 LLM
        prompt = RAG_SYSTEM_PROMPT.format(context=context, question=question)
        try:
            answer_text = self.llm.chat(prompt)
        except Exception as e:  # noqa: BLE001
            logger.error("LLM 生成失败: %s", e)
            return RAGAnswer(
                answer=REFUSAL_ANSWER,
                sources=self._to_sources(hits),
                refused=True,
            )

        return RAGAnswer(
            answer=answer_text.strip(),
            sources=self._to_sources(hits),
            refused=False,
        )

    # ---------- helpers ----------
    @staticmethod
    def _build_context(hits: List[dict]) -> str:
        blocks: List[str] = []
        for i, h in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] 来源：{h.get('filename','?')}，第{h.get('page',1)}页，"
                f"chunk_id={h.get('chunk_id','?')}\n{h.get('content','')}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _to_sources(hits: List[dict]) -> List[SourceItem]:
        sources: List[SourceItem] = []
        for h in hits:
            content = h.get("content", "") or ""
            sources.append(
                SourceItem(
                    filename=h.get("filename", "unknown"),
                    page=int(h.get("page", 1)),
                    chunk_id=h.get("chunk_id", ""),
                    score=float(h.get("score", 0.0)),
                    content_preview=content[:160].replace("\n", " "),
                )
            )
        return sources
