"""Chat / Query 相关 FastAPI 路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import ChatQueryRequest, ChatQueryResponse, WorkUnitCandidate
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# 复用单例避免每次请求都新建 embedding/llm client
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.post("/query", response_model=ChatQueryResponse)
def query(req: ChatQueryRequest) -> ChatQueryResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")
    try:
        result = get_rag_service().answer(
            question=req.question.strip(),
            top_k=req.top_k,
            use_rerank=req.use_rerank,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("RAG 查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG 查询失败: {e}") from e

    # 构造 Work Unit 候选对象：仅作为「可一键保存」的快照附带，不自动落盘。
    # 不额外调用 retriever / 不重跑 RAG；sources 复用本次响应已有的结果。
    work_unit_candidate = WorkUnitCandidate(
        entry="rag",
        question=req.question.strip(),
        answer=result.answer,
        sources=[s.model_dump() for s in result.sources],
        refused=result.refused,
        replay_payload={
            "endpoint": "/api/chat/query",
            "body": {
                "question": req.question.strip(),
                "top_k": req.top_k,
                "use_rerank": req.use_rerank,
            },
        },
    )

    return ChatQueryResponse(
        answer=result.answer,
        sources=result.sources,
        refused=result.refused,
        work_unit_candidate=work_unit_candidate,
    )
