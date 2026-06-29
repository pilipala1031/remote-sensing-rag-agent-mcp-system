"""Agent 相关 FastAPI 路由。

POST /api/agent/query  —— Agent 问答（RAG as Tool）
POST /api/agent/verify —— 独立执行 Evidence Verification

API 层只负责请求校验和调用 AgentService / verify_answer，不写业务逻辑。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.agents.agent_service import get_agent_service
from app.agents.verification import verify_answer
from app.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentVerifyRequest,
    AgentVerifyResponse,
    WorkUnitCandidate,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/agent", tags=["Agent"])


@router.post("/query", response_model=AgentQueryResponse)
def query(req: AgentQueryRequest) -> AgentQueryResponse:
    """Agent 问答接口。

    与 /api/chat/query 并存：
    - /api/chat/query 使用 RAGService（固定检索 → LLM 生成）
    - /api/agent/query 使用 AgentService（LLM 自主决定是否检索）
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    try:
        result = get_agent_service().query(
            req.question.strip(),
            include_trace=req.include_trace,
            use_rerank=req.use_rerank,
            enable_cache=req.enable_cache,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Agent 查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent 查询失败: {e}") from e

    # 构造 Work Unit 候选对象：仅作为「可一键保存」的快照附带，不自动落盘。
    # 不为 Work Unit 再跑一次 Agent；include_trace=False 时轨迹为空属正常，如实保存。
    # result 是 AgentService 返回的 dict，统一用 .get 容错（兜底场景字段可能缺失）。
    work_unit_candidate = WorkUnitCandidate(
        entry="agent",
        question=req.question.strip(),
        answer=result.get("answer"),
        sources=result.get("sources", []),
        refused=result.get("refused", False),
        tool_calls=result.get("tool_calls", []),
        trace_events=result.get("trace_events", []),
        timing=result.get("timing", {}),
        verification=result.get("verification", {}),
        errors=result.get("errors", []),
        replay_payload={
            "endpoint": "/api/agent/query",
            "body": {
                "question": req.question.strip(),
                "include_trace": req.include_trace,
                "use_rerank": req.use_rerank,
                "enable_cache": req.enable_cache,
            },
        },
    )

    return AgentQueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        refused=result.get("refused", False),
        tool_calls=result.get("tool_calls", []),
        agent_trace=result.get("agent_trace", []),
        trace_events=result.get("trace_events", []),
        errors=result.get("errors", []),
        timing=result.get("timing", {}),
        verification=result.get("verification", {}),
        work_unit_candidate=work_unit_candidate,
    )


@router.post("/verify", response_model=AgentVerifyResponse)
def verify(req: AgentVerifyRequest) -> AgentVerifyResponse:
    """独立执行 Evidence Verification。

    在 deferred 模式下，前端获取 Agent 回答后调用此端点获取校验结果。
    该端点独立于 /api/agent/query，不影响 /api/chat/query。
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")
    if not req.answer or not req.answer.strip():
        raise HTTPException(status_code=400, detail="answer 不能为空")

    try:
        verification_result = verify_answer(
            question=req.question.strip(),
            answer=req.answer,
            sources=req.sources,
            tool_calls=req.tool_calls,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Agent verification 失败: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Agent verification 失败: {e}"
        ) from e

    return AgentVerifyResponse(verification=verification_result)
