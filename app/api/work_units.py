"""Work Unit 相关 FastAPI 路由。

POST   /api/work_units             手动保存 Work Unit（前端「保存为 Work Unit」按钮触发）
GET    /api/work_units             列出 Work Unit（支持按 entry 过滤）
GET    /api/work_units/{id}        查看单个 Work Unit（复盘）
DELETE /api/work_units/{id}        删除 Work Unit

设计要点（详见 docs/work_unit_design.md）：
    1. Work Unit 只通过 POST /api/work_units 手动保存，RAG / Agent 查询不自动落盘；
    2. Replay v1 不实现端点（replay_payload 只保存，不执行）；
    3. API 层只负责请求校验 + 调用 work_unit_store，不写业务逻辑；
    4. 不调用 RAGService / AgentService / MCP / LLM。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.schemas import (
    WorkUnit,
    WorkUnitListResponse,
    WorkUnitSaveRequest,
    WorkUnitSaveResponse,
)
from app.services import work_unit_store
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/work_units", tags=["Work Unit"])

# 合法的 entry 取值（与 WorkUnit.entry 的 Literal 保持一致）
_VALID_ENTRIES = ("rag", "agent", "mcp")


def _get_work_unit_dir() -> Path:
    """解析 Work Unit 持久化目录。

    独立成函数便于测试通过 monkeypatch 替换为临时目录，避免污染真实 data/work_units/。
    """
    return get_settings().work_unit_path


def _delete_message(work_unit_id: str) -> dict:
    """DELETE 成功的统一返回结构。"""
    return {"message": "Work Unit 已删除", "work_unit_id": work_unit_id}


@router.post("", response_model=WorkUnitSaveResponse)
def save_work_unit(req: WorkUnitSaveRequest) -> WorkUnitSaveResponse:
    """手动保存 Work Unit。

    接收前端提交的候选对象，落盘后返回 work_unit_id。
    """
    try:
        saved = work_unit_store.save_work_unit(req, base_dir=_get_work_unit_dir())
    except Exception as e:  # noqa: BLE001
        logger.error("Work Unit 保存失败: %s", e)
        raise HTTPException(status_code=500, detail=f"Work Unit 保存失败: {e}") from e

    logger.info("Work Unit 保存成功: work_unit_id=%s", saved.work_unit_id)
    return WorkUnitSaveResponse(work_unit_id=saved.work_unit_id)


@router.get("", response_model=WorkUnitListResponse)
def list_work_units(
    entry: Optional[str] = Query(default=None, description="按 entry 过滤：rag / agent / mcp"),
    limit: int = Query(default=50, ge=1, le=500, description="最多返回条数"),
) -> WorkUnitListResponse:
    """列出 Work Unit，支持按 entry 过滤，按保存时间倒序返回。"""
    if entry is not None and entry not in _VALID_ENTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"entry 取值非法，只能是 {list(_VALID_ENTRIES)} 或不传",
        )

    items = work_unit_store.list_work_units(
        entry=entry,
        limit=limit,
        base_dir=_get_work_unit_dir(),
    )
    return WorkUnitListResponse(total=len(items), work_units=items)


@router.get("/{work_unit_id}", response_model=WorkUnit)
def get_work_unit(work_unit_id: str) -> WorkUnit:
    """查看单个 Work Unit（复盘）。"""
    try:
        return work_unit_store.get_work_unit(work_unit_id, base_dir=_get_work_unit_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Work Unit 不存在: {work_unit_id}")


@router.delete("/{work_unit_id}")
def delete_work_unit(work_unit_id: str) -> dict:
    """删除单个 Work Unit。"""
    deleted = work_unit_store.delete_work_unit(work_unit_id, base_dir=_get_work_unit_dir())
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Work Unit 不存在: {work_unit_id}")
    logger.info("Work Unit 已删除: work_unit_id=%s", work_unit_id)
    return _delete_message(work_unit_id)
