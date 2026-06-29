"""Work Unit JSON 文件持久化层（产品层对象，非推理层）。

职责边界（详见 docs/work_unit_design.md）：
    1. 只做文件 I/O —— save / get / list / delete；
    2. 不调用 RAGService / Agent / MCP / LLM；
    3. 不复制 retrieval / metrics 逻辑；
    4. 将 Work Unit 保存到 data/work_units/{work_unit_id}.json。

本模块是 Work Unit 的唯一落盘点。RAG / Agent 查询、MCP 工具都不会写这里；
只有 POST /api/work_units（由前端「保存为 Work Unit」触发）会调用 save_work_unit。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.schemas import WorkUnit, WorkUnitSaveRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _default_base_dir(base_dir: Optional[Path | str]) -> Path:
    """解析持久化目录：传入则用传入值，否则读取 Settings 中的配置目录。"""
    if base_dir is not None:
        p = Path(base_dir)
    else:
        p = get_settings().work_unit_path
    p.mkdir(parents=True, exist_ok=True)
    return p


def _generate_work_unit_id(entry: str, question: str) -> str:
    """生成 work_unit_id = "wu_" + md5(entry:question:time)[:16]。

    加入 time 使同一条目 + 同一问题的多次保存也能得到不同 id。
    """
    raw = f"{entry}:{question}:{datetime.now().isoformat()}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"wu_{digest}"


def _work_unit_file(base_dir: Path, work_unit_id: str) -> Path:
    """单个 Work Unit 的 JSON 文件路径。"""
    return base_dir / f"{work_unit_id}.json"


def save_work_unit(
    payload: WorkUnitSaveRequest,
    base_dir: Optional[Path | str] = None,
) -> WorkUnit:
    """将候选对象 payload 保存为 Work Unit。

    Args:
        payload: 前端提交的保存请求（字段同 WorkUnitCandidate）。
        base_dir: 持久化目录；为 None 则使用 Settings 中的 work_unit_dir。
                  测试可注入临时目录以避免污染真实 data/work_units/。

    Returns:
        已落盘的 WorkUnit（含 work_unit_id 与 created_at）。
    """
    base_dir = _default_base_dir(base_dir)

    work_unit_id = _generate_work_unit_id(payload.entry, payload.question)
    created_at = datetime.now().isoformat()

    work_unit = WorkUnit(
        work_unit_id=work_unit_id,
        created_at=created_at,
        entry=payload.entry,
        question=payload.question,
        answer=payload.answer,
        sources=payload.sources,
        refused=payload.refused,
        tool_calls=payload.tool_calls,
        trace_events=payload.trace_events,
        timing=payload.timing,
        verification=payload.verification,
        errors=payload.errors,
        replay_payload=payload.replay_payload,
    )

    file_path = _work_unit_file(base_dir, work_unit_id)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(work_unit.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info("Work Unit 已保存: work_unit_id=%s, entry=%s, file=%s", work_unit_id, payload.entry, file_path)
    return work_unit


def get_work_unit(
    work_unit_id: str,
    base_dir: Optional[Path | str] = None,
) -> WorkUnit:
    """按 work_unit_id 读取单个 Work Unit。

    Args:
        work_unit_id: Work Unit 唯一标识。
        base_dir: 持久化目录；为 None 则使用 Settings 中的 work_unit_dir。

    Raises:
        FileNotFoundError: 当指定 id 的文件不存在时，抛出带中文信息的异常。
    """
    base_dir = _default_base_dir(base_dir)
    file_path = _work_unit_file(base_dir, work_unit_id)

    if not file_path.exists():
        logger.warning("Work Unit 不存在: work_unit_id=%s", work_unit_id)
        raise FileNotFoundError(f"Work Unit 不存在: {work_unit_id}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Work Unit 已读取: work_unit_id=%s", work_unit_id)
    return WorkUnit(**data)


def list_work_units(
    entry: Optional[str] = None,
    limit: int = 50,
    base_dir: Optional[Path | str] = None,
) -> list[WorkUnit]:
    """列出 Work Unit，可选按 entry 过滤，按创建时间倒序返回，最多 limit 条。

    Args:
        entry: 仅返回该 entry（"rag" / "agent" / "mcp"）的 Work Unit；None 表示不过滤。
        limit: 最多返回条数。
        base_dir: 持久化目录；为 None 则使用 Settings 中的 work_unit_dir。
    """
    base_dir = _default_base_dir(base_dir)

    files = sorted(
        base_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    results: list[WorkUnit] = []
    for fp in files:
        if len(results) >= limit:
            break
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            work_unit = WorkUnit(**data)
        except Exception as e:  # noqa: BLE001
            # 跳过损坏 / 非法文件，避免一条坏数据影响整体列表
            logger.warning("跳过无法解析的 Work Unit 文件 %s: %s", fp, e)
            continue
        if entry is not None and work_unit.entry != entry:
            continue
        results.append(work_unit)

    logger.info("Work Unit 列表: entry=%s, 返回 %d 条", entry, len(results))
    return results


def delete_work_unit(
    work_unit_id: str,
    base_dir: Optional[Path | str] = None,
) -> bool:
    """删除单个 Work Unit。

    Args:
        work_unit_id: Work Unit 唯一标识。
        base_dir: 持久化目录；为 None 则使用 Settings 中的 work_unit_dir。

    Returns:
        True 表示已删除；False 表示文件原本就不存在。
    """
    base_dir = _default_base_dir(base_dir)
    file_path = _work_unit_file(base_dir, work_unit_id)

    if not file_path.exists():
        logger.warning("Work Unit 不存在，删除跳过: work_unit_id=%s", work_unit_id)
        return False

    file_path.unlink()
    logger.info("Work Unit 已删除: work_unit_id=%s", work_unit_id)
    return True
