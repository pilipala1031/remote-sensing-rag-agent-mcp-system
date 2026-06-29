"""Pydantic 请求 / 响应模型定义。"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- 文档相关 ----------
class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    saved_path: str
    message: str = "文件上传成功"


class IngestRequest(BaseModel):
    doc_id: Optional[str] = Field(
        default=None, description="指定要入库的 doc_id；为空则入库 raw 目录下所有新文件"
    )


class ChunkInfo(BaseModel):
    chunk_id: str
    page: int
    content_preview: str


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    chunks: List[ChunkInfo] = Field(default_factory=list)
    message: str = "文档入库成功"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
    message: str = "文档已删除"


# ---------- 聊天相关 ----------
class SourceItem(BaseModel):
    filename: str
    page: int
    chunk_id: str
    score: float
    content_preview: str


class ChatQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_rerank: Optional[bool] = Field(
        default=None,
        description="是否启用 rerank 重排序。None 则使用 .env 中的 USE_RERANK 配置。",
    )


class ChatQueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    refused: bool = False
    # Work Unit 候选对象：仅作为「可一键保存」的快照附带，不自动落盘。
    # 前端点击「保存为 Work Unit」后，才会调用 POST /api/work_units 保存。
    work_unit_candidate: Optional["WorkUnitCandidate"] = None


# ---------- Agent 相关 ----------
class AgentQueryRequest(BaseModel):
    question: str
    include_trace: bool = Field(
        default=True,
        description="是否返回 agent_trace / trace_events / tool_calls 等调试信息。"
        "生产环境可设为 False 以减少响应体积。",
    )
    use_rerank: Optional[bool] = Field(
        default=None,
        description="是否启用 rerank 重排序。None 则使用 .env 中的 USE_RERANK 配置。",
    )
    enable_cache: Optional[bool] = Field(
        default=None,
        description="是否启用 LLM 响应缓存（仅 Agent 路径 ChatOpenAI 生效）。"
        "None 则使用 .env 中的 ENABLE_AGENT_CACHE 配置。"
        "首次问相同问题时写入缓存，后续相同问题直接返回缓存结果。",
    )


class AgentQueryResponse(BaseModel):
    answer: str = Field(description="Agent 生成的最终回答")
    sources: list = Field(default_factory=list, description="引用来源列表")
    refused: bool = Field(default=False, description="是否拒答")
    tool_calls: list = Field(default_factory=list, description="工具调用记录")
    agent_trace: list = Field(default_factory=list, description="Agent 执行轨迹（字符串列表）")
    trace_events: list = Field(
        default_factory=list,
        description="结构化轨迹事件列表，每条包含 step / event / timestamp / detail",
    )
    errors: list[str] = Field(default_factory=list, description="错误信息")
    timing: dict = Field(default_factory=dict, description="耗时统计")
    verification: dict = Field(
        default_factory=dict,
        description="Evidence Verification 证据校验结果",
    )
    # Work Unit 候选对象：仅作为「可一键保存」的快照附带，不自动落盘。
    # 前端点击「保存为 Work Unit」后，才会调用 POST /api/work_units 保存。
    work_unit_candidate: Optional["WorkUnitCandidate"] = None


# ---------- Agent Verify 独立端点 ----------
class AgentVerifyRequest(BaseModel):
    question: str = Field(description="用户原始问题")
    answer: str = Field(description="Agent 生成的回答")
    sources: list = Field(default_factory=list, description="引用来源列表")
    tool_calls: list = Field(default_factory=list, description="工具调用记录")


class AgentVerifyResponse(BaseModel):
    verification: dict = Field(description="Evidence Verification 证据校验结果")


# ---------- Work Unit 相关 ----------
# 设计要点（详见 docs/work_unit_design.md）：
#   1. Work Unit 是产品层对象，不是新 Agent，也不是 MCP tool；
#   2. 手动沉淀：RAG / Agent 查询不自动保存，只在响应中附带候选对象 work_unit_candidate；
#   3. 字段统一使用 entry（取值 rag / agent / mcp）；
#   4. Replay 第一版不做端点，仅在数据模型里预留 replay_payload 字段；
#   5. MCP 第一版只返回 work_unit_fragment，不落盘（此处不涉及 MCP 字段）。
class WorkUnitCandidate(BaseModel):
    """RAG / Agent 查询响应中携带的 Work Unit 候选对象。

    仅作为「可一键保存」的快照，不包含 work_unit_id，也不自动落盘。
    前端点击「保存为 Work Unit」后，将其 POST 到 /api/work_units 才会保存。
    """

    entry: Literal["rag", "agent", "mcp"]
    question: str
    answer: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    refused: bool = False
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    timing: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    # 重放所需的完整输入配置。v1 存而不用，仅供 v2 Replay 端点消费。
    replay_payload: dict[str, Any] = Field(default_factory=dict)


class WorkUnit(BaseModel):
    """已保存的完整 Work Unit。

    相比 WorkUnitCandidate，额外携带 work_unit_id 与 created_at，
    表示已被持久化到 data/work_units/ 下。
    """

    work_unit_id: str
    entry: Literal["rag", "agent", "mcp"]
    question: str
    answer: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    refused: bool = False
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    timing: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    # 重放所需的完整输入配置。v1 存而不用，仅供 v2 Replay 端点消费。
    replay_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class WorkUnitSaveRequest(BaseModel):
    """POST /api/work_units 的请求体。

    字段基本等同于 WorkUnitCandidate：由前端把候选对象原样提交，
    保存时由 store 补上 work_unit_id 与 created_at。
    """

    entry: Literal["rag", "agent", "mcp"]
    question: str
    answer: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    refused: bool = False
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    timing: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    # 重放所需的完整输入配置。v1 存而不用，仅供 v2 Replay 端点消费。
    replay_payload: dict[str, Any] = Field(default_factory=dict)


class WorkUnitSaveResponse(BaseModel):
    """保存 Work Unit 的响应。"""

    work_unit_id: str
    message: str = "Work Unit 已保存"


class WorkUnitListResponse(BaseModel):
    """Work Unit 列表响应。"""

    total: int
    work_units: list[WorkUnit]
