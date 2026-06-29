"""Agent 层内部数据结构定义。

使用 Pydantic BaseModel，与 app/schemas.py 风格保持一致。
这些结构供 AgentService 内部使用，以及 API 响应 / 前端展示 / 测试消费。

结构层级：
    TraceEvent      —— 单条结构化轨迹事件（含时间戳）
    AgentToolCall   —— 单次工具调用记录
    AgentSource     —— 与 app.schemas.SourceItem 字段对齐的来源信息
    AgentRunResult  —— Agent 单次运行的完整结果
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# 默认拒答文案，与 core/prompts.py REFUSAL_ANSWER 保持一致
DEFAULT_REFUSAL_ANSWER = "根据当前知识库内容，无法确定该问题的答案。"


class TraceEvent(BaseModel):
    """Agent 执行轨迹中的一条结构化事件。

    比 AgentRunResult.agent_trace (list[str]) 更丰富：
    每条事件携带 step 序号、event 类型、relative_timestamp（相对于 Agent 启动）
    以及可选的 detail（如工具名称）。
    """

    step: int
    """事件序号，从 1 开始递增。"""

    event: str
    """事件类型：agent_started / tool_called / tool_result_parsed / agent_finished / no_tool_called / agent_error。"""

    timestamp: float = 0.0
    """相对于 Agent 启动的时间戳（秒），由 _parse_agent_result 填充。"""

    detail: str | None = None
    """事件附加信息，如 tool_called 事件中为工具名。"""


class AgentToolCall(BaseModel):
    """Agent 执行过程中的一次工具调用记录。

    用于前端展示 Agent 的推理链路和工具使用情况。
    """

    tool: str
    """工具名称，如 "rag_search"。"""

    input: str | dict[str, Any] | None = None
    """工具输入参数，可以是字符串、字典或 None。"""

    status: str
    """调用状态，如 "success" / "error"。"""

    output_summary: str | None = None
    """工具输出摘要（截断后的文本），供前端展示。"""

    elapsed: float | None = None
    """工具执行耗时（秒），从 ToolMessage 中解析获取。"""

    error: str | None = None
    """调用失败时的错误信息，成功时为 None。"""


class AgentSource(BaseModel):
    """Agent 引用的知识库来源，与 app.schemas.SourceItem 字段对齐。

    比 SourceItem 更宽松：page 和 score 允许 None，
    因为 Agent 从 Tool 输出中解析时可能缺失部分字段。
    """

    filename: str
    page: int | None = None
    chunk_id: str
    score: float | None = None
    content_preview: str


class AgentRunResult(BaseModel):
    """Agent 单次运行的完整结果。

    对标 RAGService.RAGAnswer，但增加了 tool_calls / agent_trace / errors，
    以支持 Agent 的多步工具调用和可观测性。
    """

    answer: str
    """Agent 最终回答文本。"""

    sources: list[AgentSource] = Field(default_factory=list)
    """引用的知识库来源列表。"""

    refused: bool = False
    """是否拒答（检索为空或信息不足时为 True）。"""

    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    """Agent 执行过程中的所有工具调用记录。"""

    agent_trace: list[str] = Field(default_factory=list)
    """Agent 执行轨迹的人类可读摘要（list[str]），供前端逐步展示。"""

    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    """结构化轨迹事件列表，每条包含 step / event / timestamp / detail。

    与 agent_trace 互补：agent_trace 是简短字符串列表用于快速概览，
    trace_events 携带时间戳和附加信息用于精确分析。
    """

    errors: list[str] = Field(default_factory=list)
    """执行过程中遇到的非致命错误列表。"""


def build_refusal_result(reason: str | None = None) -> AgentRunResult:
    """构建拒答结果。

    当 Agent 未调用工具、工具返回空、或信息不足时使用。
    所有列表字段均为空，refused 固定为 True。

    Args:
        reason: 自定义拒答文案，为 None 则使用 DEFAULT_REFUSAL_ANSWER。

    Returns:
        AgentRunResult: refused=True 的拒答结果。
    """
    return AgentRunResult(
        answer=reason or DEFAULT_REFUSAL_ANSWER,
        sources=[],
        refused=True,
        tool_calls=[],
        agent_trace=[],
        trace_events=[],
        errors=[],
    )
