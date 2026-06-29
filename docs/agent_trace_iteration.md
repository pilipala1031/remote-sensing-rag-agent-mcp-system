# Agent Trace 迭代记录

## 1. 背景与动机

Agent（Multi-Tool）路径相比 Plain RAG 的核心区别在于：LLM 自主决定调用哪些工具、调用几次、以什么顺序。这种自主性带来了强大的灵活性和复杂的可观测性需求：

- **为什么需要 trace**：当 Agent 回答错误或拒答时，仅看最终 answer 无法定位问题——是工具选错了？是检索结果为空？还是 LLM 生成质量差？trace 数据（tool_calls + agent_trace + trace_events）是诊断的唯一依据。
- **为什么需要结构化 trace_events**：原有的 `agent_trace` 是 `list[str]`（如 `["agent_started", "tool_called:knowledge_base_search", "tool_result_parsed", "agent_finished"]`），只能表达"发生了什么"，无法表达"何时发生"和"持续多久"。结构化的 `trace_events` 补充了时间戳和工具名详情，使得：
  - 可以计算每个工具的实际耗时占比
  - 可以对比不同问题的工具选择路径差异
  - 可以驱动工具选择准确率分析

## 2. 迭代阶段

### Phase 0：基础 trace（已有）

在 Block 3 之前，Agent 已经返回以下 trace 相关字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `agent_trace` | `list[str]` | 简短事件标签列表 |
| `tool_calls` | `list[dict]` | 每次工具调用的记录 |
| `timing` | `dict` | 总耗时 / invoke 耗时 / 工具检索总耗时 |
| `verification` | `dict` | 证据校验结果 |

**已有 tool_calls 每条记录包含**：`tool` / `input` / `status` / `output_summary` / `elapsed` / `error`

**已有 agent_trace 事件**：
- `agent_started` — Agent 开始执行
- `tool_called:<tool_name>` — LLM 决定调用某工具
- `tool_result_parsed` — 工具返回结果已解析
- `agent_finished` — Agent 执行完成
- `no_tool_called` — Agent 未调用任何工具
- `agent_error` — 执行异常

**局限**：
- 无时间戳，无法知道每个事件的相对时间
- 无 step 序号，无法快速定位事件顺序
- 工具名未从 trace 字符串中分离，前端需要字符串切割

### Phase 1：结构化 trace_events（本次迭代）

本次迭代在保持完全向后兼容的前提下，新增了以下能力：

#### 2.1 新增 `trace_events` 字段

每条事件为 `dict`，包含：

```json
{
  "step": 1,
  "event": "agent_started",
  "timestamp": 0.0,
  "detail": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | `int` | 事件序号，从 1 递增 |
| `event` | `str` | 事件类型（同 agent_trace 的标签值，但不含 `:tool_name` 后缀） |
| `timestamp` | `float` | 相对时间戳（秒），基于工具 `elapsed` 累积估算 |
| `detail` | `str?`? | 附加信息（如工具名、错误摘要） |

**时间戳估算逻辑**：
- `agent_started` → `0.0`
- `tool_called` → 前序所有工具的 `elapsed` 之和
- `tool_result_parsed` → 当前工具 `elapsed` 完成后的累积时间
- `agent_finished` / `no_tool_called` → `invoke_elapsed`（agent.invoke 总耗时）

注意：由于 LangChain `create_agent` 是同步阻塞调用，无法从外部插入中间件获取真实的 per-step wall-clock 时间。上述时间戳是基于工具自身报告的 `elapsed` 累积估算的，对于诊断工具耗时占比已足够准确。

#### 2.2 新增 `include_trace` API 参数

```python
class AgentQueryRequest(BaseModel):
    question: str
    include_trace: bool = True
```

- `include_trace=true`（默认）：返回完整的 `tool_calls` / `agent_trace` / `trace_events`
- `include_trace=false`：这三个字段返回空列表，减少响应体积

前端 "显示调试信息" 复选框直接映射到 `include_trace`：关闭时不请求 trace 数据。

#### 2.3 修复 `run_agent_eval.py` 的 `verification_elapsed` bug

**Bug**：原代码从 `timing.verification_elapsed` 读取，但该字段实际嵌套在 `verification.timing.verification_elapsed` 中。

```python
# Bug（原）
verification_elapsed = timing_data.get("verification_elapsed", 0.0)

# Fix（新）
v_timing = verification.get("timing", {})
verification_elapsed = v_timing.get("verification_elapsed", 0.0)
```

#### 2.4 AgentToolCall 类型补充

`app/agents/types.py` 中的 `AgentToolCall` 新增 `elapsed` 字段（此前代码中动态使用但未在 Pydantic model 中声明）。

### Phase 2：工具选择分析（本次迭代）

新增 `experiments/agent_trace_analyzer.py`，支持：

- **live 模式**：逐题调用 `/api/agent/query`（`include_trace=true`），收集工具选择数据
- **offline 模式**：从已保存的 `eval/results/agent_eval_result.json` 读取

**分析维度**：

| 维度 | 指标 |
|------|------|
| 整体准确率 | `tool_hit_rate`（实际工具与期望工具有交集的比例） |
| 误用率 | `misuse_rate`（使用了 `unexpected_tools` 的比例） |
| 无工具率 | `no_tool_rate`（Agent 未调用任何工具的比例） |
| 工具频率 | 每个工具的使用次数 |
| 工具耗时 | 每个工具的平均执行耗时 |
| 按题型分组 | 每个 `question_type` 的命中率 |
| 混淆矩阵 | expected_tool → actual_tools 的映射关系 |

## 3. 涉及文件

### 修改的文件

| 文件 | 变更说明 |
|------|----------|
| `app/agents/types.py` | 新增 `TraceEvent` 模型；`AgentToolCall` 加 `elapsed` 字段；`AgentRunResult` 加 `trace_events` 字段 |
| `app/agents/langchain_agent.py` | `_parse_agent_result` 新增 `invoke_elapsed` 参数和 `trace_events` 生成逻辑；异常路径也返回 `trace_events` |
| `app/agents/agent_service.py` | `query()` 方法新增 `include_trace` 参数；为 False 时清除 trace 字段 |
| `app/schemas.py` | `AgentQueryRequest` 新增 `include_trace` 字段；`AgentQueryResponse` 新增 `trace_events` 字段 |
| `app/api/agent.py` | 传递 `include_trace` 参数到 service；返回 `trace_events` 字段 |
| `frontend/streamlit_app.py` | 新增 `render_trace_events` 函数；API 调用传递 `include_trace` 参数 |
| `eval/run_agent_eval.py` | 修复 `verification_elapsed` 读取路径 bug；新增 `trace_events` 字段采集 |

### 新增的文件

| 文件 | 说明 |
|------|------|
| `experiments/agent_trace_analyzer.py` | Agent 工具选择轨迹分析器（live / offline 双模式） |
| `docs/agent_trace_iteration.md` | 本文档 |

## 4. 数据流

```
用户提问
  │
  ▼
POST /api/agent/query {question, include_trace: true}
  │
  ▼
AgentService.query(question, include_trace)
  │
  ▼
run_langchain_agent(question)
  │
  ├─ agent.invoke() ──→ messages 列表
  │
  ├─ _parse_agent_result(result, invoke_elapsed)
  │     │
  │     ├─ 遍历 messages，提取 answer / sources / tool_calls
  │     ├─ 生成 agent_trace (list[str])
  │     ├─ 生成 trace_events (list[dict]) with timestamps
  │     └─ 返回 dict（含所有字段）
  │
  ├─ Evidence Verification（off/sync/deferred）
  │
  └─ timing 统计
  │
  ▼
AgentQueryResponse → 前端
  │
  ├─ render_agent_trace(agent_trace)        ← 简短时间线
  ├─ render_trace_events(trace_events)      ← 结构化时间戳表
  ├─ render_tool_calls(tool_calls)          ← 工具调用详情
  └─ _render_timing_panel(timing, ...)      ← 耗时面板
```

## 5. 如何运行分析

### Live 模式（推荐）

```bash
# 1. 启动后端
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 2. 运行分析器
python -m experiments.agent_trace_analyzer

# 输出：
# - 控制台摘要报告
# - experiments/results/agent_trace_analysis.json
```

### Offline 模式

```bash
# 前提：已运行 python eval/run_agent_eval.py 生成了 eval/results/agent_eval_result.json

python -m experiments.agent_trace_analyzer --offline
```

## 6. 当前限制

1. **时间戳为估算值**：`trace_events` 中的 `timestamp` 基于工具 `elapsed` 累积计算，不是真实的 wall-clock 时间。LLM 推理时间未计入工具之间的间隔。
2. **工具调用次数不受控**：Agent 的工具调用次数由 LLM 自主决定，无上限。高频调用会增加延迟和 API 成本。
3. **无流式 trace**：后端为同步返回，前端 `trace_events` 只能在请求完成后整体渲染，无法实时展示执行进度。
4. **混淆矩阵依赖标签质量**：`expected_tool` 是单选标签，但实际一道题可能合理使用多种工具（如 `knowledge_base_search` + `dataset_spec_lookup`），严格判定可能低估准确率。
5. **`include_trace=false` 仅清除 trace 三个字段**：`timing` 和 `verification` 仍然返回。如需进一步缩减响应体积，可在后续迭代中增加更细粒度的字段选择。

## 7. 未来迭代方向

- **流式 trace**：使用 SSE / WebSocket 推送 trace 事件，实现真正的实时执行可视化
- **工具调用预算**：限制单次 Agent 执行的最大工具调用次数，防止无限循环
- **rerank / query rewrite**：在工具层增加 rerank 或 query rewrite 能力，提升检索精度
- **多轮对话 trace**：支持多轮上下文中的 trace 追踪
- **trace 可视化增强**：在前端使用时间轴图表（如 Gantt chart）展示工具调用时间线

## 8. 面试表述

> Agent 的可观测性是 Multi-Tool 架构的关键挑战。我们设计了三层 trace 体系：第一层 `agent_trace`（`list[str]`）提供快速概览；第二层 `trace_events`（`list[dict]`）携带时间戳和工具名详情，支持精确耗时分析；第三层 `tool_calls` 记录每次工具调用的完整输入输出。同时引入 `include_trace` API 参数，生产环境可关闭 trace 以减少响应体积。基于这些 trace 数据，我开发了 `agent_trace_analyzer.py`，通过对比 expected_tool 标签计算工具选择命中率、误用率和混淆矩阵，量化 Agent 的工具路由决策质量。
