# Work Unit v1 产品层设计文档

> 把一次 AI 对话变成可持续、可复用、可复盘的工作单元。
>
> 本文档只描述**设计与边界**，不含实现。本文档描述的所有 Work Unit 能力均为 **v1 规划**，尚未实现；其中明确标注为 v2 的部分，连数据消费逻辑都不在 v1 范围内。Replay 是 **v2**，本文不实现。

---

## 0. 已确认决策（速查）

以下 9 条是本设计已锁定的决策，后续实现必须遵守：

1. **Work Unit 是产品层对象**，不是新 Agent，也不是 MCP tool。
2. **RAG / Agent 查询不自动落盘**——查询响应中不触发任何文件写入。
3. RAG / Agent 查询响应中**新增可选字段 `work_unit_candidate`**，它只是候选对象，是否落盘由前端（用户）决定。
4. 前端点击「保存为 Work Unit」后，调用 `POST /api/work_units` 保存；保存成功返回 `work_unit_id`。
5. **字段统一使用 `entry`**（不用 `kind`），取值为 `"rag"` / `"agent"` / `"mcp"`。
6. **API 路径统一使用 `/api/work_units`**（下划线，不用 `/api/work-units`）。
7. **Replay 第一版不实现端点**，只在数据模型里**预留 `replay_payload` 字段**（先存而不用）。
8. **MCP 第一版只返回 `work_unit_fragment`**，不保存完整 Work Unit。
9. **MCP tool 不调用 WorkUnitStore、不写入 `data/work_units/`、不调用 RAGService / Agent / LLM**——MCP 的 fragment 由宿主或上层产品层组装，不在工具内部落盘。

---

## 1. 为什么需要 Work Unit

本项目当前的 RAG / Agent / MCP 三类能力都是**无记忆的一次性请求**：

- `/api/chat/query`（RAG）和 `/api/agent/query`（Agent）每次请求都从零开始，回答生成完就丢弃；
- 用户无法回看「我上次问过什么、得到了什么、依据是哪些来源」；
- 同一个有价值的问答（例如一次复杂的多工具 Agent 推理、一次带证据校验的回答）无法被复用、对比、归档。

Work Unit 要解决的就是这个缺口：**把一次 RAG / Agent / MCP 调用过程沉淀为一个结构化对象**，使其可查看（列表 + 详情）、可复盘（来源 / 轨迹 / 校验一目了然）、可重放（v2）。

它的定位是**产品层抽象**：它不改变推理层（RAG / Agent 怎么检索、怎么生成），只是把推理层产出的结果「装订成册」。

---

## 2. Work Unit 的定义

> **Work Unit = 一次 RAG / Agent / MCP 工具调用过程沉淀下来的、可查看、可复盘、可重放的结构化工作单元。**

三个关键词：

- **沉淀（persist）**：落盘成 JSON 文件，拥有唯一 `work_unit_id`，可被列出与读取。
- **复盘（review）**：包含回答、来源、工具调用、轨迹、耗时、证据校验等，足以让人（或下游程序）理解「这次结果是怎么来的」。
- **重放（replay）**：v2 能力。基于保存的 `replay_payload` 用相同配置重跑一遍，对比新旧结果。

Work Unit **不是**：

- ❌ 不是新 Agent（它不推理、不调用 LLM、不选工具）；
- ❌ 不是 MCP tool（它不被宿主 Agent 调用，它是被「保存」出来的对象）；
- ❌ 不是新的检索/计算逻辑（它不复制 `Retriever` / `core/metrics`，只是装载它们的产出）。

---

## 3. RAG / Agent / MCP / Work Unit 的边界

四者关系：前三者是**能力**（ability），Work Unit 是**容器**（container）。

| 对象 | 层 | 职责 | 是否推理 | 是否落盘 Work Unit（v1） |
| --- | --- | --- | :---: | :---: |
| **RAG** | 推理层（services） | 固定检索 → LLM 生成 + 拒答 | 是 | 否（仅产出 `work_unit_candidate`） |
| **Agent** | 推理层（agents） | LLM 自主编排 7 工具 + 证据校验 | 是 | 否（仅产出 `work_unit_candidate`） |
| **MCP** | 能力暴露层（mcp_server） | 给宿主 Agent 提供检索/计算原子能力 | 否 | 否（仅返回 `work_unit_fragment`） |
| **Work Unit** | **产品层**（work_units API + store） | 把上述产出装订成可复盘对象 | **否** | —— |

边界要点：

- **沉淀动作只发生在产品层**：只有 `POST /api/work_units` 会写 `data/work_units/`。RAG/Agent 查询、MCP 工具调用都不会落盘。
- **Work Unit 不回调推理层**：v1 中 Work Unit 的「查看 / 复盘」是纯读取，不触发任何 LLM 或检索；只有 v2 的 Replay 才会重新触发推理层，且 Replay 走 `replay_payload` 显式重放，不是 Work Unit 自身的行为。
- **MCP 与 Work Unit 解耦**：MCP 工具返回 `work_unit_fragment`，是否组装成完整 Work Unit 由宿主（Claude Code 等）或上层产品决定，工具内部不参与。

---

## 4. Work Unit v1 的最小范围

v1 只做「**可查看、可复盘**」，不做「可重放」：

### v1 包含

- RAG / Agent 响应中携带可选字段 `work_unit_candidate`（候选对象，不落盘）。
- `POST /api/work_units`：保存候选对象为 Work Unit，返回 `work_unit_id`。
- `GET /api/work_units`：Work Unit 列表（支持按 `entry` 过滤）。
- `GET /api/work_units/{work_unit_id}`：Work Unit 详情（复盘视图）。
- `DELETE /api/work_units/{work_unit_id}`：删除 Work Unit。
- MCP 两个工具的返回 dict 追加 `work_unit_fragment` 字段（不落盘）。
- 前端：结果下方「保存为 Work Unit」按钮；sidebar 增加 Work Unit 列表 / 详情入口。
- 数据模型里**预留 `replay_payload` 字段**（v1 存而不用）。

### v1 明确不包含

- ❌ Replay 端点（`POST /api/work_units/{id}/replay`）——见 §7。
- ❌ Replay 结果对比。
- ❌ MCP 端的 `save_work_unit` 工具——见 §6 / §9。
- ❌ Work Unit Gallery、定时执行、多用户隔离——见 §9。
- ❌ 任何后台自动沉淀 / 定时清理。

---

## 5. WorkUnitCandidate 与 WorkUnit 的区别

二者字段几乎相同，但**生命周期与归属完全不同**：

| 维度 | `WorkUnitCandidate`（候选） | `WorkUnit`（已落盘） |
| --- | --- | --- |
| 出现位置 | RAG / Agent 的**响应体**里 | `data/work_units/{id}.json` 文件里 |
| 是否落盘 | **否**，只是一个内存中的对象 | **是**，有真实文件 |
| 有无 `work_unit_id` | **无**（还没保存） | **有** |
| 有无 `created_at` | **无**（保存时才生成） | **有** |
| 谁创建它 | RAG / Agent 查询时自动附带 | 前端 POST `/api/work_units` 时由 store 生成 |
| 用途 | 给前端一个「可一键保存」的快照 | 可查看、可复盘、（v2）可重放 |

一句话：**`work_unit_candidate` 是「待保存的草稿」，`WorkUnit` 是「保存后的正式记录」**。前端把 candidate 原样 POST 给 `/api/work_units`，store 补上 `work_unit_id` + `created_at` 后落盘，就成为 WorkUnit。

> 字段统一用 `entry`（取值 `"rag"` / `"agent"` / `"mcp"`）。MCP 的 fragment 不是 candidate——它更轻、且不落盘（见 §6）。

---

## 6. MCP 为什么只返回 fragment

MCP 工具（`search_remote_sensing_kb` / `calculate_remote_sensing_metric`）的返回 dict 会**追加一个 `work_unit_fragment` 子对象**，但**绝不保存完整 Work Unit**。理由：

1. **MCP 工具是无状态、确定性的**：每次调用新建 `Retriever`、纯计算，不持有状态、不读写文件。落盘 Work Unit 会破坏这个性质。

2. **一次有意义的 Work Unit 是宿主编排后的产物**：MCP 工具只参与了「检索」或「计算」一个原子步骤，而完整 Work Unit 需要「问题 → 编排 → 答案」的闭环——这个闭环由宿主 Agent（Claude Code）完成，工具无法预知也不该承担。

3. **职责边界**：MCP 是「能力提供者」，Work Unit 是「产品层对象」。让工具内部落盘 Work Unit，等于把产品层对象泄漏进推理层工具，违反决策 1（Work Unit 是产品层对象）。

因此 MCP 工具的边界是硬性的（决策 9）：

> **MCP tool 不调用 WorkUnitStore，不写入 `data/work_units/`，不调用 RAGService、Agent 或 LLM。**

`work_unit_fragment` 是**素材**：包含 `fragment_id` / `entry:"mcp"` / `tool_name` / `inputs` / `outputs` / `elapsed`，交给宿主或上层产品决定是否、何时组装成完整 Work Unit。把 fragment 拼成 Work Unit 的能力，是 v2 的 `save_work_unit` MCP 工具的职责（见 §9）。

---

## 7. 为什么 v1 不做 Replay

Replay 的语义是「用保存的 `replay_payload` 重跑一遍 RAG / Agent，拿到新结果」。v1 不做端点的原因：

1. **非确定性问题需要专门设计**：LLM 输出、检索排序（尤其 rerank）都可能因时间、模型版本、缓存命中而变化，重放结果「不完全可复现」。需要先想清楚结果对比的展示方式（差异高亮？相似度？）。
2. **Agent 响应缓存会干扰**：`ENABLE_AGENT_RESPONSE_CACHE` 开启时，相同问题会命中缓存直接返回旧结果，让「重放」名不副实。Replay 端点需要显式绕过缓存，这属于额外逻辑。
3. **v1 先验证「沉淀 + 复盘」是否被使用**：在没人看 Work Unit 列表之前，做重放是过度设计。

所以 v1 的做法是：**数据模型里预留 `replay_payload` 字段并照常保存，但不实现任何消费它的端点**。这样 v2 加 Replay 时无需改 schema，向后兼容。

---

## 8. 数据结构 JSON 示例

### 8.1 已保存的 Work Unit（`entry = "agent"`）

```jsonc
{
  "work_unit_id": "a3f9c2b1e7d40568",
  "entry": "agent",
  "question": "请比较 U-Net 和 DeepLabV3+ 在遥感语义分割中的适用场景",
  "answer": "U-Net 适合样本量小的场景，编码器-解码器 + skip connection 对边界敏感；DeepLabV3+ 依赖空洞卷积捕获多尺度上下文，适合复杂多类别场景……",
  "sources": [
    {
      "filename": "02_models.md",
      "page": 1,
      "chunk_id": "1a2b3c4d5e6f",
      "score": 0.8421,
      "content_preview": "U-Net 由对称的编码器-解码器构成，通过 skip connection 融合浅层细节与深层语义……"
    }
  ],
  "refused": false,
  "tool_calls": [
    { "name": "knowledge_base_search", "elapsed": 1.23 },
    { "name": "model_comparison_table", "elapsed": 0.02 }
  ],
  "trace_events": [
    { "step": 1, "event": "agent_started", "timestamp": "2026-06-29T10:00:01.12", "detail": "问题已接收" },
    { "step": 2, "event": "tool_called", "timestamp": "2026-06-29T10:00:01.35", "detail": "knowledge_base_search" },
    { "step": 3, "event": "agent_finished", "timestamp": "2026-06-29T10:00:04.88", "detail": "已生成回答" }
  ],
  "timing": {
    "total_elapsed": 3.78,
    "agent_invoke_elapsed": 3.52,
    "response_cache_hit": false
  },
  "verification": {
    "enabled": true,
    "mode": "deferred",
    "verified": true,
    "confidence": 0.91,
    "ungrounded_claims": []
  },
  "errors": [],
  "replay_payload": {
    "question": "请比较 U-Net 和 DeepLabV3+ 在遥感语义分割中的适用场景",
    "include_trace": true,
    "use_rerank": null,
    "enable_cache": null
  },
  "created_at": "2026-06-29T10:00:05.02"
}
```

> 说明：v1 中 `replay_payload` 被保存但**不被任何端点消费**；它是为 v2 Replay 预留的。RAG 类型的 Work Unit（`entry: "rag"`）通常 `tool_calls` / `trace_events` / `verification` 为空数组。

### 8.2 RAG / Agent 响应中的候选对象（未保存）

RAG / Agent 查询响应体在原有字段之外，**追加可选字段 `work_unit_candidate`**，结构与上面的 Work Unit 几乎一致，但**没有 `work_unit_id` 与 `created_at`**：

```jsonc
{
  "answer": "...",
  "sources": [ ... ],
  "refused": false,
  "work_unit_candidate": {
    "entry": "agent",
    "question": "请比较 U-Net 和 DeepLabV3+ 在遥感语义分割中的适用场景",
    "answer": "...",
    "sources": [ ... ],
    "refused": false,
    "tool_calls": [ ... ],
    "trace_events": [ ... ],
    "timing": { ... },
    "verification": { ... },
    "errors": [],
    "replay_payload": { "question": "...", "include_trace": true, "use_rerank": null, "enable_cache": null }
  }
}
```

### 8.3 MCP 返回的 fragment（不落盘）

MCP 工具返回 dict **追加 `work_unit_fragment`**，结构比 candidate 更轻，且**不会进入 `data/work_units/`**：

```jsonc
{
  "success": true,
  "query": "LoveDA 数据集包含哪些类别？",
  "contexts": [ ... ],
  "sources": [ ... ],
  "elapsed": 1.23,
  "work_unit_fragment": {
    "fragment_id": "mcp_search_1a2b3c4d",
    "entry": "mcp",
    "tool_name": "search_remote_sensing_kb",
    "inputs": { "query": "LoveDA 数据集包含哪些类别？", "top_k": 5, "enable_rerank": false },
    "outputs_summary": { "hits": 5 },
    "elapsed": 1.23
  }
}
```

---

## 9. v2 Roadmap

以下均为 **未实现**，仅作为方向。每一项都标注了它要解决的 v1 遗留问题：

- **Replay endpoint**：`POST /api/work_units/{work_unit_id}/replay`，从落盘对象取 `replay_payload` 重跑 RAG / Agent。需解决 LLM 非确定性、Agent 响应缓存命中（需显式绕过）。
- **Replay 结果对比**：把 Replay 新结果与原 Work Unit 并排展示（字段级差异、答案相似度、来源是否变化），让「可重放」真正可视化。
- **`save_work_unit` MCP tool**：新增一个 MCP 工具，让宿主 Agent（Claude Code）能主动把若干 `work_unit_fragment` 组装并保存为完整 Work Unit——把「组装权」交给宿主，而不是在原子工具里落盘。
- **Work Unit Gallery**：前端提供一个可浏览、可检索、可打标签的 Work Unit 画廊，支持按 entry / 时间 / 关键词筛选。
- **定时执行**：基于 `replay_payload` 定时重跑某些 Work Unit（例如每日监测同一问题的答案漂移）。
- **多用户隔离**：Work Unit 按 user 维度隔离，`work_unit_id` 命名空间与访问权限分用户。

---

## 相关文件（v1 实现将涉及，本文档不修改它们）

- `app/schemas.py` —— 新增 `WorkUnitCandidate` / `WorkUnit` / `WorkUnitSaveRequest` 等模型，并给 `ChatQueryResponse` / `AgentQueryResponse` 追加可选 `work_unit_candidate`
- `app/services/work_unit_store.py`（新增）—— JSON 文件持久化，写 `data/work_units/{work_unit_id}.json`
- `app/api/work_units.py`（新增）—— `/api/work_units` 路由（save / list / get / delete）
- `app/main.py` —— 注册 work_units router
- `app/api/chat.py` / `app/api/agent.py` —— 查询响应附带 `work_unit_candidate`（不落盘）
- `mcp_server/server.py` —— 两个工具返回 dict 追加 `work_unit_fragment`（不落盘）
- `frontend/streamlit_app.py` —— 保存按钮 + Work Unit 列表 / 详情入口
