# MCP 设计文档：为什么是两个原子工具，而不是一个 Agent

> 本文档记录 `mcp_server/server.py` 的设计取舍。这不是「为了 JD 加 MCP」，而是基于 MCP 协议模型与本项目现有架构做出的判断。适合在面试 / 技术评审时作为讲述提纲。

## TL;DR

我没有把 `/api/agent/query`（整个 LangChain Agent）直接封装成 MCP tool，因为 **Claude Code 本身已经是宿主 Agent**。如果 MCP tool 内部再调用 LangChain Agent，会形成 Agent-in-Agent 的嵌套结构，造成重复推理、延迟增加、工具控制权不清晰。因此我选择**只暴露两个原子能力——知识库检索与确定性指标计算——让 Claude Code 自己负责编排和生成**。

---

## 1. 为什么没有把整个 LangChain Agent 包成 MCP tool？

本项目已经有一个完整的 Multi-Tool Agent（`/api/agent/query`，基于 LangChain 1.0 `create_agent`，7 个工具 + ReAct 循环 + Evidence Verification + 双层缓存）。最省事的做法似乎是把这整个 Agent 包成一个 `run_agent(question)` 的 MCP tool 暴露出去。**我没有这么做。**

### 核心原因：Agent-in-Agent 嵌套是反模式

MCP 协议的隐含假设是：**宿主（host）是编排者，MCP server 提供被编排的原子能力**。当 Claude Code 作为宿主调用一个 MCP tool 时，宿主 LLM 已经在「思考、规划、决定调用哪个工具」。此时如果被调用的 tool 内部又跑一个完整的 LangChain Agent（它自己也会思考、规划、选工具），就出现了两层 Agent 嵌套：

```
Claude Code（宿主 Agent：推理 + 编排）
   └─ mcp tool: run_agent(question)
        └─ LangChain Agent（又一个 Agent：推理 + 编排 + 选 7 个工具 + 生成）
             └─ LLM 生成最终答案
```

这带来三个具体问题：

| 问题 | 具体表现 |
| --- | --- |
| **重复推理** | 宿主 LLM 和内嵌 Agent 各跑一轮 LLM 推理，token 与费用翻倍，而第二轮推理的「规划」其实是冗余的——宿主已经规划过了 |
| **延迟叠加** | 一次回答要等两层 Agent 各自的 LLM 往返，端到端延迟显著变长 |
| **控制权不清晰** | 出问题时无法判断是宿主选错了 tool，还是内嵌 Agent 选错了内部 tool；工具选择、重试、Rerank 开关等控制权被锁死在内嵌 Agent 里，宿主无法干预 |

更深的问题是**职责重叠**：两个 Agent 干的是同一件事——「理解问题 → 检索/计算 → 生成答案」。把其中一层做成另一层的「工具」，等于让两个编排者抢方向盘。

### 判断

当宿主已经是 Agent（Claude Code / Cursor / Claude Desktop 都是），MCP server 的正确角色是**提供宿主自己不具备、且不该自己实现的能力**——也就是本项目的领域数据与检索/计算内核——而不是再塞一个 Agent 进去跟宿主抢编排权。

> 对比：如果宿主**不是** Agent（比如一个只会转发请求的简单脚本、或一个传统 Web 前端），那么把完整 Agent 封装成 MCP tool 是合理的——因为没有人负责编排。本项目面向 Claude Code 这类宿主 Agent，结论相反。**「该不该包成 Agent」取决于宿主有没有编排能力，而不是取决于实现省不省事。**

---

## 2. 为什么 MCP 只暴露这两个原子工具？

暴露的是这两个，且**刻意只有这两个**：

| MCP Tool | 能力 | 是否调 LLM | 共享内核 |
| --- | --- | :---: | --- |
| `search_remote_sensing_kb` | 知识库语义检索，返回原始 chunks | **否** | `app.services.retriever.Retriever` |
| `calculate_remote_sensing_metric` | IoU/Precision/Recall/F1 计算 | **否** | `app.core.metrics.calculate_metric` |

### 2.1 这两个工具的共同特征：确定性、无 LLM、可编排

它们满足三个条件，恰好是「适合作为 MCP 原子工具」的特征：

1. **确定性 / 无 LLM**：检索是向量相似度 + 可选 cross-encoder rerank，计算是纯公式。结果是确定的、可复现的、可单测的。宿主调用它们**不会触发额外 LLM 推理**，也就避免了 Agent-in-Agent 里「重复推理」的成本。
2. **原子 / 不可再分**：检索和计算都是「给定输入，产出结构化输出」的单步操作，没有内部规划。它们是宿主 Agent 编排的基本积木，而不是一个自带大脑的黑盒。
3. **领域专有 / 宿主不可能自带**：Claude Code 不知道本项目的 Chroma 向量库里有什么文档，也不可能凭空算出用户给的混淆矩阵在该领域怎么打分——这正是 MCP server 该补的能力空缺。

### 2.2 为什么 `search_remote_sensing_kb` 只检索、不生成？

这是最关键的设计决定。它调 `Retriever.retrieve()` 拿到 chunks 后**直接返回**，不拼 prompt、不调 LLM、不生成答案：

```python
# 简化后的核心逻辑：检索完就返回，不生成
hits = retriever.retrieve(query=query, top_k=top_k, use_rerank=enable_rerank)
return {"contexts": [...], "sources": [...], "elapsed": elapsed}
```

**理由：把生成交还给宿主 LLM。** 如果这个 tool 内部再调一次 LLM 生成答案，就回到了 §1 的「重复推理」问题——宿主拿到一个已经生成好的答案，但它自己也是 LLM，等于「本项目生成一遍 + 宿主再加工一遍」的双重生成。让 tool 只返回原始证据（chunks），由宿主 LLM 自己读、自己推理、自己写答案，才是单层编排。

> 对比本项目的 RAG/Agent HTTP 接口：`/api/chat/query` 和 `/api/agent/query` **会**生成答案，因为它们的宿主是「人」或「简单脚本」——没有人替它们生成。MCP 的宿主是 Claude Code，它自己就能生成，所以 MCP 的检索工具就不该抢着生成。**同一个检索内核，在不同入口下是否生成，取决于宿主有没有生成能力。**

### 2.3 为什么参数形态和 Agent 工具刻意不同？

`calculate_remote_sensing_metric` 用**类型化参数**（`tp/fp/fn` 为 `float`），而 Agent 的 `metrics_calculator` 用**字符串风格**（`"TP=80, FP=10"`）。这不是不一致，是有意区分：

- **Agent 工具面向 LLM 自由文本调用**：LLM 喜欢在自然语言里塞参数，字符串解析更宽松、容错好；
- **MCP 工具面向宿主 Agent 的结构化 tool-call**：宿主（如 Claude Code）是按 schema 构造调用的，类型化参数能被 schema 校验，无需字符串解析，更稳健、更省 token。

两个工具共享同一个 `core/metrics.py` 计算内核——**计算逻辑只有一份**，只是入口参数适配层不同。

### 2.4 为什么没有把另外 5 个 Agent 工具也暴露？

Agent 还有 `dataset_overview` / `dataset_spec_lookup` / `model_comparison_table` / `metric_formula_lookup` / `plan_and_search` 五个工具。没有暴露为 MCP 的原因：

- 前四个是**对静态 JSON 的结构化查询**。Claude Code 作为宿主，拿到 `search_remote_sensing_kb` 的检索结果（这些 JSON 内容本身也在知识库里）已经能自己读取和总结，再单独包一层 MCP tool 是冗余的薄封装。
- `plan_and_search` 是**查询分解 + 多次检索 + 合并**，本质是「检索的编排」——而编排正是宿主 Agent 该做的事。把它做成 MCP tool 又回到了 §1 的「编排权归属」问题。

所以最终留下两个：一个补「宿主不知道的数据」、一个补「宿主算不了的领域计算」。**少即是多——每多一个 MCP tool，宿主的选择空间和决策成本就多一分，工具数量应当服从「宿主真正缺什么」。**

---

## 3. RAG / Agent / MCP 三者的职责边界

三者共享同一套后端内核（`Retriever` 检索、`core/metrics` 计算、Chroma 向量库、domain_data 静态知识），区别在于**谁来编排、谁来生成、谁负责可信度**。

| 维度 | 普通 RAG (`/api/chat/query`) | Multi-Tool Agent (`/api/agent/query`) | MCP Server (`remote-sensing-kb`) |
| --- | --- | --- | --- |
| **宿主 / 编排者** | 固定流水线（无编排） | 本项目的 LangChain Agent | **外部宿主 Agent（Claude Code 等）** |
| **谁来选工具** | 不选，固定检索→生成 | LLM 自主选 7 个工具 | 宿主自主决定调哪个 MCP tool |
| **谁生成答案** | 本项目 LLM | 本项目 LLM | **宿主 LLM**（本项目不生成） |
| **暴露的能力** | 完整问答 | 完整 Agent + 工具 + 校验 | 仅 2 个原子能力（检索 / 计算） |
| **防幻觉机制** | 3 层（阈值拒答 + Prompt + 拒答模板） | 4 层（+ Evidence Verification） | **交给宿主**（tool 只返回原始证据） |
| **典型调用方** | 人 / 脚本 / Streamlit | 人 / 脚本 / Streamlit | 其他 Agent |
| **是否调本项目 LLM** | 是 | 是 | **否** |

### 一句话边界

- **RAG**：宿主是「人」。固定流水线，本项目包办检索 + 生成 + 拒答，延迟最低。
- **Agent**：宿主是「人，但想要可解释、可追溯的推理」。本项目用 7 个工具自主编排，生成 + 证据校验，输出结构化工作单元。
- **MCP**：宿主是「另一个 Agent」。本项目退化为**原子能力提供者**，只给检索和计算，把编排与生成完全让给宿主——因为宿主自己就是 Agent，不该被嵌套。

### 内核复用关系（同一份逻辑，三种入口）

```
                        ┌─ /api/chat/query     (RAG: 检索 + 本项目生成)
Retriever (检索内核) ────┼─ /api/agent/query    (Agent knowledge_base_search tool)
                        └─ MCP search_remote_sensing_kb  (只检索，不生成)

                          ┌─ /api/agent/query    (Agent metrics_calculator tool)
core/metrics (计算内核) ──┤
                          └─ MCP calculate_remote_sensing_metric
```

> 注意一个工程细节：MCP server **内联复制**了 `tools.py` 里的 `_truncate` / `_hit_to_context` / `_hit_to_source` 三个辅助函数，而不是 `import app.agents.tools`。这是有意的——避免 `mcp_server` 反向依赖 `app.agents`（Agent 层），保持 MCP server 作为「能力提供者」的依赖方向干净：它只依赖 `app.services.retriever`（检索）和 `app.core.metrics`（计算）这两个最底层的内核，不依赖上层的 Agent 编排逻辑。**依赖方向反映了职责边界：MCP 是 Agent 的「同行能力源」，不是 Agent 的「子模块」。**

---

## 附：设计自检清单

复盘这套设计时，我用以下问题自检，结论都是「是」：

- [x] 宿主是否已经是 Agent？是（Claude Code）→ 则 MCP 不该再包 Agent。
- [x] 每个 MCP tool 是否都无 LLM、确定性？是 → 避免重复推理。
- [x] 暴露的能力是否「宿主不可能自带」？是（领域数据 / 领域计算）→ 不是冗余薄封装。
- [x] 编排权是否清晰归属宿主？是（查询分解、生成、可信度判断都在宿主）。
- [x] 计算内核是否只有一份？是（`core/metrics.py`，Agent 与 MCP 共享）。
- [x] MCP server 是否避免反向依赖 Agent 层？是（内联辅助函数而非 import）。

---

## 相关文件

- [`mcp_server/server.py`](../mcp_server/server.py) — MCP server 实现（FastMCP，2 个 `@mcp.tool`）
- [`.mcp.json`](../.mcp.json) — Claude Code 注册配置
- [`app/services/retriever.py`](../app/services/retriever.py) — 检索内核（三入口共享）
- [`app/core/metrics.py`](../app/core/metrics.py) — 计算内核（Agent 与 MCP 共享）
- [`app/agents/tools.py`](../app/agents/tools.py) — Agent 的 `knowledge_base_search`（与 MCP 检索工具对照）
- [`CLAUDE.md`](../CLAUDE.md) — 整体架构与 MCP 定位
