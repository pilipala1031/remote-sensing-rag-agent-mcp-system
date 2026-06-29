# Remote Sensing Agentic Workspace 🛰️

> **RAG + Multi-Tool Agent + MCP Server + Work Unit v1 Prototype** —— 知识库问答、多工具编排、能力开放，以及把一次 AI 对话沉淀为可追踪、可验证、可复盘的 Work Unit。

这是一个面向遥感语义分割领域的 **Agentic Knowledge Workspace 原型**，支持 RAG 问答、Multi-Tool Agent、Claude Code 可调用的 MCP Server，并探索将一次 AI 对话沉淀为可追踪、可验证、可复盘的 Work Unit。

基于 FastAPI + LangChain 1.0 + Chroma + SiliconFlow bge-m3 + OpenAI 兼容 LLM：从文档上传到智能问答的端到端 RAG 流水线；Multi-Tool Agent 把语义检索、数据集概览、结构化查询、模型对比、指标公式/计算、复杂问题分解封装为 7 个独立 Tool，辅以 Evidence Verification 与双层缓存；并额外暴露一个 MCP Server，让 Claude Code / Cursor / Claude Desktop 等宿主 Agent 直接复用同一套领域原子能力；Work Unit v1 把上述任意一次调用沉淀为可保存、可查看、可复盘的结构化对象。

---

## 四类能力入口

| 能力 | 面向对象 | 核心流程 | 价值 |
| --- | --- | --- | --- |
| **RAG** | 项目内部问答 | 检索 → 生成 → 来源引用 | 快速可信问答 |
| **Agent** | 项目内部复杂任务 | LLM 选择工具 → 工具执行 → 证据校验 | 多步骤任务编排 |
| **MCP Server** | Claude Code / Claude Desktop | 外部客户端调用检索/计算工具 | 把项目能力开放给外部 Agent |
| **Work Unit v1** | 产品层 | 保存候选对象 → 列表查看 → 详情复盘 | 让对话可沉淀、可查看、可复盘 |

---

## 三种推理层入口，一套后端

上面四类能力中，前三类（RAG / Agent / MCP）是**推理层**的三种入口，同一个遥感知识后端提供三种递进的使用方式 —— 从「固定流水线」到「自主 Agent」到「可被其他 Agent 编排的原子能力」；Work Unit 则横跨三者之上，作为产品层把它们沉淀为可复盘对象：

| 入口 | 接口 | 编排者 | 流程 | 特点 |
| --- | --- | --- | --- | --- |
| **普通 RAG** | `POST /api/chat/query` | 固定流水线 | 检索 → Prompt 拼接 → LLM 生成 | 流程固定，延迟低 |
| **Multi-Tool Agent** | `POST /api/agent/query` | LLM 自主编排 | LLM 自主选择 7 种工具 → 基于工具结果生成 → 证据校验 | 可解释、可追溯、可扩展 |
| **MCP Server** | `remote-sensing-kb` | 宿主 Agent（Claude/Cursor）编排 | 宿主 LLM 自主调用检索/计算原子工具 → 自行推理生成 | 把本项目能力变成其他 Agent 的「工具箱」 |

```
文档上传 → 解析（PDF/TXT/MD）→ 文本清洗 → 递归切分 → Embedding → 向量入库
                                                              ↓
入口① 普通 RAG：检索 → 阈值过滤 → [可选] Rerank 精排 → LLM 生成 → 来源引用 / 拒答
                                                              ↓
入口② Multi-Tool Agent：响应缓存 → create_agent → 工具选择（7 种）→ 生成 → Evidence Verification → 结构化工作单元
                                                              ↓
入口③ MCP Server：宿主 Agent 调用 search_remote_sensing_kb / calculate_remote_sensing_metric 原子工具 → 宿主自行编排生成
```

> **后两种入口共享同一套内核**：Agent 的 `knowledge_base_search` Tool 与 MCP 的 `search_remote_sensing_kb` 复用同一个 `Retriever`；Agent 的 `metrics_calculator` 与 MCP 的 `calculate_remote_sensing_metric` 复用同一个 `core/metrics.py`。一次实现，三处复用 —— 这正是「工作单元」的核心：能力原子化、可被任意 Agent 编排。

---

## 项目亮点

### 1. 端到端 RAG 全流水线 + 双层防幻觉

从文档上传到带来源引用的智能问答，每个组件可独立替换。防幻觉覆盖 3 层（RAG）/ 4 层（Agent）：
- **工程层**：检索结果低于阈值（默认 0.3）时直接返回拒答，不调用 LLM
- **Prompt 层**：系统指令强制模型只基于上下文回答
- **Agent 第 4 层**：`verify_answer()` 事后核查回答与 sources/tool outputs 的证据一致性

### 2. Cross-encoder Rerank 重排序

采用 SiliconFlow BAAI/bge-reranker-v2-m3，向量检索召回 candidate_k=10 → cross-encoder 精排 → 保留 top_k=5。消融实验验证 rerank 使 source_recall 提升 +15.74%，MRR 提升 +10%。API 失败时自动降级到原始向量顺序。

### 3. Multi-Tool Agent：7 种工具自主选择

| 工具 | 数据来源 | 调用 LLM | 用途 |
| --- | --- | :---: | --- |
| `knowledge_base_search` | Chroma 向量库 | 否 | 开放性知识语义检索 |
| `plan_and_search` | Chroma + LLM 分解 | 是 | 复杂问题分解为子查询多次检索 |
| `dataset_overview` | 静态 JSON | 否 | 数据集共性概览 |
| `dataset_spec_lookup` | 静态 JSON | 否 | 具体数据集属性查询 |
| `model_comparison_table` | 静态 JSON | 否 | 模型架构对比 |
| `metric_formula_lookup` | 静态 JSON | 否 | 指标定义/公式查询 |
| `metrics_calculator` | 纯数值计算 | 否 | IoU/F1 等指标计算 |

### 4. 双层缓存（Agent 路径性能优化）

| 层级 | 位置 | 缓存粒度 | 默认 | 命中效果 |
| --- | --- | --- | --- | --- |
| **L1 Response Cache** | `agent_service.py` | 完整 Agent 响应 | 开启 | 零 LLM/工具调用，亚毫秒返回 |
| **L2 LLM Cache** | `langchain_agent.py` | 单次 LLM 调用 | 关闭 | 跳过 Round 1 LLM 推理（~15s） |

L1 缓存 key 包含归一化问题文本 + 检索/模型/校验配置 + 语料库版本哈希 + 领域数据哈希，确保配置变更或知识库更新后不会返回过期回答。文档入库/删除时自动清空。

### 5. Evidence Verification（证据校验）

Agent 回答生成后，对 answer / sources / tool_calls 进行证据一致性检查，输出 verified / confidence / ungrounded_claims。支持 `off` / `sync` / `deferred` 三种模式和 `lightweight` / `full` 轻量化级别。

### 6. MCP Server：把自己变成其他 Agent 的工具箱

除 HTTP API 外，项目还以 **MCP（Model Context Protocol）Server** 形式暴露能力，让 Claude Code / Cursor / Claude Desktop 等「宿主 Agent」直接调用，而不必经过本项目的 LLM 生成 —— **宿主 Agent 是编排者，本项目只提供检索与计算两类原子能力**：

| MCP 工具 | 能力 | 共享内核 |
| --- | --- | --- |
| `search_remote_sensing_kb` | 知识库语义检索，仅返回 chunks，不调 LLM 生成（避免双重 LLM） | `app.services.retriever.Retriever`（与 Agent `knowledge_base_search` 共用） |
| `calculate_remote_sensing_metric` | 评价指标确定性计算（IoU/Precision/Recall/F1，类型化参数） | `app.core.metrics.calculate_metric`（与 Agent `metrics_calculator` 共用） |

配置方式见根目录 [`.mcp.json`](.mcp.json)：

```json
{
  "mcpServers": {
    "remote-sensing-kb": { "command": "python", "args": ["-m", "mcp_server.server"] }
  }
}
```

> 设计取舍：MCP 的检索工具「只检索不生成」—— 把推理交还给宿主 LLM，避免「本项目生成一遍 + 宿主再生成一遍」的双重生成；计算工具用类型化参数（`tp/fp/fn` 为 float），与 Agent 的字符串风格（`"TP=80, FP=10"`）区分。这是「能力原子化、编排权归宿主」的 work unit 思路在协议层的落地。

### 7. 工程化架构

`core`（模型客户端）→ `services`（RAG 业务逻辑）→ `api`（HTTP 路由）→ `agents`（Agent 层）清晰分层。统一日志、异常处理、配置单例、Pydantic 全链路校验、确定性 chunk_id、Chroma 持久化、Docker 部署。

---

## Work Unit v1：把对话沉淀为可复盘的工作单元

Work Unit 是**产品层对象**，不是新 Agent，也不是 MCP tool。它把一次 RAG / Agent / MCP 调用过程装订成可保存、可查看、可复盘的结构化记录（设计详见 [`docs/work_unit_design.md`](docs/work_unit_design.md)）。

- **手动沉淀，不自动落盘**：RAG / Agent 查询响应中**新增可选字段 `work_unit_candidate`**（候选对象，含本次回答、来源、工具调用、轨迹、校验结果，以及为 v2 预留的 `replay_payload`）；
- 前端在结果下方显示「💾 保存为 Work Unit」按钮，**点击后**才调用 `POST /api/work_units` 落盘，保存成功返回 `work_unit_id`；
- 列表与详情可在前端「🗂️ 工作单元」入口查看，复用来源 / 工具调用 / 轨迹等现有渲染器。

Work Unit v1 的 HTTP 接口（路径统一用下划线）：

| 方法 | 路径 | 说明 |
| :---: | --- | --- |
| `POST` | `/api/work_units` | 保存候选对象为 Work Unit，返回 `work_unit_id` |
| `GET` | `/api/work_units` | 列表（支持 `?entry=rag/agent/mcp` 过滤） |
| `GET` | `/api/work_units/{work_unit_id}` | 详情（复盘视图） |
| `DELETE` | `/api/work_units/{work_unit_id}` | 删除 |

> **v1 不支持 Replay**：`replay_payload` 字段已**预留**并随 Work Unit 一同保存，但 v1 不实现任何重放端点，Replay 属于 v2。

字段统一用 `entry`（取值 `rag` / `agent` / `mcp`）标识 Work Unit 来源。

---

## MCP + Work Unit Fragment：能力开放，但不承担产品层状态

两个 MCP 工具的返回 dict 都**额外附带一个 `work_unit_fragment` 字段**，作为本次调用的「素材」：

| MCP 工具 | 附带的 fragment 内容 |
| --- | --- |
| `search_remote_sensing_kb` | `inputs={query, top_k}`、`outputs={contexts_count, sources_count}`、`elapsed` |
| `calculate_remote_sensing_metric` | `inputs={metric, tp, fp, fn, tn}`、`outputs={value}`、`elapsed` |

设计边界（详见 [`docs/mcp_design.md`](docs/mcp_design.md)）：

- **MCP fragment 不落盘**：MCP tool 不调用 WorkUnitStore、不写入 `data/work_units/`，也不调用 RAGService / Agent / LLM；
- **完整 Work Unit 由产品层保存**：fragment 只是碎片，是否、何时组装成完整 Work Unit 由宿主 Agent（Claude Code）或上层产品决定；
- 这样**避免让 MCP tool 承担产品层状态**，保持其无状态、确定性的性质（设计取舍见 [MCP 设计文档](docs/mcp_design.md)）。

---

## 技术栈

| 分类 | 技术 | 用途 |
| --- | --- | --- |
| Web 框架 | FastAPI + Uvicorn | RESTful API 后端 |
| Agent 框架 | LangChain 1.0 + langchain-openai | `create_agent`、`ChatOpenAI`、`@tool` |
| 向量数据库 | ChromaDB | 本地持久化（cosine HNSW） |
| Embedding | SiliconFlow BAAI/bge-m3（1024 维） | 多语言 Embedding |
| Rerank | SiliconFlow BAAI/bge-reranker-v2-m3 | Cross-encoder 精排 |
| LLM | OpenAI 兼容 API（智谱 GLM / DeepSeek / Qwen） | RAG 问答 + Agent 推理 |
| 前端 | Streamlit | 可视化问答界面 + Work Unit 列表/详情 |
| MCP Server | MCP SDK + FastMCP | 把检索/计算能力暴露给 Claude Code/Cursor 等宿主 Agent |
| Work Unit v1 | JSON 文件持久化（`data/work_units/`） | 把对话沉淀为可保存/可查看/可复盘的工作单元 |
| 测试 | pytest（442 个） | 全链路单元测试（mock 外部 API） |

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env            # Windows: copy .env.example .env
```

编辑 `.env`，填入必要配置：

```dotenv
# ===== 必填 =====
SILICONFLOW_API_KEY=你的_SiliconFlow_API_Key
LLM_API_KEY=你的_LLM_API_Key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4-flash

# ===== RAG 参数（可选，均有默认值）=====
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=5
SIMILARITY_THRESHOLD=0.3

# ===== Rerank =====
USE_RERANK=false                         # 是否全局开启 rerank
RERANK_CANDIDATE_K=10

# ===== Agent 缓存 =====
ENABLE_AGENT_RESPONSE_CACHE=true         # L1 响应级缓存（默认开启）
AGENT_RESPONSE_CACHE_TTL_SECONDS=600
AGENT_RESPONSE_CACHE_MAX_SIZE=100
ENABLE_AGENT_CACHE=false                 # L2 LLM 缓存（默认关闭，前端可逐请求覆盖）

# ===== Agent Evidence Verification =====
ENABLE_AGENT_VERIFICATION=true
AGENT_VERIFICATION_MODE=deferred          # off / sync / deferred
AGENT_VERIFICATION_LEVEL=lightweight      # lightweight / full
AGENT_MAX_TOKENS=1000
```

### 3. 启动服务

```bash
# 后端（FastAPI on :8000，Swagger at /docs）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 前端（Streamlit on :8501）
streamlit run frontend/streamlit_app.py
```

### 4. 使用

**方式一：Streamlit 界面**（推荐）— 上传文档 → 一键入库 → 选择 RAG / Agent 模式 → 提问

**方式二：REST API**

```bash
# 上传并入库
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@examples/sample_docs/01_datasets.md"
curl -X POST http://127.0.0.1:8000/api/documents/ingest \
  -H "Content-Type: application/json" -d '{"doc_id":"返回的doc_id"}'

# 普通 RAG 问答
curl -X POST http://127.0.0.1:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question":"LoveDA 数据集包含哪些类别？"}'

# Agent 问答
curl -X POST http://127.0.0.1:8000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question":"请比较 U-Net 和 DeepLabV3+ 在遥感语义分割中的适用场景"}'
```

**方式三：MCP Server**（让 Claude Code / Cursor / Claude Desktop 直接调用本项目能力）

```bash
python -m mcp_server.server          # 独立运行（stdio）
```

或在宿主 Agent 配置里引用根目录 `.mcp.json`（Claude Code 已内置 `remote-sensing-kb` server）。配好后，宿主 Agent 即可自主调用 `search_remote_sensing_kb`（知识库检索）和 `calculate_remote_sensing_metric`（指标计算）两个原子工具，把本项目当作它的工作单元工具箱。

---

## 快速公网演示（Cloudflare Tunnel）

> 适用场景：**临时演示**，给不在同一局域网的同事/朋友/老师远程查看你的前端界面。**不适合长期生产部署**。完整文档见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

### 1. 配置密钥与（可选）访问码

```bash
cp .env.example .env
```

编辑 `.env`，填入真实的 `SILICONFLOW_API_KEY` / `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`。

如需访问码保护（推荐），追加：

```env
DEMO_PASSWORD=your-demo-password
```

设置后，访问者必须输入此访问码才能进入 Streamlit 界面；不设置则保持原有体验。

### 2. 启动项目

```bash
docker compose up -d --build
docker compose ps                # 等待 backend / frontend 都 healthy
```

本机验证：浏览器打开 <http://127.0.0.1:8501>。

> 后端调试入口 <http://127.0.0.1:8000/docs> 仅本机可访问（端口已绑定 `127.0.0.1`），局域网其他人无法直接访问你的后端 API。

### 3. 启动 Cloudflare Tunnel

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

终端会输出形如：

```
https://random-words-1234.trycloudflare.com
```

把**该链接**和**演示访问码**（如果设置了）一起发给你信任的用户。

### 4. 关闭

```text
Ctrl + C            # 停止 cloudflared
docker compose down # 停止项目
```

### 5. 安全提醒

- ⚠️ **只把链接发给可信用户**——演示期间任何人都能通过前端上传文档、提问、删除知识库。
- ⚠️ **不要在演示时上传敏感文件**——演示数据存在你本机的持久化卷里。
- ⚠️ Cloudflare Quick Tunnel 地址通常是**临时的**，每次 `cloudflared` 重启都会变化。
- ⚠️ 演示期间你的电脑和 Docker 容器必须**保持运行**。
- ⚠️ 后端 8000 端口**只用于本机调试**，不应作为对外访问入口。

### 6. 常见问题（精简版）

| 症状 | 排查 |
| --- | --- |
| `http://127.0.0.1:8501` 打不开 | `docker compose ps` 看容器是否 healthy；`docker compose logs frontend` 看启动错误 |
| 页面能打开但问答失败 | `docker compose logs backend`；多半是 `.env` 密钥未配置或填错 |
| `cloudflared: command not found` | cloudflared 是独立客户端，不是 pip 包；安装方式见 [`docs/DEPLOY.md`](docs/DEPLOY.md) §7 |
| Tunnel 链接打开后白屏或断开 | 等几秒等 WebSocket 重连；长时间不操作会触发空闲断开，刷新页面即可 |
| 大文件上传或入库失败 | Cloudflare Quick Tunnel 对长请求不友好；建议演示时只上传小 PDF（< 5MB） |

完整问题排查见 [`docs/DEPLOY.md`](docs/DEPLOY.md) §8。

---

## API 接口

> 以下 HTTP 接口面向人 / 脚本调用；MCP Server（见上）则是面向其他 Agent 的同一后端能力暴露，二者共享 `Retriever` 与 `core/metrics.py` 内核。

| 方法 | 路径 | 说明 |
| :---: | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/documents/upload` | 上传文档（multipart/form-data） |
| `POST` | `/api/documents/ingest` | 入库（解析→切分→Embedding→Chroma） |
| `GET` | `/api/documents` | 列出知识库文档 |
| `DELETE` | `/api/documents/{doc_id}` | 删除文档 |
| `POST` | `/api/chat/query` | 普通 RAG 问答 |
| `POST` | `/api/agent/query` | Multi-Tool Agent 问答 |
| `POST` | `/api/agent/verify` | 独立 Evidence Verification |
| `POST` | `/api/work_units` | 保存 Work Unit（手动沉淀） |
| `GET` | `/api/work_units` | Work Unit 列表（支持 `?entry=` 过滤） |
| `GET` | `/api/work_units/{work_unit_id}` | Work Unit 详情（复盘） |
| `DELETE` | `/api/work_units/{work_unit_id}` | 删除 Work Unit |

Agent 问答返回字段：`answer` / `sources` / `refused` / `tool_calls` / `agent_trace` / `trace_events` / `verification` / `timing` / `errors` / `work_unit_candidate`（候选对象，可选）

---

## 目录结构

```
remote-sensing-rag/
├── app/
│   ├── main.py                        # FastAPI 入口
│   ├── config.py                      # Pydantic Settings 配置单例
│   ├── schemas.py                     # 请求/响应模型
│   ├── api/                           # HTTP 路由（documents / chat / agent）
│   ├── agents/                        # Agent 层
│   │   ├── langchain_agent.py         # create_agent + run_langchain_agent + L2 LLM Cache
│   │   ├── response_cache.py          # L1 Response Cache（TTL + max_size）
│   │   ├── agent_service.py           # AgentService（缓存调度 + 异常兜底）
│   │   ├── tools.py                   # knowledge_base_search Tool
│   │   ├── planning_tools.py          # plan_and_search Tool
│   │   ├── domain_tools.py            # 5 个结构化领域 Tool
│   │   ├── verification.py            # Evidence Verification
│   │   ├── prompts.py                 # Agent 系统提示词
│   │   └── types.py                   # Pydantic 数据模型
│   ├── core/                          # LLM / Embedding 客户端 + RAG Prompt
│   ├── services/                      # RAG 编排 / 检索 / 向量库 / Rerank / 文档解析
│   └── utils/                         # 日志 / 文件工具
├── frontend/streamlit_app.py          # Streamlit 前端
├── mcp_server/server.py               # MCP Server（暴露给 Claude Code/Cursor 的原子工具）
├── .mcp.json                          # MCP Server 注册配置
├── domain_data/                       # 结构化知识 JSON（datasets/models/metrics）
├── eval/                              # 评估 Harness + 21 道评估题
├── experiments/                       # 消融实验（参数 + Rerank）
├── tests/                             # pytest 单元测试（14 文件 + conftest，415 个测试）
├── examples/sample_docs/              # 10 篇遥感领域知识文档 + 示例问题
├── docs/                              # 设计文档（深度讲解 + Agent Trace 迭代记录 + 部署）
├── Dockerfile
├── docker-compose.yml                 # 后端 + 前端一键编排
├── requirements.txt
└── .env.example
```

---

## 测试

```bash
pytest -v                                    # 全部测试
pytest tests/test_agent_response_cache.py -v  # 响应缓存测试
pytest tests/test_rag.py::test_rag_refuse_when_empty -v  # 单个测试
```

415 个测试全部通过（1 个 skipped：`test_embeddings.py` 需真实 API），覆盖：文档解析、文本切分、RAG 拒答/正常路径、Agent 构建/执行/多工具解析、工具压缩/缓存、结构化查询、指标计算、查询分解、Evidence Verification、Agent API、**响应缓存（TTL/max_size/缓存隔离/异常不缓存/文档变更清空）**。

---

## 评估与消融实验

```bash
# 评估 Harness（需后端运行）
python eval/run_rag_eval.py     # 普通 RAG
python eval/run_agent_eval.py   # Agent

# 消融实验（独立 Chroma 临时 DB，不污染生产数据）
python -m experiments.rag_param_ablation.run_ablation    # 参数消融
python -m experiments.rag_rerank_ablation.run_rerank_ablation  # Rerank 消融
```

Rerank 消融关键结论：

| 指标 | baseline | rerank_k10 | 提升 |
|------|----------|------------|------|
| source_recall@k | 0.8241 | 0.9815 | **+15.74%** |
| MRR | 0.8444 | 0.9444 | **+10.00%** |
| answer_score | 0.8214 | 0.8666 | **+4.52%** |

---

## Docker 部署

```bash
docker build -t remote-sensing-rag .
docker run -d --name rs-rag -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data remote-sensing-rag
# Windows: 将 $(pwd) 替换为 %cd%
```

---

## Roadmap

**已完成**

- [x] RAG 知识库问答（双层拒答 + 来源引用）
- [x] Multi-Tool Agent（7 工具 + ReAct 循环 + 工具选择门控）
- [x] Cross-encoder Rerank（消融实验验证 + 生产集成）
- [x] Evidence Verification（off / sync / deferred + lightweight / full）
- [x] Agent Trace（trace_events + include_trace 控制 + 工具选择分析器）
- [x] **双层缓存（L1 Response Cache + L2 LLM Cache）**
- [x] **MCP Server（把检索/计算能力暴露给 Claude Code 等宿主 Agent 编排）**
- [x] **Work Unit v1（手动保存 / 列表 / 详情复盘，不含 Replay）**
- [x] **MCP Work Unit Fragment（MCP 工具返回 work_unit_fragment，不落盘）**
- [x] 参数消融 + Rerank 消融实验

**未来计划**

- [ ] Work Unit Replay（基于已保存的 `replay_payload` 重放）
- [ ] Replay 结果对比（新旧 Work Unit 并排差异）
- [ ] `save_work_unit` MCP tool（让宿主 Agent 主动组装并保存完整 Work Unit）
- [ ] Work Unit Gallery（可浏览 / 检索 / 打标签的画廊）
- [ ] 定时执行 Work Unit（基于 replay_payload 定时重跑，监测答案漂移）
- [ ] Streaming Output（流式输出）
- [ ] Self-Correction（校验未通过时自动修正）
- [ ] Query Rewrite（查询改写 / 扩展）
- [ ] 多用户知识库隔离 / OCR PDF / Redis 缓存

---

## 面试说明：这个项目想证明什么

这个项目不只是「一个 RAG demo」，而是用四类递进的能力入口，证明从「知识库问答」到「Agent 编排」到「能力开放」再到「产品抽象」的完整理解：

- **RAG 证明知识库问答能力**：端到端流水线（解析→切分→Embedding→检索→生成）+ 3 层防幻觉（阈值拒答 / Prompt 约束 / 拒答模板）+ 来源引用；
- **Agent 证明多工具编排能力**：LangChain 1.0 `create_agent` + 7 个 `@tool`（检索 / 分解 / 结构化查询 / 模型对比 / 指标查询 / 指标计算）+ ReAct 循环 + 工具选择门控；
- **MCP Server 证明能把项目能力开放给 Claude Code**：明确**没有**把整个 Agent 包成 MCP tool（避免 Agent-in-Agent 嵌套），而是暴露两个无状态、确定性的原子工具（检索 + 计算），让宿主 Agent 自己负责编排与生成（设计取舍见 [`docs/mcp_design.md`](docs/mcp_design.md)）；
- **Work Unit 证明对 agentic OS / thinking space 的产品理解**：把一次对话沉淀为可保存、可查看、可复盘的结构化工作单元，并刻意把「沉淀」做成产品层对象（手动保存、不污染推理层），而非又一个 Agent；
- **工程深度**：Evidence Verification（off / sync / deferred 三模式 + lightweight / full）、双层缓存（L1 响应缓存 + L2 LLM 缓存）、Cross-encoder Rerank 消融实验（source_recall +15.74%、MRR +10%）、参数消融、442 个单元测试（全链路 mock 外部 API，离线可跑）。

> 边界声明：Work Unit v1 **不含 Replay**（`replay_payload` 已预留但未实现端点）；MCP 工具**不落盘**完整 Work Unit（只返回 fragment）。两者均属 v2 范围，详见 [Roadmap](#roadmap)。

---

## License

MIT
