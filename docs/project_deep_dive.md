# Remote Sensing RAG 项目深度讲解

> 本文档面向已了解基础 RAG 流程（文档加载 → 切分 → 向量化 → 检索 → LLM 生成）的读者，着重讲解 **Rerank 重排序** 和 **Multi-Tool Agent** 两大进阶模块的设计理念与代码实现。

---

## 目录

- [第一部分：Rerank 重排序](#第一部分rerank-重排序)
  - [1.1 为什么需要 Rerank](#11-为什么需要-rerank)
  - [1.2 Bi-encoder vs Cross-encoder](#12-bi-encoder-vs-cross-encoder)
  - [1.3 实验版 reranker.py 代码逐行讲解](#13-实验版-rerankerpy-代码逐行讲解)
  - [1.4 三阶段消融实验设计](#14-三阶段消融实验设计)
  - [1.5 实验核心流程代码讲解](#15-实验核心流程代码讲解)
  - [1.6 实验结果与结论](#16-实验结果与结论)
  - [1.7 生产环境落地：从实验到上线](#17-生产环境落地从实验到上线)
- [第二部分：Multi-Tool Agent](#第二部分multi-tool-agent)
  - [2.1 Agent 与 RAG 的本质区别](#21-agent-与-rag-的本质区别)
  - [2.2 LangChain create_agent 的 ReAct 循环](#22-langchain-create_agent-的-react-循环)
  - [2.3 七大工具详解](#23-七大工具详解)
  - [2.4 System Prompt：如何指挥 LLM 选工具](#24-system-prompt如何指挥-llm-选工具)
  - [2.5 Agent 执行与结果解析](#25-agent-执行与结果解析)
  - [2.6 Evidence Verification（证据校验）](#26-evidence-verification证据校验)
  - [2.7 Agent Trace（执行轨迹）](#27-agent-trace执行轨迹)
  - [2.8 完整请求生命周期](#28-完整请求生命周期)
- [第三部分：Agent 双层缓存](#第三部分agent-双层缓存)
  - [3.1 为什么 Agent 需要缓存](#31-为什么-agent-需要缓存)
  - [3.2 L2 LLM Cache：单次调用级缓存](#32-l2-llm-cache单次调用级缓存)
  - [3.3 L1 Response Cache：完整响应级缓存](#33-l1-response-cache完整响应级缓存)
  - [3.4 缓存 key 设计](#34-缓存-key-设计)
  - [3.5 缓存失效与文档更新](#35-缓存失效与文档更新)

---

# 第一部分：Rerank 重排序

## 1.1 为什么需要 Rerank

在标准 RAG 流程中，检索阶段使用的是 **向量检索**（也叫密集检索）。它的工作方式是：

```
用户问题 → Embedding 模型 → 768/1024 维向量 → 与知识库中所有 chunk 向量计算余弦相似度 → 取 top-K
```

这有一个根本性的精度瓶颈：**Embedding 模型是"双编码器"（Bi-encoder），问题和文档是独立编码的**。

打个比方：就像考试时，你只看了题目摘要（没看完整题目），然后在一堆试卷中凭印象选出最相关的 5 份。你能选出大致相关的，但可能会混入"看起来相关但实际上答非所问"的卷子。

Rerank 就是来解决这个问题：**在向量检索选出的一批候选结果中，用更精确的模型重新打分排序**。

## 1.2 Bi-encoder vs Cross-encoder

这是理解 Rerank 的核心知识点：

```
┌─────────────────────────────────────────────────────────────┐
│                    Bi-encoder（向量检索）                      │
│                                                             │
│  Query ──→ Encoder ──→ [0.2, 0.8, ...] ──┐                 │
│                                           ├──→ cos(θ) = 0.87│
│  Doc   ──→ Encoder ──→ [0.3, 0.7, ...] ──┘                 │
│                                                             │
│  特点：Query 和 Doc 独立编码，速度快（O(1) 比较）              │
│  缺点：无法捕捉 Query-Doc 之间的细粒度语义交互                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Cross-encoder（Rerank）                     │
│                                                             │
│  [Query + Doc] ──→ Encoder ──→ relevance_score = 0.94      │
│                                                             │
│  特点：Query 和 Doc 拼接后联合编码，精度高                     │
│  缺点：每对 (Query, Doc) 都要跑一次模型，速度慢（O(N) 推理）   │
└─────────────────────────────────────────────────────────────┘
```

**两阶段检索流程**（本项目的做法）：

```
用户问题
  │
  ▼
向量检索（Bi-encoder, bge-m3）  ──→  候选结果 candidate_k=10
  │                                      │
  │  速度快，从数千 chunk 中粗筛            │
  ▼                                      ▼
阈值过滤（score >= 0.3）           Rerank（Cross-encoder, bge-reranker-v2-m3）
  │                                      │
  │                               逐对精排，保留 final_top_k=5
  ▼                                      │
最终 top_k=5 结果  ◄──────────────────────┘
```

为什么不直接用 Cross-encoder 检索全部文档？因为如果你有 1000 个 chunk，对每个 chunk 都跑一次 Cross-encoder 推理，延迟会高到无法接受。所以先用快速的向量检索缩小范围，再用精确的 Cross-encoder 精排。

## 1.3 实验版 reranker.py 代码逐行讲解

文件路径：`experiments/rag_rerank_ablation/reranker.py`

### 配置读取：`_get_rerank_config()`

```python
def _get_rerank_config() -> Tuple[str, str, str]:
    # 步骤 1：加载 .env 文件到环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 步骤 2：优先读 RERANK_* 环境变量
    api_key = os.getenv("RERANK_API_KEY", "")
    base_url = os.getenv("RERANK_BASE_URL", "")
    model = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)

    # 步骤 3：留空则回退到 SILICONFLOW_* 配置（通过 pydantic-settings 读取 .env）
    if not api_key or not base_url:
        from app.config import get_settings
        settings = get_settings()
        if not api_key:
            api_key = settings.siliconflow_api_key
        if not base_url:
            base_url = settings.siliconflow_base_url

    base_url = base_url.rstrip("/")
    return api_key, base_url, model
```

**关键设计：三级回退机制**
1. 先尝试 `RERANK_API_KEY`（专用 rerank 密钥）
2. 留空则回退到 `SILICONFLOW_API_KEY`（复用 embedding 的密钥）
3. model 默认 `BAAI/bge-reranker-v2-m3`

这就是为什么你的 `.env` 中不配置 `RERANK_*` 变量，实验也能正常运行：SiliconFlow 的 rerank API 和 embedding API 共用同一套认证。

### 核心 API 调用：`rerank()`

```python
def rerank(query, documents, top_n=None, ...):
    url = f"{base_url}/rerank"
    payload = {
        "model": model,               # "BAAI/bge-reranker-v2-m3"
        "query": query,               # 用户问题
        "documents": documents,       # ["文档1文本", "文档2文本", ...]
        "return_documents": False,    # 不返回文档全文，只返回 index + score
        "max_chunks_per_doc": 512,    # 每篇文档最大分块数
    }
    if top_n is not None:
        payload["top_n"] = min(top_n, len(documents))

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    # ...
    results = data.get("results", [])
    # results 格式：[{"index": 2, "relevance_score": 0.95}, {"index": 0, ...}, ...]
```

**API 请求和响应格式**：
- 请求：发送 `query` + `documents[]`（纯文本列表）
- 响应：返回 `results[]`，每项包含 `index`（原始文档索引）和 `relevance_score`（0~1 的相关性分数）
- `index` 是关键：它告诉你原始 `documents` 列表中的第几个文档最相关

### 优雅降级：`rerank_search_results()`

```python
def rerank_search_results(query, search_results, final_top_k, ...):
    # search_results 是向量检索的结果列表（dict 列表，含 content/score/filename 等）

    # 提取纯文本用于 rerank API
    documents = [r.get("content", "") for r in search_results]

    try:
        rerank_results = rerank(query=query, documents=documents, top_n=final_top_k)
        # 按 rerank 结果重排序
        reranked = []
        for item in rerank_results:
            idx = item["index"]           # 原始文档索引
            score = item["relevance_score"]
            result = dict(search_results[idx])  # 浅拷贝原始结果（保留向量 score）
            result["rerank_score"] = score      # 新增 rerank_score 字段
            reranked.append(result)
        return reranked, elapsed, False       # used_fallback=False

    except Exception as e:
        # ⭐ 优雅降级：API 失败时回退到原始向量顺序
        logger.warning("Rerank 调用失败，回退到原始向量顺序: %s", e)
        return search_results[:final_top_k], 0.0, True  # used_fallback=True
```

**三个关键设计决策**：

1. **保留原始向量 score，新增 rerank_score**：不覆盖原始相似度分数，而是添加 `rerank_score` 字段。这样你可以同时看到 bi-encoder 和 cross-encoder 的打分。

2. **返回 `used_fallback` 标志**：让调用方知道这次结果是否经过 rerank。实验中用这个标志统计 fallback 次数。

3. **try-except 包裹整个 rerank 调用**：任何异常（网络超时、API 限流、JSON 解析错误）都降级为原始向量顺序，保证系统可用性。

## 1.4 三阶段消融实验设计

文件路径：`experiments/rag_rerank_ablation/run_rerank_ablation.py`

### 三组配置

| 配置名 | use_rerank | candidate_k | final_top_k | 含义 |
|--------|-----------|-------------|-------------|------|
| baseline | False | 5 | 5 | 纯向量检索 top_k=5 |
| rerank_k10 | True | 10 | 5 | 向量检索 10 条 → rerank → 取前 5 |
| rerank_k20 | True | 20 | 5 | 向量检索 20 条 → rerank → 取前 5 |

**为什么有 rerank_k10 和 rerank_k20？** 因为 rerank 的效果取决于候选池大小。候选池太小（k=5），rerank 没有调整空间；候选池太大（k=20），可能引入更多噪声，且延迟更高。实验需要找到最佳平衡点。

### 三阶段评估

```
阶段一：Retrieval-level（只看检索质量，不调 LLM）
  ├─ 输入：18 个领域内问题
  ├─ 指标：source_hit_rate, recall@k, MRR, avg_score, latency
  └─ 目的：量化 rerank 对检索精度的直接提升

阶段二：Out-of-scope 安全性（只看检索质量，不调 LLM）
  ├─ 输入：全部 21 个问题（含 3 个领域外问题）
  ├─ 指标：false_refusal_rate, false_accept_rate
  └─ 目的：检测 rerank 是否影响拒答行为

阶段三：Answer-level（端到端，调用 LLM 生成回答）
  ├─ 输入：全部 21 个问题
  ├─ 对比：baseline vs 最佳 rerank 配置（阶段一选出）
  ├─ 指标：keyword_coverage, source_hit_rate, refusal_accuracy
  └─ 目的：验证检索精度的提升是否传导到最终回答质量
```

### 评分公式

```python
# 阶段一：检索评分
retrieval_score = (
    0.45 * source_hit_rate           # 是否命中正确文档（最重要）
    + 0.25 * source_recall_at_k      # 命中了多少比例的正确文档
    + 0.15 * mrr                     # 正确文档排在第几位
    + 0.10 * avg_top_score           # 最高相似度分数
    - 0.05 * latency_norm            # 延迟惩罚（权重很小）
)

# 阶段二：拒答评分
refusal_score = (
    0.35 * in_scope_recall           # 领域内问题不应被拒
    + 0.45 * out_refusal_acc         # 领域外问题应该被拒
    - 0.05 * false_refusal_rate      # 错误拒答惩罚
    - 0.15 * false_accept_rate       # 错误接受惩罚（权重最大）
)

# 阶段三：回答评分
answer_score = (
    0.50 * keyword_coverage          # 期望关键词覆盖率（最重要）
    + 0.25 * source_hit_rate         # 来源命中率
    + 0.15 * refusal_accuracy        # 拒答准确率
    + 0.10 * min_length_satisfied    # 回答长度达标率
)
```

## 1.5 实验核心流程代码讲解

### 检索核心：`_retrieve_with_config()`

这是三组配置共用的检索函数，是整个实验的核心：

```python
def _retrieve_with_config(store, query, config, threshold, raw_results=None, raw_latency=0.0):
    candidate_k = config["candidate_k"]      # baseline=5, rerank_k10=10, rerank_k20=20
    final_top_k = config["final_top_k"]      # 都等于 5

    # 步骤 1：获取候选结果（从预检索缓存中取前 candidate_k 条）
    if raw_results is not None:
        candidates_raw = raw_results[:candidate_k]  # 从缓存切片，公平比较
    else:
        candidates_raw, search_latency = _search_raw(store, query, candidate_k)

    # 步骤 2：阈值过滤（与生产环境一致，score >= 0.3）
    filtered = [r for r in candidates_raw if r.get("score", 0.0) >= threshold]

    if not filtered:
        return [], search_latency, False  # 全部被阈值过滤 → 空结果

    # 步骤 3：是否 rerank
    if config.get("use_rerank", False):
        reranked, rerank_elapsed, used_fallback = rerank_search_results(
            query=query,
            search_results=filtered,
            final_top_k=final_top_k,
        )
        return reranked, search_latency + rerank_elapsed, used_fallback
    else:
        return filtered[:final_top_k], search_latency, False
```

**关键设计：预检索缓存 + 切片**

在阶段一中，代码不是为每个配置独立检索一次，而是：

```python
# 预检索：取 max_candidate_k=20 条原始结果（用极低 threshold=-1 获取全部）
for q in in_scope_questions:
    raw_results, latency = _search_raw(store, q["question"], max_candidate_k=20)
    raw_cache[q["id"]] = (raw_results, latency)

# 然后每个配置从缓存中切片
# baseline 取 raw_cache[:5]
# rerank_k10 取 raw_cache[:10]
# rerank_k20 取 raw_cache[:20]（全部）
```

这确保了三组配置看到的是**完全相同的候选池**，只是切片大小不同，保证了比较的公平性。

### 阶段三的两种回答路径

**Baseline 路径**（`_run_baseline_answers`）：
```python
# 直接复用生产环境的 RAGService
retriever = Retriever(store=store)
rag = RAGService(retriever=retriever)
answer_obj = rag.answer(question=q["question"], top_k=top_k, similarity_threshold=threshold)
# 内部流程：retrieve → refuse-if-empty → LLM.chat()
```

**Rerank 路径**（`_run_rerank_answers`）：
```python
# 手动构建 rerank 流程（因为生产环境 RAGService 不含 rerank）
raw_results = store.search(query=query, top_k=candidate_k, similarity_threshold=-1.0)
filtered = [r for r in raw_results if r["score"] >= threshold]

if not filtered:
    answer = REFUSAL_ANSWER  # 阈值过滤后为空 → 拒答
else:
    reranked, _, _ = rerank_search_results(query, filtered, final_top_k)
    # 构建 context → 调用 LLM
    context = "\n\n".join([r["content"] for r in reranked])
    answer = llm.chat(prompt=query, system=RAG_SYSTEM_PROMPT, context=context)
```

## 1.6 实验结果与结论

| 指标 | baseline | rerank_k10 | 差异 |
|------|----------|------------|------|
| source_recall_at_k | 0.8241 | 0.9815 | **+15.74%** |
| MRR | 0.8444 | 0.9444 | **+10.00%** |
| retrieval_score | 0.8486 | 0.8569 | +0.83% |
| keyword_coverage | 0.7762 | 0.8571 | **+8.09%** |
| answer_score | 0.8214 | 0.8666 | **+4.52%** |
| 延迟 | 0.12s | 1.29s | 11.17x |

**结论**：
- **Recall 提升显著**（+15.74%）：rerank 能从更大的候选池中找回向量检索遗漏的正确文档
- **MRR 提升显著**（+10%）：rerank 将正确文档排到了更靠前的位置
- **Answer-level 提升明显**（+4.52%）：检索精度的提升传导到了最终回答质量
- **延迟代价可控**：rerank 增加 ~1.2s，但 LLM 生成需 ~20s，rerank 延迟占比很小

## 1.7 生产环境落地：从实验到上线

消融实验验证了 rerank_k10（candidate_k=10 → rerank → top_k=5）是最佳配置。接下来讲解如何将这一结论**最小侵入式**地集成到生产代码中。

> **设计原则**：不修改 `VectorStore`、不修改现有测试逻辑，仅在 `Retriever.retrieve()` 这一个入口点增加 rerank 分支。普通 RAG 和 Agent 两条路径都流经此入口，因此一处改动覆盖全局。

### 1.7.1 生产版 `app/services/reranker.py`

实验版和生产版的区别很小，但很重要：

| 对比项 | 实验版 `experiments/.../reranker.py` | 生产版 `app/services/reranker.py` |
|--------|--------------------------------------|-------------------------------------|
| 配置读取 | `os.getenv()` + `from dotenv import load_dotenv` | `get_settings()`（pydantic-settings 单例） |
| 凭证回退 | 三级回退（`RERANK_*` → `SILICONFLOW_*`） | 直接复用 `settings.siliconflow_api_key` |
| 返回值 | `(reranked, elapsed, used_fallback)` | 相同 |
| 降级策略 | API 失败 → 返回原始向量顺序 | 相同 |

**生产版 `_get_rerank_config()`**：

```python
def _get_rerank_config() -> Tuple[str, str, str]:
    settings = get_settings()
    api_key = settings.siliconflow_api_key        # 复用 Embedding 的 SiliconFlow 密钥
    base_url = settings.siliconflow_base_url.rstrip("/")
    model = settings.rerank_model or DEFAULT_RERANK_MODEL
    return api_key, base_url, model
```

为什么生产版不需要三级回退？因为 pydantic-settings 的 `Settings` 类中已经定义了 `rerank_model` 字段（默认值 `BAAI/bge-reranker-v2-m3`），而 `siliconflow_api_key` / `siliconflow_base_url` 已是必填项（Embedding 功能依赖）。所以 rerank 天然复用同一套凭证，无需额外配置。

### 1.7.2 `app/config.py` 新增三个字段

```python
class Settings(BaseSettings):
    # ... 原有字段 ...

    # ---- Rerank（Cross-encoder 重排序） ----
    use_rerank: bool = Field(default=False, alias="USE_RERANK")
    rerank_candidate_k: int = Field(default=10, alias="RERANK_CANDIDATE_K")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL"
    )
```

- `USE_RERANK`：全局开关，默认 `False`（不开启 rerank 时行为与之前完全一致）
- `RERANK_CANDIDATE_K`：向量检索召回的候选数量，默认 `10`（消融实验最佳值）
- `RERANK_MODEL`：rerank 模型名称，默认 `BAAI/bge-reranker-v2-m3`

> **pydantic-settings 特性**：这些字段如果不写进 `.env`，会静默使用 `Field(default=...)` 的默认值，不会报错。所以你可以在 `.env` 中只加 `USE_RERANK=true` 就开启 rerank，其余参数使用默认值。

### 1.7.3 `Retriever.retrieve()`：唯一的改动入口

```python
class Retriever:
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_rerank: Optional[bool] = None,          # ← 新增参数
    ) -> List[dict]:
        settings = get_settings()
        final_k = top_k or settings.top_k
        threshold = similarity_threshold if similarity_threshold is not None \
            else settings.similarity_threshold

        # ⭐ 优先级规则：前端传入 use_rerank > .env 中的 USE_RERANK
        effective_rerank = use_rerank if use_rerank is not None \
            else settings.use_rerank

        if effective_rerank:
            # 两阶段检索：先取 candidate_k 条候选 → rerank 精排 → 保留 final_k 条
            candidate_k = settings.rerank_candidate_k
            hits = self.store.search(query, top_k=candidate_k, ...)
            if not hits:
                return []
            reranked, rerank_elapsed, used_fallback = rerank_search_results(
                query=query, search_results=hits, final_top_k=final_k,
            )
            return reranked
        else:
            # 原始流程：纯向量检索
            hits = self.store.search(query, top_k=final_k, ...)
            return hits
```

**核心逻辑讲解**：

1. **`use_rerank` 参数的三态语义**：
   - `None`（默认）：跟随 `.env` 中的 `USE_RERANK` 配置
   - `True`：强制开启 rerank（前端勾选了复选框）
   - `False`：强制关闭 rerank（前端取消了复选框）

2. **两阶段检索流程**（`effective_rerank=True` 时）：
   ```
   用户问题
     │
     ▼
   向量检索 candidate_k=10 条（阈值过滤后）
     │
     ▼
   rerank_search_results()：调用 bge-reranker-v2-m3 API
     │   提取每条结果的 content 纯文本
     │   发送 query + documents[] 给 rerank API
     │   API 返回 [{index, relevance_score}, ...]
     │   按 relevance_score 降序重排原始 hit 字典
     │   保留前 final_top_k=5 条
     ▼
   返回重排序后的 5 条结果（含 rerank_score 字段）
   ```

3. **优雅降级**：如果 rerank API 调用失败（网络超时、密钥无效、服务不可用），`rerank_search_results()` 会 catch 异常并返回原始向量检索顺序的前 `final_top_k` 条，同时设置 `used_fallback=True`。**用户不会看到错误，只是失去了 rerank 增益**。

### 1.7.4 前端开关：端到端数据流

前端增加了一个 rerank 复选框，用户可以**逐次请求**控制是否启用 rerank。这需要在请求中传递 `use_rerank` 字段：

**普通 RAG 路径**（直接传递）：

```
前端 checkbox → ChatQueryRequest.use_rerank → chat.py → RAGService.answer(use_rerank=...)
    → Retriever.retrieve(use_rerank=...) → rerank 分支
```

**Agent 路径**（模块级标志位）：

Agent 的工具调用经由 LangChain `create_agent` 内部循环，无法直接在 `knowledge_base_search` 工具函数中传递 `use_rerank` 参数。解决方案是使用**模块级标志位**：

```python
# app/agents/tools.py
_rerank_override: bool | None = None      # 模块级全局变量

def set_rerank_override(value: bool | None) -> None:
    """由 AgentService 在调用 Agent 前设置。"""
    global _rerank_override
    _rerank_override = value

def _retrieve(query, top_k=None):
    retriever = Retriever()
    hits = retriever.retrieve(
        query, top_k=actual_top_k,
        similarity_threshold=settings.similarity_threshold,
        use_rerank=_rerank_override,        # ← 读取模块级标志
    )
    return hits
```

```python
# app/agents/agent_service.py
def query(self, question, include_trace=True, use_rerank=None):
    try:
        set_rerank_override(use_rerank)      # 设置标志位
        result = run_langchain_agent(question)  # Agent 内部调用 _retrieve
        set_rerank_override(None)             # ⭐ 用完立即重置
    # ...
```

**为什么 Agent 路径用标志位而不是参数传递？**

因为 LangChain `create_agent` 的工具调用循环是封装好的：LLM 决定调用 `knowledge_base_search(query="...")` 时，只传 `query` 参数。如果要在工具签名中增加 `use_rerank` 参数，LLM 也会尝试自己决定传什么值，而不是由用户控制。模块级标志位绕过了这个问题：`AgentService` 在调用前设置，Agent 工具内部读取，调用后重置。

**完整数据流图**：

```
┌─────────────── 前端 Streamlit ───────────────┐
│  ☑ 启用 Rerank 重排序（use_rerank=True）       │
└───────────────────┬───────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    /api/chat/query       /api/agent/query
          │                    │
          ▼                    ▼
   RAGService.answer     AgentService.query
   (use_rerank=True)     (use_rerank=True)
          │                    │
          │                    ├── set_rerank_override(True)
          │                    │        ↓
          ▼                    │  run_langchain_agent()
   Retriever.retrieve          │        ↓
   (use_rerank=True)           │  knowledge_base_search()
          │                    │        ↓
          ▼                    │  _retrieve()
   ✅ rerank 分支               │        ↓
                               │  Retriever.retrieve(use_rerank=_rerank_override)
                               │        ↓
                               │  ✅ rerank 分支
                               │        ↓
                               │  set_rerank_override(None)  ← 重置
```

### 1.7.5 与实验版的差异总结

| 维度 | 实验版 | 生产版 |
|------|--------|--------|
| 文件位置 | `experiments/rag_rerank_ablation/reranker.py` | `app/services/reranker.py` |
| 配置来源 | `os.getenv()` + dotenv | `get_settings()` (pydantic-settings) |
| 调用入口 | `run_rerank_ablation.py`（独立脚本） | `Retriever.retrieve()`（生产入口） |
| 开关控制 | 硬编码在实验脚本中 | `.env` 配置 + 前端逐次请求覆盖 |
| 侵入性 | 独立运行，不碰生产代码 | 最小侵入：仅改 `Retriever.retrieve()` |
| 测试 | 实验结果 JSON | 406 个 pytest 全部通过 |

---

# 第二部分：Multi-Tool Agent

## 2.1 Agent 与 RAG 的本质区别

```
┌─────────────────────────── RAG 路径 ───────────────────────────┐
│                                                                │
│  用户问题 → 向量检索（可选 rerank 精排）→ 拼 context → LLM 生成│
│                                                                │
│  特点：流程固定，LLM 只做一次生成，没有自主决策权               │
│  缺点：不管问什么，都去向量库里搜；无法利用结构化数据           │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────── Agent 路径 ─────────────────────────┐
│                                                                │
│  用户问题 → LLM 自主思考 → 选择合适的工具 → 执行工具           │
│                ↑               ↓                               │
│                │     是否需要更多信息？                          │
│                └───── 是 ←─────┘                               │
│                          │                                     │
│                          否                                    │
│                          ↓                                     │
│                     生成最终回答                                │
│                                                                │
│  特点：LLM 是"大脑"，自主决定用什么工具、调用几次              │
│  优势：可以根据问题类型选择最优信息来源                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**举个具体例子**：

| 问题 | RAG 做法 | Agent 做法 |
|------|---------|-----------|
| "mIoU 怎么计算？" | 去向量库搜索 "mIoU" 相关 chunk，可能搜到也可能搜不到 | 调用 `metric_formula_lookup("mIoU")`，直接从结构化 JSON 中精确查找 |
| "帮我算 IoU, TP=80, FP=10, FN=20" | 去向量库搜索（完全无法计算） | 调用 `metrics_calculator("IoU", "TP=80, FP=10, FN=20")`，精确计算 |
| "对比 U-Net 和 DeepLabV3+" | 去向量库搜索，可能只搜到一个模型的信息 | 调用 `model_comparison_table("U-Net, DeepLabV3+")`，直接返回对比表 |
| "遥感语义分割的主要挑战是什么？" | 去向量库搜索 → 合适 | 调用 `knowledge_base_search(...)` → 合适 |

## 2.2 LangChain create_agent 的 ReAct 循环

### 什么是 ReAct

ReAct = **Reasoning + Acting**，即"边想边做"。LLM 在每一步先思考（Reasoning），然后决定下一步行动（Acting），看到行动结果后再继续思考，直到得出最终答案。

```
循环流程：
  ┌──────────────────────────────────────────────────────┐
  │  User: "对比 U-Net 和 DeepLabV3+ 的优缺点"            │
  │                                                      │
  │  Step 1 - LLM 思考:                                   │
  │    "用户要对比两个模型，我应该用 model_comparison_table"│
  │  Step 1 - LLM 行动:                                   │
  │    调用 model_comparison_table(models="U-Net,DeepLabV3+")│
  │  Step 1 - 工具返回:                                   │
  │    {comparison: [{name: "U-Net", ...}, {name: "DeepLabV3+", ...}]}│
  │                                                      │
  │  Step 2 - LLM 思考:                                   │
  │    "我已经拿到了两个模型的对比信息，可以生成回答了"      │
  │  Step 2 - LLM 行动:                                   │
  │    生成最终文本回答                                    │
  │                                                      │
  │  循环结束（LLM 没有再调用工具，而是直接输出了文本）      │
  └──────────────────────────────────────────────────────┘
```

### create_agent 的内部机制

本项目使用 LangChain 1.0 的 `create_agent` 函数：

```python
# langchain_agent.py:147-153
agent = create_agent(
    model=model,                          # ChatOpenAI 实例
    tools=DEFAULT_TOOLS,                  # 7 个 @tool 装饰的函数
    system_prompt=REMOTE_SENSING_AGENT_SYSTEM_PROMPT,  # 系统提示词
)
```

`create_agent` 内部返回一个 `CompiledStateGraph`（来自 LangGraph），其核心是一个**有限状态机循环**：

```
状态1: model_node    → 调用 LLM，LLM 决定是否调用工具
         │
         ├─ LLM 输出包含 tool_calls → 转到 状态2
         └─ LLM 输出纯文本（无 tool_calls）→ 转到 状态3（结束）

状态2: tools_node    → 执行所有 tool_calls，将结果追加到 messages
         │
         └─ 返回 状态1（让 LLM 看到工具结果后继续思考）

状态3: END            → 返回最终 messages 列表
```

**你不直接操作 LangGraph**，`create_agent` 封装了所有 StateGraph 逻辑。你只需要：
1. 提供支持 `bind_tools()` 的 LLM
2. 提供用 `@tool` 装饰的工具函数
3. 调用 `agent.invoke({"messages": [{"role": "user", "content": question}]})`

### 为什么用 ChatOpenAI 而不用项目已有的 OpenAICompatibleLLMClient

```python
# langchain_agent.py:73-117
def build_chat_model() -> ChatOpenAI:
    llm = ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.agent_max_tokens,
    )
    return llm
```

原因在注释中写得很清楚：`create_agent` 要求模型支持 `bind_tools()` 方法。`ChatOpenAI` 原生支持 function calling / tool calling，而项目原有的 `OpenAICompatibleLLMClient` 虽然继承了 `BaseChatModel`，但没有实现 `bind_tools()`。

**这就是项目中存在两个 LLM 客户端的原因**：
- RAG 路径：`OpenAICompatibleLLMClient.chat()` — 只需简单文本生成
- Agent 路径：`ChatOpenAI` — 需要 `bind_tools()` 支持工具调用

这两个客户端不应合并，因为它们服务于完全不同的需求。

### 单例化 Agent

```python
# langchain_agent.py:173-196
@lru_cache(maxsize=1)
def get_remote_sensing_agent() -> Any:
    model = build_chat_model()
    agent = create_agent(model=model, tools=DEFAULT_TOOLS, system_prompt=...)
    return agent
```

`create_agent` 涉及模型初始化 + LangGraph 图编译，开销不小。用 `@lru_cache(maxsize=1)` 确保全局只构建一次。

**代价**：如果 `.env` 中的 `LLM_API_KEY` / `LLM_MODEL` 变了，需要重启服务（lru_cache 不会自动刷新）。

## 2.3 七大工具详解

### 工具总览

```python
# langchain_agent.py:52-60
DEFAULT_TOOLS = [
    knowledge_base_search,     # 1. 知识库语义检索
    plan_and_search,           # 2. 复杂问题分解+多次检索
    dataset_overview,          # 3. 数据集共性概览
    dataset_spec_lookup,       # 4. 具体数据集属性查询
    model_comparison_table,    # 5. 模型对比表
    metric_formula_lookup,     # 6. 指标定义/公式查询
    metrics_calculator,        # 7. 指标数值计算
]
```

| 工具 | 调用 LLM？ | 调用向量库？ | 数据来源 | 用途 |
|------|-----------|-------------|---------|------|
| knowledge_base_search | 否 | 是 | Chroma | 开放性知识检索 |
| plan_and_search | **是**（分解查询） | 是 | Chroma | 复杂多维度检索 |
| dataset_overview | 否 | 否 | 静态 JSON | 数据集共性总结 |
| dataset_spec_lookup | 否 | 否 | 静态 JSON | 具体数据集精确查询 |
| model_comparison_table | 否 | 否 | 静态 JSON | 模型对比 |
| metric_formula_lookup | 否 | 否 | 静态 JSON | 指标定义查询 |
| metrics_calculator | 否 | 否 | 纯数值计算 | 指标数值计算 |

### 工具 1：knowledge_base_search

文件：`agents/tools.py`

```python
@tool
def knowledge_base_search(query: str) -> str:
    """Search the local remote sensing semantic segmentation knowledge base..."""
    normalized = normalize_query(query)  # strip + lowercase + whitespace collapse
    if not normalized:
        return _build_empty_json(query, 0.0)
    return _cached_search(normalized, settings.top_k)
```

**关键设计**：

1. **`@tool` 装饰器**：LangChain 的 `@tool` 会读取函数的 **docstring**（必须是英文）作为工具描述，告诉 LLM 什么时候该用这个工具。函数的 **类型注解**（`query: str`）会被转换为 JSON Schema，告诉 LLM 怎么调用。

2. **`normalize_query`**：`strip().lower()` + 连续空白合并。这不是为了 NLP 处理，纯粹是为了 **最大化 LRU 缓存命中率**。"mIoU 怎么算" 和 " mIoU 怎么算 " 归一化后命中同一条缓存。

3. **`_cached_search` 带 `@lru_cache(maxsize=128)`**：相同查询只检索一次。后续 Agent 如果对同一问题重复调用（比如 plan_and_search 的子查询恰好相同），直接返回缓存。

4. **内容压缩**：
   ```python
   _MAX_CONTEXT_CHARS = 500    # contexts[].content 最多 500 字符
   _MAX_PREVIEW_CHARS = 150    # sources[].content_preview 最多 150 字符
   ```
   因为工具输出会进入 LLM 的 context window，太长会浪费 token 和影响注意力。

5. **返回 JSON 字符串**：所有工具返回 `str` 类型（LangChain 要求），但内容是结构化 JSON：
   ```json
   {
     "success": true,
     "query": "mIoU",
     "summary": "检索到 5 个相关片段",
     "contexts": [{"source_id": "source_1", "content": "...", "source": "...", "score": 0.78}],
     "sources": [{"filename": "03_metrics.md", "page": 1, "chunk_id": "abc123", "score": 0.78, "content_preview": "..."}],
     "timing": {"search_elapsed": 0.123}
   }
   ```

6. **缓存失效**：当文档入库或删除时，`documents.py` 会调用 `clear_agent_search_cache()` 清空缓存。

### 工具 2：plan_and_search

文件：`agents/planning_tools.py`

这是最复杂的工具——它内部会**调用 LLM 来分解查询**，然后对每个子查询分别检索，最后合并去重。

```python
@tool
def plan_and_search(query: str) -> str:
    """Use this tool only for complex multi-entity, multi-aspect comparison..."""
    # 步骤 1：准入门控（不调用 LLM，纯规则判断）
    suitable, reason = should_use_plan_and_search(query)
    if not suitable:
        return json.dumps({"success": False, "reason": reason, ...})

    # 步骤 2：LLM 分解查询
    sub_queries, planning_elapsed = _decompose_query(query)
    # 例如："对比 U-Net 和 DeepLabV3+ 在 LoveDA 上的表现"
    #   → ["U-Net 架构和特点", "DeepLabV3+ 架构和特点", "LoveDA 数据集特征", "两个模型在 LoveDA 上的表现对比"]

    # 步骤 3：对每个子查询检索（复用 _cached_search）
    for sq in sub_queries:
        result_json = _cached_search(normalize_query(sq), top_k)

    # 步骤 4：合并去重
    merged_contexts, merged_sources, total_elapsed = _merge_search_results(sub_results)
```

**准入门控的设计**（`should_use_plan_and_search`）：

这个函数用**纯规则**判断一个问题是否需要分解，避免对简单问题浪费一次 LLM 调用：

```python
def should_use_plan_and_search(query: str) -> Tuple[bool, str]:
    # 条件 A：出现比较关键词（"对比"、"比较"、"差异"、"优缺点"等）
    for kw in _COMPARISON_KEYWORDS:
        if re.search(kw, q):
            return True, f"检测到比较关键词：{kw}"

    # 条件 B：出现两个及以上已知实体（"deeplabv3+" + "loveda"）
    entity_count = 0
    for entity in _KNOWN_MODEL_ENTITIES + _KNOWN_DATASET_ENTITIES:
        if entity in q:
            entity_count += 1
            if entity_count >= 2:
                return True, "检测到多个已知实体"

    # 条件 C：多方面分析模式（"架构.*指标" 同时出现）
    for pattern, desc in _MULTI_ASPECT_PATTERNS:
        if re.search(pattern, q):
            return True, f"检测到多方面分析需求：{desc}"

    return False, "该问题不需要复杂分解"
```

**合并去重逻辑**（`_merge_search_results`）：

多个子查询可能检索到同一个 chunk。合并时按 `chunk_id` 去重，保留最高分的版本：

```python
def _merge_search_results(sub_query_results):
    seen = {}  # chunk_id → {context, source, score}
    for sub_json in sub_query_results:
        data = json.loads(sub_json)
        for ctx, src in zip(data["contexts"], data["sources"]):
            chunk_id = src["chunk_id"]
            score = src["score"]
            if chunk_id in seen:
                if score > seen[chunk_id]["score"]:  # 保留更高分
                    seen[chunk_id] = {"context": ctx, "source": src, "score": score}
            else:
                seen[chunk_id] = {"context": ctx, "source": src, "score": score}

    # 按分数降序排列，重新编号 source_id
    sorted_entries = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    # ...
```

**循环依赖处理**：

`planning_tools.py` 需要调用 `build_chat_model()` 来分解查询，但 `langchain_agent.py` 又导入了 `plan_and_search`。这形成了循环依赖：

```
planning_tools → langchain_agent（build_chat_model）
langchain_agent → planning_tools（plan_and_search）
```

解决方案是 **lazy import**（延迟导入）：

```python
# planning_tools.py:209
def _decompose_query(query):
    # 不在文件顶部 import，而是在函数内部 import
    from app.agents.langchain_agent import build_chat_model
    llm = build_chat_model()
    # ...
```

这样 Python 在调用 `_decompose_query()` 时才执行 import，此时 `langchain_agent.py` 已经完全加载，不会触发循环依赖。

### 工具 3-6：结构化领域工具

文件：`agents/domain_tools.py`

这四个工具不调用 LLM，不调用向量库，纯确定性操作：

```python
@tool
def dataset_overview(query: str = "") -> str:
    """Use this tool when the user asks about general characteristics..."""
    # 返回硬编码的共性特征 + 挑战列表

@tool
def dataset_spec_lookup(dataset_name: str) -> str:
    """Look up structured specifications for a remote sensing dataset..."""
    # 从 datasets.json 中按名称查找

@tool
def model_comparison_table(models: str) -> str:
    """Compare one or more semantic segmentation models side by side..."""
    # 解析逗号分隔的模型名，从 models.json 中逐个查找

@tool
def metric_formula_lookup(metric_name: str) -> str:
    """Look up the definition, formula, advantages, and limitations..."""
    # 从 metrics.json 中按名称查找

@tool
def metrics_calculator(metric_name: str, values: str) -> str:
    """Calculate a semantic segmentation evaluation metric from raw values..."""
    # 解析 "TP=80, FP=10, FN=20" 格式，纯数值计算
```

**查找策略**（`_find_by_name`）：先精确匹配，后包含匹配，都大小写不敏感：

```python
def _find_by_name(items, query):
    query_clean = query.strip().lower()
    # 第一轮：精确匹配
    for item in items:
        if item["name"].lower() == query_clean:
            return item
    # 第二轮：包含匹配（用户输入是名称的子串）
    for item in items:
        if query_clean in item["name"].lower():
            return item
    return None
```

**指标计算**（`_try_calculate`）：支持多种输入模式 + 零分母保护：

```python
if metric_norm == "IoU":
    if not all(k in inputs for k in ("TP", "FP", "FN")):
        return None  # 缺参数
    denom = tp + fp + fn
    if denom == 0:
        return None  # 零分母保护
    result = round(tp / denom, 4)
```

**关键设计原则**：`model_comparison_table` 的 docstring 明确说 **"Does NOT fabricate quantitative metrics like mIoU, parameters, or FLOPs"**。它只返回 JSON 中已有的定性信息（架构类型、优缺点等），不编造任何数字。

### 工具描述（docstring）的重要性

```python
@tool
def dataset_overview(query: str = "") -> str:
    """Use this tool when the user asks about general characteristics, common challenges,
    or overall patterns of remote sensing semantic segmentation datasets.
    Use dataset_spec_lookup only when the user explicitly names a specific dataset
    such as LoveDA, iSAID, DeepGlobe, Potsdam, or Vaihingen.
    ...
    """
```

注意 `"""..."""` 中的内容**全英文**且非常详细。这段文字会被发送给 LLM，是 LLM 决定"是否调用此工具"的唯一依据。如果描述含糊，LLM 会选错工具。

docstring 中的 `"Use dataset_spec_lookup only when..."` 是在**指导 LLM 区分相似工具**。这种"对比式描述"是 prompt engineering 的重要技巧。

## 2.4 System Prompt：如何指挥 LLM 选工具

文件：`agents/prompts.py`

系统提示词（`REMOTE_SENSING_AGENT_SYSTEM_PROMPT`）是 Agent 的"岗位说明书"，长达 137 行。核心结构：

```
1. 角色定位："你是一个严谨的遥感语义分割领域研究助手"
2. 工具选择原则（10 条规则）
3. 工具决策表（问题类型 → 首选工具）
4. 每个工具的详细使用指南 + 示例
5. 回答长度控制
6. 回答约束（禁止编造、必须基于工具返回）
7. 来源引用规则
```

### 工具决策表

这是 prompt 中最关键的导航机制：

```
| 用户问题类型 | 首选工具 |
|---|---|
| 数据集有什么特点/共性/难点 | dataset_overview |
| 某个具体数据集的属性（用户点名） | dataset_spec_lookup |
| 模型对比 | model_comparison_table |
| 指标定义、公式、优缺点 | metric_formula_lookup |
| 用户给出数值要求计算 | metrics_calculator |
| 开放式概念解释或文档检索 | knowledge_base_search |
| 复杂多实体、多方面比较 | plan_and_search |
```

这张表直接告诉 LLM "什么问题用什么工具"，减少 LLM 的推理负担和错误率。

### 防幻觉约束

```
1. 回答必须基于工具返回的内容，不要编造工具结果中不存在的信息。
2. 不要编造 mIoU、参数量、FLOPs 等量化指标，除非由 metrics_calculator 计算得出。
3. 如果所有工具都没有返回足够证据，请回答：
   "根据当前知识库内容，无法确定该问题的答案。"
```

第 2 条特别重要：LLM 的"本能"是在回答中补充它预训练知识中的数字（比如某个模型在某个数据集上的 mIoU），但这些数字可能过时或不适用于遥感领域。prompt 明确禁止这种行为。

### 工具去重规则

```
3. 不要对同一实体重复调用同一结构化工具（如对同一数据集连续调用多次 dataset_spec_lookup）。
4. 不要连续调用多个 dataset_spec_lookup 来回答泛化数据集问题；泛化问题用 dataset_overview。
```

这防止 LLM 陷入"循环调用"：比如为了回答"数据集有什么共性"，LLM 可能试图逐个调用 `dataset_spec_lookup("LoveDA")`、`dataset_spec_lookup("iSAID")`...，而不是直接用 `dataset_overview()`。

## 2.5 Agent 执行与结果解析

### 调用入口

```python
# langchain_agent.py:548-630
def run_langchain_agent(question: str, agent=None) -> dict:
    actual_agent = agent if agent is not None else get_remote_sensing_agent()

    # 调用 Agent（触发 ReAct 循环）
    result = actual_agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # 解析 messages 列表
    parsed = _parse_agent_result(result, invoke_elapsed=agent_invoke_elapsed)

    # Evidence Verification
    if v_mode == "sync":
        parsed["verification"] = verify_answer(...)
    elif v_mode == "deferred":
        parsed["verification"] = make_deferred_pending_result(...)
    else:
        parsed["verification"] = make_off_result()

    return parsed
```

### 结果解析：`_parse_agent_result()`

Agent 返回的 `result` 是一个 `{"messages": [...]}` dict，messages 列表包含了完整的 ReAct 对话历史：

```
messages = [
  HumanMessage("对比 U-Net 和 DeepLabV3+"),                     # 用户问题
  AIMessage(tool_calls=[{"name": "model_comparison_table", ...}]), # LLM 决定调用工具
  ToolMessage(content="{comparison: [...]}"),                    # 工具返回结果
  AIMessage("U-Net 和 DeepLabV3+ 的对比如下：..."),              # LLM 最终回答
]
```

解析函数遍历 messages，按类型提取信息：

```python
def _parse_agent_result(result, invoke_elapsed=0.0):
    messages = result.get("messages", [])
    answer = ""
    sources = []
    tool_calls = []
    agent_trace = ["agent_started"]

    # 暂存 tool_call 请求，按 id 匹配后续 ToolMessage
    pending_calls = {}

    for msg in messages:
        if msg.type == "ai":
            # AIMessage：提取 tool_calls 请求 + 最终回答
            for tc in msg.tool_calls:
                # 记录工具调用
                call_record = {"tool": tc["name"], "input": ..., "status": "success"}
                tool_calls.append(call_record)
                pending_calls[tc["id"]] = call_record
                agent_trace.append(f"tool_called:{tc['name']}")

            # 最终回答（最后一条有 content 的 AIMessage）
            if msg.content:
                answer = msg.content

        elif msg.type == "tool":
            # ToolMessage：解析工具返回的 JSON
            parsed = parse_tool_result(msg.content)
            sources.extend(parsed.get("sources", []))  # 累积来源

            # 更新对应的 tool_call 记录（通过 tool_call_id 匹配）
            call_record = pending_calls.get(msg.tool_call_id)
            if call_record:
                call_record["output_summary"] = parsed["summary"]
                call_record["elapsed"] = parsed.get("elapsed")

            agent_trace.append("tool_result_parsed")

    # 后处理
    sources = deduplicate_and_trim_sources(sources)  # 去重 + 裁剪
    tool_calls = trim_tool_calls(tool_calls)          # 截断长文本
    refused = REFUSAL_MARKER in answer                # 判断是否拒答
```

**为什么用 `pending_calls` 字典？** 因为 AIMessage 中的 tool_call 请求和后续的 ToolMessage 响应是**异步匹配**的——一条 AIMessage 可能同时请求多个工具调用，每个调用有唯一的 `id`，后续的 ToolMessage 通过 `tool_call_id` 字段指明它回应的是哪个请求。

### sources 去重裁剪

Agent 可能调用多个工具（比如先 `knowledge_base_search` 再 `plan_and_search`），多个工具返回的 sources 可能有重复。`deduplicate_and_trim_sources` 负责：

```python
def deduplicate_and_trim_sources(sources, max_sources=5, preview_max_chars=150):
    # 按 chunk_id 去重，保留最高分的
    seen = {}
    for src in sources:
        chunk_id = src.get("chunk_id", "")
        score = float(src.get("score", 0.0))
        if chunk_id in seen:
            if score > seen[chunk_id]["score"]:
                seen[chunk_id] = src
        else:
            seen[chunk_id] = src

    # 按分数降序，最多保留 5 条
    sorted_sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    # 截断 content_preview 到 150 字符
    # ...
```

## 2.6 Evidence Verification（证据校验）

文件：`agents/verification.py`

这是 Agent 路径独有的第 4 层防幻觉机制。RAG 只有 3 层（阈值过滤 → 空结果拒答 → System Prompt 约束），Agent 增加了**事后事实核查**。

### 三种模式

```python
# 通过 AGENT_VERIFICATION_MODE 环境变量控制
if v_mode == "off" or not settings.enable_agent_verification:
    parsed["verification"] = make_off_result()           # 不校验
elif v_mode == "sync":
    parsed["verification"] = verify_answer(...)           # 同步校验（阻塞请求）
else:  # "deferred"（默认）
    parsed["verification"] = make_deferred_pending_result(...)  # 延迟校验
```

| 模式 | 时机 | 体验 | 适用场景 |
|------|------|------|---------|
| off | 不校验 | 最快 | 开发/测试 |
| sync | 在 `/api/agent/query` 内同步执行 | 慢（多一次 LLM 调用） | 对准确性要求极高 |
| **deferred**（默认） | `/api/agent/query` 先返回回答，前端再调 `/api/agent/verify` | 用户先看到回答，稍后看到校验结果 | 生产环境 |

### 校验逻辑

```python
def verify_answer(question, answer, sources, tool_calls=None, ...):
    # 短路 1：拒答回答 → 直接通过（拒答不涉及事实扩展）
    if _is_refusal(answer):
        return {"verified": True, "confidence": "high", "reason": "拒答不涉及事实扩展。"}

    # 短路 2：既无 sources 也无 tool_calls → 无法验证
    if not has_sources and not has_tool_outputs:
        return {"verified": False, "confidence": "low", "reason": "无来源片段..."}

    # 正常流程：构建 prompt → 调用 LLM → 解析 JSON
    user_prompt = VERIFICATION_USER_TEMPLATE.format(
        question=question,
        answer=trimmed_answer,        # 带验证的回答
        sources_text=sources_text,    # 参考资料来源
        tool_outputs_text=...,        # 工具输出摘要
    )
    raw_response = llm_client.chat(prompt=user_prompt, system=VERIFICATION_SYSTEM_PROMPT)
    parsed = _parse_verification_json(raw_response)
    # 返回 {verified: true/false, confidence: "high"/"medium"/"low", ungrounded_claims: [...], reason: "..."}
```

**Verification Prompt 的核心规则**：
```
1. 回答中的具体数值（如 mIoU、参数量、分辨率等）必须有参考资料直接支撑。
2. 回答中有但参考资料中没有的论断，属于"未证实论断"。
3. 如果回答完全基于参考资料，verified=true。
4. 如果存在未被支撑的论断，verified=false，并列出这些论断。
```

### 轻量化级别

```python
# AGENT_VERIFICATION_LEVEL 控制
if level == "lightweight":
    # answer 截断到 800 字，sources 最多 5 条，tool_calls 最多 6 条
    # 减少 verification LLM 的输入 token
else:  # "full"
    # answer 截断到 1500 字，sources 最多 8 条，tool_calls 最多 8 条
```

## 2.7 Agent Trace（执行轨迹）

### 两种轨迹格式

项目提供两种互补的轨迹记录：

**1. agent_trace（简短字符串列表）**：
```python
["agent_started", "tool_called:model_comparison_table", "tool_result_parsed", "agent_finished"]
```

**2. trace_events（结构化事件列表）**：
```python
[
    {"step": 1, "event": "agent_started", "timestamp": 0.0, "detail": None},
    {"step": 2, "event": "tool_called", "timestamp": 0.0, "detail": "model_comparison_table"},
    {"step": 3, "event": "tool_result_parsed", "timestamp": 0.123, "detail": "model_comparison_table"},
    {"step": 4, "event": "agent_finished", "timestamp": 3.456, "detail": None},
]
```

`trace_events` 比 `agent_trace` 多了 **step 序号**、**时间戳**和**附加详情**，用于前端可视化展示和性能分析。

### 时间戳估算

```python
def _parse_agent_result(result, invoke_elapsed=0.0):
    cumulative_tool_time = 0.0

    for msg in messages:
        if msg.type == "tool":
            elapsed = parsed.get("elapsed")
            if isinstance(elapsed, (int, float)):
                cumulative_tool_time += float(elapsed)
            _add_trace_event("tool_result_parsed", timestamp=cumulative_tool_time)
```

注意：trace_events 的 timestamp **不是真实 wall-clock 时间**，而是基于工具 `elapsed` 累积的估算值。因为 LangChain 的 messages 中没有携带精确的每步时间戳。

### include_trace 控制

```python
# schemas.py:77-81
class AgentQueryRequest(BaseModel):
    question: str
    include_trace: bool = Field(
        default=True,
        description="是否返回 agent_trace / trace_events / tool_calls 等调试信息。"
    )
```

```python
# agent_service.py:92-95
if not include_trace:
    result["tool_calls"] = []
    result["agent_trace"] = []
    result["trace_events"] = []
```

生产环境可以设 `include_trace=false` 来减少响应体积（trace 和 tool_calls 可能很大），但 answer / sources / refused / timing / verification 不受影响。

## 2.8 完整请求生命周期

以"对比 U-Net 和 DeepLabV3+ 的优缺点"为例，完整走一遍：

```
1. 用户发送 POST /api/agent/query
   body: {"question": "对比 U-Net 和 DeepLabV3+ 的优缺点", "include_trace": true}

2. FastAPI 路由层 (api/agent.py:27)
   ├─ 校验 question 非空
   └─ 调用 get_agent_service().query(question, include_trace=true)

3. AgentService (agent_service.py:34)
   ├─ 校验输入
   └─ 调用 run_langchain_agent(question)

4. run_langchain_agent (langchain_agent.py:548)
   ├─ 获取 Agent 单例 get_remote_sensing_agent()
   │
   ├─ agent.invoke({"messages": [{"role": "user", "content": question}]})
   │   │
   │   │  ===== LangChain ReAct 循环（内部）=====
   │   │
   │   ├─ [第 1 轮] LLM 思考 → 决定调用 model_comparison_table
   │   │  AIMessage(tool_calls=[{"name": "model_comparison_table",
   │   │                            "args": {"models": "U-Net, DeepLabV3+"}}])
   │   │
   │   ├─ tools_node 执行 model_comparison_table("U-Net, DeepLabV3+")
   │   │  ├─ 从 models.json 查找 U-Net → 找到
   │   │  ├─ 从 models.json 查找 DeepLabV3+ → 找到
   │   │  └─ 返回 JSON: {success: true, comparison: [{name: "U-Net", ...}, ...]}
   │   │
   │   ├─ [第 2 轮] LLM 看到工具结果 → 决定不再调用工具，生成最终回答
   │   │  AIMessage("U-Net 和 DeepLabV3+ 的对比如下：\n\n**U-Net**：\n- 架构：编码器-解码器...\n\n
   │   │             **DeepLabV3+**：\n- 架构：ASPP + 编码器-解码器...\n\n
   │   │             （使用工具：model_comparison_table）")
   │   │
   │   └─ 循环结束（LLM 输出了纯文本，没有 tool_calls）
   │
   ├─ _parse_agent_result(result)
   │   ├─ 遍历 messages
   │   │   ├─ AIMessage: 提取 tool_calls + 最终 answer
   │   │   └─ ToolMessage: parse_tool_result → sources=[]（结构化工具无 sources）
   │   ├─ 去重裁剪 sources → []
   │   ├─ 裁剪 tool_calls → [{tool: "model_comparison_table", ...}]
   │   ├─ 构建 agent_trace → ["agent_started", "tool_called:model_comparison_table",
   │   │                       "tool_result_parsed", "agent_finished"]
   │   ├─ 构建 trace_events → [{step: 1, ...}, {step: 2, ...}, ...]
   │   └─ 判断拒答 → refused=False（answer 中不含拒答关键词）
   │
   ├─ Evidence Verification (deferred 模式)
   │   └─ 返回 pending=true（前端稍后调用 /api/agent/verify）
   │
   ├─ 构建 timing: {total_elapsed, agent_invoke_elapsed, tool_search_elapsed_total}
   │
   └─ 返回 dict

5. AgentService 返回 → api/agent.py 构建 AgentQueryResponse → 返回 HTTP 200

6. （deferred 模式）前端调用 POST /api/agent/verify
   body: {"question": "...", "answer": "...", "sources": [], "tool_calls": [...]}
   └─ verify_answer() 同步执行 LLM 校验 → 返回 {verified: true, confidence: "high", ...}
```

### 防幻觉的 4 层保障（Agent 路径）

```
第 1 层 - VectorStore.search 阈值过滤
  ↓ （score < 0.3 的 chunk 不会进入结果）
第 2 层 - knowledge_base_search 工具返回 success=false 时，LLM 看到没有结果
  ↓
第 3 层 - System Prompt 指示 LLM："如果所有工具都没有返回足够证据，回答拒答"
  ↓
第 4 层 - verify_answer() 事后核查：LLM 回答中的每个论断是否有 sources/工具输出支撑
```

---

# 第三部分：Agent 双层缓存

## 3.1 为什么 Agent 需要缓存

Agent 的 ReAct 循环通常需要 2 轮 LLM 调用：

```
Round 1: System Prompt + User Question → LLM 决策调用哪些工具（~15-20s）
         ↓ 工具执行（~0.5s）
Round 2: System Prompt + User Question + AIMessage(tool_calls) + ToolMessage(results) → LLM 生成最终回答（~25-40s）
```

同一个问题第二次调用时，上述 ~40-60s 的过程应该完全跳过。但直接缓存存在两个难点：

1. **Round 2 的 cache key 不可控**：LangGraph 的 ToolNode 执行并行工具调用时返回 ToolMessage 的顺序不确定，且每条 ToolMessage 携带随机的 `tool_call_id`，导致序列化后的 cache key 每次都不同。
2. **完整的 Agent 响应才是用户需要的结果**：缓存单次 LLM 调用的粒度太细，不如直接缓存整个响应。

本项目采用**双层缓存**架构解决此问题：

```
请求 → L1 Response Cache（命中=零开销直接返回）
         ↓ miss
      L2 LLM Cache（Round 1 命中=跳过 LLM 推理）
         ↓ miss
      Agent 执行（LLM + 工具调用）
         ↓ 完成
      写入 L1 + L2
```

## 3.2 L2 LLM Cache：单次调用级缓存

文件：`agents/langchain_agent.py`

L2 缓存基于 LangChain 的 `InMemoryCache`，通过 `ChatOpenAI.cache` 属性注入。当 `enable_cache=True` 时，LangChain 的 `_generate_with_cache` 方法会自动查找/写入缓存。

```python
class _TrackingInMemoryCache(InMemoryCache):
    """带命中统计和 tool_call_id 归一化的 InMemoryCache 子类。"""

    def lookup(self, prompt, llm_string):
        normalized = _normalize_cache_key(prompt)  # 去除 tool_call_id 干扰
        result = super().lookup(normalized, llm_string)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
```

**`_normalize_cache_key` 的作用**：Agent Round 2 的消息序列中包含 AIMessage 的 `tool_calls[].id`（如 `"call_abc123"`）和 ToolMessage 的 `tool_call_id`，这些 ID 每次调用都不同。`_normalize_cache_key` 用正则将这些 ID 替换为空字符串，使相同语义的消息产生相同的 cache key。

```python
_TOOL_CALL_ID_RE = re.compile(r'("id":\s*)"[^"]*"')
_TOOL_CALL_REF_RE = re.compile(r'("tool_call_id":\s*)"[^"]*"')

def _normalize_cache_key(key: str) -> str:
    key = _TOOL_CALL_ID_RE.sub(r'\1""', key)
    key = _TOOL_CALL_REF_RE.sub(r'\1""', key)
    return key
```

**L2 的局限**：即使归一化了 `tool_call_id`，Round 2 的 cache key 仍然会因为 LangGraph ToolNode 返回 ToolMessage 的**顺序不确定**而无法命中。因此 L2 实际上只对 Round 1（System Prompt + User Question，无 tool_calls）有效。

**`get_agent_llm()` 共享模式**：`planning_tools.py` 的 `_decompose_query()` 也需要调用 LLM。为避免创建独立的 ChatOpenAI 实例（导致缓存无法共享），通过 `get_agent_llm()` 复用 Agent 单例的 ChatOpenAI：

```python
# langchain_agent.py
_agent_llm: ChatOpenAI | None = None

def get_agent_llm() -> ChatOpenAI:
    global _agent_llm
    if _agent_llm is None:
        get_remote_sensing_agent()  # 触发单例构建，顺带初始化 _agent_llm
    return _agent_llm
```

```python
# planning_tools.py
def _decompose_query(query):
    from app.agents.langchain_agent import get_agent_llm  # lazy import 避免循环依赖
    llm = get_agent_llm()  # 复用同一个 ChatOpenAI（含缓存）
```

## 3.3 L1 Response Cache：完整响应级缓存

文件：`agents/response_cache.py`

L1 缓存在 `AgentService.query()` 层拦截，命中时直接返回缓存的完整 Agent 响应 dict，零 LLM/工具调用。

```python
# agent_service.py
class RemoteSensingAgentService:
    def query(self, question, include_trace=True, use_rerank=None, enable_cache=None):
        # ---------- 响应级缓存检查（L1） ----------
        settings = get_settings()
        if settings.enable_agent_response_cache:
            cache = get_agent_response_cache()
            cache_key = build_agent_response_cache_key(question, use_rerank, include_trace)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info("Agent 响应缓存命中，直接返回缓存结果")
                cached_result["timing"]["response_cache_hit"] = True
                return cached_result

        # ---------- 缓存未命中，执行 Agent ----------
        result = run_langchain_agent(question)

        # ---------- 写入响应级缓存（L1） ----------
        if settings.enable_agent_response_cache:
            result["timing"]["response_cache_hit"] = False
            cache.put(cache_key, result)

        return result
```

**`AgentResponseCache` 数据结构**：基于 `OrderedDict` 实现的 TTL + max_size 缓存：

```python
class AgentResponseCache:
    def __init__(self, ttl_seconds=600, max_size=100):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None  # 未命中

        cached_at, value = entry
        if self._ttl > 0 and (time.time() - cached_at) > self._ttl:
            del self._store[key]  # TTL 过期，删除并返回 None
            return None

        self._store.move_to_end(key)  # LRU 更新
        return value

    def put(self, key, value):
        # 异常结果不缓存
        if value.get("errors") and any("异常" in str(e) for e in value["errors"]):
            return

        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # 淘汰最旧条目

        self._store[key] = (time.time(), value)
```

**设计要点**：
- TTL 过期自动失效（默认 600 秒）
- max_size 满时淘汰最旧条目（LRU，默认 100 条）
- 异常结果（errors 含"异常"）不缓存，避免缓存错误响应
- 模块级单例（`get_agent_response_cache()`），进程生命周期内共享

## 3.4 缓存 key 设计

`build_agent_response_cache_key()` 生成 sha256 前 32 字符的 key，包含以下因素：

```python
key_parts = [
    f"q={normalized_question}",       # 归一化问题（strip + lowercase + collapse）
    f"rerank={actual_rerank}",        # use_rerank 配置
    f"top_k={settings.top_k}",        # 检索参数
    f"thresh={settings.similarity_threshold}",
    f"cand_k={settings.rerank_candidate_k}",
    f"model={settings.llm_model}",    # 模型配置
    f"max_tok={settings.agent_max_tokens}",
    f"v_mode={settings.agent_verification_mode}",   # 校验配置
    f"v_level={settings.agent_verification_level}",
    f"corpus={corpus_version}",       # 语料库版本哈希（30s 缓存）
    f"domain={domain_data_hash}",     # 领域 JSON 哈希
    f"trace={include_trace}",         # 是否包含调试信息
]
raw_key = "|".join(key_parts)
return hashlib.sha256(raw_key.encode()).hexdigest()[:32]
```

**为什么要这么多因素？** 因为缓存的是**完整响应**，任何影响响应内容的因素都必须进入 key：
- 切换 LLM 模型 → 回答不同 → key 必须不同
- 切换 `use_rerank` → 检索结果不同 → sources 不同 → key 必须不同
- 修改 `include_trace` → 返回的 tool_calls / trace_events 不同 → key 必须不同

**`corpus_version`（语料库版本哈希）**：基于 Chroma 中所有文档的 `doc_id + chunk_count` 列表计算 md5，带 30 秒短期缓存避免每次请求都查 Chroma。文档变化时哈希值变化 → key 自然不同 → 不会命中旧缓存。

**`domain_data_hash`（领域数据哈希）**：基于 `domain_data/*.json` 的文件内容哈希。修改 datasets.json/models.json/metrics.json 后 key 自然变化。

## 3.5 缓存失效与文档更新

文档入库和删除时，`documents.py` 同时清除三层缓存：

```python
# api/documents.py（ingest 和 delete 两个端点都调用）
clear_agent_search_cache()          # 清除 tools.py 的 @lru_cache(128)
invalidate_corpus_version()         # 刷新语料库版本缓存（使下次 key 计算使用新哈希）
clear_agent_response_cache()        # 清除 L1 Response Cache 全部条目
```

**为什么需要三种清除？**
1. `clear_agent_search_cache()`：`knowledge_base_search` 工具的 LRU 缓存。不清除的话，Agent 工具调用仍会返回旧的检索结果。
2. `invalidate_corpus_version()`：`_compute_corpus_version()` 有 30 秒缓存。不清除的话，下一次 `build_agent_response_cache_key()` 仍会使用旧的语料库哈希（虽然 L1 已清空，但 key 不变，理论上不影响——这是双重保险）。
3. `clear_agent_response_cache()`：L1 响应缓存。不清除的话，相同问题会返回旧的完整 Agent 响应。

> **设计原则**：宁可过度清除（影响性能极小），也不允许返回过期回答。

---

## 附录：关键文件索引

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| `agents/langchain_agent.py` | ~830 | Agent 构建、执行、结果解析、L2 LLM Cache |
| `agents/response_cache.py` | ~210 | L1 Response Cache（TTL + max_size + key 构建） |
| `agents/tools.py` | ~324 | knowledge_base_search 工具 + LRU 缓存 |
| `agents/planning_tools.py` | ~480 | plan_and_search 工具 + 查询分解 |
| `agents/domain_tools.py` | ~640 | 5 个结构化领域工具 |
| `agents/verification.py` | ~530 | Evidence Verification 证据校验 |
| `agents/prompts.py` | ~154 | Agent 系统提示词 + Verification 模板 |
| `agents/types.py` | ~136 | Pydantic 数据结构定义 |
| `agents/agent_service.py` | ~250 | Agent 服务层 + 缓存调度 + 异常兜底 |
| `api/agent.py` | ~84 | FastAPI 路由（/query + /verify） |
| `experiments/rag_rerank_ablation/reranker.py` | ~245 | Rerank API 客户端 |
| `experiments/rag_rerank_ablation/run_rerank_ablation.py` | ~600+ | 三阶段消融实验主脚本 |
