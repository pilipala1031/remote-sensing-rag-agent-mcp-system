# RAG Rerank 消融实验
## 1. 实验目的
本实验验证 SiliconFlow `BAAI/bge-reranker-v2-m3` rerank 模型是否能提升 RAG 系统的检索精度和最终回答质量。
通过三阶段分层评估（Retrieval → Out-of-scope safety → Answer），量化 rerank 相比纯向量检索的增益与代价。
## 2. 为什么需要 rerank
- **向量检索的局限**：bi-encoder（如 bge-m3）将 query 和 document 独立编码，通过余弦相似度排序，速度快但精度有限，容易在 top-K 中混入语义表面相似但实际无关的 chunk。
- **Rerank 的优势**：cross-encoder（如 bge-reranker-v2-m3）将 query 和 document 拼接后联合编码，能捕捉更细粒度的语义交互，精度更高。
- **代价**：rerank 是逐对计算，延迟高于向量检索。本实验测量其精度增益是否值得延迟代价。
## 3. 实验设置
- 固定参数：chunk_size=800, chunk_overlap=120, similarity_threshold=0.3, top_k=5
- 模型：BAAI/bge-reranker-v2-m3（SiliconFlow API）
- 评估集：21 题（18 领域内 + 3 领域外）

配置列表：
| 配置 | use_rerank | candidate_k | final_top_k | 说明 |
|------|-----------|-------------|-------------|------|
| baseline | False | 5 | 5 | 原始向量检索 top_k=5，不使用 rerank |
| rerank_k10 | True | 10 | 5 | 向量检索 candidate_k=10，rerank 后保留 top_k=5 |
| rerank_k20 | True | 20 | 5 | 向量检索 candidate_k=20，rerank 后保留 top_k=5 |

## 4. 指标设计
本实验分为三层评估：
| 层级 | 用途 | 主要指标 | 综合分数 |
|------|------|----------|----------|
| Retrieval-level | 对比检索精度 | source_hit_rate, source_recall_at_k, mrr, avg_top_score | retrieval_score |
| Out-of-scope safety | 检测拒答行为变化 | in_scope_recall, out_refusal_acc, false_refusal_rate, false_accept_rate | refusal_score |
| Answer-level | 最终回答质量 | keyword_coverage, source_hit_rate, refusal_accuracy, min_length_satisfied | answer_score |

**Scoring 公式**：
```
retrieval_score = 0.45*source_hit_rate + 0.25*source_recall_at_k + 0.15*mrr + 0.10*avg_top_score - 0.05*latency_norm
refusal_score   = 0.35*in_scope_recall + 0.45*out_refusal_acc - 0.05*false_refusal_rate - 0.15*false_accept_rate
answer_score    = 0.50*keyword_coverage + 0.25*source_hit_rate + 0.15*refusal_accuracy + 0.10*min_length_satisfied
```
## 5. Retrieval-level 结果
| 配置 | source_hit_rate | source_recall_at_k | mrr | avg_top_score | avg_latency | retrieval_score |
|------|-----------------|---------------------|-----|---------------|-------------|-----------------|
| baseline | 1.0000 | 0.8241 | 0.8444 | 0.6588 | 0.1159s | 0.8486 |
| rerank_k10 | 1.0000 | 0.9815 | 0.9444 | 0.6363 | 1.2948s | 0.8569 |
| rerank_k20 | 1.0000 | 0.9537 | 0.9444 | 0.6363 | 1.4631s | 0.8437 |

**最佳 rerank 配置**：`rerank_k10`
- retrieval_score 差异：+0.0083 (rerank=0.8569 vs baseline=0.8486)
- source_hit_rate 差异：+0.0000
- mrr 差异：+0.1000
- 延迟倍数：11.17x (rerank=1.2948s vs baseline=0.1159s)

## 6. Out-of-scope 安全性分析
| 配置 | in_scope_recall | out_refusal_acc | false_refusal_rate | false_accept_rate | refusal_score |
|------|-----------------|-----------------|--------------------|-------------------|---------------|
| baseline | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| rerank_k10 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| rerank_k20 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |

由于 similarity_threshold 过滤发生在 rerank 之前，rerank 理论上不改变拒答行为。但如果 candidate_k > top_k，rerank 配置可能从更大的候选池中找到通过阈值的结果，从而降低 false_refusal_rate（代价是可能增加 false_accept_rate）。

## 7. Answer-level 最终验证
对比配置：baseline vs rerank_k10

| 配置 | keyword_coverage_avg | source_hit_rate | refusal_accuracy | min_length_satisfied_rate | avg_latency | answer_score |
|------|---------------------|-----------------|------------------|--------------------------|-------------|--------------|
| baseline | 0.7762 | 0.8571 | 0.8571 | 0.9048 | 21.1894s | 0.8214 |
| rerank_k10 | 0.8571 | 0.8571 | 0.8571 | 0.9524 | 20.4936s | 0.8666 |

**answer_score 差异**：+0.0452 (rerank=0.8666 vs baseline=0.8214)
**keyword_coverage 差异**：+0.0809

## 8. 结论与建议
- **Retrieval-level**：rerank (`rerank_k10`) 相比 baseline retrieval_score 仅提升 +0.0083，增益不显著。
- **Answer-level**：rerank answer_score 提升 +0.0452，端到端回答质量有改善。

> **注意**：以上结论基于 21 题小规模评估集，统计显著性有限。建议在生产环境中用更大评估集验证后再决定是否启用 rerank。

## 9. 当前限制
- 评估集仅 21 题，统计结果可能不稳定，rerank 增益可能被噪声掩盖。
- rerank API 调用增加额外延迟，对实时性要求高的场景需权衡。
- 本实验仅测试 SiliconFlow bge-reranker-v2-m3 单一模型，未对比其他 rerank 模型。
- similarity_threshold 过滤发生在 rerank 之前，可能过滤掉 rerank 本能纠正的低向量相似度高语义相关 chunk。
- Stage 3 answer-level 受 LLM 输出稳定性和 API 波动影响。
- 未测试 rerank 与 query rewrite / hybrid search 等手段的组合效果。
- Chroma 返回的是 cosine distance，本实验中所有 score 均已通过 `similarity = 1.0 - distance` 转换。

## 10. 面试表述
> 在 RAG 参数消融实验确定了 chunk_size、similarity_threshold 等基础参数后，我进一步设计了 rerank 消融实验，验证 SiliconFlow bge-reranker-v2-m3 cross-encoder 是否能提升检索精度。实验固定基础参数，对比 baseline（纯向量检索 top_k=5）、rerank_k10（candidate_k=10 → rerank → top_k=5）和 rerank_k20（candidate_k=20 → rerank → top_k=5）三组配置。通过 Retrieval-level（source_hit_rate、MRR）、Out-of-scope safety（false_refusal_rate）和 Answer-level（keyword_coverage）三层评估，量化 rerank 的精度增益与延迟代价。实验设计了优雅降级机制：rerank API 调用失败时自动回退到原始向量顺序，保证可用性。
