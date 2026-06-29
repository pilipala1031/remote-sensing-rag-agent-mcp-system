# RAG 参数消融实验
## 1. 实验目的
本实验用于确定 RAG 系统中 `chunk_size`、`chunk_overlap`、`similarity_threshold` 三个参数的合理取值。通过分层评估（Retrieval → Refusal → Answer）逐步缩小参数空间，避免暴力全组合实验的高成本。
## 2. 为什么需要消融
- **chunk 太小**：语义片段被截断，同一概念分散在多个 chunk 中，召回率下降。
- **chunk 太大**：单个 chunk 包含过多无关信息，噪声稀释信号，精确度下降。
- **overlap 太小**：跨段信息丢失，边界处的上下文断裂。
- **overlap 太大**：冗余 chunk 增多，存储和检索成本上升。
- **threshold 太低**：低相关证据混入上下文，可能误导 LLM。
- **threshold 太高**：正常问题被误拒答，用户体验受损。
## 3. 实验设置
- `c400_o80`: chunk_size=400, overlap=80
- `c600_o100`: chunk_size=600, overlap=100
- `c800_o120`: chunk_size=800, overlap=120
- `c1000_o150`: chunk_size=1000, overlap=150
- `c1200_o180`: chunk_size=1200, overlap=180

Threshold 候选值：0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
- 当前默认参数：{'chunk_size': 800, 'chunk_overlap': 120, 'similarity_threshold': 0.3}
## 4. 指标设计
本实验分为三层评估：
| 层级 | 用途 | 主要指标 |
|------|------|----------|
| Retrieval-level | 选择 chunk_size / chunk_overlap | source_hit_rate, source_recall_at_k, mrr, avg_top_score |
| Refusal-level | 选择 similarity_threshold | in_scope_recall, out_of_scope_refusal_accuracy, false_refusal_rate, false_accept_rate |
| Answer-level | 最终候选参数验证 | keyword_coverage, source_hit_rate, refusal_accuracy, min_length_satisfied |
## 5. Chunk 参数实验结果
| config | chunk_size | chunk_overlap | source_hit_rate | source_recall_at_k | mrr | avg_top_score | avg_latency | retrieval_score |
|--------|------------|---------------|-----------------|---------------------|-----|---------------|-------------|-----------------|
| c1000_o150 | 1000 | 150 | 1.0000 | 0.8704 | 0.8583 | 0.6552 | 0.1130s | 0.8603 |
| c1200_o180 | 1200 | 180 | 1.0000 | 0.8981 | 0.8148 | 0.6331 | 0.1858s | 0.8101 |
| c400_o80 | 400 | 80 | 1.0000 | 0.8704 | 0.8704 | 0.6606 | 0.1125s | 0.8630 |
| c600_o100 | 600 | 100 | 1.0000 | 0.8148 | 0.8444 | 0.6640 | 0.1106s | 0.8468 |
| c800_o120 | 800 | 120 | 1.0000 | 0.8241 | 0.8444 | 0.6587 | 0.1106s | 0.8486 |

**推荐**：`c400_o80` (chunk_size=400, overlap=80)
## 6. Threshold 敏感性分析结果
| threshold | in_scope_recall | out_of_scope_refusal_accuracy | false_refusal_rate | false_accept_rate | refusal_score |
|-----------|-----------------|-------------------------------|--------------------|-------------------|---------------|
| 0.1 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 0.2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 0.3 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 0.4 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 0.5 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 0.6 | 0.9444 | 0.6667 | 0.0556 | 0.3333 | 0.5778 |
| 0.7 | 0.1667 | 1.0000 | 0.8333 | 0.0000 | 0.4667 |

**推荐**：similarity_threshold = 0.6

threshold 是「正常问题召回」和「超纲问题拒答」之间的 trade-off：threshold 越高，正常问题的召回率（in_scope_recall）可能下降（被误拒），但超纲问题的拒答准确率（out_of_scope_refusal_accuracy）会上升。refusal_score 综合权衡两者，选择使整体表现最优的阈值。
## 7. 最终候选参数 answer-level 验证
| 参数组合 | keyword_coverage_avg | source_hit_rate | refusal_accuracy | min_length_satisfied_rate | avg_latency | answer_score |
|----------|---------------------|-----------------|------------------|--------------------------|-------------|--------------|
| 推荐 (400/80/0.6) | 0.7222 | 0.8095 | 0.9048 | 0.8095 | 15.1217s | 0.7801 |
| 默认 (800/120/0.3) | 0.8548 | 0.8571 | 0.8571 | 1.0000 | 16.9291s | 0.8702 |
## 8. 推荐参数
- recommended_chunk_size: 400
- recommended_chunk_overlap: 80
- recommended_similarity_threshold: 0.6

**Answer-level 对比**: 推荐参数 answer_score=0.7801 vs 默认参数 answer_score=0.8702

推荐参数 400/80/0.6 在 retrieval/refusal 层级更优，但 answer-level 验证显示默认参数 (0.8702) 优于推荐参数 (0.7801)。
**建议保持当前默认值 800/120/0.3 不变**，因为 end-to-end 质量更高。
## 9. 当前限制
- 评估集规模较小（21 题），统计结果可能不稳定。
- 标签中的 `expected_source_files` 和 `required_keywords` 不能完全等价于人工完整答案评估。
- retrieval-only 指标只能评估证据检索质量，不能反映 LLM 理解和生成能力。
- answer-level 会受到 LLM 输出稳定性和 API 波动影响。
- 本实验未引入 rerank、query rewrite、hybrid search 等增强手段。
- Chroma 返回的是 cosine distance，本实验中所有 score 均已通过 `similarity = 1.0 - distance` 转换。
## 10. 面试表述
> RAG 系统的 chunk_size、chunk_overlap 和 similarity_threshold 并非拍脑袋设定，而是通过小规模标注集消融实验确定的。实验分为三层：先用 Retrieval-level 指标（source_hit_rate、MRR、recall@k）在 5 组 chunk 参数中选出最优切分策略，再用 Refusal-level 指标（in_scope_recall、out_of_scope_refusal_accuracy）扫描 7 个 threshold 值确定最佳拒答边界，最后用 Answer-level 指标（keyword_coverage、refusal_accuracy）做最终验证。每层指标加权合成复合 score，逐步缩小搜索空间，避免了 5×7 全组合的暴力成本。
