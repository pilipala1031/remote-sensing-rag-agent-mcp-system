# LLM 辅助评估标签生成工具

利用项目主 LLM（.env 配置，默认 GLM-5.1）自动生成评估标签，减少人工标注工作量。

---

## 快速开始

### 前置条件

1. **LLM 配置**：在 `.env` 中配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`
2. **知识库文档**：确保 `examples/sample_docs/` 下有 `.md` 文档
3. **评估题集**：确保 `eval/eval_questions.json` 存在

### 三秒启动

```bash
# 从项目根目录运行

# 1. 快速生成初版（自动接受所有 LLM 标签）
python eval/generate_eval_labels.py --auto-accept

# 2. 生成 + 人工校验
python eval/generate_eval_labels.py

# 3. 中断后继续
python eval/generate_eval_labels.py --resume
```

---

## 生成流程

```
eval/eval_questions.json          LLM (.env 配置, 默认 GLM-5.1)    人工校验 CLI
       │                               │                          │
       ├─ 加载 21 道问题 ──────────────→│                          │
       │                               ├─ 生成结构化标签            │
       │                               │  (should_refuse)          │
       │                               │  (required_keywords)      │
       │                               │  (relevant_docs)          │
       │                               │  (question_type)          │
       │                               │  (expected_tool)          │
       │                               │  (notes)                  │
       │                               ├─ JSON 解析 (含重试 ×3)     │
       │                               │                          │
       │                               ├─────────────────────────→│ [y] 接受
       │                               │                          │ [e] 编辑
       │                               │                          │ [s] 跳过
       │                               │                          │ [q] 保存退出
       │                               │                          │
       ├───────────────────────────────┴──────────────────────────┘
       │
       └─→ eval/eval_questions_with_labels.json
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--auto-accept` | 自动接受所有 LLM 生成的标签，跳过人工校验（适合快速生成初版） |
| `--resume` | 从上次中断处继续，跳过已完成且 `human_verified: true` 的问题 |
| `--kb-dir <path>` | 指定知识库文档目录（默认：`examples/sample_docs`） |

组合使用：

```bash
# 快速生成初版后中断，继续生成剩余的
python eval/generate_eval_labels.py --auto-accept
python eval/generate_eval_labels.py --resume

# 初版全部自动生成后，重新人工校验
# （删除 eval_questions_with_labels.json 后重新运行）
```

---

## 人工校验指南

### 操作界面

每个问题会展示 LLM 生成的标签，然后提供四个选项：

```
============================================================
  ID: q001
  类别: dataset
  问题: LoveDA 数据集包含哪些地物类别？它的空间分辨率是多少？
------------------------------------------------------------
  生成的评估标签：
    should_refuse:     False
    question_type:     structured
    expected_tool:     dataset_spec_lookup
    required_keywords: ['建筑', '道路', '水体', '0.3']
    relevant_docs:     ['01_datasets.md']
    min_answer_length: 50
    notes:             结构化查询，应使用 dataset_spec_lookup
============================================================

  [y]接受  [e]编辑  [s]跳过  [q]保存退出:
```

### 选项说明

| 按键 | 行为 |
|---|---|
| `y` 或回车 | 接受当前标签，标记 `human_verified: true` |
| `e` | 进入编辑模式，逐字段修改 |
| `s` | 跳过（保留标签但标记为未校验） |
| `q` | 保存当前进度并退出 |

### 编辑模式

输入 `e` 后进入逐字段编辑，直接回车保留原值：

```
  进入编辑模式（直接回车保留原值）：

  should_refuse [False]:
  required_keywords [建筑, 道路, 水体, 0.3]: 建筑, 道路, 水体, 林地, 0.3
  relevant_docs [01_datasets.md]:
  question_type [structured]:
  expected_tool [dataset_spec_lookup]:
  min_answer_length [50]: 80
  notes [结构化查询...]:
  ✓ 编辑完成
```

- **required_keywords** / **relevant_docs**：逗号分隔
- **should_refuse**：true/false
- **min_answer_length**：整数

---

## 标注字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `should_refuse` | bool | 问题是否超出领域范围（应拒答） |
| `required_keywords` | list[str] | 正确答案必须包含的 3-5 个关键词 |
| `relevant_docs` | list[str] | 应被检索到的知识库文档 |
| `question_type` | str | 问题类型：basic / structured / calculation / comparison / out_of_scope |
| `min_answer_length` | int | 合理答案的最小字符数 |
| `expected_tool` | str | Agent 最应该调用的工具名 |
| `notes` | str | 评估要点说明 |

---

## 标注质量检查清单

完成生成后，建议逐项检查：

- [ ] **should_refuse 一致性**：out_of_scope 问题的 `should_refuse=true`，其余为 `false`
- [ ] **required_keywords 覆盖性**：关键词是否覆盖了答案的核心信息点
- [ ] **relevant_docs 准确性**：标注的文档是否确实是答案来源
- [ ] **expected_tool 合理性**：
  - 结构化查询 → `dataset_spec_lookup` / `metric_formula_lookup`
  - 计算问题 → `metrics_calculator`
  - 对比问题 → `model_comparison_table` / `plan_and_search`
  - 开放式问题 → `knowledge_base_search`
  - 泛化数据集问题 → `dataset_overview`
- [ ] **question_type 分类正确**：特别是 calculation 和 comparison 不要误分为 basic
- [ ] **min_answer_length 合理**：简单概念 50+，对比分析 200+，超出范围 0
- [ ] **notes 有指导性**：是否说明了该问题的评估重点

---

## 如何在后续实验中使用标签

### 1. 导入标签

```python
import json

with open("eval/eval_questions_with_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for q in data["questions"]:
    if not q["human_verified"]:
        continue  # 跳过未校验的

    labels = q["eval_labels"]
    question = q["question"]

    # 使用标签做评估
    # - labels["required_keywords"]: 关键词命中率
    # - labels["relevant_docs"]: 来源命中率
    # - labels["expected_tool"]: 工具命中率
    # - labels["should_refuse"]: 拒答准确率
    # - labels["min_answer_length"]: 回答长度检查
```

### 2. 与评估脚本集成

`eval_labels` 中的字段可与 `eval/metrics.py` 中的指标函数配合使用：

| eval_labels 字段 | 对应的 metrics.py 函数 |
|---|---|
| `required_keywords` | `keyword_hit_rate(answer, required_keywords)` |
| `relevant_docs` | `source_hit_rate(sources, relevant_docs)` |
| `should_refuse` | `refusal_correct(refused, should_refuse)` |
| `expected_tool` | `tool_hit_rate(tool_calls, [expected_tool])` |

### 3. LLM 生成 vs 人工标注对比

输出文件同时保留 `eval_labels`（LLM 生成）和 `original_expected_*`（题集原始标注），
可用于分析 LLM 生成标签与人工标注的差异：

```python
for q in data["questions"]:
    llm_kw = set(q["eval_labels"]["required_keywords"])
    orig_kw = set(q.get("original_expected_keywords", []))
    overlap = llm_kw & orig_kw
    print(f"{q['id']}: 重叠 {len(overlap)}/{len(orig_kw)}")
```

---

## 输出文件格式

```json
{
  "generator": "generate_eval_labels.py",
  "llm_model": "GLM-5.1",
  "generated_at": "2025-06-25 12:00:00",
  "questions": [
    {
      "id": "q001",
      "category": "dataset",
      "question": "LoveDA 数据集包含哪些地物类别？",
      "eval_labels": {
        "should_refuse": false,
        "required_keywords": ["建筑", "道路", "水体", "0.3"],
        "relevant_docs": ["01_datasets.md"],
        "question_type": "structured",
        "min_answer_length": 50,
        "expected_tool": "dataset_spec_lookup",
        "notes": "结构化查询，Agent 应优先使用 dataset_spec_lookup"
      },
      "human_verified": true,
      "original_expected_keywords": ["建筑", "道路", "水体", "0.3"],
      "original_expected_tools": ["dataset_spec_lookup"]
    }
  ]
}
```

---

## 错误处理

| 场景 | 行为 |
|---|---|
| LLM 返回非 JSON | 自动重试 3 次，仍失败则使用题集原始标签回退 |
| LLM API 调用失败 | 自动重试 3 次（间隔 2 秒），仍失败则回退 |
| 知识库目录不存在 | 报错退出，提示正确路径 |
| eval_questions.json 格式错误 | 报错退出，说明期望格式 |
| LLM_API_KEY 未设置 | 报错退出，提示 .env 配置方法 |
| 用户 Ctrl+C | 自动保存当前进度并退出 |

---

## 成本估算

- 模型：`.env` 中配置的 `LLM_MODEL`（默认 GLM-5.1）
- 每题约消耗：~800 input tokens + ~300 output tokens
- 21 道题总计：约 23k tokens
- 预估费用：取决于所使用的 LLM 供应商定价
