"""标签适配器：把 eval_questions_with_labels.json 转成 metrics.py 可消费的 flat 结构。

eval_questions_with_labels.json 采用嵌套结构（eval_labels.*），
而 eval/metrics.py 的 load_questions() 期望 flat 结构（expected_keywords 等）。
本模块做纯数据转换，不依赖任何 eval 脚本、RAG/Agent 代码或网络请求。

字段映射规则：
    eval_labels.required_keywords  → expected_keywords      (list)
    eval_labels.relevant_docs      → expected_source_files   (list)
    eval_labels.expected_tool      → expected_tools          (str → [str])
    eval_labels.should_refuse      → should_refuse           (bool)
    eval_labels.question_type      → question_type           (str)
    eval_labels.min_answer_length  → min_answer_length       (int)
    eval_labels.notes              → notes                   (str)

CLI 用法：
    python -m experiments.eval_with_labels
    python -m experiments.eval_with_labels --path custom_labels.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# 路径常量
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "eval" / "eval_questions_with_labels.json"


# --------------------------------------------------------------------------- #
# 核心函数
# --------------------------------------------------------------------------- #
def load_labeled_questions(
    path: str | Path = DEFAULT_LABELS_PATH,
) -> List[Dict[str, Any]]:
    """加载 eval_questions_with_labels.json，返回 questions 列表。

    该文件顶层是 dict（含 generator / questions 等键），
    本函数只提取 ``questions`` 数组。

    Args:
        path: JSON 文件路径，默认指向 eval/eval_questions_with_labels.json。

    Returns:
        questions 列表，每个元素是原始嵌套结构的 dict。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件格式不是 dict 或缺少 ``questions`` 键。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"标签文件不存在：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"标签文件格式错误：期望顶层为 dict，实际为 {type(data).__name__}"
        )
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError(
            "标签文件格式错误：缺少 questions 数组或类型不正确"
        )
    return questions


def flatten_labeled_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """把单条嵌套标签转成 flat 结构。

    从 eval_labels 子对象中提取字段并展平到顶层，
    同时应用字段重命名和类型转换。
    对缺失或类型不匹配的字段使用安全默认值。

    Args:
        item: 单条题目的原始 dict（含 eval_labels 子对象）。

    Returns:
        flat dict，包含以下键：
        ``id``, ``category``, ``question``,
        ``expected_keywords``, ``expected_source_files``,
        ``expected_tools``, ``should_refuse``,
        ``question_type``, ``min_answer_length``, ``notes``。
    """
    labels = item.get("eval_labels", {})
    if not isinstance(labels, dict):
        labels = {}

    # --- 从 eval_labels 提取，带安全默认值 ---
    expected_keywords = labels.get("required_keywords", [])
    if not isinstance(expected_keywords, list):
        expected_keywords = []

    expected_source_files = labels.get("relevant_docs", [])
    if not isinstance(expected_source_files, list):
        expected_source_files = []

    # expected_tool 是单个字符串，包装成列表
    expected_tool = labels.get("expected_tool", "")
    if isinstance(expected_tool, str) and expected_tool:
        expected_tools = [expected_tool]
    elif isinstance(expected_tool, list):
        # 防御性处理：万一已经是列表
        expected_tools = expected_tool
    else:
        expected_tools = []

    should_refuse = labels.get("should_refuse", False)
    if not isinstance(should_refuse, bool):
        should_refuse = bool(should_refuse)

    question_type = labels.get("question_type", "")
    if not isinstance(question_type, str):
        question_type = str(question_type)

    min_answer_length = labels.get("min_answer_length", 0)
    if not isinstance(min_answer_length, int):
        try:
            min_answer_length = int(min_answer_length)
        except (TypeError, ValueError):
            min_answer_length = 0

    notes = labels.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)

    return {
        "id": item.get("id", ""),
        "category": item.get("category", ""),
        "question": item.get("question", ""),
        "expected_keywords": expected_keywords,
        "expected_source_files": expected_source_files,
        "expected_tools": expected_tools,
        "should_refuse": should_refuse,
        "question_type": question_type,
        "min_answer_length": min_answer_length,
        "notes": notes,
    }


def get_labeled_question_stats(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统计标签题目分布概况。

    接收 flatten 后的 flat 结构列表（flatten_labeled_question 的输出），
    也可以接收原始嵌套结构（内部自动 flatten）。

    Args:
        questions: 题目列表（flat 或嵌套均可）。

    Returns:
        统计 dict，包含：
        ``total``, ``in_scope_count``, ``out_of_scope_count``,
        ``question_type_distribution``, ``expected_tool_distribution``。
    """
    # 如果元素含 eval_labels 键，先 flatten
    flat_questions: List[Dict[str, Any]] = []
    for q in questions:
        if "eval_labels" in q:
            flat_questions.append(flatten_labeled_question(q))
        else:
            flat_questions.append(q)

    total = len(flat_questions)
    in_scope = sum(1 for q in flat_questions if not q.get("should_refuse", False))
    out_scope = total - in_scope

    # question_type 分布
    type_dist: Dict[str, int] = {}
    for q in flat_questions:
        qt = q.get("question_type", "unknown")
        type_dist[qt] = type_dist.get(qt, 0) + 1

    # expected_tool 分布（取每个题目 expected_tools 的第一个元素）
    tool_dist: Dict[str, int] = {}
    for q in flat_questions:
        tools = q.get("expected_tools", [])
        if tools:
            primary_tool = tools[0]
        else:
            primary_tool = "(none)"
        tool_dist[primary_tool] = tool_dist.get(primary_tool, 0) + 1

    return {
        "total": total,
        "in_scope_count": in_scope,
        "out_of_scope_count": out_scope,
        "question_type_distribution": type_dist,
        "expected_tool_distribution": tool_dist,
    }


# --------------------------------------------------------------------------- #
# CLI 入口
# --------------------------------------------------------------------------- #
def _print_stats(stats: Dict[str, Any]) -> None:
    """以可读格式打印统计结果。"""
    print("=" * 50)
    print("  标签题目统计")
    print("=" * 50)
    print(f"  总数         total             : {stats['total']}")
    print(f"  领域内       in_scope_count    : {stats['in_scope_count']}")
    print(f"  领域外       out_of_scope_count: {stats['out_of_scope_count']}")

    print("-" * 50)
    print("  题型分布 question_type_distribution:")
    for qt, count in sorted(stats["question_type_distribution"].items()):
        print(f"    {qt:20s}: {count}")

    print("-" * 50)
    print("  期望工具分布 expected_tool_distribution:")
    for tool, count in sorted(stats["expected_tool_distribution"].items()):
        print(f"    {tool:30s}: {count}")
    print("=" * 50)


def main(argv: List[str] | None = None) -> int:
    """CLI 入口：加载标签文件并打印统计。

    Args:
        argv: 命令行参数列表（不含脚本名）。支持 ``--path``。

    Returns:
        0 成功 / 1 失败。
    """
    args = argv if argv is not None else sys.argv[1:]
    path = DEFAULT_LABELS_PATH
    for i, arg in enumerate(args):
        if arg == "--path" and i + 1 < len(args):
            path = Path(args[i + 1])

    try:
        raw_questions = load_labeled_questions(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    flat_questions = [flatten_labeled_question(q) for q in raw_questions]
    stats = get_labeled_question_stats(flat_questions)
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
