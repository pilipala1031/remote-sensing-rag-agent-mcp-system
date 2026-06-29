"""tests/test_eval_with_labels.py — 标签适配器单元测试。

测试覆盖：
1. flatten_labeled_question 完整字段映射
2. flatten_labeled_question 缺失 eval_labels 的安全默认值
3. flatten_labeled_question 部分缺失字段的安全默认值
4. load_labeled_questions 加载真实标签文件
5. get_labeled_question_stats 统计正确性
6. get_labeled_question_stats 对嵌套/flat 结构的兼容性
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.eval_with_labels import (
    DEFAULT_LABELS_PATH,
    flatten_labeled_question,
    get_labeled_question_stats,
    load_labeled_questions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sample_nested_question() -> dict:
    """一条完整的嵌套标签题目。"""
    return {
        "id": "q_test_001",
        "category": "dataset",
        "question": "测试题目？",
        "eval_labels": {
            "should_refuse": False,
            "required_keywords": ["建筑", "0.3"],
            "relevant_docs": ["01_datasets.md"],
            "question_type": "structured",
            "min_answer_length": 50,
            "expected_tool": "dataset_spec_lookup",
            "notes": "测试备注",
        },
    }


@pytest.fixture()
def sample_out_of_scope_question() -> dict:
    """一条领域外拒答题目。"""
    return {
        "id": "q_test_002",
        "category": "out_of_scope",
        "question": "如何在 Linux 配置 Nginx？",
        "eval_labels": {
            "should_refuse": True,
            "required_keywords": [],
            "relevant_docs": [],
            "question_type": "out_of_scope",
            "min_answer_length": 20,
            "expected_tool": "knowledge_base_search",
            "notes": "领域外问题",
        },
    }


# --------------------------------------------------------------------------- #
# 测试 flatten_labeled_question
# --------------------------------------------------------------------------- #
class TestFlattenLabeledQuestion:
    """测试字段映射与安全默认值。"""

    def test_full_mapping(self, sample_nested_question: dict) -> None:
        """完整嵌套结构应正确映射所有字段。"""
        flat = flatten_labeled_question(sample_nested_question)

        # 顶层字段直传
        assert flat["id"] == "q_test_001"
        assert flat["category"] == "dataset"
        assert flat["question"] == "测试题目？"

        # eval_labels 字段映射 + 重命名
        assert flat["expected_keywords"] == ["建筑", "0.3"]
        assert flat["expected_source_files"] == ["01_datasets.md"]
        assert flat["should_refuse"] is False
        assert flat["question_type"] == "structured"
        assert flat["min_answer_length"] == 50
        assert flat["notes"] == "测试备注"

        # expected_tool (str) → expected_tools (list)
        assert flat["expected_tools"] == ["dataset_spec_lookup"]

        # 不应保留 eval_labels 嵌套键
        assert "eval_labels" not in flat

    def test_missing_eval_labels(self) -> None:
        """完全缺失 eval_labels 时应使用全部默认值。"""
        item = {"id": "q_min", "category": "test", "question": "q?"}
        flat = flatten_labeled_question(item)

        assert flat["expected_keywords"] == []
        assert flat["expected_source_files"] == []
        assert flat["expected_tools"] == []
        assert flat["should_refuse"] is False
        assert flat["question_type"] == ""
        assert flat["min_answer_length"] == 0
        assert flat["notes"] == ""

    def test_partial_eval_labels(self) -> None:
        """部分字段缺失时，缺失部分用默认值填充。"""
        item = {
            "id": "q_partial",
            "category": "test",
            "question": "q?",
            "eval_labels": {
                "should_refuse": True,
                # 缺少 required_keywords / relevant_docs / expected_tool 等
            },
        }
        flat = flatten_labeled_question(item)

        assert flat["should_refuse"] is True
        assert flat["expected_keywords"] == []
        assert flat["expected_source_files"] == []
        assert flat["expected_tools"] == []
        assert flat["min_answer_length"] == 0
        assert flat["notes"] == ""

    def test_empty_expected_tool(self) -> None:
        """expected_tool 为空字符串时应映射为空列表。"""
        item = {
            "id": "q_empty_tool",
            "category": "test",
            "question": "q?",
            "eval_labels": {"expected_tool": ""},
        }
        flat = flatten_labeled_question(item)
        assert flat["expected_tools"] == []

    def test_expected_tool_list_defensive(self) -> None:
        """expected_tool 已经是列表时应防御性保留。"""
        item = {
            "id": "q_list_tool",
            "category": "test",
            "question": "q?",
            "eval_labels": {"expected_tool": ["tool_a", "tool_b"]},
        }
        flat = flatten_labeled_question(item)
        assert flat["expected_tools"] == ["tool_a", "tool_b"]


# --------------------------------------------------------------------------- #
# 测试 load_labeled_questions
# --------------------------------------------------------------------------- #
class TestLoadLabeledQuestions:
    """测试标签文件加载。"""

    def test_load_real_file(self) -> None:
        """加载真实 eval_questions_with_labels.json。"""
        questions = load_labeled_questions(DEFAULT_LABELS_PATH)
        assert isinstance(questions, list)
        assert len(questions) > 0
        # 每条都应有 id 和 eval_labels
        for q in questions:
            assert "id" in q
            assert "eval_labels" in q

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """文件不存在时应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_labeled_questions(tmp_path / "nonexistent.json")

    def test_load_invalid_format_top_level_list(self, tmp_path: Path) -> None:
        """顶层是 list 而非 dict 时应抛出 ValueError。"""
        bad_file = tmp_path / "bad_list.json"
        bad_file.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="顶层为 dict"):
            load_labeled_questions(bad_file)

    def test_load_missing_questions_key(self, tmp_path: Path) -> None:
        """缺少 questions 键时应抛出 ValueError。"""
        bad_file = tmp_path / "bad_no_questions.json"
        bad_file.write_text('{"meta": "x"}', encoding="utf-8")
        with pytest.raises(ValueError, match="questions"):
            load_labeled_questions(bad_file)


# --------------------------------------------------------------------------- #
# 测试 get_labeled_question_stats
# --------------------------------------------------------------------------- #
class TestGetLabeledQuestionStats:
    """测试统计函数。"""

    def test_stats_with_flat_input(
        self,
        sample_nested_question: dict,
        sample_out_of_scope_question: dict,
    ) -> None:
        """flat 输入（先 flatten）统计应正确。"""
        flat_q1 = flatten_labeled_question(sample_nested_question)
        flat_q2 = flatten_labeled_question(sample_out_of_scope_question)
        stats = get_labeled_question_stats([flat_q1, flat_q2])

        assert stats["total"] == 2
        assert stats["in_scope_count"] == 1
        assert stats["out_of_scope_count"] == 1
        assert stats["question_type_distribution"]["structured"] == 1
        assert stats["question_type_distribution"]["out_of_scope"] == 1
        assert stats["expected_tool_distribution"]["dataset_spec_lookup"] == 1
        assert stats["expected_tool_distribution"]["knowledge_base_search"] == 1

    def test_stats_with_nested_input(
        self,
        sample_nested_question: dict,
        sample_out_of_scope_question: dict,
    ) -> None:
        """嵌套输入（未 flatten）也应正确统计。"""
        stats = get_labeled_question_stats(
            [sample_nested_question, sample_out_of_scope_question]
        )
        assert stats["total"] == 2
        assert stats["in_scope_count"] == 1
        assert stats["out_of_scope_count"] == 1

    def test_stats_real_file(self) -> None:
        """对真实标签文件统计，验证题目总数。"""
        raw = load_labeled_questions(DEFAULT_LABELS_PATH)
        flat = [flatten_labeled_question(q) for q in raw]
        stats = get_labeled_question_stats(flat)

        assert stats["total"] == len(raw)
        # 领域内 + 领域外 = 总数
        assert stats["in_scope_count"] + stats["out_of_scope_count"] == stats["total"]
        # 至少有 3 道领域外题目
        assert stats["out_of_scope_count"] >= 3
        # 题型分布的值之和 = 总数
        assert sum(stats["question_type_distribution"].values()) == stats["total"]
        # 工具分布的值之和 = 总数
        assert sum(stats["expected_tool_distribution"].values()) == stats["total"]

    def test_stats_empty_list(self) -> None:
        """空列表统计应返回全零。"""
        stats = get_labeled_question_stats([])
        assert stats["total"] == 0
        assert stats["in_scope_count"] == 0
        assert stats["out_of_scope_count"] == 0
