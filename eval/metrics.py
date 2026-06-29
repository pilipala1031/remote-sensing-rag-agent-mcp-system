"""轻量级 RAG / Agent 评估指标与公共工具。

本模块只实现最简单、可解释的评估指标，不依赖 pytest 或任何复杂评估框架。
两个评估脚本（run_rag_eval.py / run_agent_eval.py）共享这里的：

- 指标函数：keyword_hit_rate / source_hit_rate / refusal_correct /
            average_latency / average_sources_count / tool_call_rate /
            tool_hit_rate / unexpected_tool_rate
- 公共工具：load_questions / check_backend / build_summary / save_result /
            print_summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
API_BASE = "http://127.0.0.1:8000"
RAG_QUERY_URL = f"{API_BASE}/api/chat/query"
AGENT_QUERY_URL = f"{API_BASE}/api/agent/query"
HEALTH_TIMEOUT = 3  # 后端健康探测超时（秒）

# 评估题集 / 结果目录（相对本文件定位，便于从任意 cwd 运行）
EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
RESULTS_DIR = EVAL_DIR / "results"


# --------------------------------------------------------------------------- #
# 指标函数
# --------------------------------------------------------------------------- #
def keyword_hit_rate(answer: str, expected_keywords: Sequence[str]) -> float:
    """关键词命中率：答案中命中的关键词占比。

    大小写不敏感。当 expected_keywords 为空时返回 1.0
    （该题不考查关键词，例如“应拒答”的题目）。
    """
    if not expected_keywords:
        return 1.0
    if not answer:
        return 0.0
    lower_answer = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_answer)
    return hits / len(expected_keywords)


def source_hit_rate(sources: Sequence[Dict[str, Any]],
                    expected_source_files: Sequence[str]) -> float:
    """来源命中率：检索到的来源文件覆盖期望文件的比例。

    采用子串匹配，因此即使真实文件名带 doc_id 前缀
    （如 18bb34ec0cecb0c7_01_datasets.md）也能命中期望的 01_datasets.md。
    expected_source_files 为空时返回 1.0。
    """
    if not expected_source_files:
        return 1.0
    actual_filenames = [
        str(s.get("filename", "")) for s in (sources or [])
    ]
    hits = 0
    for expected in expected_source_files:
        for actual in actual_filenames:
            if expected in actual:
                hits += 1
                break
    return hits / len(expected_source_files)


def refusal_correct(refused: bool, should_refuse: bool) -> bool:
    """拒答判断是否正确：refused 与 should_refuse 一致即正确。"""
    return bool(refused) == bool(should_refuse)


def average_latency(latencies: Sequence[float]) -> float:
    """平均延迟（秒）。"""
    latencies = list(latencies)
    return sum(latencies) / len(latencies) if latencies else 0.0


def average_sources_count(counts: Sequence[int]) -> float:
    """平均来源数量。"""
    counts = list(counts)
    return sum(counts) / len(counts) if counts else 0.0


def tool_call_rate(tool_call_counts: Sequence[int], total: int) -> float:
    """工具调用率：有至少一次工具调用的题目占比（仅 Agent 使用）。

    tool_call_counts 为每题返回的 tool_calls 列表长度序列。
    """
    if total <= 0:
        return 0.0
    used = sum(1 for c in tool_call_counts if c and c > 0)
    return used / total


def tool_hit_rate(
    tool_calls: Sequence[Dict[str, Any]],
    expected_tools: Sequence[str],
) -> float | None:
    """工具命中率：tool_calls 中是否出现了任一 expected_tools。

    逻辑：
    - expected_tools 为空时返回 None（该题不考查工具选择）。
    - tool_calls 中出现任一 expected_tools 中的工具名，返回 1.0。
    - 否则返回 0.0。

    Args:
        tool_calls: Agent 返回的 tool_calls 列表，每个元素包含 "tool" 字段。
        expected_tools: 期望出现的工具名列表（OR 逻辑）。

    Returns:
        1.0 / 0.0 / None
    """
    if not expected_tools:
        return None
    actual_tools: set[str] = set()
    for tc in (tool_calls or []):
        if isinstance(tc, dict):
            tool_name = str(tc.get("tool", ""))
            if tool_name:
                actual_tools.add(tool_name)
    for expected in expected_tools:
        if expected in actual_tools:
            return 1.0
    return 0.0


def unexpected_tool_rate(
    tool_calls: Sequence[Dict[str, Any]],
    unexpected_tools: Sequence[str],
) -> float | None:
    """非预期工具误用率：tool_calls 中是否出现了 unexpected_tools 中的工具。

    逻辑：
    - unexpected_tools 为空时返回 None（该题不检查非预期工具）。
    - tool_calls 中出现了任一 unexpected_tools 中的工具，返回 1.0
      （表示有误用，值越低越好）。
    - 否则返回 0.0（表示无误用）。

    与 tool_hit_rate 的关系：
    - tool_hit_rate 衡量"该用的工具是否用了"（越高越好）。
    - unexpected_tool_rate 衡量"不该用的工具是否误用了"（越低越好）。

    Args:
        tool_calls: Agent 返回的 tool_calls 列表，每个元素包含 "tool" 字段。
        unexpected_tools: 不应出现的工具名列表。

    Returns:
        1.0（有误用）/ 0.0（无误用）/ None（不检查）
    """
    if not unexpected_tools:
        return None
    actual_tools: set[str] = set()
    for tc in (tool_calls or []):
        if isinstance(tc, dict):
            tool_name = str(tc.get("tool", ""))
            if tool_name:
                actual_tools.add(tool_name)
    for unexpected in unexpected_tools:
        if unexpected in actual_tools:
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """加载评估题集 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"评估题集格式错误：{path} 应为 JSON 数组")
    return data


def check_backend(is_agent: bool = False) -> None:
    """探测后端是否可用，不可用则以清晰中文提示退出。

    通过实际尝试对应接口的健康可达性来判断（FastAPI 默认提供 /docs）。
    """
    target = AGENT_QUERY_URL if is_agent else RAG_QUERY_URL
    health_url = f"{API_BASE}/docs"
    try:
        resp = requests.get(health_url, timeout=HEALTH_TIMEOUT)
        # /docs 返回 200 即说明 FastAPI 已启动
        if resp.status_code != 200:
            raise ConnectionError()
    except Exception:  # noqa: BLE001
        endpoint = "/api/agent/query" if is_agent else "/api/chat/query"
        print(
            f"无法连接 {API_BASE}，请先启动 FastAPI 后端。\n"
            f"（评估脚本需要调用 {endpoint}，"
            f"可执行：uvicorn app.main:app --reload --host 0.0.0.0 --port 8000）",
            file=sys.stderr,
        )
        sys.exit(1)


def _safe_avg(values: Sequence[float], n: int) -> float:
    """安全求均值：n<=0 时返回 0.0。"""
    if n <= 0:
        return 0.0
    return sum(values) / n


def build_summary(details: List[Dict[str, Any]],
                  include_tool_call: bool = False) -> Dict[str, Any]:
    """根据逐题明细汇总总体指标。

    - keyword_hit_rate_avg / source_hit_rate_avg 只在"应回答"的题目
      （有期望关键词/来源）上平均，避免拒答题拉高均值；
    - refusal_accuracy 在全部题目上统计；
    - avg_latency / avg_sources_count 在全部题目上统计；
    - tool_call_rate 仅 Agent 评估时计算；
    - 当 include_tool_call=True 时，额外计算 Agent 专属指标：
      tool_hit_rate_avg / verification_pass_rate /
      avg_agent_total_elapsed / avg_tool_calls_count。
    """
    total = len(details)

    kw_items = [
        d["keyword_hit_rate"] for d in details
        if d.get("expected_keyword_count", 0) > 0
    ]
    src_items = [
        d["source_hit_rate"] for d in details
        if d.get("expected_source_count", 0) > 0
    ]
    refusal_ok = sum(1 for d in details if d.get("refusal_correct", False))

    summary: Dict[str, Any] = {
        "total": total,
        "keyword_hit_rate_avg": round(
            _safe_avg(kw_items, len(kw_items)), 4),
        "source_hit_rate_avg": round(
            _safe_avg(src_items, len(src_items)), 4),
        "refusal_accuracy": round(refusal_ok / total, 4) if total else 0.0,
        "avg_latency": round(
            _safe_avg([d.get("latency", 0.0) for d in details], total), 4),
        "avg_sources_count": round(
            _safe_avg([d.get("sources_count", 0) for d in details], total), 4),
    }

    if include_tool_call:
        summary["tool_call_rate"] = round(tool_call_rate(
            [d.get("tool_calls_count", 0) for d in details], total), 4)

        # ---------- Agent 专属指标 ----------

        # 工具命中率平均值（只在有 expected_tools 的题目上统计）
        tool_hit_items = [
            d["tool_hit"] for d in details
            if d.get("tool_hit") is not None
        ]
        summary["tool_hit_rate_avg"] = round(
            _safe_avg(tool_hit_items, len(tool_hit_items)), 4
        ) if tool_hit_items else 0.0

        # 证据校验通过率（只在 verification.enabled=True 的题目上统计）
        verif_enabled = [
            d for d in details
            if isinstance(d.get("verification"), dict)
            and d["verification"].get("enabled")
        ]
        verif_passed = sum(
            1 for d in verif_enabled
            if d["verification"].get("verified") is True
        )
        summary["verification_pass_rate"] = round(
            verif_passed / len(verif_enabled), 4
        ) if verif_enabled else 0.0

        # 后端 Agent 总耗时平均值（timing.total_elapsed）
        summary["avg_agent_total_elapsed"] = round(
            _safe_avg(
                [d.get("agent_total_elapsed", 0.0) for d in details], total
            ), 4
        )

        # 平均工具调用次数
        summary["avg_tool_calls_count"] = round(
            _safe_avg(
                [d.get("tool_calls_count", 0) for d in details], total
            ), 4
        )

        # 非预期工具误用率平均值（只在有 unexpected_tools 的题目上统计，越低越好）
        unexpected_items = [
            d["unexpected_tool_hit"] for d in details
            if d.get("unexpected_tool_hit") is not None
        ]
        summary["unexpected_tool_rate_avg"] = round(
            _safe_avg(unexpected_items, len(unexpected_items)), 4
        ) if unexpected_items else 0.0

        # 证据校验耗时平均值
        summary["avg_verification_elapsed"] = round(
            _safe_avg(
                [d.get("verification_elapsed", 0.0) for d in details], total
            ), 4
        )

        # 平均答案长度（字符数）
        summary["avg_answer_length"] = round(
            _safe_avg(
                [d.get("answer_length", 0) for d in details], total
            ), 4
        )
    else:
        summary["tool_call_rate"] = 0.0

    return summary


def save_result(payload: Dict[str, Any], filename: str) -> Path:
    """把评估结果写入 eval/results/<filename>，返回路径。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def print_summary(title: str, summary: Dict[str, Any]) -> None:
    """以可读格式在控制台打印汇总结果。

    当 summary 中包含 Agent 专属指标时（tool_hit_rate_avg 等），
    会额外打印这些指标行。
    """
    print("\n" + "=" * 60)
    print(f"  {title} 评估汇总（共 {summary.get('total', 0)} 题）")
    print("=" * 60)
    print(f"  关键词命中率 keyword_hit_rate_avg : "
          f"{summary.get('keyword_hit_rate_avg', 0.0):.4f}")
    print(f"  来源命中率   source_hit_rate_avg  : "
          f"{summary.get('source_hit_rate_avg', 0.0):.4f}")
    print(f"  拒答准确率   refusal_accuracy      : "
          f"{summary.get('refusal_accuracy', 0.0):.4f}")
    print(f"  平均延迟     avg_latency (s)       : "
          f"{summary.get('avg_latency', 0.0):.4f}")
    print(f"  平均来源数   avg_sources_count     : "
          f"{summary.get('avg_sources_count', 0.0):.4f}")
    print(f"  工具调用率   tool_call_rate        : "
          f"{summary.get('tool_call_rate', 0.0):.4f}")

    # ---------- Agent 专属指标（存在时才打印） ----------
    if "tool_hit_rate_avg" in summary:
        print("-" * 60)
        print(f"  工具命中率   tool_hit_rate_avg     : "
              f"{summary.get('tool_hit_rate_avg', 0.0):.4f}")
        print(f"  非预期误用   unexpected_tool_rate  : "
              f"{summary.get('unexpected_tool_rate_avg', 0.0):.4f}"
              f"  (越低越好)")
        print(f"  证据校验通过 verification_pass_rate: "
              f"{summary.get('verification_pass_rate', 0.0):.4f}")
        print(f"  Agent总耗时  avg_agent_total_elapsed: "
              f"{summary.get('avg_agent_total_elapsed', 0.0):.4f}")
        print(f"  校验耗时     avg_verification_elapsed: "
              f"{summary.get('avg_verification_elapsed', 0.0):.4f}")
        print(f"  平均工具调用 avg_tool_calls_count   : "
              f"{summary.get('avg_tool_calls_count', 0.0):.4f}")
        print(f"  平均答案长度 avg_answer_length      : "
              f"{summary.get('avg_answer_length', 0.0):.1f} 字符")

    print("=" * 60)
