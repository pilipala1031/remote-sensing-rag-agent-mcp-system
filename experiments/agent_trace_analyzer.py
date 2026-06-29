"""Agent 工具选择轨迹分析器。

对比 Agent 实际调用的工具与 eval_questions_with_labels.json 中的 expected_tool 标签，
量化 Agent 的工具选择准确率、混淆模式和分类性能。

支持两种运行模式：
1. **live 模式**（默认）：逐题调用 POST /api/agent/query（include_trace=true），
   收集 tool_calls / trace_events 后分析。
2. **offline 模式**：从已保存的 eval/results/agent_eval_result.json 读取结果，
   要求该文件已包含 tool_calls 和 trace_events 字段。

CLI 用法：
    # live 模式（需后端运行在 :8000）
    python -m experiments.agent_trace_analyzer

    # offline 模式（从已保存结果分析）
    python -m experiments.agent_trace_analyzer --offline

    # 指定后端地址
    python -m experiments.agent_trace_analyzer --base-url http://127.0.0.1:8000

输出：
    experiments/results/agent_trace_analysis.json  —— 完整分析结果
    控制台摘要报告

注意：
- 本脚本不修改 Agent 逻辑、不改 RAG 参数、不重写任何现有模块。
- out_of_scope 题目的 expected_tool 为 knowledge_base_search（用于检测是否检索后拒答）。
- 工具命中率判定逻辑：actual_tools 与 expected_tools 有交集即为命中（OR 语义）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# 复用标签适配器
from experiments.eval_with_labels import (
    DEFAULT_LABELS_PATH,
    flatten_labeled_question,
    load_labeled_questions,
)

# --------------------------------------------------------------------------- #
# 路径常量
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
SAVED_EVAL_PATH = PROJECT_ROOT / "eval" / "results" / "agent_eval_result.json"

AGENT_QUERY_URL_DEFAULT = "http://127.0.0.1:8000/api/agent/query"

# 全部 7 个工具
ALL_TOOLS = [
    "knowledge_base_search",
    "plan_and_search",
    "dataset_overview",
    "dataset_spec_lookup",
    "model_comparison_table",
    "metric_formula_lookup",
    "metrics_calculator",
]

# 工具中文名（与前端 TOOL_NAME_CN 一致）
TOOL_NAME_CN = {
    "knowledge_base_search": "知识库语义检索",
    "dataset_overview": "数据集共性概览",
    "dataset_spec_lookup": "数据集结构化查询",
    "model_comparison_table": "模型对比工具",
    "metric_formula_lookup": "指标公式查询",
    "metrics_calculator": "指标计算器",
    "plan_and_search": "复杂问题分解检索",
}


# --------------------------------------------------------------------------- #
# 数据采集
# --------------------------------------------------------------------------- #
def query_agent_trace(
    question: str,
    base_url: str = AGENT_QUERY_URL_DEFAULT,
    timeout: int = 120,
) -> Dict[str, Any]:
    """调用 /api/agent/query 并返回完整响应（include_trace=true）。

    Args:
        question: 用户问题。
        base_url: 后端 API 地址。
        timeout: 请求超时（秒）。

    Returns:
        后端返回的 JSON dict。
    """
    resp = requests.post(
        base_url,
        json={"question": question, "include_trace": True},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def load_saved_eval(path: Path = SAVED_EVAL_PATH) -> List[Dict[str, Any]]:
    """从已保存的 eval 结果 JSON 中读取逐题详情。

    Args:
        path: agent_eval_result.json 路径。

    Returns:
        details 列表，每个元素包含 id / question / tool_calls / trace_events 等。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件格式不正确。
    """
    if not path.exists():
        raise FileNotFoundError(f"已保存的 eval 结果不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", [])
    if not isinstance(details, list):
        raise ValueError("eval 结果格式错误：缺少 details 数组")
    return details


# --------------------------------------------------------------------------- #
# 分析逻辑
# --------------------------------------------------------------------------- #
def extract_actual_tools(tool_calls: List[Dict]) -> List[str]:
    """从 tool_calls 中提取工具名列表（保持顺序，不去重）。"""
    tools = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            name = tc.get("tool")
            if name:
                tools.append(str(name))
    return tools


def check_tool_hit(actual_tools: List[str], expected_tools: List[str]) -> bool:
    """检查实际工具是否命中期望工具（OR 语义：有交集即命中）。"""
    if not expected_tools:
        return True  # 无期望工具，默认命中
    actual_set = set(actual_tools)
    expected_set = set(expected_tools)
    return bool(actual_set & expected_set)


def check_unexpected_tools(
    actual_tools: List[str], unexpected_tools: List[str]
) -> List[str]:
    """检查是否误用了不应使用的工具，返回误用工具列表。"""
    if not unexpected_tools:
        return []
    actual_set = set(actual_tools)
    unexpected_set = set(unexpected_tools)
    return sorted(actual_set & unexpected_set)


def analyze_single(
    question_id: str,
    question: str,
    expected_tools: List[str],
    unexpected_tools: List[str],
    question_type: str,
    should_refuse: bool,
    actual_tools: List[str],
    tool_calls: List[Dict],
    trace_events: List[Dict],
    refused: bool,
    timing: Dict,
    error: str | None = None,
) -> Dict[str, Any]:
    """分析单道题的工具选择情况。

    Returns:
        包含完整分析数据的 dict。
    """
    tool_hit = check_tool_hit(actual_tools, expected_tools)
    misused = check_unexpected_tools(actual_tools, unexpected_tools)
    total_elapsed = timing.get("total_elapsed", 0.0) if isinstance(timing, dict) else 0.0

    # 工具调用次数
    call_count = len(actual_tools)
    unique_tools = list(dict.fromkeys(actual_tools))  # 去重保持顺序

    # 从 trace_events 中提取时间线
    timeline = []
    if trace_events:
        for ev in trace_events:
            if isinstance(ev, dict):
                timeline.append({
                    "step": ev.get("step", 0),
                    "event": ev.get("event", ""),
                    "timestamp": ev.get("timestamp", 0.0),
                    "detail": ev.get("detail"),
                })

    return {
        "id": question_id,
        "question": question,
        "question_type": question_type,
        "should_refuse": should_refuse,
        "refused": refused,
        "expected_tools": expected_tools,
        "unexpected_tools": unexpected_tools,
        "actual_tools": actual_tools,
        "unique_tools": unique_tools,
        "tool_hit": tool_hit,
        "misused_tools": misused,
        "has_misuse": len(misused) > 0,
        "tool_call_count": call_count,
        "unique_tool_count": len(unique_tools),
        "total_elapsed": round(total_elapsed, 4),
        "timeline": timeline,
        "error": error,
        # 各工具的耗时
        "tool_timings": _extract_tool_timings(tool_calls),
    }


def _extract_tool_timings(tool_calls: List[Dict]) -> List[Dict]:
    """从 tool_calls 中提取每个工具的耗时信息。"""
    timings = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        elapsed = tc.get("elapsed")
        timings.append({
            "step": i + 1,
            "tool": tc.get("tool", "unknown"),
            "elapsed": round(float(elapsed), 4) if isinstance(elapsed, (int, float)) else None,
            "status": tc.get("status", "unknown"),
        })
    return timings


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总分析所有题目的工具选择情况。

    Returns:
        汇总统计 dict。
    """
    total = len(results)
    if total == 0:
        return {"total": 0}

    # --- 整体指标 ---
    tool_hit_count = sum(1 for r in results if r["tool_hit"])
    misuse_count = sum(1 for r in results if r["has_misuse"])
    no_tool_count = sum(1 for r in results if r["tool_call_count"] == 0)
    refused_count = sum(1 for r in results if r["refused"])

    # --- 工具使用频率 ---
    tool_usage: Dict[str, int] = {t: 0 for t in ALL_TOOLS}
    tool_usage["(no_tool)"] = 0
    for r in results:
        if r["tool_call_count"] == 0:
            tool_usage["(no_tool)"] += 1
        for t in r["actual_tools"]:
            tool_usage[t] = tool_usage.get(t, 0) + 1

    # --- 按 question_type 分组命中率 ---
    type_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        qt = r["question_type"]
        if qt not in type_stats:
            type_stats[qt] = {"total": 0, "hit": 0, "misuse": 0, "no_tool": 0}
        type_stats[qt]["total"] += 1
        if r["tool_hit"]:
            type_stats[qt]["hit"] += 1
        if r["has_misuse"]:
            type_stats[qt]["misuse"] += 1
        if r["tool_call_count"] == 0:
            type_stats[qt]["no_tool"] += 1

    # 计算每组的命中率
    for qt, s in type_stats.items():
        s["hit_rate"] = round(s["hit"] / s["total"], 4) if s["total"] > 0 else 0.0

    # --- 工具混淆矩阵（expected → actual） ---
    confusion: Dict[str, Dict[str, int]] = {}
    for r in results:
        for exp_t in r["expected_tools"]:
            if exp_t not in confusion:
                confusion[exp_t] = {}
            for act_t in r["actual_tools"]:
                confusion[exp_t][act_t] = confusion[exp_t].get(act_t, 0) + 1
            if r["tool_call_count"] == 0:
                confusion[exp_t]["(no_tool)"] = confusion[exp_t].get("(no_tool)", 0) + 1

    # --- 平均工具调用次数 ---
    avg_calls = sum(r["tool_call_count"] for r in results) / total
    avg_unique = sum(r["unique_tool_count"] for r in results) / total

    # --- 平均耗时 ---
    avg_elapsed = sum(r["total_elapsed"] for r in results) / total

    # --- per-tool 平均耗时 ---
    tool_elapsed_map: Dict[str, List[float]] = {}
    for r in results:
        for tt in r.get("tool_timings", []):
            if tt["elapsed"] is not None:
                tool_elapsed_map.setdefault(tt["tool"], []).append(tt["elapsed"])
    tool_avg_elapsed = {
        t: round(sum(v) / len(v), 4)
        for t, v in tool_elapsed_map.items()
        if v
    }

    return {
        "total": total,
        "tool_hit_count": tool_hit_count,
        "tool_hit_rate": round(tool_hit_count / total, 4),
        "misuse_count": misuse_count,
        "misuse_rate": round(misuse_count / total, 4),
        "no_tool_count": no_tool_count,
        "no_tool_rate": round(no_tool_count / total, 4),
        "refused_count": refused_count,
        "refusal_rate": round(refused_count / total, 4),
        "avg_tool_calls": round(avg_calls, 2),
        "avg_unique_tools": round(avg_unique, 2),
        "avg_total_elapsed": round(avg_elapsed, 4),
        "tool_usage_frequency": tool_usage,
        "tool_avg_elapsed": tool_avg_elapsed,
        "per_question_type": type_stats,
        "confusion_matrix": confusion,
    }


# --------------------------------------------------------------------------- #
# 报告生成
# --------------------------------------------------------------------------- #
def print_report(results: List[Dict], summary: Dict) -> None:
    """打印分析报告到控制台。"""
    print("=" * 70)
    print("  Agent 工具选择轨迹分析报告")
    print("=" * 70)
    print(f"  分析题目数   total            : {summary['total']}")
    print(f"  工具命中数   tool_hit_count   : {summary['tool_hit_count']}")
    print(f"  工具命中率   tool_hit_rate    : {summary['tool_hit_rate']:.2%}")
    print(f"  误用工具题数 misuse_count     : {summary['misuse_count']}")
    print(f"  误用率       misuse_rate      : {summary['misuse_rate']:.2%}")
    print(f"  未调工具题数 no_tool_count    : {summary['no_tool_count']}")
    print(f"  拒答题数     refused_count    : {summary['refused_count']}")
    print(f"  平均工具调用 avg_tool_calls   : {summary['avg_tool_calls']}")
    print(f"  平均独立工具 avg_unique_tools : {summary['avg_unique_tools']}")
    print(f"  平均总耗时   avg_total_elapsed: {summary['avg_total_elapsed']}s")

    print("-" * 70)
    print("  工具使用频率 tool_usage_frequency:")
    for tool, count in sorted(summary["tool_usage_frequency"].items(), key=lambda x: -x[1]):
        cn = TOOL_NAME_CN.get(tool, tool)
        print(f"    {tool:30s} ({cn:14s}): {count}")

    print("-" * 70)
    print("  各工具平均耗时 tool_avg_elapsed:")
    for tool, elapsed in sorted(summary.get("tool_avg_elapsed", {}).items(), key=lambda x: -x[1]):
        cn = TOOL_NAME_CN.get(tool, tool)
        print(f"    {tool:30s} ({cn:14s}): {elapsed:.4f}s")

    print("-" * 70)
    print("  按题型分组 per_question_type:")
    print(f"    {'题型':20s} {'总数':>4s} {'命中':>4s} {'命中率':>8s} {'误用':>4s} {'无工具':>6s}")
    for qt, s in sorted(summary["per_question_type"].items()):
        print(f"    {qt:20s} {s['total']:>4d} {s['hit']:>4d} {s['hit_rate']:>8.2%} "
              f"{s['misuse']:>4d} {s['no_tool']:>6d}")

    print("-" * 70)
    print("  工具混淆矩阵 confusion_matrix (expected → actual):")
    for exp_t, act_map in sorted(summary["confusion_matrix"].items()):
        exp_cn = TOOL_NAME_CN.get(exp_t, exp_t)
        parts = [f"{act_cn_or_name(k)}={v}" for k, v in sorted(act_map.items(), key=lambda x: -x[1])]
        print(f"    期望 [{exp_t}] ({exp_cn})")
        print(f"      → {', '.join(parts)}")

    print("-" * 70)
    print("  逐题明细:")
    print(f"    {'ID':12s} {'题型':14s} {'命中':>4s} {'误用':>4s} {'调用数':>4s} {'耗时':>8s}  实际工具")
    for r in results:
        hit_str = "Y" if r["tool_hit"] else "N"
        misuse_str = "Y" if r["has_misuse"] else "N"
        tools_str = ", ".join(r["actual_tools"]) if r["actual_tools"] else "(none)"
        print(f"    {r['id']:12s} {r['question_type']:14s} {hit_str:>4s} {misuse_str:>4s} "
              f"{r['tool_call_count']:>4d} {r['total_elapsed']:>8.2f}s  {tools_str}")

    print("=" * 70)


def act_cn_or_name(tool: str) -> str:
    """工具名转短标签（用于混淆矩阵展示）。"""
    return TOOL_NAME_CN.get(tool, tool)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_live_mode(
    questions: List[Dict[str, Any]],
    base_url: str,
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """live 模式：逐题调用 API 收集 trace 数据。

    Args:
        questions: flatten 后的题目列表。
        base_url: 后端 API 地址。
        timeout: 请求超时。

    Returns:
        分析结果列表。
    """
    results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        expected_tools = q.get("expected_tools", [])
        unexpected_tools = q.get("unexpected_tools", [])
        question_type = q.get("question_type", "")
        should_refuse = q.get("should_refuse", False)

        print(f"[{i}/{len(questions)}] {qid}: {question[:60]}...")

        try:
            data = query_agent_trace(question, base_url=base_url, timeout=timeout)
            error = None
        except Exception as e:
            data = {}
            error = f"{type(e).__name__}: {e}"
            print(f"  ERROR: {error}")

        actual_tools = extract_actual_tools(data.get("tool_calls", []))
        tool_calls = data.get("tool_calls", [])
        trace_events = data.get("trace_events", [])
        refused = bool(data.get("refused", False))
        timing = data.get("timing", {})

        result = analyze_single(
            question_id=qid,
            question=question,
            expected_tools=expected_tools,
            unexpected_tools=unexpected_tools,
            question_type=question_type,
            should_refuse=should_refuse,
            actual_tools=actual_tools,
            tool_calls=tool_calls,
            trace_events=trace_events,
            refused=refused,
            timing=timing,
            error=error,
        )
        results.append(result)

        hit_str = "HIT" if result["tool_hit"] else "MISS"
        print(f"  -> {hit_str} | tools={actual_tools} | elapsed={result['total_elapsed']}s")

        # 避免 API 限速
        time.sleep(0.5)

    return results


def run_offline_mode(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """offline 模式：从已保存的 eval 结果分析。

    Args:
        details: eval 结果中的 details 列表。

    Returns:
        分析结果列表。
    """
    results: List[Dict[str, Any]] = []

    # 同时加载标签以获取 expected_tools / unexpected_tools
    try:
        raw_questions = load_labeled_questions()
        label_map: Dict[str, Dict] = {}
        for rq in raw_questions:
            flat = flatten_labeled_question(rq)
            label_map[flat["id"]] = flat
    except Exception:
        label_map = {}

    for d in details:
        qid = d.get("id", "")
        question = d.get("question", "")
        actual_tools = extract_actual_tools(d.get("tool_calls", []))
        tool_calls = d.get("tool_calls", [])
        trace_events = d.get("trace_events", [])
        refused = bool(d.get("refused", False))
        timing = d.get("timing", {})
        error = d.get("error")

        # 从标签获取期望工具
        labels = label_map.get(qid, {})
        expected_tools = labels.get("expected_tools", d.get("expected_tools", []))
        unexpected_tools = d.get("unexpected_tools", [])
        question_type = labels.get("question_type", d.get("category", ""))
        should_refuse = labels.get("should_refuse", d.get("should_refuse", False))

        result = analyze_single(
            question_id=qid,
            question=question,
            expected_tools=expected_tools,
            unexpected_tools=unexpected_tools,
            question_type=question_type,
            should_refuse=should_refuse,
            actual_tools=actual_tools,
            tool_calls=tool_calls,
            trace_events=trace_events,
            refused=refused,
            timing=timing,
            error=error,
        )
        results.append(result)

    return results


def save_results(results: List[Dict], summary: Dict, output_path: Path) -> Path:
    """保存完整分析结果到 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_type": "agent_trace_analysis",
        "total_questions": len(results),
        "summary": summary,
        "details": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def main(argv: List[str] | None = None) -> int:
    """CLI 入口。"""
    args = argv if argv is not None else sys.argv[1:]

    offline = "--offline" in args
    base_url = AGENT_QUERY_URL_DEFAULT
    for i, arg in enumerate(args):
        if arg == "--base-url" and i + 1 < len(args):
            base_url = args[i + 1].rstrip("/") + "/api/agent/query"
        elif arg == "--timeout" and i + 1 < len(args):
            timeout_str = args[i + 1]

    timeout = 120

    print(f"模式: {'offline' if offline else 'live'}")
    print(f"API:  {base_url}")
    print()

    if offline:
        # offline 模式
        try:
            details = load_saved_eval()
        except (FileNotFoundError, ValueError) as e:
            print(f"错误：{e}", file=sys.stderr)
            print("请先运行 python eval/run_agent_eval.py 生成评估结果。", file=sys.stderr)
            return 1

        print(f"从 {SAVED_EVAL_PATH} 加载了 {len(details)} 条评估结果")
        results = run_offline_mode(details)
    else:
        # live 模式
        try:
            raw_questions = load_labeled_questions()
        except (FileNotFoundError, ValueError) as e:
            print(f"错误：{e}", file=sys.stderr)
            return 1

        flat_questions = [flatten_labeled_question(q) for q in raw_questions]
        print(f"加载了 {len(flat_questions)} 道标注题目")

        # 检查后端可用性
        try:
            requests.get(base_url.rsplit("/api/", 1)[0] + "/api/documents", timeout=5)
        except Exception:
            print("错误：后端不可用，请先启动后端或使用 --offline 模式。", file=sys.stderr)
            return 1

        results = run_live_mode(flat_questions, base_url, timeout=timeout)

    # 汇总
    summary = build_summary(results)

    # 打印报告
    print()
    print_report(results, summary)

    # 保存
    output_path = RESULTS_DIR / "agent_trace_analysis.json"
    save_results(results, summary, output_path)
    print(f"\n分析结果已保存：{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
