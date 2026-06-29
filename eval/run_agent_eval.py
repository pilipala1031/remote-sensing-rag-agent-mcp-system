"""Agent 研究助手评估脚本。

运行方式：
    python eval/run_agent_eval.py

流程：
1. 探测本地 FastAPI 后端是否可用；
2. 读取 eval/eval_questions.json；
3. 逐题调用 POST http://127.0.0.1:8000/api/agent/query，记录
   answer / sources / refused / tool_calls / agent_trace / latency /
   verification / timing / expected_tools / tool_hit /
   unexpected_tools / unexpected_tool_hit / answer_length /
   verification_elapsed；
4. 计算逐题指标与汇总指标（额外包含 tool_call_rate /
   tool_hit_rate_avg / unexpected_tool_rate_avg /
   verification_pass_rate / avg_agent_total_elapsed /
   avg_verification_elapsed / avg_tool_calls_count /
   avg_answer_length）；
5. 保存到 eval/results/agent_eval_result.json；
6. 控制台打印汇总结果。

不依赖 pytest，不引入复杂评估框架，仅通过 requests 调用本地 API。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

try:
    from metrics import (
        build_summary,
        check_backend,
        keyword_hit_rate,
        load_questions,
        print_summary,
        refusal_correct,
        save_result,
        source_hit_rate,
        tool_hit_rate,
        unexpected_tool_rate,
        AGENT_QUERY_URL,
    )
except ImportError:  # 以脚本方式直接运行时，metrics 与本文件同目录
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrics import (  # type: ignore
        build_summary,
        check_backend,
        keyword_hit_rate,
        load_questions,
        print_summary,
        refusal_correct,
        save_result,
        source_hit_rate,
        tool_hit_rate,
        unexpected_tool_rate,
        AGENT_QUERY_URL,
    )


def query_agent(question: str, timeout: int = 90) -> Dict[str, Any]:
    """调用 /api/agent/query，返回解析后的 JSON。

    后端响应结构：{answer, sources[], refused, tool_calls[], agent_trace[], errors[]}
    Agent 调用链路较长，超时设得比 RAG 更宽。
    """
    resp = requests.post(
        AGENT_QUERY_URL,
        json={"question": question},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def evaluate_one(item: Dict[str, Any]) -> Dict[str, Any]:
    """对单道题调用 Agent 接口并计算逐题指标。"""
    qid = item["id"]
    question = item["question"]
    expected_keywords = item.get("expected_keywords", [])
    expected_sources = item.get("expected_source_files", [])
    expected_tools = item.get("expected_tools", [])
    unexpected_tools = item.get("unexpected_tools", [])
    should_refuse = item.get("should_refuse", False)

    print(f"[{qid}] 提问：{question}")

    start = time.time()
    try:
        data = query_agent(question)
        error = None
    except Exception as e:  # noqa: BLE001
        data = {
            "answer": "", "sources": [], "refused": False,
            "tool_calls": [], "agent_trace": [], "trace_events": [],
            "errors": [], "verification": {}, "timing": {},
        }
        error = f"{type(e).__name__}: {e}"
    latency = time.time() - start

    answer = data.get("answer", "")
    sources = data.get("sources", []) or []
    refused = bool(data.get("refused", False))
    tool_calls = data.get("tool_calls", []) or []
    agent_trace = data.get("agent_trace", []) or []
    trace_events = data.get("trace_events", []) or []
    errors = data.get("errors", []) or []
    verification = data.get("verification", {}) or {}
    timing_data = data.get("timing", {}) or {}
    agent_total_elapsed = timing_data.get("total_elapsed", 0.0)

    # 修正：verification_elapsed 嵌套在 verification["timing"] 中，不在顶层 timing 中
    v_timing = verification.get("timing", {}) if isinstance(verification, dict) else {}
    verification_elapsed = v_timing.get("verification_elapsed", 0.0)

    kw_rate = keyword_hit_rate(answer, expected_keywords)
    src_rate = source_hit_rate(sources, expected_sources)
    refused_ok = refusal_correct(refused, should_refuse)
    tool_hit = tool_hit_rate(tool_calls, expected_tools)
    unexpected_hit = unexpected_tool_rate(tool_calls, unexpected_tools)
    answer_length = len(answer)

    detail = {
        "id": qid,
        "category": item.get("category", ""),
        "question": question,
        "should_refuse": should_refuse,
        "refused": refused,
        "refusal_correct": refused_ok,
        "answer": answer,
        "sources": sources,
        "sources_count": len(sources),
        "tool_calls": tool_calls,
        "tool_calls_count": len(tool_calls),
        "agent_trace": agent_trace,
        "trace_events": trace_events,
        "errors": errors,
        "expected_keyword_count": len(expected_keywords),
        "expected_source_count": len(expected_sources),
        "expected_tools": expected_tools,
        "unexpected_tools": unexpected_tools,
        "tool_hit": tool_hit,
        "unexpected_tool_hit": unexpected_hit,
        "answer_length": answer_length,
        "verification": verification,
        "agent_total_elapsed": round(agent_total_elapsed, 4),
        "verification_elapsed": round(verification_elapsed, 4),
        "keyword_hit_rate": round(kw_rate, 4),
        "source_hit_rate": round(src_rate, 4),
        "latency": round(latency, 4),
    }
    if error:
        detail["error"] = error

    status = "拒答" if refused else "已回答"
    flag = "✓" if refused_ok else "✗"
    tool_hit_str = (
        f"工具命中={tool_hit:.0f}" if tool_hit is not None else "工具命中=N/A"
    )
    unexpected_str = (
        f"误用={unexpected_hit:.0f}" if unexpected_hit is not None else "误用=N/A"
    )
    verified = verification.get("verified") if isinstance(verification, dict) else None
    verif_str = (
        f"证据={'✓' if verified else '✗'}"
        if verified is not None else "证据=N/A"
    )
    print(f"       -> {flag} {status} | kw={kw_rate:.2f} "
          f"src={src_rate:.2f} | {len(sources)}条来源 | "
          f"工具调用 {len(tool_calls)} 次 | {tool_hit_str} | "
          f"{unexpected_str} | {verif_str} | "
          f"答案{answer_length}字 | {latency:.2f}s")
    return detail


def main() -> None:
    check_backend(is_agent=True)
    questions = load_questions()
    print(f"已加载 {len(questions)} 道评估题目，开始调用 /api/agent/query ...\n")

    details: List[Dict[str, Any]] = []
    for item in questions:
        details.append(evaluate_one(item))

    summary = build_summary(details, include_tool_call=True)

    payload = {
        "eval_type": "agent",
        "api_endpoint": AGENT_QUERY_URL,
        "summary": summary,
        "details": details,
    }
    out_path = save_result(payload, "agent_eval_result.json")

    print_summary("Agent 研究助手", summary)
    print(f"  结果已保存：{out_path}\n")


if __name__ == "__main__":
    main()
