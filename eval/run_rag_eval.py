"""普通 RAG 评估脚本。

运行方式：
    python eval/run_rag_eval.py

流程：
1. 探测本地 FastAPI 后端是否可用；
2. 读取 eval/eval_questions.json；
3. 逐题调用 POST http://127.0.0.1:8000/api/chat/query，记录
   answer / sources / refused / latency；
4. 计算逐题指标与汇总指标；
5. 保存到 eval/results/rag_eval_result.json；
6. 控制台打印汇总结果。

不依赖 pytest，不引入复杂评估框架，仅通过 requests 调用本地 API。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

# 让脚本既可被 import（被测试）也可直接运行
try:
    from metrics import (
        average_sources_count,
        build_summary,
        check_backend,
        keyword_hit_rate,
        load_questions,
        print_summary,
        refusal_correct,
        save_result,
        source_hit_rate,
        RAG_QUERY_URL,
    )
except ImportError:  # 以脚本方式直接运行时，metrics 与本文件同目录
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrics import (  # type: ignore
        average_sources_count,
        build_summary,
        check_backend,
        keyword_hit_rate,
        load_questions,
        print_summary,
        refusal_correct,
        save_result,
        source_hit_rate,
        RAG_QUERY_URL,
    )


def query_rag(question: str, timeout: int = 60) -> Dict[str, Any]:
    """调用 /api/chat/query，返回解析后的 JSON。

    后端响应结构：{answer, sources[], refused}
    """
    resp = requests.post(
        RAG_QUERY_URL,
        json={"question": question},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def evaluate_one(item: Dict[str, Any]) -> Dict[str, Any]:
    """对单道题调用 RAG 接口并计算逐题指标。"""
    qid = item["id"]
    question = item["question"]
    expected_keywords = item.get("expected_keywords", [])
    expected_sources = item.get("expected_source_files", [])
    should_refuse = item.get("should_refuse", False)

    print(f"[{qid}] 提问：{question}")

    start = time.time()
    try:
        data = query_rag(question)
        error = None
    except Exception as e:  # noqa: BLE001
        # 单题失败不中断整体评估，记录为错误
        data = {"answer": "", "sources": [], "refused": False}
        error = f"{type(e).__name__}: {e}"
    latency = time.time() - start

    answer = data.get("answer", "")
    sources = data.get("sources", []) or []
    refused = bool(data.get("refused", False))

    kw_rate = keyword_hit_rate(answer, expected_keywords)
    src_rate = source_hit_rate(sources, expected_sources)
    refused_ok = refusal_correct(refused, should_refuse)

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
        "expected_keyword_count": len(expected_keywords),
        "expected_source_count": len(expected_sources),
        "keyword_hit_rate": round(kw_rate, 4),
        "source_hit_rate": round(src_rate, 4),
        "latency": round(latency, 4),
    }
    if error:
        detail["error"] = error

    status = "拒答" if refused else "已回答"
    flag = "✓" if refused_ok else "✗"
    print(f"       -> {flag} {status} | kw={kw_rate:.2f} "
          f"src={src_rate:.2f} | {len(sources)}条来源 | {latency:.2f}s")
    return detail


def main() -> None:
    check_backend(is_agent=False)
    questions = load_questions()
    print(f"已加载 {len(questions)} 道评估题目，开始调用 /api/chat/query ...\n")

    details: List[Dict[str, Any]] = []
    for item in questions:
        details.append(evaluate_one(item))

    summary = build_summary(details, include_tool_call=False)
    # 补充一个仅展示用的平均来源数（与 summary 中一致，便于阅读）
    _ = average_sources_count([d["sources_count"] for d in details])

    payload = {
        "eval_type": "rag",
        "api_endpoint": RAG_QUERY_URL,
        "summary": summary,
        "details": details,
    }
    out_path = save_result(payload, "rag_eval_result.json")

    print_summary("普通 RAG", summary)
    print(f"  结果已保存：{out_path}\n")


if __name__ == "__main__":
    main()
