"""三阶段 RAG Rerank 消融实验主脚本。

阶段一：Retrieval-level 对比（baseline vs rerank）
  - 固定 chunk_size=800, chunk_overlap=120, similarity_threshold=0.3
  - 只使用 18 个领域内问题（should_refuse=false）
  - 3 组配置：baseline / rerank_k10 / rerank_k20
  - retrieval-only，不调用 LLM
  - 输出 retrieval_score

阶段二：Out-of-scope 安全性分析
  - 全部 21 个问题（含 3 个 should_refuse=true）
  - 3 组配置同上
  - 检测 rerank 是否改变拒答行为
  - 输出 refusal_score

阶段三：Answer-level 最终验证
  - 只对比 best rerank vs baseline
  - 调用 LLM 生成回答
  - 输出 answer_score

CLI 用法：
    python -m experiments.rag_rerank_ablation.run_rerank_ablation
"""
from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

from experiments.eval_with_labels import (
    DEFAULT_LABELS_PATH,
    flatten_labeled_question,
    load_labeled_questions,
)
from experiments.rag_param_ablation.reingest_helper import (
    SAMPLE_DOCS_DIR,
    build_temp_store,
)
from experiments.rag_rerank_ablation.reranker import rerank_search_results

from app.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
CONFIGS_PATH = PACKAGE_DIR / "rerank_configs.json"
README_PATH = PACKAGE_DIR / "README_rerank_ablation.md"

# 检索时使用极低 threshold 以获取原始 top-K 结果，之后手动过滤
RAW_SEARCH_THRESHOLD = -1.0


# --------------------------------------------------------------------------- #
# 临时目录管理（Windows 兼容：Chroma sqlite3 文件锁定问题）
# --------------------------------------------------------------------------- #
@contextmanager
def ablation_temp_dir(prefix: str) -> Iterator[Path]:
    """创建临时目录，退出时尽力清理。

    Windows 上 Chroma PersistentClient 会锁定 sqlite3 文件，
    导致 TemporaryDirectory 清理失败。
    此上下文管理器在退出时先触发 GC 释放 Chroma 对象引用，
    再用 shutil.rmtree(ignore_errors=True) 清理。
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.info("创建临时目录: %s", tmpdir)
    try:
        yield tmpdir
    finally:
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)
        if tmpdir.exists():
            logger.warning("临时目录清理失败（Chroma 文件锁定），将保留: %s", tmpdir)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_configs() -> dict:
    """加载 rerank 配置 JSON。"""
    with open(CONFIGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_flat_questions() -> List[Dict[str, Any]]:
    """加载富标签问题并展平。"""
    raw = load_labeled_questions(DEFAULT_LABELS_PATH)
    return [flatten_labeled_question(q) for q in raw]


def save_json(data: Any, filename: str) -> Path:
    """保存 JSON 到 results/ 目录。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("已保存结果: %s", path)
    return path


def _match_source(expected: str, actual_filename: str) -> bool:
    """子串匹配：expected 是否出现在 actual_filename 中。"""
    return expected in actual_filename if expected and actual_filename else False


def _search_raw(store, query: str, top_k: int) -> tuple[List[dict], float]:
    """用极低 threshold 检索，返回原始 top-K 结果和耗时。"""
    t0 = time.perf_counter()
    results = store.search(
        query=query,
        top_k=top_k,
        similarity_threshold=RAW_SEARCH_THRESHOLD,
    )
    latency = time.perf_counter() - t0
    return results, latency


def _retrieve_with_config(
    store,
    query: str,
    config: dict,
    threshold: float,
    raw_results: List[dict] | None = None,
    raw_latency: float = 0.0,
) -> tuple[List[dict], float, bool]:
    """按配置执行检索（含可选 rerank），返回最终结果列表和总耗时。

    Args:
        store: 向量库实例。
        query: 查询文本。
        config: 配置 dict（含 use_rerank, candidate_k, final_top_k）。
        threshold: 相似度阈值（用于过滤）。
        raw_results: 预检索的原始结果（None 则实时检索）。
        raw_latency: 预检索耗时。

    Returns:
        (final_results, total_latency, used_fallback)
    """
    candidate_k = config["candidate_k"]
    final_top_k = config["final_top_k"]

    # 获取候选结果
    if raw_results is not None:
        candidates_raw = raw_results[:candidate_k]
        search_latency = raw_latency
    else:
        candidates_raw, search_latency = _search_raw(store, query, candidate_k)

    # 按阈值过滤
    filtered = [r for r in candidates_raw if r.get("score", 0.0) >= threshold]

    if not filtered:
        return [], search_latency, False

    # rerank 或直接取 top_k
    if config.get("use_rerank", False):
        reranked, rerank_elapsed, used_fallback = rerank_search_results(
            query=query,
            search_results=filtered,
            final_top_k=final_top_k,
        )
        return reranked, search_latency + rerank_elapsed, used_fallback
    else:
        return filtered[:final_top_k], search_latency, False


# --------------------------------------------------------------------------- #
# 阶段一：Retrieval-level 对比
# --------------------------------------------------------------------------- #
def _calc_retrieval_metrics(
    q: dict, final_results: List[dict], latency: float, used_fallback: bool,
    config: dict,
) -> dict:
    """计算单题 retrieval 指标。"""
    expected_sources = q.get("expected_source_files", [])
    retrieved_filenames = [r.get("filename", "") for r in final_results]

    # source_hit: 是否命中任一期望来源
    source_hit = 0
    for exp in expected_sources:
        if any(_match_source(exp, fn) for fn in retrieved_filenames):
            source_hit = 1
            break

    # source_recall_at_k
    if expected_sources:
        found = sum(
            1 for exp in expected_sources
            if any(_match_source(exp, fn) for fn in retrieved_filenames)
        )
        recall = found / len(expected_sources)
    else:
        recall = 1.0

    # MRR
    mrr = 0.0
    for rank, fn in enumerate(retrieved_filenames, 1):
        if any(_match_source(exp, fn) for exp in expected_sources):
            mrr = 1.0 / rank
            break

    scores = [r.get("score", 0.0) for r in final_results]
    top_score = scores[0] if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # rerank_score（仅 rerank 配置有）
    rerank_scores = [r.get("rerank_score", 0.0) for r in final_results]
    avg_rerank_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0

    return {
        "question_id": q["id"],
        "question": q["question"],
        "config": config["name"],
        "use_rerank": config.get("use_rerank", False),
        "expected_source_files": expected_sources,
        "retrieved_sources": retrieved_filenames,
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "avg_rerank_score": round(avg_rerank_score, 4),
        "source_hit": source_hit,
        "source_recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "retrieved_count": len(final_results),
        "latency": round(latency, 4),
        "used_fallback": used_fallback,
    }


def _aggregate_retrieval(details: List[dict]) -> dict:
    """汇总单组配置的 retrieval 指标。"""
    n = len(details)
    if n == 0:
        return {}
    fallback_count = sum(1 for d in details if d.get("used_fallback", False))
    return {
        "source_hit_rate": round(sum(d["source_hit"] for d in details) / n, 4),
        "source_recall_at_k": round(sum(d["source_recall_at_k"] for d in details) / n, 4),
        "mrr": round(sum(d["mrr"] for d in details) / n, 4),
        "avg_top_score": round(sum(d["top_score"] for d in details) / n, 4),
        "avg_score": round(sum(d["avg_score"] for d in details) / n, 4),
        "avg_rerank_score": round(sum(d["avg_rerank_score"] for d in details) / n, 4),
        "avg_retrieved_count": round(sum(d["retrieved_count"] for d in details) / n, 4),
        "avg_latency": round(sum(d["latency"] for d in details) / n, 4),
        "fallback_count": fallback_count,
        "total_questions": n,
    }


def _calc_retrieval_score(summary: dict, latency_norm: float) -> float:
    """计算 retrieval_score（与 Block 2 公式一致）。

    retrieval_score =
        0.45 * source_hit_rate
      + 0.25 * source_recall_at_k
      + 0.15 * mrr
      + 0.10 * avg_top_score
      - 0.05 * latency_norm
    """
    score = (
        0.45 * summary["source_hit_rate"]
        + 0.25 * summary["source_recall_at_k"]
        + 0.15 * summary["mrr"]
        + 0.10 * summary["avg_top_score"]
        - 0.05 * latency_norm
    )
    return round(score, 4)


def run_stage1_retrieval(
    in_scope_questions: List[dict],
    configs: List[dict],
    fixed_threshold: float,
    top_k: int,
) -> dict:
    """阶段一：Retrieval-level 对比实验。"""
    print("\n" + "=" * 60)
    print("  阶段一：Retrieval-level 对比 (threshold=0.3, retrieval-only)")
    print("=" * 60)

    max_candidate_k = max(cfg["candidate_k"] for cfg in configs)

    all_configs: Dict[str, dict] = {}
    config_latencies: Dict[str, float] = {}

    with ablation_temp_dir("rerank_s1_") as tmpdir:
        store = build_temp_store(
            chunk_size=800,
            chunk_overlap=120,
            source_dir=SAMPLE_DOCS_DIR,
            persist_dir=Path(tmpdir),
            collection_name="rerank_stage1",
        )

        # 预检索所有问题（取 max_candidate_k 条原始结果）
        raw_cache: Dict[str, tuple[List[dict], float]] = {}
        for q in in_scope_questions:
            raw_results, latency = _search_raw(store, q["question"], max_candidate_k)
            raw_cache[q["id"]] = (raw_results, latency)

        # 遍历配置
        for cfg in configs:
            name = cfg["name"]
            print(f"\n  [{name}] use_rerank={cfg.get('use_rerank', False)}, "
                  f"candidate_k={cfg['candidate_k']}, final_top_k={cfg['final_top_k']}")

            details = []
            for q in in_scope_questions:
                raw_results, raw_latency = raw_cache[q["id"]]
                final_results, total_latency, used_fallback = _retrieve_with_config(
                    store=store,
                    query=q["question"],
                    config=cfg,
                    threshold=fixed_threshold,
                    raw_results=raw_results,
                    raw_latency=raw_latency,
                )
                detail = _calc_retrieval_metrics(
                    q, final_results, total_latency, used_fallback, cfg,
                )
                details.append(detail)

            summary = _aggregate_retrieval(details)
            all_configs[name] = {
                "config": cfg,
                "details": details,
                "summary": summary,
            }
            config_latencies[name] = summary["avg_latency"]

            fb_str = f", fallback={summary['fallback_count']}" if summary.get("fallback_count", 0) else ""
            print(f"    source_hit={summary['source_hit_rate']:.4f}, "
                  f"recall={summary['source_recall_at_k']:.4f}, "
                  f"mrr={summary['mrr']:.4f}, "
                  f"avg_latency={summary['avg_latency']:.4f}s{fb_str}")

    # latency 归一化
    latencies = list(config_latencies.values())
    lat_min = min(latencies) if latencies else 0.0
    lat_max = max(latencies) if latencies else 1.0
    lat_range = lat_max - lat_min

    for name, data in all_configs.items():
        lat = config_latencies[name]
        latency_norm = (lat - lat_min) / lat_range if lat_range > 0 else 0.0
        data["summary"]["latency_norm"] = round(latency_norm, 4)
        data["summary"]["retrieval_score"] = _calc_retrieval_score(
            data["summary"], latency_norm,
        )
        print(f"  [{name}] retrieval_score={data['summary']['retrieval_score']:.4f}")

    # 选择最佳 rerank 配置
    rerank_configs = {
        k: v for k, v in all_configs.items()
        if v["config"].get("use_rerank", False)
    }
    best_rerank_name = max(
        rerank_configs,
        key=lambda k: rerank_configs[k]["summary"]["retrieval_score"],
    ) if rerank_configs else None

    if best_rerank_name:
        best_rerank_config = all_configs[best_rerank_name]["config"]
        best_rerank_score = all_configs[best_rerank_name]["summary"]["retrieval_score"]
        baseline_score = all_configs.get("baseline", {}).get("summary", {}).get("retrieval_score", 0)
        print(f"\n  >>> 最佳 rerank 配置: {best_rerank_name} "
              f"(score={best_rerank_score:.4f} vs baseline={baseline_score:.4f})")
    else:
        best_rerank_config = None
        print("\n  >>> 无可用 rerank 配置")

    return {
        "stage": "retrieval_level",
        "configs": all_configs,
        "best_rerank_config": best_rerank_config,
        "best_rerank_name": best_rerank_name,
    }


# --------------------------------------------------------------------------- #
# 阶段二：Out-of-scope 安全性分析
# --------------------------------------------------------------------------- #
def run_stage2_out_of_scope(
    all_questions: List[dict],
    configs: List[dict],
    fixed_threshold: float,
    top_k: int,
) -> dict:
    """阶段二：检测 rerank 是否影响拒答行为。"""
    print("\n" + "=" * 60)
    print("  阶段二：Out-of-scope 安全性分析 (retrieval-only)")
    print("=" * 60)

    in_scope = [q for q in all_questions if not q.get("should_refuse", False)]
    out_scope = [q for q in all_questions if q.get("should_refuse", False)]
    n_in = len(in_scope)
    n_out = len(out_scope)
    print(f"  领域内 {n_in} 题, 领域外 {n_out} 题")

    max_candidate_k = max(cfg["candidate_k"] for cfg in configs)

    all_configs: Dict[str, dict] = {}

    with ablation_temp_dir("rerank_s2_") as tmpdir:
        store = build_temp_store(
            chunk_size=800,
            chunk_overlap=120,
            source_dir=SAMPLE_DOCS_DIR,
            persist_dir=Path(tmpdir),
            collection_name="rerank_stage2",
        )

        # 预检索
        raw_cache: Dict[str, tuple[List[dict], float]] = {}
        for q in all_questions:
            raw_results, latency = _search_raw(store, q["question"], max_candidate_k)
            raw_cache[q["id"]] = (raw_results, latency)

        for cfg in configs:
            name = cfg["name"]
            details = []

            for q in all_questions:
                raw_results, raw_latency = raw_cache[q["id"]]
                final_results, total_latency, used_fallback = _retrieve_with_config(
                    store=store,
                    query=q["question"],
                    config=cfg,
                    threshold=fixed_threshold,
                    raw_results=raw_results,
                    raw_latency=raw_latency,
                )

                retrieved_count = len(final_results)
                scores = [r.get("score", 0.0) for r in final_results]
                max_score = max(scores) if scores else 0.0
                refused = retrieved_count == 0
                should_refuse = q.get("should_refuse", False)
                false_refusal = 1 if (not should_refuse and refused) else 0
                false_accept = 1 if (should_refuse and not refused) else 0

                details.append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "should_refuse": should_refuse,
                    "config": name,
                    "max_score": round(max_score, 4),
                    "retrieved_count": retrieved_count,
                    "refused_by_retrieval": refused,
                    "false_refusal": false_refusal,
                    "false_accept": false_accept,
                    "used_fallback": used_fallback,
                })

            # 汇总
            in_not_refused = sum(
                1 for d in details
                if not d["should_refuse"] and not d["refused_by_retrieval"]
            )
            out_refused = sum(
                1 for d in details
                if d["should_refuse"] and d["refused_by_retrieval"]
            )
            false_refusals = sum(d["false_refusal"] for d in details)
            false_accepts = sum(d["false_accept"] for d in details)
            all_max_scores = [d["max_score"] for d in details]
            all_counts = [d["retrieved_count"] for d in details]
            n_total = len(details)

            in_scope_recall = in_not_refused / n_in if n_in > 0 else 0.0
            out_refusal_acc = out_refused / n_out if n_out > 0 else 0.0
            false_refusal_rate = false_refusals / n_in if n_in > 0 else 0.0
            false_accept_rate = false_accepts / n_out if n_out > 0 else 0.0

            refusal_score = (
                0.35 * in_scope_recall
                + 0.45 * out_refusal_acc
                - 0.05 * false_refusal_rate
                - 0.15 * false_accept_rate
            )

            summary = {
                "config": name,
                "in_scope_recall": round(in_scope_recall, 4),
                "out_of_scope_refusal_accuracy": round(out_refusal_acc, 4),
                "false_refusal_rate": round(false_refusal_rate, 4),
                "false_accept_rate": round(false_accept_rate, 4),
                "avg_max_score": round(sum(all_max_scores) / n_total, 4) if n_total else 0.0,
                "avg_retrieved_count": round(sum(all_counts) / n_total, 4) if n_total else 0.0,
                "refusal_score": round(refusal_score, 4),
            }
            all_configs[name] = {
                "config": cfg,
                "details": details,
                "summary": summary,
            }

            print(f"  [{name}] in_recall={summary['in_scope_recall']:.4f}, "
                  f"out_refusal={summary['out_of_scope_refusal_accuracy']:.4f}, "
                  f"false_refusal={summary['false_refusal_rate']:.4f}, "
                  f"false_accept={summary['false_accept_rate']:.4f}, "
                  f"refusal_score={summary['refusal_score']:.4f}")

    return {
        "stage": "out_of_scope_safety",
        "configs": all_configs,
    }


# --------------------------------------------------------------------------- #
# 阶段三：Answer-level 最终验证
# --------------------------------------------------------------------------- #
def _run_baseline_answers(
    store,
    questions: List[dict],
    threshold: float,
    top_k: int,
) -> dict:
    """Baseline 配置：通过 RAGService 获取 LLM 回答（与 Block 2 一致）。"""
    from app.services.retriever import Retriever
    from app.services.rag_service import RAGService

    print(f"\n  [baseline] 纯向量检索 → LLM")

    retriever = Retriever(store=store)
    rag = RAGService(retriever=retriever)

    details = []
    for q in questions:
        expected_keywords = q.get("expected_keywords", [])
        expected_sources = q.get("expected_source_files", [])
        should_refuse = q.get("should_refuse", False)
        min_len = q.get("min_answer_length", 0)

        t0 = time.perf_counter()
        try:
            answer_obj = rag.answer(
                question=q["question"],
                top_k=top_k,
                similarity_threshold=threshold,
            )
        except Exception as e:
            logger.error("[baseline] RAG 调用失败 (q=%s): %s", q["id"], e)
            answer_obj = None
        latency = time.perf_counter() - t0

        if answer_obj is None:
            details.append({
                "question_id": q["id"], "question": q["question"],
                "answer": "(调用失败)", "sources": [],
                "keyword_coverage": 0.0, "source_hit": 0,
                "refusal_accuracy": 0, "min_length_satisfied": 0,
                "latency": round(latency, 4), "error": True,
            })
            continue

        answer_text = answer_obj.answer or ""
        sources = answer_obj.sources or []
        refused = answer_obj.refused

        if expected_keywords:
            lower_ans = answer_text.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in lower_ans)
            kw_cov = hits / len(expected_keywords)
        else:
            kw_cov = 1.0

        source_filenames = [s.filename for s in sources]
        src_hit = 0
        for exp in expected_sources:
            if any(_match_source(exp, fn) for fn in source_filenames):
                src_hit = 1
                break

        refusal_acc = 1 if (refused == should_refuse) else 0
        min_len_ok = 1 if len(answer_text) >= min_len else 0

        details.append({
            "question_id": q["id"], "question": q["question"],
            "answer": answer_text[:500], "sources": source_filenames,
            "keyword_coverage": round(kw_cov, 4), "source_hit": src_hit,
            "refusal_accuracy": refusal_acc, "min_length_satisfied": min_len_ok,
            "latency": round(latency, 4),
        })

    n = len(details)
    summary = {
        "keyword_coverage_avg": round(sum(d["keyword_coverage"] for d in details) / n, 4) if n else 0.0,
        "source_hit_rate": round(sum(d["source_hit"] for d in details) / n, 4) if n else 0.0,
        "refusal_accuracy": round(sum(d["refusal_accuracy"] for d in details) / n, 4) if n else 0.0,
        "min_length_satisfied_rate": round(sum(d["min_length_satisfied"] for d in details) / n, 4) if n else 0.0,
        "avg_latency": round(sum(d["latency"] for d in details) / n, 4) if n else 0.0,
    }
    summary["answer_score"] = round(
        0.50 * summary["keyword_coverage_avg"]
        + 0.25 * summary["source_hit_rate"]
        + 0.15 * summary["refusal_accuracy"]
        + 0.10 * summary["min_length_satisfied_rate"],
        4,
    )

    print(f"  [baseline] keyword_cov={summary['keyword_coverage_avg']:.4f}, "
          f"source_hit={summary['source_hit_rate']:.4f}, "
          f"refusal_acc={summary['refusal_accuracy']:.4f}, "
          f"answer_score={summary['answer_score']:.4f}")

    return {"label": "baseline", "details": details, "summary": summary}


def _run_rerank_answers(
    store,
    questions: List[dict],
    config: dict,
    threshold: float,
    top_k: int,
) -> dict:
    """Rerank 配置：向量检索 candidate_k → rerank → top_k → LLM。"""
    from app.core.llm import OpenAICompatibleLLMClient
    from app.core.prompts import RAG_SYSTEM_PROMPT, REFUSAL_ANSWER

    candidate_k = config["candidate_k"]
    final_top_k = config["final_top_k"]
    label = config["name"]

    print(f"\n  [{label}] 向量检索 candidate_k={candidate_k} → rerank → top_k={final_top_k} → LLM")

    llm = OpenAICompatibleLLMClient()

    details = []
    for q in questions:
        expected_keywords = q.get("expected_keywords", [])
        expected_sources = q.get("expected_source_files", [])
        should_refuse = q.get("should_refuse", False)
        min_len = q.get("min_answer_length", 0)

        t0 = time.perf_counter()
        try:
            # 1. 向量检索 candidate_k
            raw_results, search_latency = _search_raw(store, q["question"], candidate_k)

            # 2. 按阈值过滤
            filtered = [r for r in raw_results if r.get("score", 0.0) >= threshold]

            # 3. 拒答判断
            if not filtered:
                latency = time.perf_counter() - t0
                details.append({
                    "question_id": q["id"], "question": q["question"],
                    "answer": REFUSAL_ANSWER, "sources": [],
                    "keyword_coverage": 1.0 if should_refuse else 0.0,
                    "source_hit": 0,
                    "refusal_accuracy": 1 if should_refuse else 0,
                    "min_length_satisfied": 1 if len(REFUSAL_ANSWER) >= min_len else 0,
                    "latency": round(latency, 4), "refused": True,
                })
                continue

            # 4. Rerank
            reranked, rerank_elapsed, used_fallback = rerank_search_results(
                query=q["question"],
                search_results=filtered,
                final_top_k=final_top_k,
            )

            # 5. 拼接上下文（复用 RAGService._build_context 逻辑）
            context_blocks = []
            for i, h in enumerate(reranked, start=1):
                context_blocks.append(
                    f"[{i}] 来源：{h.get('filename', '?')}，"
                    f"第{h.get('page', 1)}页，"
                    f"chunk_id={h.get('chunk_id', '?')}\n{h.get('content', '')}"
                )
            context = "\n\n".join(context_blocks)

            # 6. 调用 LLM
            prompt = RAG_SYSTEM_PROMPT.format(context=context, question=q["question"])
            try:
                answer_text = llm.chat(prompt).strip()
            except Exception as e:
                logger.error("[%s] LLM 生成失败 (q=%s): %s", label, q["id"], e)
                latency = time.perf_counter() - t0
                details.append({
                    "question_id": q["id"], "question": q["question"],
                    "answer": REFUSAL_ANSWER, "sources": [],
                    "keyword_coverage": 1.0 if should_refuse else 0.0,
                    "source_hit": 0,
                    "refusal_accuracy": 1 if should_refuse else 0,
                    "min_length_satisfied": 0,
                    "latency": round(latency, 4), "error": True,
                })
                continue

            latency = time.perf_counter() - t0
            source_filenames = [r.get("filename", "unknown") for r in reranked]

        except Exception as e:
            logger.error("[%s] 整体调用失败 (q=%s): %s", label, q["id"], e)
            latency = time.perf_counter() - t0
            details.append({
                "question_id": q["id"], "question": q["question"],
                "answer": "(调用失败)", "sources": [],
                "keyword_coverage": 0.0, "source_hit": 0,
                "refusal_accuracy": 0, "min_length_satisfied": 0,
                "latency": round(latency, 4), "error": True,
            })
            continue

        # 计算指标
        if expected_keywords:
            lower_ans = answer_text.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in lower_ans)
            kw_cov = hits / len(expected_keywords)
        else:
            kw_cov = 1.0

        src_hit = 0
        for exp in expected_sources:
            if any(_match_source(exp, fn) for fn in source_filenames):
                src_hit = 1
                break

        refusal_acc = 1 if (False == should_refuse) else 0  # not refused

        min_len_ok = 1 if len(answer_text) >= min_len else 0

        details.append({
            "question_id": q["id"], "question": q["question"],
            "answer": answer_text[:500], "sources": source_filenames,
            "keyword_coverage": round(kw_cov, 4), "source_hit": src_hit,
            "refusal_accuracy": refusal_acc, "min_length_satisfied": min_len_ok,
            "latency": round(latency, 4), "used_fallback": used_fallback,
        })

    n = len(details)
    summary = {
        "keyword_coverage_avg": round(sum(d["keyword_coverage"] for d in details) / n, 4) if n else 0.0,
        "source_hit_rate": round(sum(d["source_hit"] for d in details) / n, 4) if n else 0.0,
        "refusal_accuracy": round(sum(d["refusal_accuracy"] for d in details) / n, 4) if n else 0.0,
        "min_length_satisfied_rate": round(sum(d["min_length_satisfied"] for d in details) / n, 4) if n else 0.0,
        "avg_latency": round(sum(d["latency"] for d in details) / n, 4) if n else 0.0,
    }
    summary["answer_score"] = round(
        0.50 * summary["keyword_coverage_avg"]
        + 0.25 * summary["source_hit_rate"]
        + 0.15 * summary["refusal_accuracy"]
        + 0.10 * summary["min_length_satisfied_rate"],
        4,
    )

    fb_count = sum(1 for d in details if d.get("used_fallback", False))
    print(f"  [{label}] keyword_cov={summary['keyword_coverage_avg']:.4f}, "
          f"source_hit={summary['source_hit_rate']:.4f}, "
          f"refusal_acc={summary['refusal_accuracy']:.4f}, "
          f"answer_score={summary['answer_score']:.4f}"
          + (f", fallback={fb_count}" if fb_count else ""))

    return {"label": label, "details": details, "summary": summary,
            "config": config}


def run_stage3_answer_validation(
    all_questions: List[dict],
    best_rerank_config: dict | None,
    fixed_threshold: float,
    top_k: int,
) -> dict:
    """阶段三：Answer-level 最终验证（best rerank vs baseline）。"""
    print("\n" + "=" * 60)
    print("  阶段三：Answer-level 最终验证 (LLM)")
    print("=" * 60)

    if best_rerank_config is None:
        print("  ⚠️ 无可用 rerank 配置，跳过阶段三")
        return {
            "stage": "answer_validation",
            "baseline": None,
            "rerank": None,
            "skipped": True,
            "reason": "无可用 rerank 配置（阶段一无结果或全部 fallback）",
        }

    with ablation_temp_dir("rerank_s3_") as tmpdir:
        store = build_temp_store(
            chunk_size=800,
            chunk_overlap=120,
            source_dir=SAMPLE_DOCS_DIR,
            persist_dir=Path(tmpdir),
            collection_name="rerank_stage3",
        )

        baseline_result = _run_baseline_answers(
            store=store,
            questions=all_questions,
            threshold=fixed_threshold,
            top_k=top_k,
        )

        rerank_result = _run_rerank_answers(
            store=store,
            questions=all_questions,
            config=best_rerank_config,
            threshold=fixed_threshold,
            top_k=top_k,
        )

    return {
        "stage": "answer_validation",
        "baseline": baseline_result,
        "rerank": rerank_result,
        "best_rerank_config": best_rerank_config,
    }


# --------------------------------------------------------------------------- #
# Markdown 报告生成
# --------------------------------------------------------------------------- #
def generate_readme(
    stage1: dict,
    stage2: dict,
    stage3: dict,
    configs: List[dict],
    fixed_params: dict,
    actually_ran: bool,
    error_msg: str = "",
) -> str:
    """生成 README_rerank_ablation.md 报告。"""
    lines: List[str] = []
    lines.append("# RAG Rerank 消融实验\n")

    if not actually_ran:
        lines.append("> **⚠️ 实验未成功完成，以下为框架占位，无实际数据。**\n")
        lines.append(f"> 失败原因：{error_msg}\n")
        lines.append("> 请解决问题后重新运行 `python -m experiments.rag_rerank_ablation.run_rerank_ablation`\n")

    # 1. 实验目的
    lines.append("## 1. 实验目的\n")
    lines.append(
        "本实验验证 SiliconFlow `BAAI/bge-reranker-v2-m3` rerank 模型"
        "是否能提升 RAG 系统的检索精度和最终回答质量。\n"
        "通过三阶段分层评估（Retrieval → Out-of-scope safety → Answer），"
        "量化 rerank 相比纯向量检索的增益与代价。\n"
    )

    # 2. 为什么需要 rerank
    lines.append("## 2. 为什么需要 rerank\n")
    lines.append("- **向量检索的局限**：bi-encoder（如 bge-m3）将 query 和 document 独立编码，"
                 "通过余弦相似度排序，速度快但精度有限，容易在 top-K 中混入语义表面相似但实际无关的 chunk。\n")
    lines.append("- **Rerank 的优势**：cross-encoder（如 bge-reranker-v2-m3）将 query 和 document "
                 "拼接后联合编码，能捕捉更细粒度的语义交互，精度更高。\n")
    lines.append("- **代价**：rerank 是逐对计算，延迟高于向量检索。本实验测量其精度增益是否值得延迟代价。\n")

    # 3. 实验设置
    lines.append("## 3. 实验设置\n")
    lines.append(f"- 固定参数：chunk_size={fixed_params['chunk_size']}, "
                 f"chunk_overlap={fixed_params['chunk_overlap']}, "
                 f"similarity_threshold={fixed_params['similarity_threshold']}, "
                 f"top_k={fixed_params['top_k']}\n")
    lines.append("- 模型：BAAI/bge-reranker-v2-m3（SiliconFlow API）\n")
    lines.append("- 评估集：21 题（18 领域内 + 3 领域外）\n")
    lines.append("\n配置列表：\n")
    lines.append("| 配置 | use_rerank | candidate_k | final_top_k | 说明 |\n")
    lines.append("|------|-----------|-------------|-------------|------|\n")
    for cfg in configs:
        lines.append(
            f"| {cfg['name']} | {cfg.get('use_rerank', False)} | "
            f"{cfg['candidate_k']} | {cfg['final_top_k']} | "
            f"{cfg.get('description', '')} |\n"
        )

    # 4. 指标设计
    lines.append("\n## 4. 指标设计\n")
    lines.append("本实验分为三层评估：\n")
    lines.append("| 层级 | 用途 | 主要指标 | 综合分数 |\n")
    lines.append("|------|------|----------|----------|\n")
    lines.append("| Retrieval-level | 对比检索精度 | source_hit_rate, source_recall_at_k, mrr, avg_top_score | retrieval_score |\n")
    lines.append("| Out-of-scope safety | 检测拒答行为变化 | in_scope_recall, out_refusal_acc, false_refusal_rate, false_accept_rate | refusal_score |\n")
    lines.append("| Answer-level | 最终回答质量 | keyword_coverage, source_hit_rate, refusal_accuracy, min_length_satisfied | answer_score |\n")
    lines.append("\n**Scoring 公式**：\n")
    lines.append("```\n")
    lines.append("retrieval_score = 0.45*source_hit_rate + 0.25*source_recall_at_k + 0.15*mrr + 0.10*avg_top_score - 0.05*latency_norm\n")
    lines.append("refusal_score   = 0.35*in_scope_recall + 0.45*out_refusal_acc - 0.05*false_refusal_rate - 0.15*false_accept_rate\n")
    lines.append("answer_score    = 0.50*keyword_coverage + 0.25*source_hit_rate + 0.15*refusal_accuracy + 0.10*min_length_satisfied\n")
    lines.append("```\n")

    # 5. Stage 1 结果
    lines.append("## 5. Retrieval-level 结果\n")
    if actually_ran and stage1.get("configs"):
        s1_configs = stage1["configs"]
        lines.append("| 配置 | source_hit_rate | source_recall_at_k | mrr | avg_top_score | avg_latency | retrieval_score |\n")
        lines.append("|------|-----------------|---------------------|-----|---------------|-------------|-----------------|\n")
        for name in sorted(s1_configs.keys()):
            s = s1_configs[name]["summary"]
            fb = s.get("fallback_count", 0)
            fb_note = f" ⚠{fb}fb" if fb else ""
            lines.append(
                f"| {name}{fb_note} | {s['source_hit_rate']:.4f} | "
                f"{s['source_recall_at_k']:.4f} | {s['mrr']:.4f} | "
                f"{s['avg_top_score']:.4f} | {s['avg_latency']:.4f}s | "
                f"{s['retrieval_score']:.4f} |\n"
            )

        # 对比分析
        baseline_s = s1_configs.get("baseline", {}).get("summary", {})
        best_name = stage1.get("best_rerank_name", "")
        if best_name and best_name in s1_configs:
            best_s = s1_configs[best_name]["summary"]
            delta = best_s["retrieval_score"] - baseline_s.get("retrieval_score", 0)
            delta_hit = best_s["source_hit_rate"] - baseline_s.get("source_hit_rate", 0)
            delta_mrr = best_s["mrr"] - baseline_s.get("mrr", 0)
            lines.append(f"\n**最佳 rerank 配置**：`{best_name}`\n")
            lines.append(f"- retrieval_score 差异：{delta:+.4f} "
                         f"(rerank={best_s['retrieval_score']:.4f} vs baseline={baseline_s.get('retrieval_score', 0):.4f})\n")
            lines.append(f"- source_hit_rate 差异：{delta_hit:+.4f}\n")
            lines.append(f"- mrr 差异：{delta_mrr:+.4f}\n")
            lat_ratio = best_s["avg_latency"] / baseline_s["avg_latency"] if baseline_s.get("avg_latency", 0) > 0 else 0
            lines.append(f"- 延迟倍数：{lat_ratio:.2f}x "
                         f"(rerank={best_s['avg_latency']:.4f}s vs baseline={baseline_s.get('avg_latency', 0):.4f}s)\n")
    else:
        lines.append("（实验未运行，无数据）\n")

    # 6. Stage 2 结果
    lines.append("\n## 6. Out-of-scope 安全性分析\n")
    if actually_ran and stage2.get("configs"):
        s2_configs = stage2["configs"]
        lines.append("| 配置 | in_scope_recall | out_refusal_acc | false_refusal_rate | false_accept_rate | refusal_score |\n")
        lines.append("|------|-----------------|-----------------|--------------------|-------------------|---------------|\n")
        for name in sorted(s2_configs.keys()):
            s = s2_configs[name]["summary"]
            lines.append(
                f"| {name} | {s['in_scope_recall']:.4f} | "
                f"{s['out_of_scope_refusal_accuracy']:.4f} | "
                f"{s['false_refusal_rate']:.4f} | "
                f"{s['false_accept_rate']:.4f} | "
                f"{s['refusal_score']:.4f} |\n"
            )
        lines.append(
            "\n由于 similarity_threshold 过滤发生在 rerank 之前，"
            "rerank 理论上不改变拒答行为。但如果 candidate_k > top_k，"
            "rerank 配置可能从更大的候选池中找到通过阈值的结果，"
            "从而降低 false_refusal_rate（代价是可能增加 false_accept_rate）。\n"
        )
    else:
        lines.append("（实验未运行，无数据）\n")

    # 7. Stage 3 结果
    lines.append("\n## 7. Answer-level 最终验证\n")
    if actually_ran and stage3 and not stage3.get("skipped"):
        baseline = stage3.get("baseline", {}).get("summary", {})
        rerank = stage3.get("rerank", {}).get("summary", {})
        best_cfg = stage3.get("best_rerank_config", {})
        lines.append(f"对比配置：baseline vs {best_cfg.get('name', '?')}\n\n")
        lines.append("| 配置 | keyword_coverage_avg | source_hit_rate | refusal_accuracy | min_length_satisfied_rate | avg_latency | answer_score |\n")
        lines.append("|------|---------------------|-----------------|------------------|--------------------------|-------------|--------------|\n")
        lines.append(
            f"| baseline | {baseline.get('keyword_coverage_avg', 0):.4f} | "
            f"{baseline.get('source_hit_rate', 0):.4f} | "
            f"{baseline.get('refusal_accuracy', 0):.4f} | "
            f"{baseline.get('min_length_satisfied_rate', 0):.4f} | "
            f"{baseline.get('avg_latency', 0):.4f}s | "
            f"{baseline.get('answer_score', 0):.4f} |\n"
        )
        lines.append(
            f"| {best_cfg.get('name', '?')} | {rerank.get('keyword_coverage_avg', 0):.4f} | "
            f"{rerank.get('source_hit_rate', 0):.4f} | "
            f"{rerank.get('refusal_accuracy', 0):.4f} | "
            f"{rerank.get('min_length_satisfied_rate', 0):.4f} | "
            f"{rerank.get('avg_latency', 0):.4f}s | "
            f"{rerank.get('answer_score', 0):.4f} |\n"
        )

        delta_score = rerank.get("answer_score", 0) - baseline.get("answer_score", 0)
        delta_kw = rerank.get("keyword_coverage_avg", 0) - baseline.get("keyword_coverage_avg", 0)
        lines.append(f"\n**answer_score 差异**：{delta_score:+.4f} "
                     f"(rerank={rerank.get('answer_score', 0):.4f} vs baseline={baseline.get('answer_score', 0):.4f})\n")
        lines.append(f"**keyword_coverage 差异**：{delta_kw:+.4f}\n")
    elif actually_ran and stage3 and stage3.get("skipped"):
        lines.append(f"阶段三已跳过：{stage3.get('reason', '未知原因')}\n")
    else:
        lines.append("（实验未运行，无数据）\n")

    # 8. 结论与建议
    lines.append("\n## 8. 结论与建议\n")
    if actually_ran:
        s1_configs = stage1.get("configs", {})
        baseline_s1 = s1_configs.get("baseline", {}).get("summary", {})
        best_name = stage1.get("best_rerank_name", "")
        best_s1 = s1_configs.get(best_name, {}).get("summary", {}) if best_name else {}

        if best_s1 and baseline_s1:
            delta = best_s1.get("retrieval_score", 0) - baseline_s1.get("retrieval_score", 0)
            if delta > 0.01:
                lines.append(
                    f"- **Retrieval-level**：rerank (`{best_name}`) 相比 baseline "
                    f"retrieval_score 提升 {delta:+.4f}，"
                    f"建议在检索精度敏感的场景引入 rerank。\n"
                )
            elif delta > 0:
                lines.append(
                    f"- **Retrieval-level**：rerank (`{best_name}`) 相比 baseline "
                    f"retrieval_score 仅提升 {delta:+.4f}，增益不显著。\n"
                )
            else:
                lines.append(
                    f"- **Retrieval-level**：rerank (`{best_name}`) 相比 baseline "
                    f"retrieval_score 下降 {delta:+.4f}，在本评估集上未带来改善。\n"
                )

        if stage3 and not stage3.get("skipped"):
            baseline_as = stage3.get("baseline", {}).get("summary", {}).get("answer_score", 0)
            rerank_as = stage3.get("rerank", {}).get("summary", {}).get("answer_score", 0)
            delta_a = rerank_as - baseline_as
            if delta_a > 0.01:
                lines.append(
                    f"- **Answer-level**：rerank answer_score 提升 {delta_a:+.4f}，"
                    f"端到端回答质量有改善。\n"
                )
            elif delta_a > 0:
                lines.append(
                    f"- **Answer-level**：rerank answer_score 仅提升 {delta_a:+.4f}，"
                    f"端到端改善有限。\n"
                )
            else:
                lines.append(
                    f"- **Answer-level**：rerank answer_score 下降 {delta_a:+.4f}，"
                    f"端到端回答质量未改善。\n"
                )

        lines.append(
            "\n> **注意**：以上结论基于 21 题小规模评估集，统计显著性有限。"
            "建议在生产环境中用更大评估集验证后再决定是否启用 rerank。\n"
        )
    else:
        lines.append("（实验未运行，无法给出结论）\n")

    # 9. 当前限制
    lines.append("\n## 9. 当前限制\n")
    lines.append("- 评估集仅 21 题，统计结果可能不稳定，rerank 增益可能被噪声掩盖。\n")
    lines.append("- rerank API 调用增加额外延迟，对实时性要求高的场景需权衡。\n")
    lines.append("- 本实验仅测试 SiliconFlow bge-reranker-v2-m3 单一模型，未对比其他 rerank 模型。\n")
    lines.append("- similarity_threshold 过滤发生在 rerank 之前，可能过滤掉 rerank 本能纠正的低向量相似度高语义相关 chunk。\n")
    lines.append("- Stage 3 answer-level 受 LLM 输出稳定性和 API 波动影响。\n")
    lines.append("- 未测试 rerank 与 query rewrite / hybrid search 等手段的组合效果。\n")
    lines.append("- Chroma 返回的是 cosine distance，本实验中所有 score 均已通过 `similarity = 1.0 - distance` 转换。\n")

    # 10. 面试表述
    lines.append("\n## 10. 面试表述\n")
    lines.append(
        "> 在 RAG 参数消融实验确定了 chunk_size、similarity_threshold 等基础参数后，"
        "我进一步设计了 rerank 消融实验，验证 SiliconFlow bge-reranker-v2-m3 cross-encoder "
        "是否能提升检索精度。实验固定基础参数，对比 baseline（纯向量检索 top_k=5）、"
        "rerank_k10（candidate_k=10 → rerank → top_k=5）和 rerank_k20（candidate_k=20 → rerank → top_k=5）"
        "三组配置。通过 Retrieval-level（source_hit_rate、MRR）、Out-of-scope safety（false_refusal_rate）"
        "和 Answer-level（keyword_coverage）三层评估，量化 rerank 的精度增益与延迟代价。"
        "实验设计了优雅降级机制：rerank API 调用失败时自动回退到原始向量顺序，保证可用性。\n"
    )

    content = "".join(lines)
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("已生成报告: %s", README_PATH)
    return content


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> int:
    """三阶段 rerank 消融实验主入口。"""
    print("=" * 60)
    print("  RAG Rerank 消融实验")
    print("  对比: baseline vs rerank_k10 vs rerank_k20")
    print("=" * 60)

    # 加载配置
    configs_data = load_configs()
    configs = configs_data["configs"]
    fixed_params = configs_data["fixed_params"]
    fixed_threshold = fixed_params["similarity_threshold"]
    top_k = fixed_params["top_k"]

    # 加载问题
    flat_questions = load_flat_questions()
    in_scope = [q for q in flat_questions if not q.get("should_refuse", False)]
    print(f"\n  共加载 {len(flat_questions)} 题 (领域内 {len(in_scope)}，领域外 {len(flat_questions) - len(in_scope)})")
    print(f"  固定参数: chunk_size={fixed_params['chunk_size']}, "
          f"overlap={fixed_params['chunk_overlap']}, "
          f"threshold={fixed_threshold}, top_k={top_k}")

    actually_ran = False
    error_msg = ""
    stage1_result: dict = {}
    stage2_result: dict = {}
    stage3_result: dict = {}

    try:
        # ---------- 阶段一 ----------
        stage1_result = run_stage1_retrieval(
            in_scope_questions=in_scope,
            configs=configs,
            fixed_threshold=fixed_threshold,
            top_k=top_k,
        )
        save_json(stage1_result, "stage1_retrieval_results.json")

        best_rerank_config = stage1_result.get("best_rerank_config")

        # ---------- 阶段二 ----------
        stage2_result = run_stage2_out_of_scope(
            all_questions=flat_questions,
            configs=configs,
            fixed_threshold=fixed_threshold,
            top_k=top_k,
        )
        save_json(stage2_result, "stage2_out_of_scope_results.json")

        # ---------- 阶段三 ----------
        stage3_result = run_stage3_answer_validation(
            all_questions=flat_questions,
            best_rerank_config=best_rerank_config,
            fixed_threshold=fixed_threshold,
            top_k=top_k,
        )
        save_json(stage3_result, "stage3_answer_validation_results.json")

        # ---------- 推荐结论 ----------
        s1_configs = stage1_result.get("configs", {})
        recommendation = {
            "fixed_params": fixed_params,
            "best_rerank_config": best_rerank_config,
            "baseline_retrieval_score": s1_configs.get("baseline", {}).get("summary", {}).get("retrieval_score"),
            "best_rerank_retrieval_score": s1_configs.get(
                stage1_result.get("best_rerank_name", ""), {}
            ).get("summary", {}).get("retrieval_score"),
        }
        if stage3_result and not stage3_result.get("skipped"):
            recommendation["baseline_answer_score"] = stage3_result.get("baseline", {}).get("summary", {}).get("answer_score")
            recommendation["rerank_answer_score"] = stage3_result.get("rerank", {}).get("summary", {}).get("answer_score")
        save_json(recommendation, "recommendation.json")

        actually_ran = True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("实验运行失败: %s", error_msg, exc_info=True)
        print(f"\n  ❌ 实验失败: {error_msg}")

        # 保存部分结果
        if stage1_result:
            save_json(stage1_result, "stage1_retrieval_results.json")
        if stage2_result:
            save_json(stage2_result, "stage2_out_of_scope_results.json")
        if stage3_result:
            save_json(stage3_result, "stage3_answer_validation_results.json")

    # ---------- 生成报告 ----------
    generate_readme(
        stage1=stage1_result,
        stage2=stage2_result,
        stage3=stage3_result,
        configs=configs,
        fixed_params=fixed_params,
        actually_ran=actually_ran,
        error_msg=error_msg,
    )

    # ---------- 控制台输出 ----------
    print("\n" + "=" * 60)
    if actually_ran:
        print("  ✅ 实验完成！")
        best_name = stage1_result.get("best_rerank_name", "N/A")
        print(f"  最佳 rerank 配置: {best_name}")
        print(f"  报告: {README_PATH}")
    else:
        print(f"  ❌ 实验未完成: {error_msg}")
        print("  请检查 API Key、网络连接和文档路径后重试。")
        print(f"  运行命令: python -m experiments.rag_rerank_ablation.run_rerank_ablation")
        print(f"  报告（占位）: {README_PATH}")
    print("=" * 60)

    return 0 if actually_ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
