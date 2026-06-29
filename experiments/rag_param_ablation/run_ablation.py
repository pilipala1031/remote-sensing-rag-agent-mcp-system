"""三阶段 RAG 参数消融实验主脚本。

阶段一：chunk_size / chunk_overlap 对比（Retrieval-level）
  - 固定 similarity_threshold = 0.3
  - 只使用 18 个领域内问题（should_refuse=false）
  - 每组 chunk 参数重新切分写入独立临时 Chroma
  - retrieval-only，不调用 LLM
  - 输出 retrieval_score，选最佳 chunk 参数

阶段二：similarity_threshold 敏感性分析（Refusal-level）
  - 使用阶段一推荐的 chunk 参数
  - 全部 21 个问题（含 3 个 should_refuse=true）
  - 遍历 threshold=[0.1..0.7]
  - retrieval-only，不调用 LLM
  - 输出 refusal_score，选最佳 threshold

阶段三：answer-level 最终验证
  - 只对比推荐参数 vs 默认参数(800/120/0.3)
  - 调用 RAGService（含 LLM）
  - 输出 answer_score

CLI 用法：
    python -m experiments.rag_param_ablation.run_ablation
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
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

from app.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
PARAM_CONFIGS_PATH = PACKAGE_DIR / "param_configs.json"
README_PATH = PACKAGE_DIR / "README_rag_param_ablation.md"

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
    如果清理失败，临时目录会留在系统 temp 目录中，不影响实验正确性。
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.info("创建临时目录: %s", tmpdir)
    try:
        yield tmpdir
    finally:
        # 触发垃圾回收，释放 Chroma sqlite3 连接引用
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)
        if tmpdir.exists():
            logger.warning("临时目录清理失败（Chroma 文件锁定），将保留: %s", tmpdir)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_param_configs() -> dict:
    """加载参数配置 JSON。"""
    with open(PARAM_CONFIGS_PATH, "r", encoding="utf-8") as f:
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


# --------------------------------------------------------------------------- #
# 阶段一：Chunk 参数实验
# --------------------------------------------------------------------------- #
def _search_raw(store, query: str, top_k: int) -> tuple[List[dict], float]:
    """用极低 threshold 检索，返回原始 top-K 结果和耗时。

    store.search 内部已做 score = 1.0 - distance 转换。
    """
    t0 = time.perf_counter()
    results = store.search(
        query=query,
        top_k=top_k,
        similarity_threshold=RAW_SEARCH_THRESHOLD,
    )
    latency = time.perf_counter() - t0
    return results, latency


def _calc_retrieval_per_question(
    q: dict, raw_results: List[dict], latency: float, threshold: float
) -> dict:
    """计算单题 retrieval 指标。

    先用 threshold 过滤 raw_results，再统计指标。
    """
    expected_sources = q.get("expected_source_files", [])

    # 按阈值过滤
    filtered = [r for r in raw_results if r.get("score", 0.0) >= threshold]
    retrieved_filenames = [r.get("filename", "") for r in filtered]

    # source_hit: 是否命中任一期望来源
    source_hit = 0
    for exp in expected_sources:
        if any(_match_source(exp, fn) for fn in retrieved_filenames):
            source_hit = 1
            break

    # source_recall_at_k: 期望来源中被命中的比例
    if expected_sources:
        found = sum(
            1 for exp in expected_sources
            if any(_match_source(exp, fn) for fn in retrieved_filenames)
        )
        recall = found / len(expected_sources)
    else:
        recall = 1.0

    # MRR: 第一个匹配来源的倒数排名
    mrr = 0.0
    for rank, fn in enumerate(retrieved_filenames, 1):
        if any(_match_source(exp, fn) for exp in expected_sources):
            mrr = 1.0 / rank
            break

    scores = [r.get("score", 0.0) for r in filtered]
    top_score = scores[0] if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "question_id": q["id"],
        "question": q["question"],
        "expected_source_files": expected_sources,
        "retrieved_sources": retrieved_filenames,
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "source_hit": source_hit,
        "source_recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "retrieved_count": len(filtered),
        "latency": round(latency, 4),
    }


def _aggregate_stage1(details: List[dict]) -> dict:
    """汇总单组 chunk 参数的 retrieval 指标。"""
    n = len(details)
    if n == 0:
        return {}
    return {
        "source_hit_rate": round(sum(d["source_hit"] for d in details) / n, 4),
        "source_recall_at_k": round(sum(d["source_recall_at_k"] for d in details) / n, 4),
        "mrr": round(sum(d["mrr"] for d in details) / n, 4),
        "avg_top_score": round(sum(d["top_score"] for d in details) / n, 4),
        "avg_score": round(sum(d["avg_score"] for d in details) / n, 4),
        "avg_retrieved_count": round(sum(d["retrieved_count"] for d in details) / n, 4),
        "avg_latency": round(sum(d["latency"] for d in details) / n, 4),
    }


def _calc_retrieval_score(summary: dict, latency_norm: float) -> float:
    """计算 retrieval_score。

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


def run_stage1_chunk_ablation(
    in_scope_questions: List[dict],
    chunk_configs: List[dict],
    fixed_threshold: float,
    top_k: int,
) -> dict:
    """阶段一：chunk 参数对比实验。

    Returns:
        {
            "configs": {config_name: {config, details, summary}},
            "best_config": {...},
            "stage": "chunk_ablation"
        }
    """
    print("\n" + "=" * 60)
    print("  阶段一：Chunk 参数实验 (threshold=0.3, retrieval-only)")
    print("=" * 60)

    all_configs: Dict[str, dict] = {}
    config_latencies: Dict[str, float] = {}

    for cfg in chunk_configs:
        name = cfg["name"]
        print(f"\n  [{name}] chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}")

        with ablation_temp_dir(f"ablation_s1_{name}_") as tmpdir:
            # 构建临时向量库
            store = build_temp_store(
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                source_dir=SAMPLE_DOCS_DIR,
                persist_dir=Path(tmpdir),
                collection_name=name,
            )

            # 逐题检索（先拿 raw 结果，再用固定阈值过滤）
            details = []
            for q in in_scope_questions:
                raw_results, latency = _search_raw(store, q["question"], top_k)
                detail = _calc_retrieval_per_question(
                    q, raw_results, latency, fixed_threshold
                )
                details.append(detail)

            summary = _aggregate_stage1(details)
            all_configs[name] = {
                "config": cfg,
                "details": details,
                "summary": summary,
            }
            config_latencies[name] = summary["avg_latency"]

            print(f"    source_hit_rate={summary['source_hit_rate']:.4f}, "
                  f"recall={summary['source_recall_at_k']:.4f}, "
                  f"mrr={summary['mrr']:.4f}, "
                  f"avg_latency={summary['avg_latency']:.4f}s")

    # latency 0-1 归一化
    latencies = list(config_latencies.values())
    lat_min = min(latencies) if latencies else 0.0
    lat_max = max(latencies) if latencies else 1.0
    lat_range = lat_max - lat_min

    for name, data in all_configs.items():
        lat = config_latencies[name]
        latency_norm = (lat - lat_min) / lat_range if lat_range > 0 else 0.0
        data["summary"]["latency_norm"] = round(latency_norm, 4)
        data["summary"]["retrieval_score"] = _calc_retrieval_score(
            data["summary"], latency_norm
        )
        print(f"  [{name}] retrieval_score={data['summary']['retrieval_score']:.4f}")

    # 选择最佳
    best_name = max(
        all_configs,
        key=lambda k: all_configs[k]["summary"]["retrieval_score"],
    )
    best_config = all_configs[best_name]["config"]
    print(f"\n  >>> 推荐 chunk 参数: {best_name} "
          f"(size={best_config['chunk_size']}, overlap={best_config['chunk_overlap']})")

    return {
        "stage": "chunk_ablation",
        "configs": all_configs,
        "best_config": best_config,
    }


# --------------------------------------------------------------------------- #
# 阶段二：Threshold 敏感性分析
# --------------------------------------------------------------------------- #
def run_stage2_threshold_sensitivity(
    all_questions: List[dict],
    best_chunk_config: dict,
    thresholds: List[float],
    top_k: int,
) -> dict:
    """阶段二：threshold 敏感性分析。

    使用阶段一推荐的 chunk 参数构建一个临时向量库，
    对每个问题只检索一次（threshold=-1），然后模拟不同阈值的过滤效果。
    """
    print("\n" + "=" * 60)
    print("  阶段二：Threshold 敏感性分析 (retrieval-only)")
    print("=" * 60)

    in_scope = [q for q in all_questions if not q.get("should_refuse", False)]
    out_scope = [q for q in all_questions if q.get("should_refuse", False)]
    n_in = len(in_scope)
    n_out = len(out_scope)
    print(f"  领域内 {n_in} 题, 领域外 {n_out} 题")

    with ablation_temp_dir("ablation_s2_") as tmpdir:
        store = build_temp_store(
            chunk_size=best_chunk_config["chunk_size"],
            chunk_overlap=best_chunk_config["chunk_overlap"],
            source_dir=SAMPLE_DOCS_DIR,
            persist_dir=Path(tmpdir),
            collection_name="stage2_threshold",
        )

        # 预检索所有问题的原始结果
        raw_cache: Dict[str, tuple[List[dict], float]] = {}
        for q in all_questions:
            results, latency = _search_raw(store, q["question"], top_k)
            raw_cache[q["id"]] = (results, latency)

        # 遍历阈值
        threshold_summaries: Dict[str, dict] = {}
        all_threshold_details: List[dict] = []

        for thr in thresholds:
            thr_key = str(thr)
            details = []

            for q in all_questions:
                raw_results, latency = raw_cache[q["id"]]
                filtered = [r for r in raw_results if r.get("score", 0.0) >= thr]
                retrieved_count = len(filtered)
                scores = [r.get("score", 0.0) for r in filtered]
                max_score = max(scores) if scores else 0.0
                refused = retrieved_count == 0
                should_refuse = q.get("should_refuse", False)
                false_refusal = 1 if (not should_refuse and refused) else 0
                false_accept = 1 if (should_refuse and not refused) else 0

                details.append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "should_refuse": should_refuse,
                    "threshold": thr,
                    "max_score": round(max_score, 4),
                    "retrieved_count": retrieved_count,
                    "refused_by_retrieval": refused,
                    "false_refusal": false_refusal,
                    "false_accept": false_accept,
                    "latency": round(latency, 4),
                })

            all_threshold_details.extend(details)

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
            all_lats = [d["latency"] for d in details]
            n_total = len(details)

            in_scope_recall = in_not_refused / n_in if n_in > 0 else 0.0
            out_refusal_acc = out_refused / n_out if n_out > 0 else 0.0
            false_refusal_rate = false_refusals / n_in if n_in > 0 else 0.0
            false_accept_rate = false_accepts / n_out if n_out > 0 else 0.0

            # refusal_score
            refusal_score = (
                0.35 * in_scope_recall
                + 0.45 * out_refusal_acc
                - 0.05 * false_refusal_rate
                - 0.15 * false_accept_rate
            )

            summary = {
                "threshold": thr,
                "in_scope_recall": round(in_scope_recall, 4),
                "out_of_scope_refusal_accuracy": round(out_refusal_acc, 4),
                "false_refusal_rate": round(false_refusal_rate, 4),
                "false_accept_rate": round(false_accept_rate, 4),
                "avg_max_score": round(sum(all_max_scores) / n_total, 4) if n_total else 0.0,
                "avg_retrieved_count": round(sum(all_counts) / n_total, 4) if n_total else 0.0,
                "avg_latency": round(sum(all_lats) / n_total, 4) if n_total else 0.0,
                "refusal_score": round(refusal_score, 4),
            }
            threshold_summaries[thr_key] = summary

            print(f"  [thr={thr:.1f}] in_recall={summary['in_scope_recall']:.4f}, "
                  f"out_refusal={summary['out_of_scope_refusal_accuracy']:.4f}, "
                  f"false_refusal={summary['false_refusal_rate']:.4f}, "
                  f"false_accept={summary['false_accept_rate']:.4f}, "
                  f"refusal_score={summary['refusal_score']:.4f}")

    # 选择最佳 threshold
    best_thr_key = max(
        threshold_summaries,
        key=lambda k: threshold_summaries[k]["refusal_score"],
    )
    best_threshold = float(best_thr_key)
    print(f"\n  >>> 推荐 similarity_threshold: {best_threshold}")

    return {
        "stage": "threshold_sensitivity",
        "chunk_config_used": best_chunk_config,
        "threshold_summaries": threshold_summaries,
        "details": all_threshold_details,
        "best_threshold": best_threshold,
    }


# --------------------------------------------------------------------------- #
# 阶段三：Answer-level 最终验证
# --------------------------------------------------------------------------- #
def _run_rag_answers(
    questions: List[dict],
    chunk_size: int,
    chunk_overlap: int,
    similarity_threshold: float,
    top_k: int,
    label: str,
) -> dict:
    """构建临时向量库并通过 RAGService 获取 LLM 回答。

    RAGService 内部流程: 检索 → 拒答判断 → LLM 生成。
    使用注入的 Retriever 指向临时向量库，不触碰正式 data/chroma。
    """
    from app.services.retriever import Retriever
    from app.services.rag_service import RAGService

    print(f"\n  [{label}] chunk={chunk_size}/{chunk_overlap}, thr={similarity_threshold}")

    with ablation_temp_dir(f"ablation_s3_{label}_") as tmpdir:
        store = build_temp_store(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_dir=SAMPLE_DOCS_DIR,
            persist_dir=Path(tmpdir),
            collection_name=f"stage3_{label}",
        )

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
                    similarity_threshold=similarity_threshold,
                )
            except Exception as e:
                logger.error("[%s] RAG 调用失败 (q=%s): %s", label, q["id"], e)
                answer_obj = None
            latency = time.perf_counter() - t0

            if answer_obj is None:
                details.append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "answer": "(调用失败)",
                    "sources": [],
                    "keyword_coverage": 0.0,
                    "source_hit": 0,
                    "refusal_accuracy": 0,
                    "min_length_satisfied": 0,
                    "latency": round(latency, 4),
                    "error": True,
                })
                continue

            answer_text = answer_obj.answer or ""
            sources = answer_obj.sources or []
            refused = answer_obj.refused

            # keyword_coverage
            if expected_keywords:
                lower_ans = answer_text.lower()
                hits = sum(1 for kw in expected_keywords if kw.lower() in lower_ans)
                kw_cov = hits / len(expected_keywords)
            else:
                kw_cov = 1.0

            # source_hit
            source_filenames = [s.filename for s in sources]
            src_hit = 0
            for exp in expected_sources:
                if any(_match_source(exp, fn) for fn in source_filenames):
                    src_hit = 1
                    break

            # refusal_accuracy
            refusal_acc = 1 if (refused == should_refuse) else 0

            # min_length_satisfied
            min_len_ok = 1 if len(answer_text) >= min_len else 0

            details.append({
                "question_id": q["id"],
                "question": q["question"],
                "answer": answer_text[:500],
                "sources": source_filenames,
                "keyword_coverage": round(kw_cov, 4),
                "source_hit": src_hit,
                "refusal_accuracy": refusal_acc,
                "min_length_satisfied": min_len_ok,
                "latency": round(latency, 4),
            })

        # 汇总
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

        print(f"  [{label}] keyword_cov={summary['keyword_coverage_avg']:.4f}, "
              f"source_hit={summary['source_hit_rate']:.4f}, "
              f"refusal_acc={summary['refusal_accuracy']:.4f}, "
              f"answer_score={summary['answer_score']:.4f}")

        return {"label": label, "details": details, "summary": summary,
                "params": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "similarity_threshold": similarity_threshold,
                }}


def run_stage3_answer_validation(
    all_questions: List[dict],
    recommended_chunk: dict,
    recommended_threshold: float,
    default_params: dict,
    top_k: int,
) -> dict:
    """阶段三：answer-level 最终验证。

    只对比推荐参数组合与默认参数组合。
    """
    print("\n" + "=" * 60)
    print("  阶段三：Answer-level 最终验证 (LLM)")
    print("=" * 60)

    # 推荐参数
    recommended_result = _run_rag_answers(
        questions=all_questions,
        chunk_size=recommended_chunk["chunk_size"],
        chunk_overlap=recommended_chunk["chunk_overlap"],
        similarity_threshold=recommended_threshold,
        top_k=top_k,
        label="recommended",
    )

    # 默认参数
    default_result = _run_rag_answers(
        questions=all_questions,
        chunk_size=default_params["chunk_size"],
        chunk_overlap=default_params["chunk_overlap"],
        similarity_threshold=default_params["similarity_threshold"],
        top_k=top_k,
        label="default",
    )

    return {
        "stage": "answer_validation",
        "recommended": recommended_result,
        "default": default_result,
    }


# --------------------------------------------------------------------------- #
# Markdown 报告生成
# --------------------------------------------------------------------------- #
def generate_readme(
    stage1: dict,
    stage2: dict,
    stage3: dict,
    recommended_chunk: dict,
    recommended_threshold: float,
    default_params: dict,
    actually_ran: bool,
    error_msg: str = "",
) -> str:
    """生成 README_rag_param_ablation.md 报告。"""
    lines: List[str] = []
    lines.append("# RAG 参数消融实验\n")

    if not actually_ran:
        lines.append("> **⚠️ 实验未成功完成，以下为框架占位，无实际数据。**\n")
        lines.append(f"> 失败原因：{error_msg}\n")
        lines.append("> 请解决问题后重新运行 `python -m experiments.rag_param_ablation.run_ablation`\n")

    lines.append("## 1. 实验目的\n")
    lines.append(
        "本实验用于确定 RAG 系统中 `chunk_size`、`chunk_overlap`、"
        "`similarity_threshold` 三个参数的合理取值。"
        "通过分层评估（Retrieval → Refusal → Answer）逐步缩小参数空间，"
        "避免暴力全组合实验的高成本。\n"
    )

    lines.append("## 2. 为什么需要消融\n")
    lines.append("- **chunk 太小**：语义片段被截断，同一概念分散在多个 chunk 中，召回率下降。\n")
    lines.append("- **chunk 太大**：单个 chunk 包含过多无关信息，噪声稀释信号，精确度下降。\n")
    lines.append("- **overlap 太小**：跨段信息丢失，边界处的上下文断裂。\n")
    lines.append("- **overlap 太大**：冗余 chunk 增多，存储和检索成本上升。\n")
    lines.append("- **threshold 太低**：低相关证据混入上下文，可能误导 LLM。\n")
    lines.append("- **threshold 太高**：正常问题被误拒答，用户体验受损。\n")

    lines.append("## 3. 实验设置\n")
    if actually_ran:
        configs = stage1.get("configs", {})
        for name, data in configs.items():
            cfg = data["config"]
            lines.append(
                f"- `{name}`: chunk_size={cfg['chunk_size']}, "
                f"overlap={cfg['chunk_overlap']}\n"
            )
        thr_summaries = stage2.get("threshold_summaries", {})
        thr_values = sorted(thr_summaries.keys(), key=float) if thr_summaries else ["0.1","0.2","0.3","0.4","0.5","0.6","0.7"]
        lines.append(f"\nThreshold 候选值：{', '.join(thr_values)}\n")
    else:
        lines.append("- Chunk 参数组：c400_o80, c600_o100, c800_o120, c1000_o150, c1200_o180\n")
        lines.append("- Threshold 候选值：0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7\n")
    lines.append(f"- 当前默认参数：{default_params}\n")

    lines.append("## 4. 指标设计\n")
    lines.append("本实验分为三层评估：\n")
    lines.append("| 层级 | 用途 | 主要指标 |\n")
    lines.append("|------|------|----------|\n")
    lines.append("| Retrieval-level | 选择 chunk_size / chunk_overlap | source_hit_rate, source_recall_at_k, mrr, avg_top_score |\n")
    lines.append("| Refusal-level | 选择 similarity_threshold | in_scope_recall, out_of_scope_refusal_accuracy, false_refusal_rate, false_accept_rate |\n")
    lines.append("| Answer-level | 最终候选参数验证 | keyword_coverage, source_hit_rate, refusal_accuracy, min_length_satisfied |\n")

    # Stage 1 结果
    lines.append("## 5. Chunk 参数实验结果\n")
    if actually_ran and stage1.get("configs"):
        lines.append("| config | chunk_size | chunk_overlap | source_hit_rate | source_recall_at_k | mrr | avg_top_score | avg_latency | retrieval_score |\n")
        lines.append("|--------|------------|---------------|-----------------|---------------------|-----|---------------|-------------|-----------------|\n")
        for name, data in sorted(stage1["configs"].items()):
            cfg = data["config"]
            s = data["summary"]
            lines.append(
                f"| {name} | {cfg['chunk_size']} | {cfg['chunk_overlap']} | "
                f"{s['source_hit_rate']:.4f} | {s['source_recall_at_k']:.4f} | "
                f"{s['mrr']:.4f} | {s['avg_top_score']:.4f} | "
                f"{s['avg_latency']:.4f}s | {s['retrieval_score']:.4f} |\n"
            )
        best = stage1["best_config"]
        lines.append(f"\n**推荐**：`{best['name']}` (chunk_size={best['chunk_size']}, overlap={best['chunk_overlap']})\n")
    else:
        lines.append("（实验未运行，无数据）\n")

    # Stage 2 结果
    lines.append("## 6. Threshold 敏感性分析结果\n")
    if actually_ran and stage2.get("threshold_summaries"):
        lines.append("| threshold | in_scope_recall | out_of_scope_refusal_accuracy | false_refusal_rate | false_accept_rate | refusal_score |\n")
        lines.append("|-----------|-----------------|-------------------------------|--------------------|-------------------|---------------|\n")
        for thr_key in sorted(stage2["threshold_summaries"].keys(), key=float):
            s = stage2["threshold_summaries"][thr_key]
            lines.append(
                f"| {s['threshold']:.1f} | {s['in_scope_recall']:.4f} | "
                f"{s['out_of_scope_refusal_accuracy']:.4f} | "
                f"{s['false_refusal_rate']:.4f} | "
                f"{s['false_accept_rate']:.4f} | "
                f"{s['refusal_score']:.4f} |\n"
            )
        lines.append(f"\n**推荐**：similarity_threshold = {stage2.get('best_threshold', '?')}\n")
        lines.append(
            '\nthreshold 是「正常问题召回」和「超纲问题拒答」之间的 trade-off：'
            'threshold 越高，正常问题的召回率（in_scope_recall）可能下降（被误拒），'
            '但超纲问题的拒答准确率（out_of_scope_refusal_accuracy）会上升。'
            'refusal_score 综合权衡两者，选择使整体表现最优的阈值。\n'
        )
    else:
        lines.append("（实验未运行，无数据）\n")

    # Stage 3 结果
    lines.append("## 7. 最终候选参数 answer-level 验证\n")
    if actually_ran and stage3:
        rec = stage3.get("recommended", {}).get("summary", {})
        dft = stage3.get("default", {}).get("summary", {})
        lines.append("| 参数组合 | keyword_coverage_avg | source_hit_rate | refusal_accuracy | min_length_satisfied_rate | avg_latency | answer_score |\n")
        lines.append("|----------|---------------------|-----------------|------------------|--------------------------|-------------|--------------|\n")
        rec_params = stage3.get("recommended", {}).get("params", {})
        dft_params = stage3.get("default", {}).get("params", {})
        lines.append(
            f"| 推荐 ({rec_params.get('chunk_size','?')}/{rec_params.get('chunk_overlap','?')}/{rec_params.get('similarity_threshold','?')}) | "
            f"{rec.get('keyword_coverage_avg', 0):.4f} | {rec.get('source_hit_rate', 0):.4f} | "
            f"{rec.get('refusal_accuracy', 0):.4f} | {rec.get('min_length_satisfied_rate', 0):.4f} | "
            f"{rec.get('avg_latency', 0):.4f}s | {rec.get('answer_score', 0):.4f} |\n"
        )
        lines.append(
            f"| 默认 ({dft_params.get('chunk_size','?')}/{dft_params.get('chunk_overlap','?')}/{dft_params.get('similarity_threshold','?')}) | "
            f"{dft.get('keyword_coverage_avg', 0):.4f} | {dft.get('source_hit_rate', 0):.4f} | "
            f"{dft.get('refusal_accuracy', 0):.4f} | {dft.get('min_length_satisfied_rate', 0):.4f} | "
            f"{dft.get('avg_latency', 0):.4f}s | {dft.get('answer_score', 0):.4f} |\n"
        )
    else:
        lines.append("（实验未运行，无数据）\n")

    # 推荐
    lines.append("## 8. 推荐参数\n")
    if actually_ran:
        lines.append(f"- recommended_chunk_size: {recommended_chunk.get('chunk_size', '?')}\n")
        lines.append(f"- recommended_chunk_overlap: {recommended_chunk.get('chunk_overlap', '?')}\n")
        lines.append(f"- recommended_similarity_threshold: {recommended_threshold}\n")
        default_str = f"{default_params['chunk_size']}/{default_params['chunk_overlap']}/{default_params['similarity_threshold']}"
        rec_str = f"{recommended_chunk.get('chunk_size','?')}/{recommended_chunk.get('chunk_overlap','?')}/{recommended_threshold}"
        rec_score = stage3.get("recommended", {}).get("summary", {}).get("answer_score", 0)
        dft_score = stage3.get("default", {}).get("summary", {}).get("answer_score", 0)
        lines.append(f"\n**Answer-level 对比**: 推荐参数 answer_score={rec_score:.4f} vs 默认参数 answer_score={dft_score:.4f}\n")
        if rec_str != default_str:
            if rec_score > dft_score:
                lines.append(f"\n建议替换当前默认值 {default_str} → {rec_str}（answer-level 验证支持替换）。\n")
            else:
                lines.append(
                    f"\n推荐参数 {rec_str} 在 retrieval/refusal 层级更优，"
                    f"但 answer-level 验证显示默认参数 ({dft_score:.4f}) 优于推荐参数 ({rec_score:.4f})。"
                    f"\n**建议保持当前默认值 {default_str} 不变**，因为 end-to-end 质量更高。\n"
                )
        else:
            lines.append(f"\n推荐参数与当前默认值一致，建议保持不变。\n")
    else:
        lines.append("（实验未运行，无法给出推荐）\n")

    # 限制
    lines.append("## 9. 当前限制\n")
    lines.append("- 评估集规模较小（21 题），统计结果可能不稳定。\n")
    lines.append("- 标签中的 `expected_source_files` 和 `required_keywords` 不能完全等价于人工完整答案评估。\n")
    lines.append("- retrieval-only 指标只能评估证据检索质量，不能反映 LLM 理解和生成能力。\n")
    lines.append("- answer-level 会受到 LLM 输出稳定性和 API 波动影响。\n")
    lines.append("- 本实验未引入 rerank、query rewrite、hybrid search 等增强手段。\n")
    lines.append("- Chroma 返回的是 cosine distance，本实验中所有 score 均已通过 `similarity = 1.0 - distance` 转换。\n")

    # 面试表述
    lines.append("## 10. 面试表述\n")
    lines.append(
        "> RAG 系统的 chunk_size、chunk_overlap 和 similarity_threshold "
        "并非拍脑袋设定，而是通过小规模标注集消融实验确定的。"
        "实验分为三层：先用 Retrieval-level 指标（source_hit_rate、MRR、recall@k）"
        "在 5 组 chunk 参数中选出最优切分策略，再用 Refusal-level 指标"
        "（in_scope_recall、out_of_scope_refusal_accuracy）扫描 7 个 threshold "
        "值确定最佳拒答边界，最后用 Answer-level 指标（keyword_coverage、"
        "refusal_accuracy）做最终验证。每层指标加权合成复合 score，"
        "逐步缩小搜索空间，避免了 5×7 全组合的暴力成本。\n"
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
    """三阶段消融实验主入口。"""
    print("=" * 60)
    print("  RAG 参数消融实验")
    print("  参数: chunk_size, chunk_overlap, similarity_threshold")
    print("=" * 60)

    # 加载配置
    configs = load_param_configs()
    chunk_configs = configs["chunk_configs"]
    thresholds = configs["thresholds"]
    fixed_threshold = configs["fixed_threshold_stage1"]
    top_k = configs["top_k"]
    default_params = configs["default_params"]

    # 加载问题
    flat_questions = load_flat_questions()
    in_scope = [q for q in flat_questions if not q.get("should_refuse", False)]
    print(f"\n  共加载 {len(flat_questions)} 题 (领域内 {len(in_scope)}，领域外 {len(flat_questions) - len(in_scope)})")

    actually_ran = False
    error_msg = ""
    stage1_result = {}
    stage2_result = {}
    stage3_result = {}

    try:
        # ---------- 阶段一 ----------
        stage1_result = run_stage1_chunk_ablation(
            in_scope_questions=in_scope,
            chunk_configs=chunk_configs,
            fixed_threshold=fixed_threshold,
            top_k=top_k,
        )
        save_json(stage1_result, "chunk_ablation_results.json")

        best_chunk_config = stage1_result["best_config"]

        # ---------- 阶段二 ----------
        stage2_result = run_stage2_threshold_sensitivity(
            all_questions=flat_questions,
            best_chunk_config=best_chunk_config,
            thresholds=thresholds,
            top_k=top_k,
        )
        save_json(stage2_result, "threshold_sensitivity_results.json")

        best_threshold = stage2_result["best_threshold"]

        # ---------- 阶段三 ----------
        stage3_result = run_stage3_answer_validation(
            all_questions=flat_questions,
            recommended_chunk=best_chunk_config,
            recommended_threshold=best_threshold,
            default_params=default_params,
            top_k=top_k,
        )
        save_json(stage3_result, "final_answer_validation_results.json")

        # ---------- 推荐参数 ----------
        recommended = {
            "recommended_chunk_size": best_chunk_config["chunk_size"],
            "recommended_chunk_overlap": best_chunk_config["chunk_overlap"],
            "recommended_similarity_threshold": best_threshold,
            "recommended_chunk_config_name": best_chunk_config["name"],
            "retrieval_score": stage1_result["configs"][best_chunk_config["name"]]["summary"]["retrieval_score"],
            "refusal_score": stage2_result["threshold_summaries"][str(best_threshold)]["refusal_score"],
            "recommended_answer_score": stage3_result["recommended"]["summary"]["answer_score"],
            "default_answer_score": stage3_result["default"]["summary"]["answer_score"],
            "default_params": default_params,
        }
        save_json(recommended, "recommended_params.json")

        actually_ran = True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("实验运行失败: %s", error_msg, exc_info=True)
        print(f"\n  ❌ 实验失败: {error_msg}")

        # 仍然保存部分结果
        if stage1_result:
            save_json(stage1_result, "chunk_ablation_results.json")
        if stage2_result:
            save_json(stage2_result, "threshold_sensitivity_results.json")
        if stage3_result:
            save_json(stage3_result, "final_answer_validation_results.json")

    # ---------- 生成报告 ----------
    recommended_chunk = (
        stage1_result.get("best_config", {}) if stage1_result else {}
    )
    recommended_threshold = (
        stage2_result.get("best_threshold", 0.0) if stage2_result else 0.0
    )

    generate_readme(
        stage1=stage1_result,
        stage2=stage2_result,
        stage3=stage3_result,
        recommended_chunk=recommended_chunk,
        recommended_threshold=recommended_threshold,
        default_params=default_params,
        actually_ran=actually_ran,
        error_msg=error_msg,
    )

    # ---------- 控制台输出 ----------
    print("\n" + "=" * 60)
    if actually_ran:
        print("  ✅ 实验完成！")
        print(f"  推荐 chunk_size: {recommended_chunk.get('chunk_size', '?')}")
        print(f"  推荐 chunk_overlap: {recommended_chunk.get('chunk_overlap', '?')}")
        print(f"  推荐 similarity_threshold: {recommended_threshold}")
        print(f"  默认参数: {default_params}")
        print(f"  报告: {README_PATH}")
    else:
        print(f"  ❌ 实验未完成: {error_msg}")
        print("  请检查 API Key、网络连接和文档路径后重试。")
        print(f"  运行命令: python -m experiments.rag_param_ablation.run_ablation")
        print(f"  报告（占位）: {README_PATH}")
    print("=" * 60)

    return 0 if actually_ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
