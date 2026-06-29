"""Remote Sensing Knowledge Base MCP Server。

通过 MCP 协议（Model Context Protocol）暴露两个原子能力，
让 Claude Desktop / Cursor / Claude Code 等宿主 LLM 可以直接调用：

1. search_remote_sensing_kb       ——  知识库语义检索（仅检索，不调 LLM 生成）
2. calculate_remote_sensing_metric ——  评价指标确定性计算（纯数值，无 LLM）

设计原则：
    - 宿主 LLM（Claude）是编排者，MCP 工具提供原子能力
    - search_remote_sensing_kb 只返回检索 chunks，由宿主 LLM 自行推理生成答案
      （不调用 RAG 全流程，避免双重 LLM 生成）
    - calculate_remote_sensing_metric 使用类型化参数（tp/fp/fn 为 float），
      无需字符串解析（与 Agent @tool 的 "TP=80, FP=10" 字符串风格区分）
    - 两个工具的核心逻辑与 Agent @tool 共享同一套内核：
        search       →  app.services.retriever.Retriever（与 tools.py::_retrieve 共用）
        calculate    →  app.core.metrics.calculate_metric（与 domain_tools.py::metrics_calculator 共用）

运行方式：
    cd <项目根目录>
    python -m mcp_server.server

    或在 Claude Desktop 的 claude_desktop_config.json 中配置：
    {
      "mcpServers": {
        "remote-sensing-kb": {
          "command": "python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "<项目根目录的绝对路径>"
        }
      }
    }
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.core.metrics import calculate_metric
from app.services.retriever import Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP 实例                                                                #
# --------------------------------------------------------------------------- #

mcp = FastMCP("remote-sensing-kb")

# contexts 中每个 content 最大字符数（与 tools.py _MAX_CONTEXT_CHARS 保持一致）
_MAX_CONTEXT_CHARS = 500

# sources 中每个 content_preview 最大字符数（与 tools.py _MAX_PREVIEW_CHARS 保持一致）
_MAX_PREVIEW_CHARS = 150


# --------------------------------------------------------------------------- #
#  辅助函数                                                                    #
# --------------------------------------------------------------------------- #

def _truncate(text: str | None, max_chars: int) -> str:
    """安全截断文本，超出长度时追加省略号。

    逻辑与 app.agents.tools.truncate_text 一致，此处内联以避免
    MCP Server 依赖 Agent 层（mcp_server → app.agents.tools 的跨层依赖）。
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _hit_to_context(h: dict[str, Any], idx: int) -> dict[str, Any]:
    """将单个检索 hit 转为 contexts 数组元素（压缩后，供宿主 LLM 阅读）。

    逻辑与 app.agents.tools._hit_to_context 一致。
    """
    content = (h.get("content", "") or "").replace("\n", " ")
    return {
        "source_id": f"source_{idx + 1}",
        "content": _truncate(content, _MAX_CONTEXT_CHARS),
        "source": (
            f"{h.get('filename', 'unknown')}，"
            f"第{h.get('page', 1)}页，"
            f"chunk_id={h.get('chunk_id', '')}"
        ),
        "score": round(float(h.get("score", 0.0)), 4),
    }


def _hit_to_source(h: dict[str, Any]) -> dict[str, Any]:
    """将单个检索 hit 转为 sources 数组元素（含压缩后的 content_preview）。

    逻辑与 app.agents.tools._hit_to_source 一致。
    """
    content = (h.get("content", "") or "").replace("\n", " ")
    return {
        "filename": h.get("filename", "unknown"),
        "page": h.get("page", 1),
        "chunk_id": h.get("chunk_id", ""),
        "score": round(float(h.get("score", 0.0)), 4),
        "content_preview": _truncate(content, _MAX_PREVIEW_CHARS),
    }


def _make_fragment(
    tool_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    """构造 work_unit_fragment（MCP 工作单元碎片）。

    设计要点（详见 docs/work_unit_design.md §6）：
        - fragment 只是「素材」，由宿主 Agent / 上层产品决定是否组装成完整 Work Unit；
        - 此处仅生成，绝不落盘、不写 data/work_units/、不调用 WorkUnitStore；
        - fragment_id = "frag_" + md5(tool_name + inputs)[:12]，保证可追溯且无需随机源。
    """
    raw = f"{tool_name}:{inputs}"
    fragment_id = "frag_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return {
        "fragment_id": fragment_id,
        "entry": "mcp",
        "tool_name": tool_name,
        "inputs": inputs,
        "outputs": outputs,
        "elapsed": elapsed,
    }


# --------------------------------------------------------------------------- #
#  MCP Tool 1: 知识库语义检索                                                  #
# --------------------------------------------------------------------------- #

@mcp.tool()
def search_remote_sensing_kb(
    query: str,
    top_k: int = 5,
    enable_rerank: bool = False,
) -> dict[str, Any]:
    """Search the remote sensing semantic segmentation knowledge base.

    Retrieves semantically relevant text chunks from the vector database.
    Does NOT generate an answer — returns raw context chunks for the host LLM
    to read and reason with.

    Use this tool when the user asks about remote sensing datasets, segmentation
    models, evaluation metrics, challenges, or methods and needs factual
    information from the knowledge base.

    Args:
        query: Natural language search query (Chinese or English).
        top_k: Number of chunks to return (default 5).
        enable_rerank: Use cross-encoder reranking (bge-reranker-v2-m3) for
                       better precision at the cost of latency (default False).

    Returns:
        dict with:
        - success (bool): whether any chunks were found
        - query (str): the search query
        - summary (str): human-readable result count
        - contexts (list): context chunks with source_id, content, source, score
        - sources (list): source metadata with filename, page, chunk_id, score
        - elapsed (float): retrieval time in seconds
        - work_unit_fragment (dict): MCP 工作单元碎片（entry=mcp，不落盘）
        - error (str, optional): error message on failure
    """
    try:
        settings = get_settings()
        retriever = Retriever()

        start = time.time()
        hits = retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=settings.similarity_threshold,
            use_rerank=enable_rerank,
        )
        elapsed = round(time.time() - start, 4)

        contexts = [_hit_to_context(h, i) for i, h in enumerate(hits)]
        sources = [_hit_to_source(h) for h in hits]

        logger.info(
            "search_remote_sensing_kb: query=%r, hits=%d, elapsed=%.4fs, rerank=%s",
            query, len(hits), elapsed, enable_rerank,
        )

        return {
            "success": len(hits) > 0,
            "query": query,
            "summary": (
                f"检索到 {len(hits)} 个相关片段"
                if hits
                else "未检索到相关知识库内容"
            ),
            "contexts": contexts,
            "sources": sources,
            "elapsed": elapsed,
            # 追加 work_unit_fragment（不落盘，仅供宿主 / 上层组装完整 Work Unit）
            "work_unit_fragment": _make_fragment(
                tool_name="search_remote_sensing_kb",
                inputs={"query": query, "top_k": top_k},
                outputs={"contexts_count": len(contexts), "sources_count": len(sources)},
                elapsed=elapsed,
            ),
        }

    except Exception as e:
        logger.error("search_remote_sensing_kb 异常: %s", e)
        return {
            "success": False,
            "query": query,
            "summary": "检索失败",
            "contexts": [],
            "sources": [],
            "error": str(e),
            # 异常分支也附带 fragment，保持返回结构一致（不落盘）
            "work_unit_fragment": _make_fragment(
                tool_name="search_remote_sensing_kb",
                inputs={"query": query, "top_k": top_k},
                outputs={"contexts_count": 0, "sources_count": 0},
                elapsed=0.0,
            ),
        }


# --------------------------------------------------------------------------- #
#  MCP Tool 2: 评价指标计算                                                    #
# --------------------------------------------------------------------------- #

@mcp.tool()
def calculate_remote_sensing_metric(
    metric: str,
    tp: float | None = None,
    fp: float | None = None,
    fn: float | None = None,
    tn: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
) -> dict[str, Any]:
    """Calculate a semantic segmentation evaluation metric from raw values.

    Computes IoU, Precision, Recall, or F1-score from confusion matrix values.
    This is a deterministic tool — no LLM involved, results are exact.

    Supported metrics:
    - IoU:       requires tp, fp, fn
    - Precision: requires tp, fp
    - Recall:    requires tp, fn
    - F1-score:  mode 1 — precision + recall
                 mode 2 — tp, fp, fn

    Args:
        metric: Metric name. Case-insensitive (accepts "iou", "f1", "dice",
                "precision", "recall", etc.).
        tp: True Positives.
        fp: False Positives.
        fn: False Negatives.
        tn: True Negatives (reserved for future use, not currently in formulas).
        precision: Pre-computed Precision value (F1-score mode 1 only).
        recall: Pre-computed Recall value (F1-score mode 1 only).

    Returns:
        dict with:
        - success (bool): whether calculation succeeded
        - metric (str): normalized metric name
        - inputs (dict): values used in calculation
        - result (float): calculated value (on success)
        - formula (str): formula text (on success)
        - summary (str): human-readable calculation trace or error hint
        - error (str, optional): error description (on failure)
        - supported_metrics (list, optional): when metric is unsupported
        - work_unit_fragment (dict): MCP 工作单元碎片（entry=mcp，不落盘）
    """
    try:
        start = time.time()
        result = calculate_metric(
            metric=metric,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            recall=recall,
        )
        elapsed = round(time.time() - start, 4)

        logger.info(
            "calculate_remote_sensing_metric: metric=%s, success=%s",
            result.get("metric"),
            result.get("success"),
        )
        # 追加 work_unit_fragment（不落盘）。
        # 注意：core/metrics.py 的返回值字段名为 "result"，此处映射到 fragment.outputs.value。
        result["work_unit_fragment"] = _make_fragment(
            tool_name="calculate_remote_sensing_metric",
            inputs={"metric": metric, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
            outputs={"value": result.get("result")},
            elapsed=elapsed,
        )
        return result

    except Exception as e:
        logger.error("calculate_remote_sensing_metric 异常: %s", e)
        return {
            "success": False,
            "metric": metric,
            "inputs": {},
            "error": str(e),
            "summary": "工具执行异常。",
            # 异常分支也附带 fragment，保持返回结构一致（不落盘）
            "work_unit_fragment": _make_fragment(
                tool_name="calculate_remote_sensing_metric",
                inputs={"metric": metric, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
                outputs={"value": None},
                elapsed=0.0,
            ),
        }


# --------------------------------------------------------------------------- #
#  入口                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run()
