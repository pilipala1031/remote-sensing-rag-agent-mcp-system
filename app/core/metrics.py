"""遥感语义分割评价指标计算内核（共享逻辑）。

从 app/agents/domain_tools.py 抽取的纯计算层，无框架依赖（不依赖 LangChain / MCP）。
Agent @tool（LangChain）和 MCP @tool 共同调用此模块，避免逻辑重复。

核心函数：
    calculate_metric()  ——  接受类型化参数，返回标准化结果 dict

设计原则：
    - 纯计算，无 LLM 调用，无 I/O 依赖
    - 返回 dict 格式统一，上层（@tool / @mcp.tool）可按需适配
    - 支持 None 值（表示"未提供"），与 0 区分
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# ============================================================================ #
#  常量                                                                         #
# ============================================================================ #

#: calculate_metric 支持的指标列表
SUPPORTED_METRICS: list[str] = ["IoU", "Precision", "Recall", "F1-score"]

#: 指标名称别名映射（大写归一化 key → 标准名称）
METRIC_ALIASES: dict[str, str] = {
    "IOU": "IoU",
    "PRECISION": "Precision",
    "RECALL": "Recall",
    "F1": "F1-score",
    "F1-SCORE": "F1-score",
    "F1SCORE": "F1-score",
    "F1_SCORE": "F1-score",
    "DICE": "F1-score",
}

#: 缺少参数时的提示信息
_PARAM_HINTS: dict[str, str] = {
    "IoU": (
        "IoU 需要 TP, FP, FN。"
        "示例: values='TP=80, FP=10, FN=20'"
    ),
    "Precision": (
        "Precision 需要 TP, FP。"
        "示例: values='TP=80, FP=10'"
    ),
    "Recall": (
        "Recall 需要 TP, FN。"
        "示例: values='TP=80, FN=20'"
    ),
    "F1-score": (
        "F1-score 支持两种输入模式: "
        "(1) precision=0.85, recall=0.90; "
        "(2) TP=80, FP=10, FN=20"
    ),
}


# ============================================================================ #
#  辅助函数                                                                     #
# ============================================================================ #

def normalize_metric_name(metric_name: str) -> str:
    """将指标名称归一化为标准名称。

    "f1" → "F1-score", "dice" → "F1-score", "iou" → "IoU"
    """
    key = metric_name.strip().upper().replace("_", "-").replace(" ", "")
    return METRIC_ALIASES.get(key, metric_name.strip())


def _fmt_num(v: float) -> str:
    """格式化数值用于 summary 字符串：整数显示为整数，小数保持原样。"""
    if v == int(v):
        return str(int(v))
    return str(v)


def _clean_inputs(inputs: Dict[str, float]) -> Dict[str, Any]:
    """将 float 输入值转换为 int（如果为整数），使 JSON 输出更干净。"""
    cleaned: Dict[str, Any] = {}
    for k, v in inputs.items():
        cleaned[k] = int(v) if isinstance(v, float) and v == int(v) else v
    return cleaned


# ============================================================================ #
#  核心计算函数                                                                 #
# ============================================================================ #

def calculate_metric(
    metric: str,
    tp: Optional[float] = None,
    fp: Optional[float] = None,
    fn: Optional[float] = None,
    tn: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
) -> Dict[str, Any]:
    """计算遥感语义分割评价指标。

    这是 Agent @tool（domain_tools.metrics_calculator）和
    MCP @tool（mcp_server.calculate_remote_sensing_metric）的共享内核。
    纯计算，无 LLM 调用，无 I/O 依赖。

    Args:
        metric: 指标名称（支持别名：iou, f1, dice, precision, recall 等）。
        tp: True Positives。None 表示未提供（区别于 0）。
        fp: False Positives。
        fn: False Negatives。
        tn: True Negatives（当前指标计算未使用，保留接口）。
        precision: 已知 Precision 值（仅 F1-score 模式 1 使用）。
        recall: 已知 Recall 值（仅 F1-score 模式 1 使用）。

    Returns:
        dict，固定包含 ``success`` 键。成功时额外包含::

            metric   —— 标准化指标名称
            inputs   —— 实际使用的输入值（int/float，已清洗）
            result   —— 计算结果（float，保留 4 位小数）
            formula  —— 公式文本
            summary  —— 人类可读的计算过程

        失败时额外包含::

            metric             —— 标准化或原始指标名称
            inputs             —— 已提供的输入值（可能部分）
            error              —— 错误描述
            summary            —— 参数提示或错误说明
            supported_metrics  —— （仅不支持该指标时）支持的指标列表
    """
    metric_norm = normalize_metric_name(metric)

    # ---- 1. 检查是否支持该指标 ----
    if metric_norm not in SUPPORTED_METRICS:
        return {
            "success": False,
            "metric": metric.strip(),
            "inputs": {},
            "error": f"暂不支持的指标: {metric.strip()}",
            "supported_metrics": list(SUPPORTED_METRICS),
            "summary": f"当前支持计算的指标: {', '.join(SUPPORTED_METRICS)}",
        }

    # ---- 2. 构建输入字典（仅包含非 None 的值） ----
    inputs: Dict[str, float] = {}
    if tp is not None:
        inputs["TP"] = tp
    if fp is not None:
        inputs["FP"] = fp
    if fn is not None:
        inputs["FN"] = fn
    if tn is not None:
        inputs["TN"] = tn
    if precision is not None:
        inputs["PRECISION"] = precision
    if recall is not None:
        inputs["RECALL"] = recall

    # ---- 3. 按指标类型计算 ----
    _tp = tp or 0
    _fp = fp or 0
    _fn = fn or 0

    if metric_norm == "IoU":
        if tp is None or fp is None or fn is None:
            return _build_fail(metric_norm, inputs)
        denom = _tp + _fp + _fn
        if denom == 0:
            return _build_fail(metric_norm, inputs)
        result = round(_tp / denom, 4)
        return _build_ok(
            metric_norm, inputs, result,
            "IoU = TP / (TP + FP + FN)",
            f"IoU = {_fmt_num(_tp)} / "
            f"({_fmt_num(_tp)} + {_fmt_num(_fp)} + {_fmt_num(_fn)}) "
            f"= {result}",
        )

    if metric_norm == "Precision":
        if tp is None or fp is None:
            return _build_fail(metric_norm, inputs)
        denom = _tp + _fp
        if denom == 0:
            return _build_fail(metric_norm, inputs)
        result = round(_tp / denom, 4)
        return _build_ok(
            metric_norm, inputs, result,
            "Precision = TP / (TP + FP)",
            f"Precision = {_fmt_num(_tp)} / "
            f"({_fmt_num(_tp)} + {_fmt_num(_fp)}) = {result}",
        )

    if metric_norm == "Recall":
        if tp is None or fn is None:
            return _build_fail(metric_norm, inputs)
        denom = _tp + _fn
        if denom == 0:
            return _build_fail(metric_norm, inputs)
        result = round(_tp / denom, 4)
        return _build_ok(
            metric_norm, inputs, result,
            "Recall = TP / (TP + FN)",
            f"Recall = {_fmt_num(_tp)} / "
            f"({_fmt_num(_tp)} + {_fmt_num(_fn)}) = {result}",
        )

    if metric_norm == "F1-score":
        # 模式 1：从 precision 和 recall 计算
        if precision is not None and recall is not None:
            denom = precision + recall
            if denom == 0:
                return _build_fail(metric_norm, inputs)
            result = round(2 * precision * recall / denom, 4)
            return _build_ok(
                metric_norm, inputs, result,
                "F1 = 2 * Precision * Recall / (Precision + Recall)",
                f"F1 = 2 * {_fmt_num(precision)} * {_fmt_num(recall)} / "
                f"({_fmt_num(precision)} + {_fmt_num(recall)}) = {result}",
            )
        # 模式 2：从 TP, FP, FN 计算
        if tp is not None and fp is not None and fn is not None:
            denom = 2 * _tp + _fp + _fn
            if denom == 0:
                return _build_fail(metric_norm, inputs)
            result = round(2 * _tp / denom, 4)
            return _build_ok(
                metric_norm, inputs, result,
                "F1 = 2*TP / (2*TP + FP + FN)",
                f"F1 = 2*{_fmt_num(_tp)} / "
                f"(2*{_fmt_num(_tp)} + {_fmt_num(_fp)} + {_fmt_num(_fn)}) "
                f"= {result}",
            )
        # 两种模式都不满足
        return _build_fail(metric_norm, inputs)

    # 理论上不会走到这里（前面已检查 supported）
    return _build_fail(metric_norm, inputs)


# ============================================================================ #
#  内部：构建返回 dict                                                          #
# ============================================================================ #

def _build_ok(
    metric_norm: str,
    inputs: Dict[str, float],
    result: float,
    formula: str,
    summary: str,
) -> Dict[str, Any]:
    """构建计算成功的返回 dict。"""
    return {
        "success": True,
        "metric": metric_norm,
        "inputs": _clean_inputs(inputs),
        "result": result,
        "formula": formula,
        "summary": summary,
    }


def _build_fail(
    metric_norm: str,
    inputs: Dict[str, float],
) -> Dict[str, Any]:
    """构建计算失败的返回 dict（缺少参数或分母为零）。"""
    return {
        "success": False,
        "metric": metric_norm,
        "inputs": _clean_inputs(inputs),
        "error": "缺少必需参数或分母为零",
        "summary": _PARAM_HINTS.get(
            metric_norm, "请检查输入参数是否完整。"
        ),
    }
