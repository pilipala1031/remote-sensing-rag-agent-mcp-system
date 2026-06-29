"""Multi-Tool 领域工具：确定性结构化查询工具。

5 个工具不调用 LLM 和向量库，直接从 JSON 数据查询或数值计算：
- dataset_overview        遥感数据集共性概览
- dataset_spec_lookup     查询遥感数据集结构化信息
- model_comparison_table  对比一个或多个语义分割模型
- metric_formula_lookup   查询评价指标定义、公式、适用场景
- metrics_calculator      根据用户提供的数值计算常见指标（调用 app.core.metrics 共享内核）

metrics_calculator 的核心计算逻辑已抽取到 app/core/metrics.py，
与 MCP Server（mcp_server/server.py）共享同一套计算内核，避免逻辑重复。

所有工具返回合法 JSON 字符串，异常时返回 success=false。
不破坏已有 app/agents/tools.py 中的 knowledge_base_search。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.agents.domain_data_loader import (
    get_datasets_data,
    get_metrics_data,
    get_models_data,
)
from app.core.metrics import calculate_metric, normalize_metric_name
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================ #
#  常量                                                                         #
# ============================================================================ #

#: model_comparison_table 提取的对比字段
_MODEL_COMPARISON_FIELDS: List[str] = [
    "name",
    "architecture_type",
    "key_modules",
    "strengths",
    "limitations",
    "suitable_scenarios",
    "remote_sensing_notes",
]

#: dataset_overview 返回的共性特征列表
_DATASET_COMMON_FEATURES: List[str] = [
    "高空间分辨率",
    "多类别地物覆盖",
    "类别不平衡",
    "尺度变化明显",
    "小目标密集",
    "边界复杂",
    "跨区域泛化困难",
    "标注成本高",
]

#: dataset_overview 返回的共性挑战列表
_DATASET_COMMON_CHALLENGES: List[str] = [
    "城乡场景差异",
    "小目标识别",
    "阴影和遮挡",
    "地物边界模糊",
    "跨数据集泛化",
]


# ============================================================================ #
#  辅助函数                                                                     #
# ============================================================================ #

def _find_by_name(
    items: List[Dict[str, Any]], query: str
) -> Optional[Dict[str, Any]]:
    """大小写不敏感的名称查找，先精确匹配后包含匹配。"""
    query_clean = query.strip().lower()
    if not query_clean:
        return None

    # 精确匹配
    for item in items:
        if item.get("name", "").lower() == query_clean:
            return item

    # 包含匹配（用户输入是名称的子串）
    for item in items:
        name = item.get("name", "").lower()
        if query_clean in name:
            return item

    return None


def _parse_values(values: str) -> Dict[str, float]:
    """解析 'TP=80, FP=10, FN=20' 格式的输入字符串。

    支持 = 和 : 作为键值分隔符，支持 , 和 ; 作为对分隔符。
    所有键名转为大写。
    """
    result: Dict[str, float] = {}
    parts = re.split(r"[,;]\s*", values.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"(\w+)\s*[=:]\s*([-+]?\d*\.?\d+)", part)
        if match:
            key = match.group(1).upper()
            val = float(match.group(2))
            result[key] = val
    return result


# ============================================================================ #
#  @tool 工具                                                                   #
# ============================================================================ #

@tool
def dataset_overview(query: str = "") -> str:
    """Use this tool when the user asks about general characteristics, common challenges, or overall patterns of remote sensing semantic segmentation datasets. Use dataset_spec_lookup only when the user explicitly names a specific dataset such as LoveDA, iSAID, DeepGlobe, Potsdam, or Vaihingen.

    Args:
        query: Optional user query string (for logging/reference only).

    Returns:
        A JSON string with common features, common challenges, and related dataset names.
    """
    try:
        datasets = get_datasets_data()
        related_datasets = [ds.get("name", "") for ds in datasets if ds.get("name")]
        dataset_count = len(related_datasets)

        if dataset_count == 0:
            result: Dict[str, Any] = {
                "success": False,
                "tool": "dataset_overview",
                "query": query.strip() if query else "",
                "summary": "本地结构化数据集信息为空。",
                "common_features": [],
                "common_challenges": [],
                "related_datasets": [],
                "summary_short": "未找到本地结构化数据集信息。",
            }
            logger.info("dataset_overview 本地数据集为空")
            return json.dumps(result, ensure_ascii=False)

        result = {
            "success": True,
            "tool": "dataset_overview",
            "query": query.strip() if query else "",
            "summary": (
                "遥感语义分割数据集通常具有高空间分辨率、多类别地物覆盖、"
                "类别不平衡、尺度差异明显、标注成本高和场景分布复杂等特点。"
            ),
            "common_features": list(_DATASET_COMMON_FEATURES),
            "common_challenges": list(_DATASET_COMMON_CHALLENGES),
            "related_datasets": related_datasets,
            "summary_short": (
                f"已总结本地结构化数据集中 {dataset_count} 个遥感语义分割数据集的共同特点。"
            ),
        }
        logger.info("dataset_overview 成功: %d 个数据集", dataset_count)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("dataset_overview 异常: %s", e)
        return json.dumps(
            {
                "success": False,
                "tool": "dataset_overview",
                "query": query if query else "",
                "summary": "工具执行异常。",
                "common_features": [],
                "common_challenges": [],
                "related_datasets": [],
                "summary_short": "工具执行异常。",
                "error": str(e),
            },
            ensure_ascii=False,
        )


@tool
def dataset_spec_lookup(dataset_name: str) -> str:
    """Look up structured specifications for a remote sensing semantic segmentation dataset.

    Use this tool when the user asks about specific remote sensing dataset properties,
    such as resolution, classes, scenes, strengths, limitations, or notes.

    Supported datasets: LoveDA, iSAID, DeepGlobe, Potsdam, Vaihingen.

    Args:
        dataset_name: The name of the dataset (e.g., "LoveDA", "iSAID").
                      Case-insensitive, supports partial matching.

    Returns:
        A JSON string with the dataset's full structured information.
    """
    try:
        datasets = get_datasets_data()
        found = _find_by_name(datasets, dataset_name)

        if found is None:
            result: Dict[str, Any] = {
                "success": False,
                "tool": "dataset_spec_lookup",
                "query": dataset_name.strip(),
                "data": None,
                "summary": "未找到该数据集的结构化信息。",
                "suggestion": "可以尝试使用 knowledge_base_search 进行语义检索。",
            }
            logger.info("dataset_spec_lookup 未找到: %s", dataset_name)
            return json.dumps(result, ensure_ascii=False)

        result = {
            "success": True,
            "tool": "dataset_spec_lookup",
            "query": dataset_name.strip(),
            "data": found,
            "summary": f"找到 {found['name']} 数据集的结构化信息。",
        }
        logger.info("dataset_spec_lookup 命中: %s", found["name"])
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("dataset_spec_lookup 异常: %s", e)
        return json.dumps(
            {
                "success": False,
                "tool": "dataset_spec_lookup",
                "query": dataset_name,
                "data": None,
                "summary": "工具执行异常。",
                "error": str(e),
            },
            ensure_ascii=False,
        )


@tool
def model_comparison_table(models: str) -> str:
    """Compare one or more semantic segmentation models side by side.

    Use this tool when the user asks to compare segmentation models, such as
    U-Net, DeepLabV3+, SegFormer, PSPNet, FCN, or Swin-Transformer.

    Provides architecture type, key modules, strengths, limitations,
    suitable scenarios, and remote sensing notes for each model.
    Does NOT fabricate quantitative metrics like mIoU, parameters, or FLOPs.

    Args:
        models: Comma-separated model names (e.g., "U-Net, DeepLabV3+").

    Returns:
        A JSON string with structured comparison data for each model found.
    """
    try:
        all_models = get_models_data()
        # 解析逗号分隔的模型名
        names = [n.strip() for n in models.split(",") if n.strip()]

        found_list: List[Dict[str, Any]] = []
        found_names: List[str] = []
        not_found_names: List[str] = []

        for name in names:
            model = _find_by_name(all_models, name)
            if model is not None:
                comparison_item = {k: model.get(k) for k in _MODEL_COMPARISON_FIELDS}
                found_list.append(comparison_item)
                found_names.append(model["name"])
            else:
                not_found_names.append(name)

        if not found_list:
            result: Dict[str, Any] = {
                "success": False,
                "tool": "model_comparison_table",
                "query": models.strip(),
                "models_found": [],
                "models_not_found": not_found_names,
                "comparison": [],
                "summary": "未找到任何匹配的模型。",
                "suggestion": (
                    f"支持的模型: {', '.join(m['name'] for m in all_models)}"
                ),
            }
            logger.info("model_comparison_table 未找到任何模型: %s", names)
            return json.dumps(result, ensure_ascii=False)

        result = {
            "success": True,
            "tool": "model_comparison_table",
            "query": models.strip(),
            "models_found": found_names,
            "models_not_found": not_found_names,
            "comparison": found_list,
            "summary": f"找到 {len(found_list)} 个模型进行对比。",
        }
        logger.info(
            "model_comparison_table 找到 %d 个模型, 未找到 %d 个",
            len(found_list),
            len(not_found_names),
        )
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("model_comparison_table 异常: %s", e)
        return json.dumps(
            {
                "success": False,
                "tool": "model_comparison_table",
                "query": models,
                "comparison": [],
                "summary": "工具执行异常。",
                "error": str(e),
            },
            ensure_ascii=False,
        )


@tool
def metric_formula_lookup(metric_name: str) -> str:
    """Look up the definition, formula, advantages, limitations, and use cases
    of a remote sensing semantic segmentation evaluation metric.

    Use this tool when the user asks about definitions, formulas, advantages,
    limitations, or use cases of remote sensing semantic segmentation metrics
    such as mIoU, IoU, FWIoU, PA, MPA, Precision, Recall, or F1-score.

    Args:
        metric_name: The metric name (e.g., "mIoU", "IoU", "F1-score").
                     Case-insensitive, supports partial matching.

    Returns:
        A JSON string with the metric's full information including formula.
    """
    try:
        metrics = get_metrics_data()
        found = _find_by_name(metrics, metric_name)

        if found is None:
            result: Dict[str, Any] = {
                "success": False,
                "tool": "metric_formula_lookup",
                "query": metric_name.strip(),
                "data": None,
                "summary": "未找到该评价指标的信息。",
                "suggestion": "可以尝试使用 knowledge_base_search 进行语义检索。",
            }
            logger.info("metric_formula_lookup 未找到: %s", metric_name)
            return json.dumps(result, ensure_ascii=False)

        result = {
            "success": True,
            "tool": "metric_formula_lookup",
            "query": metric_name.strip(),
            "data": found,
            "summary": f"找到 {found['name']} 指标的详细信息。",
        }
        logger.info("metric_formula_lookup 命中: %s", found["name"])
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error("metric_formula_lookup 异常: %s", e)
        return json.dumps(
            {
                "success": False,
                "tool": "metric_formula_lookup",
                "query": metric_name,
                "data": None,
                "summary": "工具执行异常。",
                "error": str(e),
            },
            ensure_ascii=False,
        )


@tool
def metrics_calculator(metric_name: str, values: str) -> str:
    """Calculate a semantic segmentation evaluation metric from raw values.

    Use this tool when the user provides numerical values and asks to calculate
    IoU, Precision, Recall, or F1-score.

    Supported metrics and required inputs:
    - IoU:       TP, FP, FN
    - Precision: TP, FP
    - Recall:    TP, FN
    - F1-score:  precision, recall   OR   TP, FP, FN

    Args:
        metric_name: One of "IoU", "Precision", "Recall", "F1-score".
                     Case-insensitive (accepts "f1", "dice", "iou", etc.).
        values:      Key-value pairs, e.g. "TP=80, FP=10, FN=20"
                     or "precision=0.85, recall=0.90".

    Returns:
        A JSON string with the calculated result, formula, and computation summary.
    """
    try:
        # 解析输入值字符串（@tool 特有逻辑，MCP 版本使用类型化参数无需解析）
        parsed = _parse_values(values)

        if not parsed:
            result: Dict[str, Any] = {
                "success": False,
                "tool": "metrics_calculator",
                "metric": normalize_metric_name(metric_name),
                "inputs": {},
                "error": "无法解析输入值",
                "summary": (
                    "输入格式提示: 'TP=80, FP=10, FN=20'"
                    "（用逗号分隔键值对，支持 = 或 : 分隔符）"
                ),
            }
            logger.info("metrics_calculator 无法解析: values=%r", values)
            return json.dumps(result, ensure_ascii=False)

        # 调用共享内核（app.core.metrics.calculate_metric）
        calc = calculate_metric(
            metric=metric_name,
            tp=parsed.get("TP"),
            fp=parsed.get("FP"),
            fn=parsed.get("FN"),
            tn=parsed.get("TN"),
            precision=parsed.get("PRECISION"),
            recall=parsed.get("RECALL"),
        )

        # 添加 tool 字段（@tool 输出特有，MCP 版本不需要此字段）
        calc["tool"] = "metrics_calculator"

        logger.info(
            "metrics_calculator: metric=%s, success=%s",
            calc.get("metric"),
            calc.get("success"),
        )
        return json.dumps(calc, ensure_ascii=False)

    except Exception as e:
        logger.error("metrics_calculator 异常: %s", e)
        return json.dumps(
            {
                "success": False,
                "tool": "metrics_calculator",
                "metric": metric_name,
                "inputs": {},
                "error": str(e),
                "summary": "工具执行异常。",
            },
            ensure_ascii=False,
        )
