"""Multi-Tool 领域工具测试：验证 4 个确定性工具的正确性。

覆盖：
- 工具元信息（@tool 装饰、名称、英文 description）
- dataset_spec_lookup 命中 / 未命中 / 大小写 / 部分匹配
- model_comparison_table 多模型对比 / 未找到 / 字段完整性 / 无编造指标
- metric_formula_lookup 命中 / 大小写 / 别名 / 未找到
- metrics_calculator IoU / Precision / Recall / F1 双模式 / 不支持指标 / 非法输入
- 所有工具返回值为合法 JSON

不调用 LLM，不调用向量数据库。
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.agents.domain_data_loader import clear_domain_data_cache
from app.agents.domain_tools import (
    dataset_overview,
    dataset_spec_lookup,
    metric_formula_lookup,
    metrics_calculator,
    model_comparison_table,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个测试前后清空 lru_cache，确保数据从磁盘重新加载。"""
    clear_domain_data_cache()
    yield
    clear_domain_data_cache()


# ============================================================================ #
#  工具元信息                                                                   #
# ============================================================================ #

@pytest.mark.parametrize(
    "tool_obj,expected_name",
    [
        (dataset_spec_lookup, "dataset_spec_lookup"),
        (model_comparison_table, "model_comparison_table"),
        (metric_formula_lookup, "metric_formula_lookup"),
        (metrics_calculator, "metrics_calculator"),
    ],
)
def test_tool_has_correct_name(tool_obj, expected_name: str) -> None:
    """每个 @tool 装饰后的对象具有正确的 name 属性。"""
    assert hasattr(tool_obj, "name")
    assert tool_obj.name == expected_name


@pytest.mark.parametrize(
    "tool_obj",
    [dataset_spec_lookup, model_comparison_table, metric_formula_lookup, metrics_calculator],
)
def test_tool_has_invoke(tool_obj) -> None:
    """每个工具具有 invoke 方法。"""
    assert hasattr(tool_obj, "invoke")
    assert callable(tool_obj.invoke)


@pytest.mark.parametrize(
    "tool_obj",
    [dataset_spec_lookup, model_comparison_table, metric_formula_lookup, metrics_calculator],
)
def test_tool_description_in_english(tool_obj) -> None:
    """工具 description 为英文，帮助 LLM 判断何时调用。"""
    desc = tool_obj.description
    assert isinstance(desc, str)
    assert len(desc) > 10
    # 至少包含 "remote sensing" 或 "segmentation"
    desc_lower = desc.lower()
    assert "remote sensing" in desc_lower or "segmentation" in desc_lower or "metric" in desc_lower


# ============================================================================ #
#  dataset_spec_lookup                                                          #
# ============================================================================ #

def test_dataset_lookup_loveda() -> None:
    """查询 LoveDA 返回完整结构化信息。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "LoveDA"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["tool"] == "dataset_spec_lookup"
    assert data["query"] == "LoveDA"
    assert data["data"]["name"] == "LoveDA"
    assert data["data"]["classes"] == 7
    assert "建筑" in data["data"]["class_names"]
    assert "找到" in data["summary"]


def test_dataset_lookup_case_insensitive() -> None:
    """大小写不敏感：'loveda' 匹配 'LoveDA'。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "loveda"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["data"]["name"] == "LoveDA"


def test_dataset_lookup_partial_match() -> None:
    """部分匹配：'isa' 匹配 'iSAID'。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "isa"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["data"]["name"] == "iSAID"


def test_dataset_lookup_not_found() -> None:
    """查询不存在的数据集 → success=false + suggestion。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "COCO"})
    data = json.loads(raw)

    assert data["success"] is False
    assert data["data"] is None
    assert "未找到" in data["summary"]
    assert "knowledge_base_search" in data["suggestion"]


def test_dataset_lookup_empty_string() -> None:
    """空字符串 → success=false。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": ""})
    data = json.loads(raw)

    assert data["success"] is False


def test_dataset_lookup_potsdam_has_high_res() -> None:
    """Potsdam 分辨率为 0.05 m。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "Potsdam"})
    data = json.loads(raw)

    assert data["success"] is True
    assert "0.05" in data["data"]["resolution"]


# ============================================================================ #
#  model_comparison_table                                                       #
# ============================================================================ #

def test_model_comparison_two_models() -> None:
    """对比 U-Net 和 DeepLabV3+ → 返回 2 个模型的完整字段。"""
    raw = model_comparison_table.invoke({"models": "U-Net, DeepLabV3+"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["tool"] == "model_comparison_table"
    assert len(data["comparison"]) == 2
    assert "U-Net" in data["models_found"]
    assert "DeepLabV3+" in data["models_found"]
    assert data["models_not_found"] == []


def test_model_comparison_case_insensitive() -> None:
    """大小写不敏感：'u-net' 匹配 'U-Net'。"""
    raw = model_comparison_table.invoke({"models": "u-net, deeplabv3+"})
    data = json.loads(raw)

    assert data["success"] is True
    assert len(data["comparison"]) == 2


def test_model_comparison_partial_found() -> None:
    """部分找到：U-Net 找到，UnknownModel 未找到。"""
    raw = model_comparison_table.invoke({"models": "U-Net, UnknownModel"})
    data = json.loads(raw)

    assert data["success"] is True
    assert "U-Net" in data["models_found"]
    assert "UnknownModel" in data["models_not_found"]
    assert len(data["comparison"]) == 1


def test_model_comparison_all_not_found() -> None:
    """全部未找到 → success=false + suggestion。"""
    raw = model_comparison_table.invoke({"models": "ModelA, ModelB"})
    data = json.loads(raw)

    assert data["success"] is False
    assert data["comparison"] == []
    assert len(data["models_not_found"]) == 2
    assert "支持的模型" in data["suggestion"]


def test_model_comparison_empty_string() -> None:
    """空字符串 → success=false。"""
    raw = model_comparison_table.invoke({"models": ""})
    data = json.loads(raw)

    assert data["success"] is False


def test_model_comparison_has_required_fields() -> None:
    """对比结果包含 7 个必需字段。"""
    required = {
        "name", "architecture_type", "key_modules",
        "strengths", "limitations", "suitable_scenarios", "remote_sensing_notes",
    }

    raw = model_comparison_table.invoke({"models": "SegFormer, PSPNet"})
    data = json.loads(raw)

    for item in data["comparison"]:
        assert required.issubset(item.keys()), f"缺少字段: {item.get('name')}"


def test_model_comparison_no_fabricated_metrics() -> None:
    """对比结果不包含 mIoU / params / flops 等编造指标。"""
    forbidden = {"miou", "params", "flops", "parameters", "gflops"}

    raw = model_comparison_table.invoke({"models": "U-Net, FCN"})
    data = json.loads(raw)

    for item in data["comparison"]:
        for key in item:
            assert key.lower() not in forbidden, f"不应包含编造指标字段: {key}"


def test_model_comparison_three_models() -> None:
    """对比 3 个模型。"""
    raw = model_comparison_table.invoke({"models": "U-Net, DeepLabV3+, SegFormer"})
    data = json.loads(raw)

    assert data["success"] is True
    assert len(data["comparison"]) == 3
    assert "3" in data["summary"]


# ============================================================================ #
#  metric_formula_lookup                                                        #
# ============================================================================ #

def test_metric_lookup_miou() -> None:
    """查询 mIoU 返回公式与说明。"""
    raw = metric_formula_lookup.invoke({"metric_name": "mIoU"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["tool"] == "metric_formula_lookup"
    assert data["query"] == "mIoU"
    assert data["data"]["name"] == "mIoU"
    assert "IoU" in data["data"]["formula"]
    assert "找到" in data["summary"]


def test_metric_lookup_case_insensitive() -> None:
    """大小写不敏感：'iou' 匹配 'IoU'。"""
    raw = metric_formula_lookup.invoke({"metric_name": "iou"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["data"]["name"] == "IoU"


def test_metric_lookup_partial_match() -> None:
    """部分匹配：'precision' 匹配 'Precision'。"""
    raw = metric_formula_lookup.invoke({"metric_name": "precision"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["data"]["name"] == "Precision"


def test_metric_lookup_not_found() -> None:
    """查询不存在的指标 → success=false + suggestion。"""
    raw = metric_formula_lookup.invoke({"metric_name": "mAP"})
    data = json.loads(raw)

    assert data["success"] is False
    assert data["data"] is None
    assert "未找到" in data["summary"]


def test_metric_lookup_f1_score() -> None:
    """查询 F1-score 返回完整信息。"""
    raw = metric_formula_lookup.invoke({"metric_name": "F1-score"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["data"]["name"] == "F1-score"
    assert "Precision" in data["data"]["formula"]
    assert "Recall" in data["data"]["formula"]


# ============================================================================ #
#  metrics_calculator                                                           #
# ============================================================================ #

def test_calculator_iou() -> None:
    """IoU = 80 / (80 + 10 + 20) = 0.7273。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["tool"] == "metrics_calculator"
    assert data["metric"] == "IoU"
    assert data["result"] == 0.7273
    assert data["inputs"]["TP"] == 80
    assert data["inputs"]["FP"] == 10
    assert data["inputs"]["FN"] == 20
    assert "TP" in data["formula"]
    assert "0.7273" in data["summary"]


def test_calculator_precision() -> None:
    """Precision = 80 / (80 + 10) = 0.8889。"""
    raw = metrics_calculator.invoke({
        "metric_name": "Precision",
        "values": "TP=80, FP=10",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["result"] == 0.8889
    assert data["metric"] == "Precision"


def test_calculator_recall() -> None:
    """Recall = 80 / (80 + 20) = 0.8。"""
    raw = metrics_calculator.invoke({
        "metric_name": "Recall",
        "values": "TP=80, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["result"] == 0.8


def test_calculator_f1_from_precision_recall() -> None:
    """F1 从 precision 和 recall 计算。"""
    raw = metrics_calculator.invoke({
        "metric_name": "F1-score",
        "values": "precision=0.85, recall=0.90",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["metric"] == "F1-score"
    # F1 = 2 * 0.85 * 0.90 / (0.85 + 0.90) = 1.53 / 1.75 = 0.8743
    assert data["result"] == 0.8743
    assert "Precision" in data["formula"]


def test_calculator_f1_from_tp_fp_fn() -> None:
    """F1 从 TP, FP, FN 计算（等价于 Dice）。"""
    raw = metrics_calculator.invoke({
        "metric_name": "F1-score",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    # F1 = 2*80 / (2*80 + 10 + 20) = 160 / 190 = 0.8421
    assert data["result"] == 0.8421


def test_calculator_f1_alias() -> None:
    """别名 'f1' 归一化为 'F1-score'。"""
    raw = metrics_calculator.invoke({
        "metric_name": "f1",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["metric"] == "F1-score"


def test_calculator_dice_alias() -> None:
    """别名 'dice' 归一化为 'F1-score'。"""
    raw = metrics_calculator.invoke({
        "metric_name": "dice",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["metric"] == "F1-score"


def test_calculator_iou_alias() -> None:
    """别名 'iou' 归一化为 'IoU'。"""
    raw = metrics_calculator.invoke({
        "metric_name": "iou",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["metric"] == "IoU"


def test_calculator_colon_separator() -> None:
    """冒号分隔符 ':80' 也能解析。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "TP:80, FP:10, FN:20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["result"] == 0.7273


def test_calculator_lowercase_keys() -> None:
    """小写键名 'tp=80' 也能解析。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "tp=80, fp=10, fn=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert data["result"] == 0.7273


def test_calculator_unsupported_metric() -> None:
    """不支持的指标 → success=false + supported_metrics。"""
    raw = metrics_calculator.invoke({
        "metric_name": "mAP",
        "values": "TP=80",
    })
    data = json.loads(raw)

    assert data["success"] is False
    assert "暂不支持" in data["error"]
    assert "IoU" in data["supported_metrics"]
    assert "F1-score" in data["supported_metrics"]


def test_calculator_invalid_values() -> None:
    """无法解析的 values → success=false + 格式提示。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "这是一段乱码",
    })
    data = json.loads(raw)

    assert data["success"] is False
    assert data["inputs"] == {}
    assert "无法解析" in data["error"]


def test_calculator_empty_values() -> None:
    """空 values → success=false。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "",
    })
    data = json.loads(raw)

    assert data["success"] is False


def test_calculator_missing_params() -> None:
    """IoU 只提供 TP，缺少 FP 和 FN → success=false + 提示。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "TP=80",
    })
    data = json.loads(raw)

    assert data["success"] is False
    assert "TP" in data["inputs"]
    assert data["inputs"]["TP"] == 80
    assert "缺少" in data["error"] or "分母" in data["error"]


def test_calculator_zero_denominator() -> None:
    """全零输入 → 分母为零 → success=false。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "TP=0, FP=0, FN=0",
    })
    data = json.loads(raw)

    assert data["success"] is False


def test_calculator_returns_formula() -> None:
    """计算成功时返回 formula 字段。"""
    raw = metrics_calculator.invoke({
        "metric_name": "Recall",
        "values": "TP=80, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    assert "formula" in data
    assert "Recall" in data["formula"]


def test_calculator_summary_contains_numbers() -> None:
    """summary 字符串包含计算过程。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert data["success"] is True
    summary = data["summary"]
    assert "80" in summary
    assert "110" in summary or "80 + 10 + 20" in summary


# ============================================================================ #
#  所有工具返回合法 JSON                                                         #
# ============================================================================ #

@pytest.mark.parametrize(
    "tool_obj,invoke_args",
    [
        (dataset_spec_lookup, {"dataset_name": "LoveDA"}),
        (dataset_spec_lookup, {"dataset_name": "NotFound"}),
        (model_comparison_table, {"models": "U-Net, DeepLabV3+"}),
        (model_comparison_table, {"models": "NotFound"}),
        (metric_formula_lookup, {"metric_name": "mIoU"}),
        (metric_formula_lookup, {"metric_name": "NotFound"}),
        (metrics_calculator, {"metric_name": "IoU", "values": "TP=80, FP=10, FN=20"}),
        (metrics_calculator, {"metric_name": "IoU", "values": "garbage"}),
    ],
)
def test_all_returns_valid_json(tool_obj, invoke_args) -> None:
    """所有工具在正常和异常路径下都返回可解析的 JSON。"""
    raw = tool_obj.invoke(invoke_args)
    assert isinstance(raw, str)
    data = json.loads(raw)  # 不抛异常即为合法 JSON
    assert "success" in data
    assert "tool" in data


# ============================================================================ #
#  与 knowledge_base_search 共存                                                 #
# ============================================================================ #

def test_knowledge_base_search_still_exists() -> None:
    """knowledge_base_search 仍然可正常导入和使用。"""
    from app.agents.tools import knowledge_base_search

    assert hasattr(knowledge_base_search, "name")
    assert knowledge_base_search.name == "knowledge_base_search"
    assert hasattr(knowledge_base_search, "invoke")


def test_domain_tools_imported_alongside_existing() -> None:
    """新工具与旧工具可以从各自模块独立导入。"""
    from app.agents.tools import knowledge_base_search  # noqa: F401
    from app.agents.domain_tools import (  # noqa: F401
        dataset_spec_lookup,
        metric_formula_lookup,
        metrics_calculator,
        model_comparison_table,
    )


# ============================================================================ #
#  Block 2: 工具输出长度限制验证                                                 #
# ============================================================================ #

def test_dataset_lookup_summary_within_limit() -> None:
    """dataset_spec_lookup summary 不超过 200 字符。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "LoveDA"})
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_dataset_lookup_summary_not_found_within_limit() -> None:
    """dataset_spec_lookup 未找到时 summary 不超过 200 字符。"""
    raw = dataset_spec_lookup.invoke({"dataset_name": "NotFound"})
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_model_comparison_summary_within_limit() -> None:
    """model_comparison_table summary 不超过 200 字符。"""
    raw = model_comparison_table.invoke({"models": "U-Net, DeepLabV3+, SegFormer"})
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_metric_lookup_summary_within_limit() -> None:
    """metric_formula_lookup summary 不超过 200 字符。"""
    raw = metric_formula_lookup.invoke({"metric_name": "mIoU"})
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_calculator_summary_within_limit() -> None:
    """metrics_calculator summary 不超过 200 字符。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "TP=80, FP=10, FN=20",
    })
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_calculator_summary_error_within_limit() -> None:
    """metrics_calculator 错误时 summary 不超过 200 字符。"""
    raw = metrics_calculator.invoke({
        "metric_name": "IoU",
        "values": "garbage",
    })
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200


def test_all_domain_tool_outputs_are_valid_json_and_compact() -> None:
    """所有领域工具输出为合法 JSON 且 summary 不超长。"""
    calls = [
        (dataset_overview, {"query": "数据集特点"}),
        (dataset_spec_lookup, {"dataset_name": "LoveDA"}),
        (dataset_spec_lookup, {"dataset_name": "NotFound"}),
        (model_comparison_table, {"models": "U-Net, DeepLabV3+"}),
        (model_comparison_table, {"models": "NotFound"}),
        (metric_formula_lookup, {"metric_name": "mIoU"}),
        (metric_formula_lookup, {"metric_name": "NotFound"}),
        (metrics_calculator, {"metric_name": "IoU", "values": "TP=80, FP=10, FN=20"}),
        (metrics_calculator, {"metric_name": "IoU", "values": "garbage"}),
    ]

    for tool_obj, args in calls:
        raw = tool_obj.invoke(args)
        data = json.loads(raw)  # 合法 JSON
        assert "success" in data
        summary = data.get("summary", "")
        assert len(summary) <= 200, f"{tool_obj.name} summary 过长: {len(summary)}"


# ============================================================================ #
#  Block 3: dataset_overview 工具测试                                           #
# ============================================================================ #

def test_dataset_overview_returns_valid_json() -> None:
    """dataset_overview 返回合法 JSON。"""
    raw = dataset_overview.invoke({"query": "数据集有什么特点"})
    data = json.loads(raw)

    assert data["success"] is True
    assert data["tool"] == "dataset_overview"


def test_dataset_overview_returns_common_features() -> None:
    """dataset_overview 返回 common_features 列表。"""
    raw = dataset_overview.invoke({"query": ""})
    data = json.loads(raw)

    assert data["success"] is True
    features = data.get("common_features", [])
    assert isinstance(features, list)
    assert len(features) > 0
    assert "高空间分辨率" in features
    assert "类别不平衡" in features


def test_dataset_overview_returns_common_challenges() -> None:
    """dataset_overview 返回 common_challenges 列表。"""
    raw = dataset_overview.invoke({"query": "挑战"})
    data = json.loads(raw)

    assert data["success"] is True
    challenges = data.get("common_challenges", [])
    assert isinstance(challenges, list)
    assert len(challenges) > 0


def test_dataset_overview_returns_related_datasets() -> None:
    """dataset_overview 返回 related_datasets 列表。"""
    raw = dataset_overview.invoke({"query": ""})
    data = json.loads(raw)

    assert data["success"] is True
    related = data.get("related_datasets", [])
    assert isinstance(related, list)
    assert len(related) > 0
    assert "LoveDA" in related


def test_dataset_overview_summary_within_limit() -> None:
    """dataset_overview summary 不超过 200 字符。"""
    raw = dataset_overview.invoke({"query": ""})
    data = json.loads(raw)

    assert len(data.get("summary", "")) <= 200
    assert len(data.get("summary_short", "")) <= 200


def test_dataset_overview_no_query_arg() -> None:
    """dataset_overview 无参数时正常返回。"""
    raw = dataset_overview.invoke({})
    data = json.loads(raw)

    assert data["success"] is True


def test_dataset_overview_does_not_call_llm() -> None:
    """dataset_overview 不调用 LLM（纯本地数据查询）。"""
    with patch("app.agents.domain_tools.get_datasets_data") as mock_data:
        mock_data.return_value = [{"name": "LoveDA"}, {"name": "iSAID"}]

        raw = dataset_overview.invoke({"query": "test"})
        data = json.loads(raw)

    assert data["success"] is True
    assert data["related_datasets"] == ["LoveDA", "iSAID"]


def test_dataset_overview_does_not_call_vector_db() -> None:
    """dataset_overview 不调用向量数据库。"""
    with patch("app.agents.domain_tools.get_datasets_data") as mock_data:
        mock_data.return_value = [{"name": "LoveDA"}]

        raw = dataset_overview.invoke({"query": "test"})

    data = json.loads(raw)
    assert "sources" not in data
    assert "contexts" not in data


def test_dataset_overview_empty_datasets() -> None:
    """本地数据集为空时返回 success=false。"""
    with patch("app.agents.domain_tools.get_datasets_data") as mock_data:
        mock_data.return_value = []

        raw = dataset_overview.invoke({"query": "test"})
        data = json.loads(raw)

    assert data["success"] is False
    assert data["related_datasets"] == []


def test_dataset_overview_tool_name() -> None:
    """dataset_overview 工具名为 dataset_overview。"""
    assert dataset_overview.name == "dataset_overview"


def test_dataset_overview_description_in_english() -> None:
    """dataset_overview description 为英文（便于 LLM 选择工具）。"""
    desc = dataset_overview.description
    assert "Use this tool" in desc


def test_dataset_overview_is_tool() -> None:
    """dataset_overview 是 LangChain tool。"""
    from langchain_core.tools import BaseTool

    assert isinstance(dataset_overview, BaseTool)


def test_dataset_overview_has_invoke() -> None:
    """dataset_overview 有 invoke 方法。"""
    assert hasattr(dataset_overview, "invoke")
    assert callable(dataset_overview.invoke)
