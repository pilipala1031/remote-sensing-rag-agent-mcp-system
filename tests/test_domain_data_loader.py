"""领域结构化数据加载器测试。

测试覆盖：
- load_json_data 正常 / 异常 / 边界场景
- get_datasets_data / get_models_data / get_metrics_data 返回正确数据
- @lru_cache 缓存命中与清除
- 数据内容完整性校验（字段齐全、数据量符合预期）

不调用 LLM，不依赖外部 API。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.agents.domain_data_loader import (
    _DOMAIN_DATA_DIR,
    clear_domain_data_cache,
    get_datasets_data,
    get_metrics_data,
    get_models_data,
    load_json_data,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个测试前后清空 lru_cache，避免测试间互相干扰。"""
    clear_domain_data_cache()
    yield
    clear_domain_data_cache()


# ============================================================================ #
#  load_json_data 基础功能                                                       #
# ============================================================================ #

def test_load_json_data_valid_file(tmp_path: Path) -> None:
    """正常 JSON 文件返回解析后的列表。"""
    test_file = tmp_path / "test.json"
    test_data = [{"name": "test", "value": 123}]
    test_file.write_text(json.dumps(test_data), encoding="utf-8")

    result = load_json_data(test_file)
    assert result == test_data


def test_load_json_data_file_not_found(tmp_path: Path) -> None:
    """文件不存在时返回空列表。"""
    result = load_json_data(tmp_path / "nonexistent.json")
    assert result == []


def test_load_json_data_invalid_json(tmp_path: Path) -> None:
    """非法 JSON 返回空列表。"""
    test_file = tmp_path / "bad.json"
    test_file.write_text("这不是 JSON {{{", encoding="utf-8")

    result = load_json_data(test_file)
    assert result == []


def test_load_json_data_not_a_list(tmp_path: Path) -> None:
    """JSON 顶层不是列表时返回空列表。"""
    test_file = tmp_path / "dict.json"
    test_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")

    result = load_json_data(test_file)
    assert result == []


def test_load_json_data_empty_list(tmp_path: Path) -> None:
    """空列表 JSON 返回空列表。"""
    test_file = tmp_path / "empty.json"
    test_file.write_text("[]", encoding="utf-8")

    result = load_json_data(test_file)
    assert result == []


def test_load_json_data_empty_file(tmp_path: Path) -> None:
    """空文件返回空列表。"""
    test_file = tmp_path / "blank.json"
    test_file.write_text("", encoding="utf-8")

    result = load_json_data(test_file)
    assert result == []


def test_load_json_data_chinese_content(tmp_path: Path) -> None:
    """含中文的 JSON 正确解析。"""
    test_file = tmp_path / "cn.json"
    test_data = [{"name": "遥感数据集", "task": "语义分割"}]
    test_file.write_text(json.dumps(test_data, ensure_ascii=False), encoding="utf-8")

    result = load_json_data(test_file)
    assert len(result) == 1
    assert result[0]["name"] == "遥感数据集"
    assert result[0]["task"] == "语义分割"


def test_load_json_data_accepts_string_path(tmp_path: Path) -> None:
    """接受 str 类型路径。"""
    test_file = tmp_path / "str.json"
    test_data = [{"a": 1}]
    test_file.write_text(json.dumps(test_data), encoding="utf-8")

    result = load_json_data(str(test_file))
    assert result == test_data


# ============================================================================ #
#  get_datasets_data                                                            #
# ============================================================================ #

def test_get_datasets_data_returns_list() -> None:
    """get_datasets_data 返回列表。"""
    data = get_datasets_data()
    assert isinstance(data, list)


def test_get_datasets_data_has_expected_count() -> None:
    """数据集数量符合预期（5 个）。"""
    data = get_datasets_data()
    assert len(data) == 5


def test_get_datasets_data_names() -> None:
    """包含预期的 5 个数据集名称。"""
    data = get_datasets_data()
    names = {item["name"] for item in data}
    expected = {"LoveDA", "iSAID", "DeepGlobe", "Potsdam", "Vaihingen"}
    assert names == expected


def test_get_datasets_data_required_fields() -> None:
    """每条记录包含所有必需字段。"""
    required_keys = {
        "name", "full_name", "task", "resolution", "classes",
        "class_names", "scenes", "image_count", "image_size",
        "strengths", "limitations", "notes",
    }
    data = get_datasets_data()
    for item in data:
        assert required_keys.issubset(item.keys()), f"缺少字段: {item.get('name')}"


def test_get_datasets_data_classes_are_int() -> None:
    """classes 字段为整数。"""
    data = get_datasets_data()
    for item in data:
        assert isinstance(item["classes"], int)
        assert item["classes"] > 0


def test_get_datasets_data_strengths_is_list() -> None:
    """strengths 和 limitations 为列表。"""
    data = get_datasets_data()
    for item in data:
        assert isinstance(item["strengths"], list)
        assert isinstance(item["limitations"], list)
        assert len(item["strengths"]) > 0
        assert len(item["limitations"]) > 0


def test_get_datasets_data_class_names_length_matches_classes() -> None:
    """class_names 列表长度与 classes 数值一致。"""
    data = get_datasets_data()
    for item in data:
        assert len(item["class_names"]) == item["classes"], \
            f"{item['name']}: class_names 长度({len(item['class_names'])}) != classes({item['classes']})"


# ============================================================================ #
#  get_models_data                                                              #
# ============================================================================ #

def test_get_models_data_returns_list() -> None:
    """get_models_data 返回列表。"""
    data = get_models_data()
    assert isinstance(data, list)


def test_get_models_data_has_expected_count() -> None:
    """模型数量符合预期（6 个）。"""
    data = get_models_data()
    assert len(data) == 6


def test_get_models_data_names() -> None:
    """包含预期的 6 个模型名称。"""
    data = get_models_data()
    names = {item["name"] for item in data}
    expected = {"U-Net", "DeepLabV3+", "SegFormer", "PSPNet", "FCN", "Swin-Transformer"}
    assert names == expected


def test_get_models_data_required_fields() -> None:
    """每条记录包含所有必需字段。"""
    required_keys = {
        "name", "architecture_type", "key_modules",
        "strengths", "limitations", "suitable_scenarios", "remote_sensing_notes",
    }
    data = get_models_data()
    for item in data:
        assert required_keys.issubset(item.keys()), f"缺少字段: {item.get('name')}"


def test_get_models_data_key_modules_is_list() -> None:
    """key_modules 为非空列表。"""
    data = get_models_data()
    for item in data:
        assert isinstance(item["key_modules"], list)
        assert len(item["key_modules"]) > 0


def test_get_models_data_has_remote_sensing_notes() -> None:
    """每条记录有 remote_sensing_notes 遥感说明。"""
    data = get_models_data()
    for item in data:
        assert isinstance(item["remote_sensing_notes"], str)
        assert len(item["remote_sensing_notes"]) > 0


# ============================================================================ #
#  get_metrics_data                                                             #
# ============================================================================ #

def test_get_metrics_data_returns_list() -> None:
    """get_metrics_data 返回列表。"""
    data = get_metrics_data()
    assert isinstance(data, list)


def test_get_metrics_data_has_expected_count() -> None:
    """指标数量符合预期（8 个）。"""
    data = get_metrics_data()
    assert len(data) == 8


def test_get_metrics_data_names() -> None:
    """包含预期的 8 个指标名称。"""
    data = get_metrics_data()
    names = {item["name"] for item in data}
    expected = {"PA", "MPA", "IoU", "mIoU", "FWIoU", "Precision", "Recall", "F1-score"}
    assert names == expected


def test_get_metrics_data_required_fields() -> None:
    """每条记录包含所有必需字段。"""
    required_keys = {
        "name", "full_name", "formula", "meaning",
        "advantages", "limitations", "use_cases",
    }
    data = get_metrics_data()
    for item in data:
        assert required_keys.issubset(item.keys()), f"缺少字段: {item.get('name')}"


def test_get_metrics_data_formula_not_empty() -> None:
    """每条记录的 formula 非空。"""
    data = get_metrics_data()
    for item in data:
        assert isinstance(item["formula"], str)
        assert len(item["formula"]) > 0


def test_get_metrics_data_meaning_not_empty() -> None:
    """每条记录的 meaning 非空。"""
    data = get_metrics_data()
    for item in data:
        assert isinstance(item["meaning"], str)
        assert len(item["meaning"]) > 0


# ============================================================================ #
#  @lru_cache 缓存                                                              #
# ============================================================================ #

def test_cache_hit_on_repeat_call() -> None:
    """相同函数多次调用返回同一列表对象（缓存命中）。"""
    first = get_datasets_data()
    second = get_datasets_data()
    assert first is second  # 同一对象引用


def test_cache_hit_models() -> None:
    """get_models_data 多次调用返回同一列表。"""
    first = get_models_data()
    second = get_models_data()
    assert first is second


def test_cache_hit_metrics() -> None:
    """get_metrics_data 多次调用返回同一列表。"""
    first = get_metrics_data()
    second = get_metrics_data()
    assert first is second


def test_clear_cache_reloads_data() -> None:
    """清空缓存后再次调用返回新列表对象。"""
    first = get_datasets_data()
    clear_domain_data_cache()
    second = get_datasets_data()
    assert first is not second
    # 但内容应该一样
    assert first == second


def test_clear_cache_reloads_models() -> None:
    """清空缓存后 get_models_data 返回新列表。"""
    first = get_models_data()
    clear_domain_data_cache()
    second = get_models_data()
    assert first is not second


def test_clear_cache_reloads_metrics() -> None:
    """清空缓存后 get_metrics_data 返回新列表。"""
    first = get_metrics_data()
    clear_domain_data_cache()
    second = get_metrics_data()
    assert first is not second


# ============================================================================ #
#  路径常量                                                                     #
# ============================================================================ #

def test_domain_data_dir_exists() -> None:
    """_DOMAIN_DATA_DIR 指向真实存在的目录。"""
    assert _DOMAIN_DATA_DIR.exists()
    assert _DOMAIN_DATA_DIR.is_dir()


def test_json_files_exist() -> None:
    """三个 JSON 文件均存在于 domain_data 目录。"""
    assert (_DOMAIN_DATA_DIR / "datasets.json").exists()
    assert (_DOMAIN_DATA_DIR / "models.json").exists()
    assert (_DOMAIN_DATA_DIR / "metrics.json").exists()


# ============================================================================ #
#  load_json_data 异常路径（通过 mock 验证 fallback）                             #
# ============================================================================ #

def test_load_json_data_read_error_fallback(tmp_path: Path) -> None:
    """Path.read_text 抛异常时返回空列表。"""
    test_file = tmp_path / "err.json"
    test_file.write_text("[]", encoding="utf-8")

    with patch("pathlib.Path.read_text", side_effect=OSError("模拟 IO 错误")):
        result = load_json_data(test_file)

    assert result == []


# ============================================================================ #
#  数据内容抽样校验（确保不是空壳 JSON）                                          #
# ============================================================================ #

def test_loveda_has_correct_classes() -> None:
    """LoveDA 数据集应为 7 类。"""
    data = get_datasets_data()
    loveda = next(item for item in data if item["name"] == "LoveDA")
    assert loveda["classes"] == 7
    assert "建筑" in loveda["class_names"]


def test_deeplabv3_has_aspp_module() -> None:
    """DeepLabV3+ 的 key_modules 应提及 ASPP。"""
    data = get_models_data()
    deeplab = next(item for item in data if item["name"] == "DeepLabV3+")
    aspp_found = any("ASPP" in m for m in deeplab["key_modules"])
    assert aspp_found


def test_miou_formula_contains_iou() -> None:
    """mIoU 的 formula 应包含 'IoU'。"""
    data = get_metrics_data()
    miou = next(item for item in data if item["name"] == "mIoU")
    assert "IoU" in miou["formula"]


def test_f1_formula_contains_precision_and_recall() -> None:
    """F1 的 formula 应包含 Precision 和 Recall。"""
    data = get_metrics_data()
    f1 = next(item for item in data if item["name"] == "F1-score")
    assert "Precision" in f1["formula"]
    assert "Recall" in f1["formula"]
