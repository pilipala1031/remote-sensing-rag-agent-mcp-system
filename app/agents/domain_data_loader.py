"""领域结构化数据加载器。

从 app/domain_data/*.json 加载遥感语义分割领域知识，
供 Agent 多工具架构中的结构化查询工具使用。

设计原则：
- @lru_cache 缓存，避免每次工具调用重复读磁盘
- 异常时返回空列表（不崩溃），由上层工具决定拒答
- 路径基于 __file__ 计算，不依赖 cwd
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---- 常量 --------------------------------------------------------------------

_DOMAIN_DATA_DIR = Path(__file__).resolve().parent.parent / "domain_data"

_DATASETS_PATH = _DOMAIN_DATA_DIR / "datasets.json"
_MODELS_PATH = _DOMAIN_DATA_DIR / "models.json"
_METRICS_PATH = _DOMAIN_DATA_DIR / "metrics.json"


# ---- 通用加载函数 ------------------------------------------------------------

def load_json_data(filepath: str | Path) -> List[Dict[str, Any]]:
    """从 JSON 文件加载领域数据。

    Args:
        filepath: JSON 文件路径。

    Returns:
        解析后的列表（每个元素为一条结构化记录）。
        文件不存在 / JSON 格式错误 / 内容非列表时返回空列表。
    """
    p = Path(filepath)
    if not p.exists():
        logger.warning("领域数据文件不存在: %s", p)
        return []

    try:
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("领域数据 JSON 解析失败: %s, 错误: %s", p, e)
        return []
    except Exception as e:
        logger.error("领域数据读取异常: %s, 错误: %s", p, e)
        return []

    if not isinstance(data, list):
        logger.warning("领域数据 JSON 顶层不是列表: %s, 实际类型: %s", p, type(data).__name__)
        return []

    return data


# ---- 缓存加载器 --------------------------------------------------------------
# 使用 @lru_cache(maxsize=1) 确保每个文件只读取一次。
# 与 get_remote_sensing_agent / get_settings 的单例模式一致。

@lru_cache(maxsize=1)
def get_datasets_data() -> List[Dict[str, Any]]:
    """获取遥感数据集结构化数据（LoveDA, iSAID, DeepGlobe 等）。

    结果被缓存，首次调用后后续调用直接返回内存中的列表。
    """
    data = load_json_data(_DATASETS_PATH)
    logger.info("加载领域数据集信息: %d 条", len(data))
    return data


@lru_cache(maxsize=1)
def get_models_data() -> List[Dict[str, Any]]:
    """获取语义分割模型结构化数据（U-Net, DeepLabV3+, SegFormer 等）。

    结果被缓存，首次调用后后续调用直接返回内存中的列表。
    """
    data = load_json_data(_MODELS_PATH)
    logger.info("加载领域模型信息: %d 条", len(data))
    return data


@lru_cache(maxsize=1)
def get_metrics_data() -> List[Dict[str, Any]]:
    """获取评价指标结构化数据（IoU, mIoU, F1-score 等）。

    结果被缓存，首次调用后后续调用直接返回内存中的列表。
    """
    data = load_json_data(_METRICS_PATH)
    logger.info("加载领域指标信息: %d 条", len(data))
    return data


# ---- 缓存管理 ----------------------------------------------------------------

def clear_domain_data_cache() -> None:
    """清空所有领域数据的 lru_cache。

    在文档更新 / 测试 / 需要强制重载时调用。
    """
    get_datasets_data.cache_clear()
    get_models_data.cache_clear()
    get_metrics_data.cache_clear()
    logger.info("领域数据缓存已清空")
