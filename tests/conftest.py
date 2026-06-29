"""Pytest 公共 fixture：伪造 Settings，避免测试时调用真实 API。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_text() -> str:
    return (
        "Landsat 8 卫星携带 OLI 和 TIRS 两种传感器。"
        "OLI 包括 9 个波段，其中近红外波段（Band 5）波长为 0.85–0.88 μm。"
        "TIRS 提供两个热红外波段，Band 10 中心波长约 10.9 μm，Band 11 约 12.0 μm。\n\n"
        "NDVI 计算公式为 (NIR - Red) / (NIR + Red)，常用于植被覆盖度监测。"
    )
