"""Embedding 调用测试。

需要真实 API Key 且网络可用，运行前请确保 .env 已配置 SILICONFLOW_API_KEY。
无 Key 时自动跳过。

可通过环境变量 RUN_INTEGRATION_TESTS=1 强制运行（默认在 pytest 总回归中跳过，
避免真实外部网络调用）。
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# 加载 .env 到环境变量
load_dotenv()


def _should_run_integration_tests() -> bool:
    """判断是否运行集成测试（需真实 API Key 且显式开启）。

    条件：
    1. SILICONFLOW_API_KEY 非空（从 .env 加载后检查实际值）；
    2. 环境变量 RUN_INTEGRATION_TESTS=1（默认不设置 → 跳过，
       避免总回归时真实调用外部网络）。
    """
    has_key = bool(os.getenv("SILICONFLOW_API_KEY", "").strip())
    explicitly_enabled = os.getenv("RUN_INTEGRATION_TESTS", "").strip() == "1"
    return has_key and explicitly_enabled


@pytest.mark.skipif(
    not _should_run_integration_tests(),
    reason="未配置 SILICONFLOW_API_KEY 或未设置 RUN_INTEGRATION_TESTS=1，跳过真实 Embedding 调用",
)
def test_embed_query() -> None:
    from app.core.embeddings import SiliconFlowEmbeddingClient

    client = SiliconFlowEmbeddingClient()
    vec = client.embed_query("Landsat 8 热红外波段")
    assert isinstance(vec, list)
    assert len(vec) == 1024  # bge-m3 维度
