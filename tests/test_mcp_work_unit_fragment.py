"""MCP 工具返回 work_unit_fragment 的测试。

验证两个 MCP 工具的返回 dict 含 work_unit_fragment（entry=mcp，不落盘）。
calculate 工具是纯计算，可直接调用；search 工具需 mock Retriever 以避免 Chroma 依赖。
不真实调用 LLM。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_server import server as mcp_server


def _run_tool(tool_func, *args, **kwargs):
    """统一调用工具函数。

    FastMCP 的 @mcp.tool() 装饰后，函数对象挂在 .fn 上；若已是普通函数则直接调用。
    """
    fn = getattr(tool_func, "fn", tool_func)
    return fn(*args, **kwargs)


def test_calculate_returns_work_unit_fragment() -> None:
    """4/5. calculate_remote_sensing_metric 返回含 work_unit_fragment，tool_name 正确。"""
    result = _run_tool(
        mcp_server.calculate_remote_sensing_metric,
        metric="iou",
        tp=80,
        fp=10,
        fn=20,
    )
    assert result["success"] is True
    assert "work_unit_fragment" in result
    frag = result["work_unit_fragment"]
    assert frag["tool_name"] == "calculate_remote_sensing_metric"
    # fragment 不应包含完整答案，只有 value（计算结果数值）
    assert "value" in frag["outputs"]


def test_calculate_fragment_entry_is_mcp() -> None:
    """fragment entry == mcp。"""
    result = _run_tool(mcp_server.calculate_remote_sensing_metric, metric="iou", tp=1, fp=1, fn=1)
    assert result["work_unit_fragment"]["entry"] == "mcp"
    assert result["work_unit_fragment"]["fragment_id"].startswith("frag_")


@patch("mcp_server.server.Retriever")
def test_search_returns_work_unit_fragment(mock_retriever_cls: MagicMock) -> None:
    """1/2/3. search_remote_sensing_kb 返回含 work_unit_fragment，entry/tool_name 正确。"""
    mock_inst = MagicMock()
    mock_inst.retrieve.return_value = [
        {
            "chunk_id": "abc123",
            "score": 0.85,
            "content": "U-Net 由编码器-解码器构成。",
            "filename": "02_models.md",
            "page": 1,
        }
    ]
    mock_retriever_cls.return_value = mock_inst

    result = _run_tool(mcp_server.search_remote_sensing_kb, query="U-Net 是什么", top_k=5)

    assert result["success"] is True
    assert "work_unit_fragment" in result
    frag = result["work_unit_fragment"]
    assert frag["entry"] == "mcp"
    assert frag["tool_name"] == "search_remote_sensing_kb"
    assert frag["inputs"]["query"] == "U-Net 是什么"
    assert frag["inputs"]["top_k"] == 5
    assert frag["outputs"]["contexts_count"] == 1
    assert frag["outputs"]["sources_count"] == 1


def test_mcp_does_not_persist_work_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """6. MCP 工具调用不会创建 data/work_units 文件。

    把 work_unit_store 的默认目录重定向到临时目录，调用两个工具后断言目录为空。
    （MCP 工具内部根本不 import / 调用 work_unit_store，此处为防御性回归断言。）
    """
    from app.api import work_units as work_units_api

    monkeypatch.setattr(work_units_api, "_get_work_unit_dir", lambda: tmp_path)

    # calculate（纯计算，直接调）
    _run_tool(mcp_server.calculate_remote_sensing_metric, metric="iou", tp=1, fp=1, fn=1)
    assert list(tmp_path.glob("*.json")) == []

    # search（mock Retriever）
    with patch("mcp_server.server.Retriever") as mock_retriever_cls:
        mock_retriever_cls.return_value.retrieve.return_value = []
        _run_tool(mcp_server.search_remote_sensing_kb, query="x")
    assert list(tmp_path.glob("*.json")) == []
