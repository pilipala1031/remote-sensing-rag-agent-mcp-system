"""WorkUnitStore 持久化层单元测试。

全部使用 pytest 内置 tmp_path 注入 base_dir，绝不污染真实 data/work_units/。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import WorkUnitSaveRequest
from app.services import work_unit_store


def _make_payload(entry: str = "agent", question: str = "U-Net 和 DeepLabV3+ 哪个更好？") -> WorkUnitSaveRequest:
    """构造一个带 replay_payload 的典型保存请求。"""
    return WorkUnitSaveRequest(
        entry=entry,
        question=question,
        answer="U-Net 适合小样本，DeepLabV3+ 适合多尺度场景。",
        sources=[{"filename": "02_models.md", "page": 1, "chunk_id": "abc123", "score": 0.84}],
        refused=False,
        tool_calls=[{"name": "knowledge_base_search", "elapsed": 1.2}],
        trace_events=[{"step": 1, "event": "agent_started"}],
        timing={"total_elapsed": 3.5},
        verification={"verified": True, "confidence": 0.9},
        errors=[],
        replay_payload={"question": question, "include_trace": True, "use_rerank": None, "enable_cache": None},
    )


def test_save_creates_file(tmp_path: Path) -> None:
    """1. 保存后文件存在。"""
    payload = _make_payload()
    work_unit = work_unit_store.save_work_unit(payload, base_dir=tmp_path)

    file_path = tmp_path / f"{work_unit.work_unit_id}.json"
    assert file_path.exists(), "保存后应生成对应的 JSON 文件"


def test_save_returns_readable_work_unit(tmp_path: Path) -> None:
    """2. 可以读取保存的 Work Unit。"""
    payload = _make_payload()
    saved = work_unit_store.save_work_unit(payload, base_dir=tmp_path)

    got = work_unit_store.get_work_unit(saved.work_unit_id, base_dir=tmp_path)
    assert got.work_unit_id == saved.work_unit_id
    assert got.entry == "agent"
    assert got.question == payload.question
    assert got.answer == payload.answer


def test_list_work_units(tmp_path: Path) -> None:
    """3. 可以 list。"""
    work_unit_store.save_work_unit(_make_payload(entry="rag", question="Q1"), base_dir=tmp_path)
    work_unit_store.save_work_unit(_make_payload(entry="agent", question="Q2"), base_dir=tmp_path)

    items = work_unit_store.list_work_units(base_dir=tmp_path)
    assert len(items) == 2


def test_list_filter_by_entry(tmp_path: Path) -> None:
    """4. 可以按 entry 过滤。"""
    work_unit_store.save_work_unit(_make_payload(entry="rag", question="Q1"), base_dir=tmp_path)
    work_unit_store.save_work_unit(_make_payload(entry="agent", question="Q2"), base_dir=tmp_path)
    work_unit_store.save_work_unit(_make_payload(entry="agent", question="Q3"), base_dir=tmp_path)

    rag_items = work_unit_store.list_work_units(entry="rag", base_dir=tmp_path)
    agent_items = work_unit_store.list_work_units(entry="agent", base_dir=tmp_path)

    assert len(rag_items) == 1 and rag_items[0].entry == "rag"
    assert len(agent_items) == 2 and all(w.entry == "agent" for w in agent_items)


def test_delete_work_unit(tmp_path: Path) -> None:
    """5. 可以 delete。"""
    saved = work_unit_store.save_work_unit(_make_payload(), base_dir=tmp_path)
    assert (tmp_path / f"{saved.work_unit_id}.json").exists()

    deleted = work_unit_store.delete_work_unit(saved.work_unit_id, base_dir=tmp_path)
    assert deleted is True
    assert not (tmp_path / f"{saved.work_unit_id}.json").exists()

    # 再次删除（已不存在）应返回 False，而非抛异常
    deleted_again = work_unit_store.delete_work_unit(saved.work_unit_id, base_dir=tmp_path)
    assert deleted_again is False


def test_get_nonexistent_raises(tmp_path: Path) -> None:
    """6. 不存在 id 时抛出带中文信息的 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError) as exc_info:
        work_unit_store.get_work_unit("wu_not_exist", base_dir=tmp_path)
    assert "Work Unit 不存在" in str(exc_info.value)


def test_work_unit_id_prefix(tmp_path: Path) -> None:
    """7. 保存时生成的 work_unit_id 以 wu_ 开头。"""
    saved = work_unit_store.save_work_unit(_make_payload(), base_dir=tmp_path)
    assert saved.work_unit_id.startswith("wu_")
    # 形如 wu_ + 16 个十六进制字符
    assert len(saved.work_unit_id) == len("wu_") + 16


def test_replay_payload_roundtrip(tmp_path: Path) -> None:
    """8. replay_payload 能被完整保存和读取。"""
    payload = _make_payload()
    expected_replay = {
        "question": payload.question,
        "include_trace": True,
        "use_rerank": None,
        "enable_cache": None,
        "nested": {"a": [1, 2, 3]},
    }
    payload.replay_payload = expected_replay

    saved = work_unit_store.save_work_unit(payload, base_dir=tmp_path)
    got = work_unit_store.get_work_unit(saved.work_unit_id, base_dir=tmp_path)

    assert got.replay_payload == expected_replay
    # 嵌套结构也必须原样保留
    assert got.replay_payload["nested"]["a"] == [1, 2, 3]


def test_json_file_is_utf8_and_pretty(tmp_path: Path) -> None:
    """附加校验：文件为 UTF-8、非 ASCII 转义（ensure_ascii=False）、带缩进。"""
    payload = _make_payload()
    saved = work_unit_store.save_work_unit(payload, base_dir=tmp_path)
    raw = (tmp_path / f"{saved.work_unit_id}.json").read_text(encoding="utf-8")

    # 中文未转义（说明 ensure_ascii=False）
    assert payload.question in raw
    # 带缩进（至少出现两空格缩进）
    assert '\n  "' in raw
    # 是合法 JSON
    assert json.loads(raw)["work_unit_id"] == saved.work_unit_id
