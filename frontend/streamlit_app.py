"""Streamlit 前端：文档上传 / 入库 / 问答 / 来源展示。

支持两种问答模式切换：
- 普通 RAG（/api/chat/query）：固定检索 → LLM 生成
- Agent 研究助手（/api/agent/query）：LLM 自主决定是否检索，返回工具调用轨迹

Agent 模式下额外展示执行过程可视化面板，包括：
- Agent 执行时间线（基于 agent_trace）
- 工具调用详情（基于 tool_calls）
- 检索来源卡片（基于 sources）
- 最终回答（基于 answer + refused）
- 证据校验（基于 verification）
- 错误信息（基于 errors）
- 原始返回 JSON 调试区
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

# 默认连接本地 FastAPI，可通过环境变量覆盖
API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Remote Sensing RAG", page_icon="🛰️", layout="wide")


# =====================================================================
# 演示访问码（可选，仅用于 Cloudflare Tunnel 等公网演示场景）
# =====================================================================

def require_demo_password() -> None:
    """可选的演示访问码保护（password gate）。

    通过环境变量 ``DEMO_PASSWORD`` 控制：

    - 未设置或为空字符串：**不启用访问码**，保持原有体验，
      不影响本地开发和 Docker Compose 普通启动。
    - 已设置：页面先显示访问码输入框，用户输入正确后才能进入主应用。
      验证状态记录在 ``st.session_state``，避免每次交互都要重新输入。

    仅用于 Cloudflare Tunnel 等临时公网演示场景，**不是生产级鉴权方案**。
    不引入任何复杂依赖，也不实现 JWT/OAuth/数据库用户系统。
    """
    password = os.getenv("DEMO_PASSWORD", "").strip()
    if not password:
        return

    # 已通过验证，直接放行
    if st.session_state.get("demo_password_ok"):
        return

    # 未通过验证，渲染访问码输入页面并阻止后续 UI
    st.title("🔒 访问验证")
    st.caption("Remote Sensing RAG Agent｜遥感知识库问答")
    st.write("请输入演示访问码以继续。")
    entered = st.text_input("访问码", type="password")
    if st.button("进入", type="primary"):
        if entered == password:
            st.session_state["demo_password_ok"] = True
            st.rerun()
        else:
            st.error("访问码错误，请重试。")
    # 阻止后续主 UI 渲染
    st.stop()


# 在主标题渲染前调用：未通过验证时主 UI 不会出现
require_demo_password()

st.title("🛰️ Remote Sensing RAG Agent｜遥感知识库问答")


# =====================================================================
# Agent 模式可视化：辅助渲染函数
# =====================================================================

# 常见 agent_trace 字符串 -> 中文标签映射
_TRACE_LABEL_MAP: dict[str, str] = {
    "agent_started": "Agent 已启动",
    "no_tool_called": "Agent 未调用工具，直接尝试回答",
    "tool_result_parsed": "工具结果已解析",
    "agent_finished": "Agent 已完成回答",
    "agent_error": "Agent 执行异常",
    "agent_service_error": "Agent 服务层异常",
    "fallback_to_rag": "已回退到基础 RAG 链路",
}

# 错误关键词 -> 是否提示 fallback
_FALLBACK_KEYWORDS = ("agent_failed", "agent_error", "fallback_to_rag")

#: Multi-Tool 工具中文名映射
TOOL_NAME_CN: dict[str, str] = {
    "knowledge_base_search": "知识库语义检索",
    "dataset_overview": "数据集共性概览",
    "dataset_spec_lookup": "数据集结构化查询",
    "model_comparison_table": "模型对比工具",
    "metric_formula_lookup": "指标公式查询",
    "metrics_calculator": "指标计算器",
    "plan_and_search": "复杂问题分解检索",
}

#: 结构化工具集合（返回 JSON 数据但不返回 sources）
_STRUCTURED_TOOLS: set[str] = {
    "dataset_overview",
    "dataset_spec_lookup",
    "model_comparison_table",
    "metric_formula_lookup",
}

#: 计算工具集合
_CALC_TOOLS: set[str] = {"metrics_calculator"}


def format_trace_label(trace: str) -> str:
    """将单条 agent_trace 字符串映射为中文展示标签。

    支持以下情况：
    - 形如 ``tool_called:knowledge_base_search`` 的带参 trace，会拆分出工具名；
    - 已知的固定字符串使用 ``_TRACE_LABEL_MAP`` 映射；
    - 未知 trace 原样返回，不抛异常。
    """
    if trace is None:
        return "未知轨迹"
    trace = str(trace).strip()
    if not trace:
        return "未知轨迹"

    # 处理 tool_called:<tool_name> 形式
    if trace.startswith("tool_called:"):
        tool_name = trace.split(":", 1)[1].strip() or "未知工具"
        cn_name = TOOL_NAME_CN.get(tool_name, tool_name)
        return f"调用工具：{cn_name}"

    # 命中预定义映射
    if trace in _TRACE_LABEL_MAP:
        return _TRACE_LABEL_MAP[trace]

    # 未知 trace，原样展示
    return trace


def render_agent_trace(agent_trace: object) -> None:
    """渲染 Agent 执行时间线。

    agent_trace 期望为 list[str]，但本函数能处理 None / 非列表 / 元素非字符串等情况。
    """
    st.markdown("## 🧭 Agent 执行过程")

    if not agent_trace or not isinstance(agent_trace, list) or len(agent_trace) == 0:
        st.info(
            "暂无 Agent 执行轨迹，可能是 Agent 未成功启动或后端未返回 trace。"
        )
        return

    lines: list[str] = []
    for idx, item in enumerate(agent_trace, start=1):
        label = format_trace_label(item if isinstance(item, str) else str(item))
        # 根据 trace 内容选择图标
        icon = _trace_icon(label, item if isinstance(item, str) else "")
        lines.append(f"{idx}. {icon} {label}")

    st.markdown("\n".join(lines))


def _trace_icon(label: str, raw: str) -> str:
    """根据 trace 的中文标签/原始值返回合适的图标。"""
    if "调用工具" in label or raw.startswith("tool_called:"):
        return "🔧"
    if "异常" in label or "错误" in label or raw in ("agent_error", "agent_service_error"):
        return "❌"
    if "未调用工具" in label or "回退" in label:
        return "⚠️"
    if "结果" in label:
        return "📦"
    return "✅"


def render_trace_events(trace_events: object) -> None:
    """渲染结构化轨迹事件（trace_events），展示 step / event / timestamp / detail。

    与 render_agent_trace 互补：
    - render_agent_trace 基于 list[str] 做简单时间线
    - render_trace_events 基于 list[dict] 展示时间戳和详情

    本函数能处理 None / 非列表 / 元素非 dict / 字段缺失等情况。
    """
    st.markdown("## ⏱️ 结构化轨迹事件（trace_events）")

    if not trace_events or not isinstance(trace_events, list) or len(trace_events) == 0:
        st.info(
            "暂无结构化轨迹事件。可能后端未返回 trace_events 或 include_trace=False。"
        )
        return

    # 构建展示数据
    rows: list[dict] = []
    for ev in trace_events:
        if not isinstance(ev, dict):
            continue
        event_type = str(ev.get("event") or "unknown")
        detail = ev.get("detail")
        ts = ev.get("timestamp", 0.0)
        ts_str = f"{float(ts):.4f}s" if isinstance(ts, (int, float)) else "—"

        # 中文标签
        label = _TRACE_LABEL_MAP.get(event_type, event_type)
        if event_type == "tool_called" and detail:
            tool_cn = TOOL_NAME_CN.get(detail, detail)
            label = f"调用工具：{tool_cn}"
        elif event_type == "tool_result_parsed" and detail:
            tool_cn = TOOL_NAME_CN.get(detail, detail)
            label = f"工具结果已解析（{tool_cn}）"

        icon = _trace_icon(label, event_type)

        rows.append({
            "step": ev.get("step", "?"),
            "图标": icon,
            "事件": label,
            "时间戳": ts_str,
            "详情": detail or "—",
        })

    if not rows:
        st.info("结构化轨迹事件数据为空或格式异常。")
        return

    # 用 columns 做简易表格展示
    for row in rows:
        cols = st.columns([1, 1, 4, 2, 3])
        with cols[0]:
            st.caption(f"#{row['step']}")
        with cols[1]:
            st.markdown(row["图标"])
        with cols[2]:
            st.markdown(row["事件"])
        with cols[3]:
            st.caption(row["时间戳"])
        with cols[4]:
            st.caption(row["详情"])


def render_tool_calls(tool_calls: object) -> None:
    """渲染工具调用详情。

    tool_calls 期望为 list[dict]，每个 dict 包含 tool/input/status/output_summary/error。
    本函数能处理 None / 非列表 / 元素非 dict / 字段缺失等情况。
    """
    st.markdown("## 🔧 工具调用详情")

    if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
        st.info(
            "本次 Agent 未调用工具，可能是模型判断无需检索，或工具调用失败。"
        )
        return

    for i, tc in enumerate(tool_calls, start=1):
        # 容错：非 dict 元素降级处理
        if not isinstance(tc, dict):
            with st.expander(f"工具调用 {i}：未知工具 ｜ 数据格式异常", expanded=False):
                st.warning(f"该条记录不是有效的 dict：{tc!r}")
            continue

        tool_name = str(tc.get("tool") or "未知工具")
        tool_cn = TOOL_NAME_CN.get(tool_name, tool_name)
        status = str(tc.get("status") or "unknown")
        status_icon = "✅" if status == "success" else "❌"

        with st.expander(
            f"工具调用 {i}：{tool_cn}（{tool_name}） ｜ {status_icon} {status}",
            expanded=False,
        ):
            st.markdown(f"**工具名称：** `{tool_name}`")
            st.markdown(f"**工具中文名：** {tool_cn}")

            # 工具输入：dict/list 用 json，字符串用 code/markdown
            tool_input = tc.get("input")
            st.markdown("**工具输入：**")
            if isinstance(tool_input, (dict, list)):
                st.json(tool_input, expanded=True)
            elif tool_input is None or tool_input == "":
                st.caption("无")
            else:
                st.code(str(tool_input), language="text")

            st.markdown(f"**调用状态：** `{status}`")

            # 工具耗时
            elapsed = tc.get("elapsed")
            elapsed_str = f"{float(elapsed):.4f} 秒" if isinstance(elapsed, (int, float)) else "无"
            st.markdown(f"**耗时：** `{elapsed_str}`")

            output_summary = tc.get("output_summary")
            st.markdown(
                f"**输出摘要：** {output_summary if output_summary else '无'}"
            )

            error = tc.get("error")
            st.markdown(f"**错误信息：** {error if error else '无'}")


def _render_tool_stats(tool_calls: object) -> None:
    """渲染工具调用统计区域：调用次数、工具列表、分类标签。"""
    st.markdown("## 🧰 本次 Agent 使用的工具")

    if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
        st.info("本次 Agent 未调用任何工具。")
        return

    # 提取工具名列表
    tool_names: list[str] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            name = str(tc.get("tool") or "")
            if name:
                tool_names.append(name)

    if not tool_names:
        st.info("本次 Agent 未调用任何工具。")
        return

    # 统计
    total_calls = len(tool_names)
    unique_names = list(dict.fromkeys(tool_names))  # 去重保持顺序
    cn_names = [TOOL_NAME_CN.get(n, n) for n in unique_names]

    used_kb = "knowledge_base_search" in tool_names
    used_structured = any(n in _STRUCTURED_TOOLS for n in tool_names)
    used_calc = any(n in _CALC_TOOLS for n in tool_names)
    used_plan = "plan_and_search" in tool_names

    # 展示
    st.metric("工具总调用次数", total_calls)
    st.markdown(f"**调用的工具：** {'、'.join(cn_names)}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if used_kb:
            st.success("✅ 调用了知识库检索")
        else:
            st.caption("⚪ 未调用知识库检索")
    with col2:
        if used_structured:
            st.success("✅ 调用了结构化工具")
        else:
            st.caption("⚪ 未调用结构化工具")
    with col3:
        if used_calc:
            st.success("✅ 调用了计算工具")
        else:
            st.caption("⚪ 未调用计算工具")
    with col4:
        if used_plan:
            st.success("✅ 调用了复杂分解检索")
        else:
            st.caption("⚪ 未调用复杂分解检索")


def render_agent_sources(sources: object, tool_calls: object = None) -> None:
    """渲染 Agent 检索到的来源卡片。

    sources 期望为 list[dict]，每个 dict 包含 filename/page/chunk_id/score/content_preview。
    tool_calls 可选，用于在 sources 为空时做智能提示（区分结构化工具 vs 检索失败）。
    本函数能处理 None / 非列表 / 元素非 dict / 字段缺失等情况。
    """
    st.markdown("## 📚 Agent 检索到的来源")

    if not sources or not isinstance(sources, list) or len(sources) == 0:
        # 智能提示：如果使用了结构化/计算工具但无 sources，这是正常行为
        if tool_calls and isinstance(tool_calls, list):
            tc_names = [
                str(tc.get("tool", ""))
                for tc in tool_calls
                if isinstance(tc, dict)
            ]
            used_structured_or_calc = any(
                n in _STRUCTURED_TOOLS or n in _CALC_TOOLS for n in tc_names
            )
            if used_structured_or_calc:
                st.info(
                    "本次主要使用结构化工具，无向量检索来源。"
                )
                return

        st.info(
            "未返回来源。请检查工具调用或知识库检索逻辑。"
        )
        return

    for i, s in enumerate(sources, start=1):
        if not isinstance(s, dict):
            with st.expander(f"来源 {i}：数据格式异常", expanded=False):
                st.warning(f"该条记录不是有效的 dict：{s!r}")
            continue

        filename = str(s.get("filename") or "未知文件")
        page = s.get("page")
        page_str = str(page) if page is not None else "?"
        score = s.get("score")
        score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else "无"

        with st.expander(
            f"来源 {i}：{filename} ｜ 第 {page_str} 页 ｜ score={score_str}",
            expanded=False,
        ):
            st.markdown(f"**filename：** `{filename}`")
            st.markdown(f"**page：** `{page_str}`")
            st.markdown(f"**chunk_id：** `{s.get('chunk_id') or '无'}`")
            st.markdown(f"**score：** `{score_str}`")

            preview = s.get("content_preview")
            st.markdown("**content_preview：**")
            if preview:
                st.write(preview)
            else:
                st.caption("无")


def render_agent_errors(errors: object) -> None:
    """渲染错误信息，并针对 fallback 关键词突出提示。"""
    if not errors or not isinstance(errors, list) or len(errors) == 0:
        return

    st.markdown("## ⚠️ 错误信息")

    # 检查是否包含 fallback 相关关键词
    joined = " ".join(str(e) for e in errors).lower()
    has_fallback_signal = any(kw in joined for kw in _FALLBACK_KEYWORDS)
    if has_fallback_signal:
        st.warning(
            "Agent 执行异常，但系统可能已经回退到基础 RAG 链路。"
        )

    for err in errors:
        st.error(str(err))


def render_agent_raw_json(result: object) -> None:
    """渲染原始返回 JSON 调试区。"""
    st.markdown("## 🛠️ 原始返回 JSON")
    with st.expander("查看原始 Agent 返回 JSON", expanded=False):
        if isinstance(result, (dict, list)):
            st.json(result, expanded=False)
        else:
            st.code(repr(result), language="text")


def render_agent_final_answer(answer: object, refused: object) -> None:
    """渲染 Agent 最终回答。"""
    st.markdown("## ✅ Agent 最终回答")
    answer_text = str(answer) if answer is not None else ""
    refused_flag = bool(refused)

    if refused_flag:
        st.warning(answer_text or "Agent 拒绝回答（无内容）")
    else:
        if answer_text:
            st.success("Agent 已生成回答")
            st.markdown(answer_text)
            # 回答过长提示
            if len(answer_text) > 1200:
                st.caption(
                    "⚠️ 回答较长，可能增加生成与校验耗时。"
                )
        else:
            st.warning("Agent 未返回任何回答内容")


def render_agent_verification(
    verification: object,
    question: str = "",
    answer: str = "",
    sources: object = None,
    tool_calls: object = None,
) -> None:
    """渲染 Evidence Verification 证据校验结果。

    verification 期望为 dict，包含 enabled / mode / pending / verified /
    confidence / ungrounded_claims / reason / timing 等字段。

    支持三种模式渲染：
    - off：显示"证据校验未启用"
    - sync：直接展示已有校验结果
    - deferred（pending=True）：显示"执行证据校验"按钮，点击后调 /api/agent/verify

    本函数能处理 None / 非字典 / 字段缺失等情况。
    """
    st.markdown("## 🛡️ 证据校验")

    if not verification or not isinstance(verification, dict):
        st.caption("未返回证据校验信息。")
        return

    enabled = verification.get("enabled", True)

    # ---------- 未启用 ----------
    if not enabled:
        st.info("证据校验未启用。")
        return

    pending = verification.get("pending", False)
    mode = str(verification.get("mode", "") or "")

    # ---------- deferred 待执行 ----------
    if pending or mode == "deferred":
        st.info("证据校验待执行。")
        level = verification.get("level", "")
        if level:
            st.caption(f"轻量化级别：{level}")

        if st.button("执行证据校验", key="run_verification"):
            _execute_deferred_verification(
                question=question,
                answer=answer,
                sources=sources,
                tool_calls=tool_calls,
            )
        return

    # ---------- 已有校验结果（sync 模式或 deferred 完成后） ----------
    verified = verification.get("verified")
    confidence = str(verification.get("confidence", "") or "")
    ungrounded_claims = verification.get("ungrounded_claims") or []
    reason = str(verification.get("reason", "") or "")
    timing = verification.get("timing")
    verification_elapsed = None
    if isinstance(timing, dict):
        verification_elapsed = timing.get("verification_elapsed")

    # ---------- 校验通过 ----------
    if verified is True:
        st.success("✅ 回答已通过证据校验")
        if reason:
            st.caption(f"原因：{reason}")

    # ---------- 校验未通过 ----------
    elif verified is False:
        st.warning("⚠️ 回答中存在可能未被证据支撑的论断")
        if ungrounded_claims and isinstance(ungrounded_claims, list):
            st.markdown("**未证实论断：**")
            for claim in ungrounded_claims:
                st.markdown(f"- {claim}")
        if reason:
            st.caption(f"原因：{reason}")

    # ---------- 校验结果未知 ----------
    else:
        st.info("证据校验结果未知。")
        if reason:
            st.caption(f"原因：{reason}")

    # ---------- 置信度 ----------
    if confidence:
        confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
            confidence, "⚪"
        )
        st.caption(f"置信度：{confidence_icon} {confidence}")

    # ---------- 校验耗时 ----------
    if verification_elapsed is not None and isinstance(
        verification_elapsed, (int, float)
    ):
        st.caption(f"校验耗时：{float(verification_elapsed):.4f} 秒")


# =====================================================================
# Agent / RAG 请求执行 + 响应渲染编排
# =====================================================================

def _execute_deferred_verification(
    question: str,
    answer: str,
    sources: object,
    tool_calls: object,
) -> None:
    """调用 /api/agent/verify 执行延迟的证据校验并渲染结果。

    在 deferred 模式下，用户点击"执行证据校验"按钮后触发此函数。
    """
    with st.spinner("正在进行证据校验..."):
        try:
            response = requests.post(
                f"{API_BASE}/api/agent/verify",
                json={
                    "question": question,
                    "answer": answer,
                    "sources": sources if isinstance(sources, list) else [],
                    "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
                },
                timeout=120,
            )
        except Exception as e:
            st.error(f"证据校验请求异常: {e}")
            return

    if response.status_code != 200:
        st.error(f"证据校验失败 {response.status_code}: {response.text}")
        return

    try:
        data = response.json()
    except Exception as e:
        st.error(f"证据校验响应解析失败: {e}")
        return

    verification = data.get("verification", {})
    # 递归调用 render_agent_verification，但此时 pending=False，会进入展示逻辑
    render_agent_verification(
        verification,
        question=question,
        answer=answer,
        sources=sources,
        tool_calls=tool_calls,
    )

# Agent 执行的预期流程步骤（仅用于前端状态提示，非真实后端流式状态）
_AGENT_FLOW_STEPS = [
    "正在发送问题给 Agent...",
    "Agent 正在分析问题...",
    "Agent 正在调用知识库检索工具...",
    "Agent 正在整理来源并生成回答...",
    "Agent 执行完成",
]


def _has_st_status() -> bool:
    """检测当前 Streamlit 版本是否支持 st.status。"""
    return hasattr(st, "status") and callable(getattr(st, "status"))


def _run_agent_query_with_status(question: str, include_trace: bool = True, use_rerank: bool | None = None, enable_cache: bool | None = None) -> object | None:
    """执行 Agent 查询，展示前端流程状态。

    后端为同步返回（非流式），因此这里的状态提示只是展示流程，
    真正的执行细节以请求完成后渲染的 agent_trace 为准。

    Args:
        question: 用户问题
        include_trace: 是否请求后端返回 trace/tool_calls 等调试信息。
        use_rerank: 是否启用 rerank 重排序，None 则使用后端配置。
        enable_cache: 是否启用 LLM 响应缓存，None 则使用后端配置。

    成功返回响应 dict，失败返回 None（错误已通过 st.error 展示）。
    """
    use_status = _has_st_status()

    if use_status:
        return _run_agent_with_st_status(question, include_trace=include_trace, use_rerank=use_rerank, enable_cache=enable_cache)
    # fallback：低版本 Streamlit
    return _run_agent_with_spinner(question, include_trace=include_trace, use_rerank=use_rerank, enable_cache=enable_cache)


def _run_agent_with_st_status(question: str, include_trace: bool = True, use_rerank: bool | None = None, enable_cache: bool | None = None) -> object | None:
    """使用 st.status 展示 Agent 执行状态（优先路径）。"""
    with st.status("正在发送问题给 Agent...", expanded=True) as status:
        # 展示预期流程（概念性提示，非真实中间状态）
        st.caption(
            "注：后端为同步返回，以下为执行流程提示，"
            "真实步骤请参考下方的 Agent 执行过程时间线。"
        )
        for step in _AGENT_FLOW_STEPS:
            st.markdown(f"- {step}")

        try:
            payload: dict = {"question": question, "include_trace": include_trace}
            if use_rerank is not None:
                payload["use_rerank"] = use_rerank
            if enable_cache is not None:
                payload["enable_cache"] = enable_cache
            response = requests.post(
                f"{API_BASE}/api/agent/query",
                json=payload,
                timeout=120,
            )
        except Exception as e:
            status.update(label="Agent 请求异常", state="error", expanded=False)
            st.error(f"请求异常: {e}")
            return None

        if response.status_code != 200:
            status.update(
                label=f"查询失败 {response.status_code}", state="error", expanded=False
            )
            st.error(f"查询失败 {response.status_code}: {response.text}")
            return None

        try:
            result = response.json()
        except Exception as e:
            status.update(label="响应解析失败", state="error", expanded=False)
            st.error(f"响应 JSON 解析失败: {e}")
            return None

        status.update(label="Agent 执行完成", state="complete", expanded=False)
        return result


def _run_agent_with_spinner(question: str, include_trace: bool = True, use_rerank: bool | None = None, enable_cache: bool | None = None) -> object | None:
    """fallback 路径：低版本 Streamlit 使用 st.spinner + st.expander。"""
    with st.spinner("正在发送问题给 Agent..."):
        try:
            payload: dict = {"question": question, "include_trace": include_trace}
            if use_rerank is not None:
                payload["use_rerank"] = use_rerank
            if enable_cache is not None:
                payload["enable_cache"] = enable_cache
            response = requests.post(
                f"{API_BASE}/api/agent/query",
                json=payload,
                timeout=120,
            )
        except Exception as e:
            st.error(f"请求异常: {e}")
            return None

    if response.status_code != 200:
        st.error(f"查询失败 {response.status_code}: {response.text}")
        return None

    try:
        result = response.json()
    except Exception as e:
        st.error(f"响应 JSON 解析失败: {e}")
        return None

    with st.expander("Agent 执行流程（概念提示）", expanded=False):
        st.caption(
            "注：后端为同步返回，以下为执行流程提示，"
            "真实步骤请参考下方的 Agent 执行过程时间线。"
        )
        for step in _AGENT_FLOW_STEPS:
            st.markdown(f"- {step}")
    return result


def _run_rag_query_with_status(question: str, top_k: int, use_rerank: bool | None = None) -> object | None:
    """执行普通 RAG 查询，展示加载状态。

    成功返回响应 dict，失败返回 None（错误已通过 st.error 展示）。
    """
    with st.spinner("正在检索并生成回答..."):
        try:
            payload: dict = {"question": question, "top_k": top_k}
            if use_rerank is not None:
                payload["use_rerank"] = use_rerank
            response = requests.post(
                f"{API_BASE}/api/chat/query",
                json=payload,
                timeout=120,
            )
        except Exception as e:
            st.error(f"请求异常: {e}")
            return None

    if response.status_code != 200:
        st.error(f"查询失败 {response.status_code}: {response.text}")
        return None

    try:
        return response.json()
    except Exception as e:
        st.error(f"响应 JSON 解析失败: {e}")
        return None


def _render_agent_response(result: object, show_debug: bool, frontend_elapsed: float = 0.0, question: str = "") -> None:
    """编排 Agent 响应的整体渲染顺序。

    show_debug=True 时展示 agent_trace / tool_calls / errors / 原始 JSON；
    show_debug=False 时只展示来源 + 最终回答。
    question 用于 deferred 模式下触发 /api/agent/verify。
    """
    if not isinstance(result, dict):
        st.error("Agent 返回的数据结构异常，无法渲染。")
        render_agent_raw_json(result)
        return

    if show_debug:
        # 1. Agent 执行过程时间线（简短字符串列表）
        render_agent_trace(result.get("agent_trace"))
        # 1b. 结构化轨迹事件（含时间戳）
        render_trace_events(result.get("trace_events"))
        # 2. 工具调用统计
        _render_tool_stats(result.get("tool_calls"))
        # 3. 工具调用详情
        render_tool_calls(result.get("tool_calls"))

    # 4. 检索来源（始终展示，传入 tool_calls 做智能提示）
    render_agent_sources(result.get("sources"), result.get("tool_calls"))

    # 5. 最终回答（始终展示，位于执行过程之后）
    render_agent_final_answer(result.get("answer"), result.get("refused"))

    # 6. 证据校验（始终展示，传入 question/answer/sources/tool_calls 以支持 deferred）
    render_agent_verification(
        result.get("verification"),
        question=question,
        answer=str(result.get("answer") or ""),
        sources=result.get("sources"),
        tool_calls=result.get("tool_calls"),
    )

    # 7. 耗时统计（始终展示）
    _render_timing_panel(
        result.get("timing"),
        frontend_elapsed,
        verification=result.get("verification"),
        tool_calls=result.get("tool_calls"),
    )

    # 8. 性能提示
    _render_performance_warnings(
        result.get("verification"),
        result.get("tool_calls"),
        str(result.get("answer") or ""),
    )

    if show_debug:
        # 9. 错误信息
        render_agent_errors(result.get("errors"))
        # 10. 原始返回 JSON
        render_agent_raw_json(result)


def _render_performance_warnings(
    verification: object,
    tool_calls: object,
    answer: str,
) -> None:
    """根据耗时、工具调用数和回答长度展示性能提示。

    规则：
    - verification_elapsed > 10s → 建议关闭/延迟/轻量
    - tool_calls 数量 > 5 → 可能导致响应变慢
    - answer 长度 > 1200 字符 → 可能增加生成与校验耗时
    """
    warnings_shown = False

    # verification 耗时过长
    if isinstance(verification, dict):
        v_timing = verification.get("timing")
        if isinstance(v_timing, dict):
            v_elapsed = v_timing.get("verification_elapsed")
            if isinstance(v_elapsed, (int, float)) and v_elapsed > 10:
                st.warning(
                    "证据校验耗时较长，可考虑关闭、延迟执行或使用轻量模式。"
                )
                warnings_shown = True

    # 工具调用过多
    if isinstance(tool_calls, list) and len(tool_calls) > 5:
        st.warning("本次工具调用较多，可能导致响应变慢。")
        warnings_shown = True

    # 回答过长
    if len(answer) > 1200:
        st.warning("回答较长，可能增加生成与校验耗时。")
        warnings_shown = True

    if not warnings_shown:
        st.caption("✅ 各项指标正常，无明显性能瓶颈。")


def _render_timing_panel(
    timing: object,
    frontend_elapsed: float,
    verification: object = None,
    tool_calls: object = None,
) -> None:
    """渲染耗时统计面板（前端观测 + 后端 timing + verification + per-tool）。

    verification 和 tool_calls 为可选参数，用于展示 verification 耗时和
    每个工具的单独耗时。
    """
    with st.expander("⏱️ 耗时统计", expanded=False):
        st.markdown(f"**前端观测总耗时：** `{frontend_elapsed:.2f} 秒`")

        if isinstance(timing, dict):
            total = timing.get("total_elapsed")
            invoke = timing.get("agent_invoke_elapsed")
            tool_total = timing.get("tool_search_elapsed_total")

            if isinstance(total, (int, float)):
                st.markdown(f"**后端 Agent 总耗时（total_elapsed）：** `{total:.4f} 秒`")
            if isinstance(invoke, (int, float)):
                st.markdown(f"**agent.invoke 耗时（agent_invoke_elapsed）：** `{invoke:.4f} 秒`")
            if isinstance(tool_total, (int, float)):
                st.markdown(f"**工具检索总耗时（tool_search_elapsed_total）：** `{tool_total:.4f} 秒`")

            # 缓存命中统计
            cache_enabled = timing.get("cache_enabled")
            if cache_enabled is True:
                hits = timing.get("cache_hits", 0)
                misses = timing.get("cache_misses", 0)
                if isinstance(hits, (int, float)) and hits > 0:
                    st.success(
                        f"✅ LLM 缓存命中：{hits} 次 / 未命中：{misses} 次（省去 {hits} 次 LLM 调用）"
                    )
                else:
                    st.caption(
                        f"⚪ LLM 缓存已启用：命中 {hits} 次 / 未命中 {misses} 次（首次提问，结果已写入缓存）"
                    )
        else:
            st.caption("后端未返回 timing 信息。")

        # verification 耗时
        if isinstance(verification, dict):
            v_timing = verification.get("timing")
            if isinstance(v_timing, dict):
                v_elapsed = v_timing.get("verification_elapsed")
                if isinstance(v_elapsed, (int, float)) and v_elapsed > 0:
                    st.markdown(f"**证据校验耗时（verification_elapsed）：** `{v_elapsed:.4f} 秒`")

        # 每个工具的单独耗时
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            st.markdown("**各工具耗时明细：**")
            for i, tc in enumerate(tool_calls, start=1):
                if not isinstance(tc, dict):
                    continue
                tc_name = str(tc.get("tool") or "未知工具")
                tc_cn = TOOL_NAME_CN.get(tc_name, tc_name)
                tc_elapsed = tc.get("elapsed")
                if isinstance(tc_elapsed, (int, float)):
                    st.markdown(f"  {i}. {tc_cn}：`{tc_elapsed:.4f} 秒`")
                else:
                    st.markdown(f"  {i}. {tc_cn}：`无耗时信息`")


def _render_rag_response(result: object) -> None:
    """渲染普通 RAG 响应（保持原有逻辑，不做破坏性改动）。"""
    if not isinstance(result, dict):
        st.error("RAG 返回的数据结构异常，无法渲染。")
        return

    # ---------- 回答 ----------
    st.subheader("📝 回答")
    if result.get("refused"):
        st.warning(result.get("answer") or "无回答内容")
    else:
        st.markdown(result.get("answer") or "")

    # ---------- 引用来源 ----------
    st.subheader("📚 引用来源")
    sources = result.get("sources", [])
    if not sources:
        st.caption("无引用来源")
    else:
        for i, s in enumerate(sources, start=1):
            if not isinstance(s, dict):
                continue
            score = s.get("score")
            score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else "无"
            with st.expander(
                f"[{i}] {s.get('filename', '?')} - 第{s.get('page', '?')}页 | score={score_str}"
            ):
                st.caption(f"chunk_id: {s.get('chunk_id', '无')}")
                st.write(s.get("content_preview", ""))


# =====================================================================
# Work Unit（工作单元）：保存 / 列表 / 详情（v1，不含 Replay）
# =====================================================================
def _render_save_work_unit_button(result: dict) -> None:
    """在 RAG / Agent 结果下方渲染「保存为 Work Unit」按钮。

    - 仅当响应含 work_unit_candidate 时显示；
    - 不自动保存；用 session_state 防止同一候选对象被重复提交；
    - 保存失败时显示中文错误提示，不暴露本地路径 / 密钥。
    """
    candidate = result.get("work_unit_candidate") if isinstance(result, dict) else None
    if not candidate or not isinstance(candidate, dict):
        return

    # 候选对象的指纹，用于防重复提交（同一响应多次点击只保存一次）
    fingerprint = f"{candidate.get('entry')}:{candidate.get('question')}:{candidate.get('answer', '')[:32]}"
    saved_key = f"wu_saved::{fingerprint}"

    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 保存为 Work Unit", key=f"save_wu::{fingerprint}"):
            if st.session_state.get(saved_key):
                st.info("该结果已保存为 Work Unit，无需重复保存。")
            else:
                wu_id = _post_work_unit(candidate)
                if wu_id:
                    st.session_state[saved_key] = wu_id
                    st.success(f"Work Unit 已保存：`{wu_id}`")
                # _post_work_unit 内部失败时已通过 st.error 提示，此处不重复
    with col2:
        existing = st.session_state.get(saved_key)
        if existing:
            st.caption(f"✅ 已保存：{existing}")


def _post_work_unit(candidate: dict) -> str | None:
    """调用 POST /api/work_units 保存候选对象，成功返回 work_unit_id，失败返回 None。"""
    try:
        resp = requests.post(
            f"{API_BASE}/api/work_units",
            json=candidate,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"保存 Work Unit 失败：后端连接异常（{type(e).__name__}）")
        return None

    if resp.status_code != 200:
        st.error(f"保存 Work Unit 失败：HTTP {resp.status_code}")
        return None

    try:
        return resp.json().get("work_unit_id")
    except Exception:  # noqa: BLE001
        st.error("保存 Work Unit 失败：响应解析异常。")
        return None


def _render_work_units_view() -> None:
    """工作单元视图：列表 + 可展开详情（v1，无 Replay）。"""
    st.subheader("🗂️ 工作单元历史")
    st.caption("把一次 RAG / Agent / MCP 调用沉淀为可复盘的工作单元。")

    try:
        resp = requests.get(f"{API_BASE}/api/work_units", params={"limit": 20}, timeout=30)
    except Exception as e:  # noqa: BLE001
        st.warning(f"后端连接失败：{type(e).__name__}")
        return

    if resp.status_code != 200:
        st.warning(f"获取工作单元列表失败：HTTP {resp.status_code}")
        return

    body = resp.json()
    work_units = body.get("work_units", [])
    if not work_units:
        st.info("暂无工作单元。在提问后点击「💾 保存为 Work Unit」即可沉淀。")
        return

    for wu in work_units:
        if not isinstance(wu, dict):
            continue
        wu_id = wu.get("work_unit_id", "?")
        entry = wu.get("entry", "?")
        refused = wu.get("refused", False)
        header = (
            f"[{entry}] {wu.get('question', '(无问题)')[:40]}"
            f" | {wu.get('created_at', '')[:19]}"
            f" | {'拒答' if refused else '已回答'}"
        )
        with st.expander(header, expanded=False):
            _render_work_unit_detail(wu)

    # v1 明确不实现 Replay：仅显示灰色提示，不放可点击按钮
    st.caption("🔁 Replay 将在 v2 支持。")


def _render_work_unit_detail(wu: dict) -> None:
    """渲染单个 Work Unit 的复盘详情，尽量复用现有渲染函数。"""
    st.markdown(f"**work_unit_id**：`{wu.get('work_unit_id', '')}`")

    # 问题 / 回答
    st.markdown("**❓ 问题**")
    st.write(wu.get("question", ""))
    st.markdown("**📝 回答**")
    answer = wu.get("answer")
    if wu.get("refused"):
        st.warning(answer or "无回答内容")
    else:
        st.markdown(answer or "（无回答）")

    # 来源（复用现有渲染器；候选对象里 sources 为 list[dict]，与渲染器期望一致）
    if wu.get("sources"):
        render_agent_sources(wu.get("sources"))

    # 工具调用 / 轨迹事件（复用现有渲染器；为空时渲染器自行处理）
    render_tool_calls(wu.get("tool_calls"))
    render_trace_events(wu.get("trace_events"))

    # 校验 / 耗时 / 错误
    if wu.get("verification"):
        with st.expander("🔬 证据校验（verification）"):
            st.json(wu.get("verification"))
    if wu.get("timing"):
        with st.expander("⏱️ 耗时（timing）"):
            st.json(wu.get("timing"))
    if wu.get("errors"):
        with st.expander(f"⚠️ 错误信息（{len(wu.get('errors'))}）"):
            for err in wu.get("errors"):
                st.write(f"- {err}")

    # replay_payload：v1 仅保存展示，不可执行
    with st.expander("🔁 重放配置（replay_payload，v2 才可执行）"):
        st.json(wu.get("replay_payload"))
        st.caption("Replay 将在 v2 支持，当前仅保存配置。")


# =====================================================================
# 侧边栏：文档管理
# =====================================================================
with st.sidebar:
    st.header("📄 文档管理")

    uploaded = st.file_uploader(
        "上传文档 (PDF / TXT / MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("上传到服务器"):
        for f in uploaded:
            files = {"file": (f.name, f.getvalue())}
            try:
                r = requests.post(f"{API_BASE}/api/documents/upload", files=files, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ {data['filename']} (doc_id={data['doc_id']})")
                else:
                    st.error(f"上传失败 {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"上传异常: {e}")

    if st.button("📥 一键入库"):
        try:
            r = requests.post(
                f"{API_BASE}/api/documents/ingest", json={}, timeout=300
            )
            if r.status_code == 200:
                st.success(r.json().get("message", "入库完成"))
            else:
                st.error(f"入库失败 {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"入库异常: {e}")

    st.divider()
    if st.button("🔄 刷新文档列表"):
        st.session_state["refresh_docs"] = True

    try:
        r = requests.get(f"{API_BASE}/api/documents", timeout=30)
        if r.status_code == 200:
            docs = r.json().get("documents", [])
            st.caption(f"知识库中共 {len(docs)} 个文档")
            for d in docs:
                st.write(
                    f"- `{d['doc_id']}` {d['filename']} ({d['chunk_count']} chunks)"
                )
        else:
            st.warning("无法获取文档列表，请确认后端已启动")
    except Exception as e:
        st.warning(f"后端连接失败: {e}")


# =====================================================================
# 主区域：问答
# =====================================================================
st.header("💬 提问")

# 问答模式切换：普通 RAG / Agent 研究助手 / 工作单元
qa_mode = st.radio(
    "问答模式",
    options=["agent", "rag", "work_units"],
    format_func=lambda x: {
        "agent": "Agent 研究助手（RAG as Tool）",
        "rag": "普通 RAG 检索问答",
        "work_units": "🗂️ 工作单元",
    }[x],
    horizontal=True,
    help="普通 RAG：固定检索后拼接 Prompt 生成；"
    "Agent：LLM 自主决定是否调用知识库检索工具，返回工具调用轨迹；"
    "工作单元：查看已保存的 Work Unit。",
)

# 工作单元视图：展示列表与详情，不进入提问流程
if qa_mode == "work_units":
    _render_work_units_view()
    st.stop()

# Agent 模式顶部说明
show_debug = True  # 默认值，仅 Agent 模式下会被复选框覆盖
if qa_mode == "agent":
    st.markdown("**当前模式：Agent 研究助手（Multi-Tool）**")
    st.caption(
        "Agent 研究助手会根据问题类型自主选择不同工具，"
        "例如知识库检索、数据集结构化查询、模型对比、指标公式查询或指标计算。"
    )
    # 调试信息开关（可选优化）
    show_debug = st.checkbox("显示调试信息", value=True)

question = st.text_input(
    "输入你的问题",
    placeholder="例如：Landsat 8 的热红外波段中心波长是多少？",
)
top_k = st.slider("检索 Top-K", min_value=1, max_value=10, value=5)

use_rerank = st.checkbox(
    "启用 Rerank 重排序（Cross-encoder 精排，提升检索精度，增加 ~1s 延迟）",
    value=False,
    help="开启后先向量检索 10 条候选，再用 bge-reranker-v2-m3 精排取 Top-K。"
    "消融实验表明可提升回答质量约 4.5%。",
)

# Agent 模式下额外的缓存开关
enable_cache = False
if qa_mode == "agent":
    enable_cache = st.checkbox(
        "启用 LLM 响应缓存（相同问题秒回，适合演示/重复测试）",
        value=False,
        help="开启后 Agent 路径的 ChatOpenAI LLM 调用将缓存到内存。"
        "相同问题第二次直接返回缓存结果，无需调用 LLM API。"
        "注意：仅对 Agent 模式生效，普通 RAG 模式不受影响。",
    )


if st.button("🚀 提交问题") and question.strip():
    # 记录请求耗时
    start_time = time.time()

    # ------- 执行状态展示（前端流程提示，非真正后端流式） -------
    if qa_mode == "agent":
        result_data = _run_agent_query_with_status(
            question.strip(), include_trace=show_debug, use_rerank=use_rerank, enable_cache=enable_cache,
        )
    else:
        result_data = _run_rag_query_with_status(
            question.strip(), top_k, use_rerank=use_rerank,
        )

    elapsed = time.time() - start_time

    if result_data is None:
        # _run_* 函数内部已通过 st.error 展示错误，这里直接终止
        st.stop()

    # ------- 耗时展示 -------
    st.caption(f"⏱️ 本次{'Agent' if qa_mode == 'agent' else 'RAG'}请求耗时：{elapsed:.2f} 秒")

    if qa_mode == "agent":
        _render_agent_response(result_data, show_debug, elapsed, question.strip())
    else:
        _render_rag_response(result_data)

    # Work Unit：在结果下方提供「保存为 Work Unit」按钮（不自动保存）
    _render_save_work_unit_button(result_data)
