"""Evidence Verification 模块：校验 Agent 回答是否被 sources / tool outputs 支撑。

支持三种模式（通过 app/config.py Settings.agent_verification_mode 控制）：
- off：不执行证据校验
- sync：在主请求中同步执行证据校验
- deferred：主请求跳过校验，前端再调用 /api/agent/verify 独立获取校验结果

支持两种轻量化级别（通过 Settings.agent_verification_level 控制）：
- lightweight：裁剪 answer/sources/tool_calls 后校验（默认）
- full：相对完整校验，但仍做基本截断避免超长输入

LLM 参数通过 Settings 读取（verification_model / verification_max_tokens / verification_temperature），
不硬编码模型名 / API Key / base_url。

设计要点：
- 拒答回答直接返回 verified=true（拒答不涉及事实扩展）。
- 既无 sources 也无有效 tool_calls 时返回 verified=false。
- LLM 输出 JSON 解析失败时 fallback 为 verified=false / confidence=low。
- llm_client 参数支持测试注入，避免真实调用外部 LLM。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 拒答关键词（与 app/core/prompts.py REFUSAL_ANSWER / langchain_agent REFUSAL_MARKER 一致）
_REFUSAL_KEYWORD = "无法确定该问题的答案"


# -------------------------------------------------------------------------- #
#  轻量化裁剪常量                                                              #
# -------------------------------------------------------------------------- #

#: lightweight 模式下 answer 最大字符数
_LIGHTWEIGHT_ANSWER_CHARS = 800

#: full 模式下 answer 最大字符数
_FULL_ANSWER_CHARS = 1500

#: lightweight 模式下 sources 最大条数
_LIGHTWEIGHT_MAX_SOURCES = 5

#: full 模式下 sources 最大条数
_FULL_MAX_SOURCES = 8

#: 每条 source 的 content_preview 最大字符数
_MAX_PREVIEW_CHARS = 150

#: lightweight 模式下 tool_calls 最大条数
_LIGHTWEIGHT_MAX_TOOL_CALLS = 6

#: full 模式下 tool_calls 最大条数
_FULL_MAX_TOOL_CALLS = 8

#: 每条 tool_call 的 output_summary 最大字符数
_MAX_SUMMARY_CHARS = 200

#: ungrounded_claims 最大条数
_MAX_UNGROUNDED_CLAIMS = 5

#: reason 最大中文字符数
_MAX_REASON_CHARS = 100


# -------------------------------------------------------------------------- #
#  Verification Prompt                                                        #
# -------------------------------------------------------------------------- #

VERIFICATION_SYSTEM_PROMPT = """你是一个严格的事实核查助手。你的任务是判断一段回答中的每一个关键论断是否都能在提供的参考资料中找到依据。

判断规则：
1. 回答中的具体数值（如 mIoU、参数量、分辨率等）必须有参考资料直接支撑。
2. 回答中的结论必须有参考资料推导得出。
3. 回答中有但参考资料中没有的论断，属于"未证实论断"。
4. 如果回答完全基于参考资料，verified=true。
5. 如果回答中存在未被参考资料支撑的论断，verified=false，并列出这些论断。

confidence 级别说明：
- high：所有论断都有明确依据，或完全没有论断需要验证。
- medium：大部分论断有依据，但个别论断依据不够直接。
- low：无法判断，或参考资料严重不足。

约束：
- reason 控制在 100 个中文字符以内。
- ungrounded_claims 最多 5 条，每条为一个具体论断。
- 严格输出以下 JSON 格式（不要输出任何其他内容，不要使用 markdown 代码块）：

{"verified": true/false, "confidence": "high"/"medium"/"low", "ungrounded_claims": ["未被来源支撑的具体论断"], "reason": "简要说明判断依据"}"""

VERIFICATION_USER_TEMPLATE = """## 用户问题
{question}

## 待验证的回答
{answer}

## 参考资料 — 来源片段（content_preview）
{sources_text}

## 参考资料 — 工具输出摘要（output_summary）
{tool_outputs_text}

请判断回答中的每一个关键论断是否都能在上述参考资料中找到依据，并输出 JSON。"""


# -------------------------------------------------------------------------- #
#  裁剪辅助函数                                                               #
# -------------------------------------------------------------------------- #

def _truncate_str(text: str, max_chars: int) -> str:
    """安全截断字符串，超出长度时追加 '...'。"""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _trim_answer(answer: str, level: str) -> str:
    """根据 level 裁剪 answer 长度。"""
    max_chars = _LIGHTWEIGHT_ANSWER_CHARS if level == "lightweight" else _FULL_ANSWER_CHARS
    return _truncate_str(answer, max_chars)


def _trim_sources(sources: list, level: str) -> list[dict[str, Any]]:
    """根据 level 裁剪 sources 列表。

    保留字段：filename / page / chunk_id / score / content_preview（截断到 150 字）。
    """
    max_items = _LIGHTWEIGHT_MAX_SOURCES if level == "lightweight" else _FULL_MAX_SOURCES
    result: list[dict[str, Any]] = []
    for s in sources[:max_items]:
        if not isinstance(s, dict):
            continue
        result.append({
            "filename": s.get("filename", "未知"),
            "page": s.get("page", 1),
            "chunk_id": s.get("chunk_id", ""),
            "score": s.get("score", 0.0),
            "content_preview": _truncate_str(
                s.get("content_preview", ""), _MAX_PREVIEW_CHARS
            ),
        })
    return result


def _trim_tool_calls(tool_calls: list | None, level: str) -> list[dict[str, Any]]:
    """根据 level 裁剪 tool_calls 列表。

    保留字段：tool / status / output_summary（截断到 200 字）。
    """
    if not tool_calls:
        return []
    max_items = _LIGHTWEIGHT_MAX_TOOL_CALLS if level == "lightweight" else _FULL_MAX_TOOL_CALLS
    result: list[dict[str, Any]] = []
    for tc in tool_calls[:max_items]:
        if not isinstance(tc, dict):
            continue
        result.append({
            "tool": tc.get("tool", "未知工具"),
            "status": tc.get("status", "unknown"),
            "output_summary": _truncate_str(
                tc.get("output_summary") or "", _MAX_SUMMARY_CHARS
            ),
        })
    return result


def _trim_ungrounded_claims(claims: list) -> list[str]:
    """裁剪 ungrounded_claims 到最大条数。"""
    return [str(c) for c in (claims or [])[:_MAX_UNGROUNDED_CLAIMS]]


# -------------------------------------------------------------------------- #
#  LLM 客户端构建                                                             #
# -------------------------------------------------------------------------- #

def _get_llm_client() -> Any:
    """懒加载 LLM 客户端。

    复用 OpenAICompatibleLLMClient，但覆盖 model / max_tokens / temperature，
    以支持 VERIFICATION_MODEL / VERIFICATION_MAX_TOKENS / VERIFICATION_TEMPERATURE。
    """
    from app.core.llm import OpenAICompatibleLLMClient
    from app.config import get_settings

    settings = get_settings()

    # 如果配置了独立 verification 模型，使用它；否则复用主模型
    model = settings.verification_model if settings.verification_model else settings.llm_model

    client = OpenAICompatibleLLMClient(
        model=model,
        max_tokens=settings.verification_max_tokens,
        temperature=settings.verification_temperature,
    )
    return client


# -------------------------------------------------------------------------- #
#  短路判断辅助函数                                                           #
# -------------------------------------------------------------------------- #

def _is_refusal(answer: str) -> bool:
    """判断回答是否为拒答。"""
    return _REFUSAL_KEYWORD in answer


def _format_sources(sources: list) -> str:
    """格式化 sources 的 content_preview 供 prompt 使用。"""
    if not sources:
        return "（无来源片段）"
    parts: list[str] = []
    for i, s in enumerate(sources, start=1):
        if isinstance(s, dict):
            preview = s.get("content_preview", "")
            filename = s.get("filename", "未知")
            parts.append(f"[来源{i}] 文件: {filename}\n内容: {preview}")
        else:
            parts.append(f"[来源{i}] {s}")
    return "\n\n".join(parts) if parts else "（无来源片段）"


def _format_tool_outputs(tool_calls: list | None) -> str:
    """格式化 tool_calls 的 output_summary 供 prompt 使用。"""
    if not tool_calls:
        return "（无工具输出）"
    parts: list[str] = []
    for i, tc in enumerate(tool_calls, start=1):
        if isinstance(tc, dict):
            tool_name = tc.get("tool", "未知工具")
            summary = tc.get("output_summary") or "无"
            parts.append(f"[工具{i}] {tool_name}: {summary}")
        else:
            parts.append(f"[工具{i}] {tc}")
    return "\n".join(parts) if parts else "（无工具输出）"


# -------------------------------------------------------------------------- #
#  JSON 解析辅助函数                                                          #
# -------------------------------------------------------------------------- #

def _parse_verification_json(raw: str) -> dict | None:
    """尝试从 LLM 输出中解析 JSON，支持多种格式。

    依次尝试：
    1. 直接 json.loads
    2. 从 markdown 代码块中提取
    3. 从原始文本中提取第一个 { ... } 结构
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # 尝试 1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试 2：从 markdown 代码块中提取
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试 3：提取第一个 { ... } 结构（贪婪匹配最外层花括号）
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_result(parsed: dict) -> dict:
    """验证并规范化解析结果，确保输出格式一致。"""
    # verified
    verified = parsed.get("verified", False)
    if isinstance(verified, str):
        verified = verified.lower().strip() == "true"
    elif not isinstance(verified, bool):
        verified = bool(verified)

    # confidence
    confidence = parsed.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    # ungrounded_claims
    ungrounded_claims = parsed.get("ungrounded_claims", [])
    if not isinstance(ungrounded_claims, list):
        ungrounded_claims = [str(ungrounded_claims)] if ungrounded_claims else []
    else:
        ungrounded_claims = [str(c) for c in ungrounded_claims]
    ungrounded_claims = _trim_ungrounded_claims(ungrounded_claims)

    # reason（截断到 100 字）
    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason) if reason else ""
    reason = _truncate_str(reason, _MAX_REASON_CHARS)

    return {
        "verified": verified,
        "confidence": confidence,
        "ungrounded_claims": ungrounded_claims,
        "reason": reason,
    }


def _fallback_result(reason: str, elapsed: float) -> dict:
    """构造 fallback 结果（JSON 解析失败或 LLM 调用失败时使用）。"""
    return {
        "verified": False,
        "confidence": "low",
        "ungrounded_claims": [],
        "reason": _truncate_str(reason, _MAX_REASON_CHARS),
        "timing": {"verification_elapsed": round(elapsed, 4)},
    }


# -------------------------------------------------------------------------- #
#  模式化返回（off / deferred）                                                #
# -------------------------------------------------------------------------- #

def make_off_result() -> dict:
    """verification 关闭时的标准返回。"""
    return {
        "enabled": False,
        "mode": "off",
        "level": None,
        "pending": False,
        "verified": None,
        "confidence": None,
        "ungrounded_claims": [],
        "reason": "Evidence Verification 未启用。",
        "timing": {"verification_elapsed": 0.0},
    }


def make_deferred_pending_result(level: str) -> dict:
    """deferred 模式下 /api/agent/query 返回的 pending 结构。"""
    return {
        "enabled": True,
        "mode": "deferred",
        "level": level,
        "pending": True,
        "verified": None,
        "confidence": None,
        "ungrounded_claims": [],
        "reason": "Evidence Verification 将在独立请求中执行。",
        "timing": {"verification_elapsed": 0.0},
    }


# -------------------------------------------------------------------------- #
#  核心函数                                                                   #
# -------------------------------------------------------------------------- #

def verify_answer(
    question: str,
    answer: str,
    sources: list,
    tool_calls: list | None = None,
    llm_client: Any = None,
    level: str | None = None,
) -> dict:
    """验证 Agent 回答是否被 sources / tool outputs 支撑。

    Args:
        question: 用户原始问题。
        answer: Agent 生成的回答。
        sources: 检索来源列表（list[dict]），每个 dict 包含 content_preview 等。
        tool_calls: 工具调用列表（list[dict]），每个 dict 包含 output_summary 等。
        llm_client: 可选的 LLM 客户端实例（用于测试注入），
                    为 None 则从配置懒加载 LLM 客户端。
        level: 轻量化级别（"lightweight" / "full"），为 None 则从 Settings 读取。

    Returns:
        dict，格式如下::

            {
                "enabled": True,
                "mode": "sync",
                "level": "lightweight",
                "pending": False,
                "verified": True,
                "confidence": "high",
                "ungrounded_claims": [],
                "reason": "一句话说明",
                "timing": {"verification_elapsed": 0.42}
            }

    特殊处理：
    - 拒答回答 → verified=true（拒答不涉及事实扩展）。
    - 既无 sources 也无有效 tool_calls → verified=false。
    - LLM 输出解析失败 → fallback verified=false / confidence=low。
    """
    from app.config import get_settings

    start = time.time()
    settings = get_settings()

    # 确定 level
    actual_level = level if level is not None else settings.agent_verification_level
    if actual_level not in ("lightweight", "full"):
        actual_level = "lightweight"

    # ---------- 拒答快速通过 ----------
    if _is_refusal(answer):
        elapsed = time.time() - start
        logger.info("Evidence Verification: 回答为拒答，直接通过")
        return {
            "enabled": True,
            "mode": "sync",
            "level": actual_level,
            "pending": False,
            "verified": True,
            "confidence": "high",
            "ungrounded_claims": [],
            "reason": "拒答不涉及事实扩展。",
            "timing": {"verification_elapsed": round(elapsed, 4)},
        }

    # ---------- 裁剪输入 ----------
    trimmed_answer = _trim_answer(answer, actual_level)
    trimmed_sources = _trim_sources(sources, actual_level)
    trimmed_tool_calls = _trim_tool_calls(tool_calls, actual_level)

    # ---------- 既无 sources，也无有效 tool_calls ----------
    has_sources = bool(trimmed_sources)
    has_tool_outputs = bool(trimmed_tool_calls) and any(
        isinstance(tc, dict) and tc.get("output_summary")
        for tc in trimmed_tool_calls
    )
    if not has_sources and not has_tool_outputs:
        elapsed = time.time() - start
        logger.warning("Evidence Verification: 无来源片段且无有效工具输出")
        return {
            "enabled": True,
            "mode": "sync",
            "level": actual_level,
            "pending": False,
            "verified": False,
            "confidence": "low",
            "ungrounded_claims": [],
            "reason": "既无来源片段，也无有效工具输出，无法验证回答。",
            "timing": {"verification_elapsed": round(elapsed, 4)},
        }

    # ---------- 构建 Verification Prompt ----------
    sources_text = _format_sources(trimmed_sources)
    tool_outputs_text = _format_tool_outputs(trimmed_tool_calls)

    user_prompt = VERIFICATION_USER_TEMPLATE.format(
        question=question,
        answer=trimmed_answer,
        sources_text=sources_text,
        tool_outputs_text=tool_outputs_text,
    )

    # ---------- 调用 LLM ----------
    try:
        if llm_client is None:
            llm_client = _get_llm_client()
        raw_response = llm_client.chat(
            prompt=user_prompt,
            system=VERIFICATION_SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("Evidence Verification LLM 调用失败: %s", e)
        elapsed = time.time() - start
        fb = _fallback_result(f"验证模型调用失败: {e}", elapsed)
        fb["enabled"] = True
        fb["mode"] = "sync"
        fb["level"] = actual_level
        fb["pending"] = False
        return fb

    # ---------- 解析 JSON ----------
    parsed = _parse_verification_json(raw_response)
    if parsed is None:
        logger.warning(
            "Evidence Verification JSON 解析失败，raw=%r", raw_response[:200]
        )
        elapsed = time.time() - start
        fb = _fallback_result(
            "验证模型输出解析失败，无法确认回答是否完全有据。", elapsed
        )
        fb["enabled"] = True
        fb["mode"] = "sync"
        fb["level"] = actual_level
        fb["pending"] = False
        return fb

    # ---------- 规范化结果 ----------
    result = _validate_result(parsed)
    elapsed = time.time() - start
    result["timing"] = {"verification_elapsed": round(elapsed, 4)}
    result["enabled"] = True
    result["mode"] = "sync"
    result["level"] = actual_level
    result["pending"] = False

    logger.info(
        "Evidence Verification 完成: verified=%s, confidence=%s, ungrounded=%d, "
        "elapsed=%.4fs, level=%s",
        result["verified"],
        result["confidence"],
        len(result["ungrounded_claims"]),
        elapsed,
        actual_level,
    )

    return result
