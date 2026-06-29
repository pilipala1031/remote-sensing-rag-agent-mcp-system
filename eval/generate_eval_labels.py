#!/usr/bin/env python
"""LLM 辅助评估标签生成工具。

读取 eval/eval_questions.json 中的问题，调用 .env 配置的主 LLM（默认 glm-5.1）
自动生成结构化评估标签，并提供交互式人工校验界面。

流程：
1. 加载 eval/eval_questions.json 评估题集
2. 扫描知识库文档目录，获取 .md 文档列表
3. 逐题调用 LLM API 生成评估标签（含重试）
4. 交互式人工校验（接受 / 编辑 / 跳过）
5. 保存到 eval/eval_questions_with_labels.json

使用方式：
    # 生成初版标注（自动接受所有，跳过人工校验）
    python eval/generate_eval_labels.py --auto-accept

    # 生成并人工校验
    python eval/generate_eval_labels.py

    # 从中断处继续（跳过已完成且已校验的问题）
    python eval/generate_eval_labels.py --resume

    # 打印 LLM 原始返回，便于排查 JSON 解析失败
    python eval/generate_eval_labels.py --auto-accept --debug-raw

不依赖 pytest 或后端服务，仅调用 .env 配置的 LLM API。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

# 将项目根目录加入 sys.path，以便导入 app.config
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402


# 读取配置
SETTINGS = get_settings()

# LLM 配置 — 复用 .env 中的 LLM_MODEL / LLM_API_KEY / LLM_BASE_URL
# 智谱官方模型名建议使用小写 glm-5.1
LABEL_LLM_MODEL = str(SETTINGS.llm_model or "glm-5.1").strip()
if LABEL_LLM_MODEL.upper().startswith("GLM-"):
    LABEL_LLM_MODEL = LABEL_LLM_MODEL.lower()

LABEL_LLM_TEMPERATURE = 0.0
LABEL_LLM_MAX_TOKENS = 1024
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒

# 文件路径
QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
OUTPUT_PATH = EVAL_DIR / "eval_questions_with_labels.json"

# 知识库文档目录（默认使用 examples/sample_docs）
DEFAULT_KB_DIR = PROJECT_ROOT / "examples" / "sample_docs"

# 可用工具列表（与 Agent 的 7 个工具保持一致）
AVAILABLE_TOOLS: List[str] = [
    "knowledge_base_search",
    "dataset_overview",
    "dataset_spec_lookup",
    "model_comparison_table",
    "metric_formula_lookup",
    "metrics_calculator",
    "plan_and_search",
]

VALID_QUESTION_TYPES = {
    "basic",
    "structured",
    "calculation",
    "comparison",
    "out_of_scope",
}

VALID_TOOLS = set(AVAILABLE_TOOLS)

# --------------------------------------------------------------------------- #
# LLM Prompt 模板
# --------------------------------------------------------------------------- #

LABEL_SYSTEM_PROMPT = (
    "你是遥感语义分割领域的评估专家。"
    "你的任务是为评估问题生成严格合法的 JSON 评估标签。"
    "必须只返回一个 JSON object，不要返回 Markdown，不要返回解释文字。"
)

LABEL_USER_TEMPLATE = """问题：
{question}

可用的知识库文档列表：
{knowledge_base_docs}

可用工具列表（Agent 的 7 个工具）：
{available_tools}

请严格按照以下 JSON 格式返回标注。必须是合法 JSON，不要有任何额外文字：

{{
  "should_refuse": false,
  "required_keywords": ["关键词1", "关键词2", "关键词3"],
  "relevant_docs": ["doc1.md", "doc2.md"],
  "question_type": "basic",
  "min_answer_length": 50,
  "expected_tool": "knowledge_base_search",
  "notes": "简短说明该问题的评估要点"
}}

标注说明：
- should_refuse:
  - 如果问题超出遥感语义分割领域范围，设为 true
  - 如果 should_refuse 为 true，则 question_type 应为 "out_of_scope"
- required_keywords:
  - 正确答案必须包含的 3-5 个关键词或短语
  - 必须是字符串数组
- relevant_docs:
  - 只能从上面的知识库文档列表中选择文件名
  - 不要编造不存在的文件名
  - 如果问题超出范围，可为空数组
- question_type:
  - 只能是以下之一：
    - basic: 基础概念解释
    - structured: 结构化查询（数据集属性 / 模型参数等）
    - calculation: 数值计算
    - comparison: 多实体对比分析
    - out_of_scope: 超出领域范围
- min_answer_length:
  - 合理答案的最小字符数，使用整数
- expected_tool:
  - 对于 Agent，该问题最应该调用的工具
  - 只能从可用工具列表中选择一个
- notes:
  - 评估时需要注意的特殊点
"""


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #

def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """加载评估题集 JSON。

    Returns:
        问题列表，每个元素包含 id / question / category 等字段。

    Raises:
        FileNotFoundError: 题集文件不存在。
        ValueError: 题集格式错误（非 JSON 数组或缺少必要字段）。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"评估题集文件不存在：{path}\n"
            f"请确保 eval/eval_questions.json 已创建。"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"评估题集格式错误：{path} 应为 JSON 数组，"
            f"实际为 {type(data).__name__}"
        )

    for i, item in enumerate(data):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(
                f"评估题集第 {i + 1} 条缺少 'question' 字段"
            )

    return data


def get_knowledge_base_docs(kb_dir: Path = DEFAULT_KB_DIR) -> List[str]:
    """扫描知识库目录，返回所有 .md / .markdown 文档文件名。"""
    if not kb_dir.exists():
        raise FileNotFoundError(
            f"知识库文档目录不存在：{kb_dir}\n"
            f"请确保文档已放置在 {DEFAULT_KB_DIR} 目录下，"
            f"或通过 --kb-dir 参数指定其他目录。"
        )

    docs = sorted([
        f.name for f in kb_dir.iterdir()
        if f.suffix.lower() in (".md", ".markdown") and f.is_file()
    ])

    if not docs:
        raise FileNotFoundError(
            f"知识库文档目录中没有 .md 文件：{kb_dir}\n"
            f"请先将遥感领域知识文档放入该目录。"
        )

    return docs


def load_existing_output(path: Path = OUTPUT_PATH) -> Optional[Dict[str, Any]]:
    """加载已有的输出文件（用于 --resume）。"""
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            return data

    except (json.JSONDecodeError, OSError):
        pass

    return None


# --------------------------------------------------------------------------- #
# LLM 调用与 JSON 解析
# --------------------------------------------------------------------------- #

def create_llm_client():
    """创建 OpenAI 兼容 LLM 客户端。

    复用 app/config.py 中的 llm_api_key 和 llm_base_url。
    默认使用智谱通用 OpenAI-compatible API endpoint。

    Raises:
        SystemExit: 如果 API Key 未配置。
    """
    from openai import OpenAI

    settings = get_settings()
    api_key = settings.llm_api_key

    if not api_key:
        print(
            "错误：LLM_API_KEY 未设置。\n"
            "请在项目根目录 .env 文件中配置：\n"
            "  LLM_API_KEY=your_api_key\n"
            "或通过环境变量设置。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 推荐使用智谱通用端点，而不是 coding 专用端点
    base_url = (
        settings.llm_base_url
        or os.getenv("LLM_BASE_URL")
        or "https://open.bigmodel.cn/api/coding/paas/v4"
    )

    return OpenAI(api_key=api_key, base_url=base_url)


def strip_markdown_fence(text: str) -> str:
    """去掉 ```json ... ``` 代码块外壳。"""
    text = text.strip()

    fence_match = re.match(
        r"^```(?:json|JSON)?\s*(.*?)\s*```$",
        text,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    return text


def try_load_json_object(text: str) -> Optional[Dict[str, Any]]:
    """尝试把字符串解析为 JSON object，支持去掉尾逗号。"""
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # 容错：去掉 JSON object / array 末尾多余逗号
    repaired = re.sub(r",\s*([}\]])", r"\1", text)

    if repaired != text:
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """从 LLM 返回内容中解析 JSON。

    支持以下格式：
    - 纯 JSON：{"key": "value"}
    - Markdown 代码块：```json\\n{...}\\n```
    - JSON 嵌在自然语言中

    相比简单正则 r"\\{.*\\}"，这里用 JSONDecoder.raw_decode
    从每一个 "{" 开始尝试解析，能避免贪婪匹配误伤。
    """
    if not content or not content.strip():
        return None

    text = strip_markdown_fence(content.strip())

    # 1. 先尝试整体解析
    direct = try_load_json_object(text)
    if direct is not None:
        return direct

    # 2. 再从每个 "{" 起点尝试解析第一个合法 JSON object
    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch != "{":
            continue

        candidate = text[i:]

        try:
            obj, _ = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            # 再尝试修复尾逗号
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                obj, _ = decoder.raw_decode(repaired)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    return None


def normalize_doc_names(raw_docs: Any, kb_docs: List[str]) -> List[str]:
    """规范化 relevant_docs，只保留知识库中实际存在的文档名。"""
    if not isinstance(raw_docs, list):
        return []

    kb_set = set(kb_docs)
    kb_lower_map = {doc.lower(): doc for doc in kb_docs}

    normalized: List[str] = []

    for item in raw_docs:
        doc = str(item).strip()
        if not doc:
            continue

        # 只取文件名，防止模型返回 examples/sample_docs/xxx.md
        doc_name = Path(doc).name

        matched = None
        if doc_name in kb_set:
            matched = doc_name
        elif doc_name.lower() in kb_lower_map:
            matched = kb_lower_map[doc_name.lower()]

        if matched and matched not in normalized:
            normalized.append(matched)

    return normalized


def normalize_string_list(value: Any, max_items: int = 5) -> List[str]:
    """规范化字符串数组。"""
    if not isinstance(value, list):
        return []

    result: List[str] = []

    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)

    return result[:max_items]


def normalize_and_validate_labels(
    labels: Dict[str, Any],
    kb_docs: List[str],
) -> Dict[str, Any]:
    """规范化并校验 LLM 生成的标签。

    目标：
    - 避免 JSON 虽然能解析，但字段类型不对
    - 避免 question_type / expected_tool 越界
    - 避免 relevant_docs 编造不存在的文件
    """
    result: Dict[str, Any] = {}

    should_refuse = bool(labels.get("should_refuse", False))
    result["should_refuse"] = should_refuse

    result["required_keywords"] = normalize_string_list(
        labels.get("required_keywords", []),
        max_items=5,
    )

    result["relevant_docs"] = normalize_doc_names(
        labels.get("relevant_docs", []),
        kb_docs=kb_docs,
    )

    question_type = str(labels.get("question_type", "basic")).strip().lower()
    if question_type not in VALID_QUESTION_TYPES:
        question_type = "basic"

    if should_refuse:
        question_type = "out_of_scope"

    result["question_type"] = question_type

    try:
        min_answer_length = int(labels.get("min_answer_length", 50))
    except (TypeError, ValueError):
        min_answer_length = 50

    result["min_answer_length"] = max(20, min(min_answer_length, 500))

    expected_tool = str(
        labels.get("expected_tool", "knowledge_base_search")
    ).strip()

    if expected_tool not in VALID_TOOLS:
        expected_tool = "knowledge_base_search"

    result["expected_tool"] = expected_tool

    result["notes"] = str(labels.get("notes", "")).strip()

    return result


def build_completion_kwargs(user_prompt: str) -> Dict[str, Any]:
    """构造 LLM 请求参数。

    默认启用 JSON mode，并关闭 thinking。
    """
    return {
        "model": LABEL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": LABEL_LLM_TEMPERATURE,
        "max_tokens": LABEL_LLM_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "thinking": {"type": "disabled"},
            "do_sample": False,
        },
    }


def call_llm_with_compat_fallback(client, user_prompt: str):
    """调用 LLM。

    第一优先级：
        JSON mode + thinking disabled + do_sample=false

    兼容兜底：
        部分 OpenAI-compatible 服务不支持 extra_body 或 response_format。
        如果严格参数报错，则逐级降级。
    """
    strict_kwargs = build_completion_kwargs(user_prompt)

    try:
        return client.chat.completions.create(**strict_kwargs)
    except Exception as strict_error:
        msg = str(strict_error).lower()

        maybe_param_issue = any(
            key in msg
            for key in [
                "response_format",
                "json_object",
                "thinking",
                "do_sample",
                "extra_body",
                "unsupported",
                "invalid parameter",
                "unrecognized",
            ]
        )

        if not maybe_param_issue:
            raise

        # 降级 1：保留 response_format，去掉 extra_body
        kwargs_without_extra = dict(strict_kwargs)
        kwargs_without_extra.pop("extra_body", None)

        try:
            return client.chat.completions.create(**kwargs_without_extra)
        except Exception as second_error:
            msg2 = str(second_error).lower()

            maybe_response_format_issue = any(
                key in msg2
                for key in [
                    "response_format",
                    "json_object",
                    "unsupported",
                    "invalid parameter",
                    "unrecognized",
                ]
            )

            if not maybe_response_format_issue:
                raise

            # 降级 2：完全靠 prompt 约束
            plain_kwargs = dict(kwargs_without_extra)
            plain_kwargs.pop("response_format", None)

            return client.chat.completions.create(**plain_kwargs)


def generate_labels_for_question(
    client,
    question: str,
    kb_docs: List[str],
    debug_raw: bool = False,
) -> Optional[Dict[str, Any]]:
    """对单个问题调用 LLM 生成评估标签。

    包含重试机制：
    - LLM 调用失败时重试
    - LLM 返回格式错误时重试
    - 最多尝试 MAX_RETRIES 次
    """
    user_prompt = LABEL_USER_TEMPLATE.format(
        question=question,
        knowledge_base_docs="\n".join(f"  - {d}" for d in kb_docs),
        available_tools="\n".join(f"  - {t}" for t in AVAILABLE_TOOLS),
    )

    last_content = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = call_llm_with_compat_fallback(client, user_prompt)

            content = response.choices[0].message.content or ""
            last_content = content

            if debug_raw:
                print("    ---- RAW LLM RESPONSE START ----")
                print(repr(content[:4000]))
                print("    ---- RAW LLM RESPONSE END ----")

            labels = parse_json_response(content)

            if labels is not None:
                return normalize_and_validate_labels(labels, kb_docs=kb_docs)

            if attempt < MAX_RETRIES:
                print(f"    ⚠ 第 {attempt} 次 JSON 解析失败，重试中...")
                time.sleep(RETRY_DELAY)

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"    ⚠ 第 {attempt} 次 LLM 调用失败: {e}，重试中...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ✗ LLM 调用失败（已重试 {MAX_RETRIES} 次）: {e}")
                return None

    print(f"    ✗ JSON 解析失败（已重试 {MAX_RETRIES} 次），跳过此问题")

    if not debug_raw and last_content:
        print("    提示：可加 --debug-raw 查看模型原始返回内容")
        print(f"    原始返回片段: {repr(last_content[:500])}")

    return None


# --------------------------------------------------------------------------- #
# 交互式人工校验
# --------------------------------------------------------------------------- #

def print_labels(question: Dict[str, Any], labels: Dict[str, Any]) -> None:
    """格式化打印问题和生成的标签。"""
    print(f"\n{'=' * 60}")
    print(f"  ID: {question.get('id', '?')}")
    print(f"  类别: {question.get('category', '?')}")
    print(f"  问题: {question['question']}")
    print(f"{'─' * 60}")
    print("  生成的评估标签：")
    print(f"    should_refuse:     {labels.get('should_refuse', False)}")
    print(f"    question_type:     {labels.get('question_type', '?')}")
    print(f"    expected_tool:     {labels.get('expected_tool', '?')}")
    print(f"    required_keywords: {labels.get('required_keywords', [])}")
    print(f"    relevant_docs:     {labels.get('relevant_docs', [])}")
    print(f"    min_answer_length: {labels.get('min_answer_length', '?')}")
    print(f"    notes:             {labels.get('notes', '')}")
    print(f"{'=' * 60}")


def edit_labels(labels: Dict[str, Any]) -> Dict[str, Any]:
    """交互式编辑标签。

    逐个字段提示修改，直接回车保留原值。
    """
    result = dict(labels)
    print("\n  进入编辑模式（直接回车保留原值）：\n")

    # should_refuse
    current = result.get("should_refuse", False)
    val = input(f"  should_refuse [{current}]: ").strip().lower()
    if val:
        result["should_refuse"] = val in ("true", "yes", "1", "y", "是")

    # required_keywords
    current = result.get("required_keywords", [])
    current_str = ", ".join(current) if isinstance(current, list) else str(current)
    val = input(f"  required_keywords [{current_str}]: ").strip()
    if val:
        result["required_keywords"] = [
            k.strip() for k in val.split(",") if k.strip()
        ]

    # relevant_docs
    current = result.get("relevant_docs", [])
    current_str = ", ".join(current) if isinstance(current, list) else str(current)
    val = input(f"  relevant_docs [{current_str}]: ").strip()
    if val:
        result["relevant_docs"] = [
            d.strip() for d in val.split(",") if d.strip()
        ]

    # question_type
    current = result.get("question_type", "")
    val = input(
        f"  question_type [{current}] "
        f"(basic/structured/calculation/comparison/out_of_scope): "
    ).strip().lower()
    if val:
        if val in VALID_QUESTION_TYPES:
            result["question_type"] = val
        else:
            print(f"    ⚠ 无效 question_type，保留原值 {current}")

    # expected_tool
    current = result.get("expected_tool", "")
    val = input(f"  expected_tool [{current}]: ").strip()
    if val:
        if val in VALID_TOOLS:
            result["expected_tool"] = val
        else:
            print(f"    ⚠ 无效 expected_tool，保留原值 {current}")

    # min_answer_length
    current = result.get("min_answer_length", 50)
    val = input(f"  min_answer_length [{current}]: ").strip()
    if val:
        try:
            result["min_answer_length"] = int(val)
        except ValueError:
            print(f"    ⚠ 无效数字，保留原值 {current}")

    # notes
    current = result.get("notes", "")
    val = input(f"  notes [{current}]: ").strip()
    if val:
        result["notes"] = val

    print("  ✓ 编辑完成")
    return result


def human_review(
    question: Dict[str, Any],
    labels: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """交互式校验单个问题的标签。

    提供选项：[y]接受 [e]编辑 [s]跳过 [q]保存退出

    Returns:
        - 校验后的标签字典（用户接受或编辑后）
        - None 表示用户选择保存退出
    """
    while True:
        print_labels(question, labels)

        try:
            choice = input(
                "\n  [y]接受  [e]编辑  [s]跳过  [q]保存退出: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  检测到中断信号，保存并退出...")
            return None

        if choice in ("y", ""):
            return labels
        if choice == "e":
            labels = edit_labels(labels)
        elif choice == "s":
            return {**labels, "_skipped": True}
        elif choice == "q":
            return None
        else:
            print("  ⚠ 无效选项，请输入 y/e/s/q")


# --------------------------------------------------------------------------- #
# 输出保存
# --------------------------------------------------------------------------- #

def save_output(
    questions: List[Dict[str, Any]],
    path: Path = OUTPUT_PATH,
) -> Path:
    """保存标注结果到 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generator": "generate_eval_labels.py",
        "llm_model": LABEL_LLM_MODEL,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_schema": {
            "should_refuse": "bool",
            "required_keywords": "list[str]",
            "relevant_docs": "list[str]",
            "question_type": sorted(VALID_QUESTION_TYPES),
            "min_answer_length": "int",
            "expected_tool": AVAILABLE_TOOLS,
            "notes": "str",
        },
        "questions": questions,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


# --------------------------------------------------------------------------- #
# 回退标签
# --------------------------------------------------------------------------- #

def get_first_expected_tool(question: Dict[str, Any]) -> str:
    """从原始题集 expected_tools 中取第一个合法工具名。"""
    expected_tools = question.get("expected_tools", [])

    if isinstance(expected_tools, list) and expected_tools:
        tool = str(expected_tools[0]).strip()
        if tool in VALID_TOOLS:
            return tool

    if isinstance(expected_tools, str):
        tool = expected_tools.strip()
        if tool in VALID_TOOLS:
            return tool

    return "knowledge_base_search"


def fallback_labels_from_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 失败时，从原始题集字段生成回退标签。"""
    category = str(question.get("category", "basic")).strip().lower()

    if category not in VALID_QUESTION_TYPES:
        category = "basic"

    should_refuse = bool(question.get("should_refuse", False))

    if should_refuse:
        category = "out_of_scope"

    return {
        "should_refuse": should_refuse,
        "required_keywords": normalize_string_list(
            question.get("expected_keywords", []),
            max_items=5,
        ),
        "relevant_docs": normalize_string_list(
            question.get("expected_source_files", []),
            max_items=10,
        ),
        "question_type": category,
        "min_answer_length": 50,
        "expected_tool": get_first_expected_tool(question),
        "notes": "LLM 生成失败，使用题集原始标签回退",
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> None:
    """主入口：解析参数，生成标签，人工校验，保存结果。"""
    parser = argparse.ArgumentParser(
        description="LLM 辅助评估标签生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例：\n"
            "  python eval/generate_eval_labels.py --auto-accept\n"
            "  python eval/generate_eval_labels.py\n"
            "  python eval/generate_eval_labels.py --resume\n"
            "  python eval/generate_eval_labels.py --auto-accept --debug-raw\n"
        ),
    )

    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="自动接受所有 LLM 生成的标签，跳过人工校验",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续（跳过已完成且已校验的问题）",
    )

    parser.add_argument(
        "--kb-dir",
        type=str,
        default=str(DEFAULT_KB_DIR),
        help=f"知识库文档目录（默认：{DEFAULT_KB_DIR}）",
    )

    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="打印 LLM 原始返回内容，便于排查 JSON 解析失败",
    )

    args = parser.parse_args()

    # ---- Step 1: 加载题集 ----
    try:
        questions = load_questions()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    print(f"已加载 {len(questions)} 道评估题目")

    # ---- Step 2: 扫描知识库文档 ----
    kb_dir = Path(args.kb_dir)

    try:
        kb_docs = get_knowledge_base_docs(kb_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    print(f"知识库文档：{len(kb_docs)} 个 .md 文件")
    for doc in kb_docs:
        print(f"  - {doc}")

    # ---- Step 3: 创建 LLM 客户端 ----
    client = create_llm_client()

    settings = get_settings()
    effective_base_url = (
        settings.llm_base_url
        or os.getenv("LLM_BASE_URL")
        or "https://open.bigmodel.cn/api/coding/paas/v4"
    )

    print(f"\nLLM 模型：{LABEL_LLM_MODEL}")
    print(f"LLM Base URL：{effective_base_url}")
    print(f"人工校验：{'跳过 (--auto-accept)' if args.auto_accept else '启用'}")
    print(f"断点续传：{'启用 (--resume)' if args.resume else '禁用'}")
    print(f"调试原文：{'启用 (--debug-raw)' if args.debug_raw else '禁用'}")

    # ---- Step 4: 加载已有结果（resume 模式）----
    existing_data: Optional[Dict[str, Any]] = None

    if args.resume:
        existing_data = load_existing_output()

        if existing_data:
            existing_qs = existing_data.get("questions", [])
            print(f"已加载上次结果：{len(existing_qs)} 条")
        else:
            print("未找到上次的结果文件，从头开始")

    # 构建已有标签索引（按 question 文本匹配）
    existing_map: Dict[str, Dict[str, Any]] = {}

    if existing_data:
        for existing_q in existing_data.get("questions", []):
            q_text = existing_q.get("question", "")
            if q_text and existing_q.get("human_verified"):
                existing_map[q_text] = existing_q

    # ---- Step 5: 逐题生成标签 ----
    output_questions: List[Dict[str, Any]] = []

    save_and_exit = False
    processed = 0
    skipped = 0
    reused = 0
    failed_llm = 0

    for i, q in enumerate(questions):
        q_text = q["question"]
        q_id = q.get("id", str(i + 1))
        total = len(questions)

        # resume 模式：跳过已校验的问题
        if q_text in existing_map:
            output_questions.append(existing_map[q_text])
            reused += 1
            print(f"  [{i + 1}/{total}] {q_id} ✓ 已校验（跳过）")
            continue

        print(f"\n处理 [{i + 1}/{total}] {q_id}：{q_text[:60]}...")

        # 调用 LLM 生成标签
        labels = generate_labels_for_question(
            client=client,
            question=q_text,
            kb_docs=kb_docs,
            debug_raw=args.debug_raw,
        )

        if labels is None:
            print("    ⚠ LLM 生成失败，使用题集原始标签回退")
            labels = fallback_labels_from_question(q)
            skipped += 1
            failed_llm += 1

        # 人工校验
        if args.auto_accept:
            reviewed_labels = labels
            verified = True
        else:
            reviewed = human_review(q, labels)

            if reviewed is None:
                # 用户选择保存退出
                reviewed_labels = labels
                verified = False
                save_and_exit = True
            elif reviewed.get("_skipped"):
                reviewed_labels = {
                    k: v for k, v in reviewed.items()
                    if k != "_skipped"
                }
                verified = False
                skipped += 1
            else:
                reviewed_labels = reviewed
                verified = True

        # 构建输出条目（保留原始字段 + 添加标签）
        output_entry: Dict[str, Any] = {
            "id": q_id,
            "category": q.get("category", ""),
            "question": q_text,
            "eval_labels": reviewed_labels,
            "human_verified": verified,
        }

        # 保留原始题集中的期望标签（用于对比）
        if q.get("expected_keywords"):
            output_entry["original_expected_keywords"] = q["expected_keywords"]

        if q.get("expected_tools"):
            output_entry["original_expected_tools"] = q["expected_tools"]

        if q.get("unexpected_tools"):
            output_entry["original_unexpected_tools"] = q["unexpected_tools"]

        if q.get("expected_source_files"):
            output_entry["original_expected_source_files"] = q[
                "expected_source_files"
            ]

        output_questions.append(output_entry)
        processed += 1

        if save_and_exit:
            break

    # ---- Step 6: 保存结果 ----
    # resume 模式下合并已有结果中未处理的条目
    if existing_data and not save_and_exit:
        existing_qs = existing_data.get("questions", [])
        existing_question_texts = {q.get("question") for q in output_questions}

        for existing_q in existing_qs:
            if existing_q.get("question") not in existing_question_texts:
                output_questions.append(existing_q)

    output_path = save_output(output_questions)

    print(f"\n{'=' * 60}")
    print("  标注完成！")
    print(f"  新处理: {processed}  |  复用: {reused}  |  跳过: {skipped}")
    print(f"  LLM 失败回退: {failed_llm}")
    print(f"  总计: {len(output_questions)} 条")
    print(f"  保存至: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
