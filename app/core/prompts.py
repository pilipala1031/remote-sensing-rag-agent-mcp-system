"""RAG Prompt 模板。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请只基于给定的上下文回答用户问题。

要求：
1. 如果上下文中没有足够信息，请回答："根据当前知识库内容，无法确定该问题的答案。"
2. 不要编造上下文中不存在的信息。
3. 回答要结构清晰，必要时分点说明。
4. 回答末尾列出参考来源，格式为：[来源：文件名，第X页，chunk_id]。

上下文：
{context}

用户问题：
{question}

请给出答案："""


def build_rag_prompt() -> ChatPromptTemplate:
    """构建 RAG ChatPromptTemplate。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
        ]
    )


# 拒答文案，避免在 LLM 调用失败时仍返回
REFUSAL_ANSWER = "根据当前知识库内容，无法确定该问题的答案。"
