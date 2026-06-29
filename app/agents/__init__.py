"""Agent 模块：基于 LangChain 1.0 create_agent 的 RAG-as-Tool Agent。

设计原则：
- 不修改现有 RAG 主流程（RAGService / Retriever / VectorStore）
- Agent 通过 Tool 复用现有检索能力
- 不直接使用 LangGraph StateGraph，仅通过 create_agent 间接使用
"""
