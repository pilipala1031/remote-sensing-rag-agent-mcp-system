"""集中读取 .env 配置，所有模块通过 Settings 单例访问。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Embedding (SiliconFlow) ----
    siliconflow_api_key: str = Field(default="", alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1", alias="SILICONFLOW_BASE_URL"
    )
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")

    # ---- LLM (OpenAI-compatible) ----
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_model: str = Field(default="", alias="LLM_MODEL")

    # ---- 存储与切分 ----
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    raw_data_dir: str = Field(default="./data/raw", alias="RAW_DATA_DIR")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    # Work Unit JSON 文件持久化目录（手动沉淀，RAG/Agent 查询不会自动写入）
    work_unit_dir: str = Field(default="./data/work_units", alias="WORK_UNIT_DIR")

    # ---- 检索 ----
    top_k: int = Field(default=5, alias="TOP_K")
    similarity_threshold: float = Field(default=0.3, alias="SIMILARITY_THRESHOLD")

    # ---- Agent Evidence Verification ----
    enable_agent_verification: bool = Field(
        default=True, alias="ENABLE_AGENT_VERIFICATION"
    )

    # ---- Agent Verification 模式与轻量化 ----
    # off: 不执行证据校验
    # sync: 在 /api/agent/query 中同步执行证据校验
    # deferred: /api/agent/query 先返回回答，前端再调用 /api/agent/verify
    agent_verification_mode: str = Field(
        default="deferred", alias="AGENT_VERIFICATION_MODE"
    )

    # lightweight: 裁剪 answer/sources/tool_calls 后校验
    # full: 相对完整校验，但也做基本截断
    agent_verification_level: str = Field(
        default="lightweight", alias="AGENT_VERIFICATION_LEVEL"
    )

    # Verification 独立模型配置（留空则复用主 LLM_MODEL）
    verification_model: str = Field(default="", alias="VERIFICATION_MODEL")
    verification_max_tokens: int = Field(
        default=512, alias="VERIFICATION_MAX_TOKENS"
    )
    verification_temperature: float = Field(
        default=0.0, alias="VERIFICATION_TEMPERATURE"
    )

    # ---- Rerank（Cross-encoder 重排序） ----
    # 是否启用 rerank（默认关闭，开启后复用 SILICONFLOW_API_KEY 调用 rerank API）
    use_rerank: bool = Field(default=False, alias="USE_RERANK")
    # rerank 时向量检索的候选数量（先检索 candidate_k 条，rerank 后取 top_k 条）
    rerank_candidate_k: int = Field(default=10, alias="RERANK_CANDIDATE_K")
    # rerank 模型名称
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL"
    )

    # ---- Agent 回答长度控制 ----
    # Agent ChatOpenAI max_tokens，控制最终回答长度；0 或负数表示不传
    agent_max_tokens: int = Field(default=1000, alias="AGENT_MAX_TOKENS")

    # ---- Agent LLM 响应缓存 ----
    # 是否为 Agent 路径启用 LangChain LLM Cache（InMemoryCache）。
    # 仅对 Agent 路径的 ChatOpenAI 生效；RAG 路径直接调 openai SDK，不受影响。
    # 前端可通过请求体的 enable_cache 字段逐请求覆盖。
    enable_agent_cache: bool = Field(default=False, alias="ENABLE_AGENT_CACHE")

    # ---- Agent 响应级缓存（Response Cache） ----
    # 在 AgentService.query 层缓存完整 Agent 响应，
    # 相同问题+相同配置第二次直接返回缓存结果，零 LLM / 工具调用。
    # 文档入库 / 删除时自动清空缓存，避免返回过期回答。
    enable_agent_response_cache: bool = Field(
        default=True, alias="ENABLE_AGENT_RESPONSE_CACHE"
    )
    # 缓存条目 TTL（秒），超时自动失效。0 表示不过期（仅 max_size 淘汰）
    agent_response_cache_ttl_seconds: int = Field(
        default=600, alias="AGENT_RESPONSE_CACHE_TTL_SECONDS"
    )
    # 缓存最大条目数，超出后淘汰最旧条目
    agent_response_cache_max_size: int = Field(
        default=100, alias="AGENT_RESPONSE_CACHE_MAX_SIZE"
    )

    @property
    def raw_data_path(self) -> Path:
        p = Path(self.raw_data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def work_unit_path(self) -> Path:
        p = Path(self.work_unit_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
