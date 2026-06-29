"""SiliconFlow BAAI/bge-m3 Embedding Client。

通过 OpenAI SDK 兼容方式调用 SiliconFlow embeddings 接口，
并对网络/参数异常进行统一处理。
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SiliconFlowEmbeddingClient(Embeddings):
    """封装 SiliconFlow bge-m3 embedding 调用，可作为 LangChain Embeddings 使用。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.siliconflow_api_key
        self.base_url = base_url or settings.siliconflow_base_url
        self.model = model or settings.embedding_model

        if not self.api_key:
            raise ValueError(
                "SiliconFlow API Key 未配置，请在 .env 中设置 SILICONFLOW_API_KEY"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # 向量维度 bge-m3 = 1024
        self._dim: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            vec = self.embed_query("dimension_probe")
            self._dim = len(vec)
        return self._dim  # type: ignore[return-value]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            logger.error("SiliconFlow embedding 调用失败: %s", e)
            raise RuntimeError(f"Embedding 调用失败: {e}") from e

    # ---------- LangChain Embeddings 接口 ----------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 分批，单次最多 64 条避免超时
        batch_size = 32
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[i : i + batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
