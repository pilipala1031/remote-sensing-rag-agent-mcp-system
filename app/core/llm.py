"""OpenAI-compatible Chat LLM Client。

使用 openai SDK，可对接智谱 GLM、DeepSeek、Qwen 等任意 OpenAI 兼容接口。
同时实现 LangChain BaseChatModel 标准接口，方便后续替换 / 扩展 Agent。
"""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import OpenAI

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _lc_to_openai_messages(messages: List[BaseMessage]) -> List[dict]:
    out: List[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


class OpenAICompatibleLLMClient(BaseChatModel):
    """OpenAI 兼容 Chat Completion 客户端。"""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: Optional[int] = 2048
    client: Any = None

    def __init__(self, **kwargs: Any) -> None:
        settings = get_settings()
        kwargs.setdefault("api_key", settings.llm_api_key)
        kwargs.setdefault("base_url", settings.llm_base_url)
        kwargs.setdefault("model", settings.llm_model)
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置，请检查 .env")
        if not self.base_url:
            raise ValueError("LLM_BASE_URL 未配置，请检查 .env")
        if not self.model:
            raise ValueError("LLM_MODEL 未配置，请检查 .env")
        # 显式构建 OpenAI client
        object.__setattr__(
            self, "client", OpenAI(api_key=self.api_key, base_url=self.base_url)
        )

    # ------- 直接调用便捷方法 -------
    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            logger.error("LLM 调用失败: %s", e)
            raise RuntimeError(f"LLM 调用失败: {e}") from e

    # ------- LangChain BaseChatModel 接口 -------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=_lc_to_openai_messages(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            content = resp.choices[0].message.content or ""
            msg = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=msg)])
        except Exception as e:  # noqa: BLE001
            logger.error("LLM 调用失败: %s", e)
            raise RuntimeError(f"LLM 调用失败: {e}") from e

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "base_url": self.base_url}
