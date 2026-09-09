"""Thin LLM provider adapter — swaps OpenAI / Anthropic without touching agent logic."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()


def get_llm() -> BaseChatModel:
    """Return the configured LLM as a LangChain chat model."""
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key or None,
            temperature=0.1,
            max_tokens=1024,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}. Use 'openai' or 'anthropic'.")


async def call_llm(
    messages: list[BaseMessage],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """Send a list of messages to the configured LLM and return the text response."""
    llm = get_llm()
    # Re-apply overrides when the caller wants non-default values
    if temperature != 0.1 or max_tokens != 1024:
        llm = llm.with_config(temperature=temperature, max_tokens=max_tokens)

    response = await llm.ainvoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)
