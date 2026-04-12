"""Anthropic Claude API provider with tool_use, prompt caching, native web search, and model discovery."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from models import LLMResponse, Message, ModelInfo, ToolCall

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Claude API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "",
        web_search: bool = True,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._web_search = web_search
        if model:
            logger.info("Anthropic provider: model=%s", model)

    @property
    def model_name(self) -> str:
        return self._model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model = value

    async def discover_models(self) -> list[ModelInfo]:
        """List available models from Anthropic API."""
        models: list[ModelInfo] = []
        try:
            page = await self._client.models.list(limit=100)
            for m in page.data:
                models.append(
                    ModelInfo(
                        id=m.id,
                        provider="anthropic",
                        display_name=getattr(m, "display_name", m.id),
                        created=self._parse_timestamp(getattr(m, "created_at", None)),
                    )
                )
            logger.info("Anthropic: discovered %d models", len(models))
        except Exception:
            logger.exception("Anthropic model discovery failed")
        return models

    @staticmethod
    def _parse_timestamp(val: Any) -> int:
        if val is None:
            return 0
        if isinstance(val, (int, float)):
            return int(val)
        # datetime object
        if hasattr(val, "timestamp"):
            return int(val.timestamp())
        return 0

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        api_messages = self._build_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }

        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

        all_tools: list[dict[str, Any]] = []
        if tools:
            all_tools.extend(tools)
        if self._web_search:
            all_tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
        if all_tools:
            if all_tools[-1].get("type") != "web_search_20250305":
                all_tools[-1] = {**all_tools[-1], "cache_control": {"type": "ephemeral"}}
            kwargs["tools"] = all_tools

        logger.debug("Anthropic: %d msgs, %d tools", len(api_messages), len(all_tools))
        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    def _build_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        api_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool_result":
                api_msgs.append(
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                api_msgs.append({"role": "assistant", "content": content})
            else:
                api_msgs.append({"role": msg.role, "content": msg.content})
        return api_msgs

    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        usage_dict: dict[str, int] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage_dict["cache_creation_tokens"] = response.usage.cache_creation_input_tokens or 0
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage_dict["cache_read_tokens"] = response.usage.cache_read_input_tokens or 0

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=usage_dict,
        )
