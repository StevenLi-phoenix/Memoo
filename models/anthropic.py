"""Anthropic Claude API provider with streaming, tool_use, prompt caching, web search, and model discovery."""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from models import LLMResponse, Message, ModelInfo, ToolCall

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Claude API provider with streaming support."""

    def __init__(
        self,
        api_key: str,
        model: str = "",
        web_search: bool = True,
        advisor_model: str = "",
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._web_search = web_search
        self._advisor_model = advisor_model
        if model:
            logger.info("Anthropic provider: model=%s, advisor=%s", model, advisor_model or "none")

    @property
    def model_name(self) -> str:
        return self._model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model = value

    async def discover_models(self) -> list[ModelInfo]:
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
        # Advisor strategy: consult a more capable model for hard decisions
        if self._advisor_model:
            all_tools.append({"type": "advisor_20260301", "model": self._advisor_model})
            kwargs["betas"] = ["advisor-tool-2026-03-01"]
        if all_tools:
            # Cache control on last non-special tool
            for i in range(len(all_tools) - 1, -1, -1):
                t = all_tools[i]
                if t.get("type") not in ("web_search_20250305", "advisor_20260301"):
                    all_tools[i] = {**t, "cache_control": {"type": "ephemeral"}}
                    break
            kwargs["tools"] = all_tools

        logger.debug(
            "Anthropic stream: %d msgs, %d tools, advisor=%s",
            len(api_messages),
            len(all_tools),
            bool(self._advisor_model),
        )
        return await self._stream_response(**kwargs)

    async def _stream_response(self, **kwargs: Any) -> LLMResponse:
        """Streaming API call — accumulates content blocks incrementally."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage_dict: dict[str, int] = {}
        stop_reason = "end_turn"

        # Track current tool_use block being streamed
        current_tool: dict[str, Any] | None = None
        current_tool_json = ""

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        current_tool_json = ""

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_parts.append(delta.text)
                    elif delta.type == "input_json_delta":
                        current_tool_json += delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            arguments = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        tool_calls.append(
                            ToolCall(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                arguments=arguments,
                            )
                        )
                        current_tool = None
                        current_tool_json = ""

                elif event.type == "message_delta":
                    stop_reason = getattr(event.delta, "stop_reason", stop_reason) or stop_reason

            # Get final message for usage
            final = await stream.get_final_message()
            usage_dict = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
            if hasattr(final.usage, "cache_creation_input_tokens"):
                usage_dict["cache_creation_tokens"] = final.usage.cache_creation_input_tokens or 0
            if hasattr(final.usage, "cache_read_input_tokens"):
                usage_dict["cache_read_tokens"] = final.usage.cache_read_input_tokens or 0

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage_dict,
        )

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
