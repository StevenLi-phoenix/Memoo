"""OpenAI API provider with function calling and model discovery."""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

from models import LLMResponse, Message, ModelInfo, ToolCall

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "") -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        if model:
            logger.info("OpenAI provider: model=%s", model)

    @property
    def model_name(self) -> str:
        return self._model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model = value

    async def discover_models(self) -> list[ModelInfo]:
        """List available models from OpenAI API."""
        models: list[ModelInfo] = []
        try:
            page = await self._client.models.list()
            for m in page.data:
                models.append(
                    ModelInfo(
                        id=m.id,
                        provider="openai",
                        display_name=m.id,
                        created=m.created,
                    )
                )
            logger.info("OpenAI: discovered %d models", len(models))
        except Exception:
            logger.exception("OpenAI model discovery failed")
        return models

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        api_messages = self._build_messages(messages, system)
        oai_tools = self._convert_tools(tools) if tools else None

        kwargs: dict[str, Any] = {"model": self._model, "messages": api_messages, "max_tokens": max_tokens}
        if oai_tools:
            kwargs["tools"] = oai_tools

        response = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def _build_messages(self, messages: list[Message], system: str | None) -> list[dict[str, Any]]:
        api_msgs: list[dict[str, Any]] = []
        if system:
            api_msgs.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool_result":
                api_msgs.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
            elif msg.role == "assistant" and msg.tool_calls:
                oai_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    oai_msg["content"] = msg.content
                oai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ]
                api_msgs.append(oai_msg)
            else:
                api_msgs.append({"role": msg.role, "content": msg.content})
        return api_msgs

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    def _parse_response(self, response: openai.types.chat.ChatCompletion) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
                )

        usage_dict: dict[str, int] = {}
        if response.usage:
            usage_dict = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            usage=usage_dict,
        )
