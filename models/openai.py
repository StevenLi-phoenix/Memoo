"""OpenAI API provider with streaming, function calling, and model discovery."""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

from models import LLMResponse, Message, ModelInfo, ToolCall

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI API provider with streaming support."""

    def __init__(self, api_key: str, model: str = "", base_url: str = "") -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model
        if model:
            logger.info("OpenAI provider: model=%s base_url=%s", model, base_url or "default")

    @property
    def model_name(self) -> str:
        return self._model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model = value

    async def discover_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        try:
            page = await self._client.models.list()
            for m in page.data:
                models.append(ModelInfo(id=m.id, provider="openai", display_name=m.id, created=m.created))
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
        output_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        api_messages = self._build_messages(messages, system)
        oai_tools = self._convert_tools(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        # Structured JSON output. Skip when tools are present: llama.cpp/LM Studio
        # backends reject combining tool-call grammar with response_format
        # ("Cannot combine structured output constraints with lazy grammar").
        if output_schema and not oai_tools:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_response", "strict": True, "schema": output_schema},
            }

        logger.debug("OpenAI stream: %d msgs, %d tools", len(api_messages), len(tools or []))
        return await self._stream_response(**kwargs)

    async def _stream_response(self, **kwargs: Any) -> LLMResponse:
        """Streaming API call — accumulates deltas incrementally."""
        text_parts: list[str] = []
        stop_reason = "stop"
        usage_dict: dict[str, int] = {}

        # tool_calls are streamed as indexed deltas
        tool_map: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments_json}

        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if not chunk.choices:
                # Usage-only chunk (stream_options)
                if chunk.usage:
                    usage_dict = {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                stop_reason = choice.finish_reason

            if delta.content:
                text_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_map:
                        tool_map[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments_json": "",
                        }
                    entry = tool_map[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments_json"] += tc_delta.function.arguments

        # Build tool_calls from accumulated deltas
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_map):
            entry = tool_map[idx]
            try:
                arguments = json.loads(entry["arguments_json"]) if entry["arguments_json"] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=entry["id"], name=entry["name"], arguments=arguments))

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage_dict,
        )

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
