"""Core agent with perception-decision-action-reflection loop.

The agent is the application's central entry point.
All messages flow through the agent, which orchestrates LLM, tools, memory, and hooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from core.hooks import HookRegistry
from core.tools import ToolRegistry, set_context
from models import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)

# Safety cap even when max_rounds=0 (unlimited), prevents API budget exhaustion
HARD_MAX_ROUNDS = 200

COMPRESS_DECISION_PROMPT = """\
Based on the conversation below, answer in JSON:
{{"topic_changed": true/false, "current_topic": "brief topic",
"should_compress": true/false, "reason": "brief reason"}}

Rules:
- topic_changed: true if the user shifted to a substantially different subject
- current_topic: concise 3-10 word description of what the conversation is currently focused on
- should_compress: true if older messages are no longer relevant to the current topic and can be summarized
- Only compress when the conversation is long enough to benefit from it

Conversation (last 3 turns):
{recent_turns}

Total messages in history: {total_messages}
Estimated tokens: {estimated_tokens}"""


@dataclass
class TurnResult:
    """Result of a single agent turn, including metadata decisions."""

    response: str
    topic_changed: bool = False
    current_topic: str = ""
    should_compress: bool = False
    compress_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class Agent:
    """Agentic loop: the central entry point for all interactions.

    Handles message -> LLM -> tool calls -> loop until done.
    Supports: unlimited rounds, fallback LLMs, tool authorization hooks,
    cancellation via new messages, and smart end-of-turn compression decisions.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str = "You are Memoo, a helpful AI assistant.",
        max_rounds: int = 0,
        fallback_llms: list[LLMProvider] | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_rounds = max_rounds  # 0 = unlimited
        self._fallback_llms = fallback_llms or []
        self._hooks = hooks or HookRegistry()
        self._cancel_events: dict[str, asyncio.Event] = {}

    def cancel(self, run_id: str | None = None) -> None:
        """Cancel an agent run by *run_id* (e.g. chat_id).

        If *run_id* is ``None``, cancel **all** active runs.
        """
        if run_id is not None:
            ev = self._cancel_events.get(run_id)
            if ev:
                ev.set()
                logger.info("Agent run cancelled (run_id=%s)", run_id)
        else:
            for rid, ev in self._cancel_events.items():
                ev.set()
            if self._cancel_events:
                logger.info("All agent runs cancelled (%d)", len(self._cancel_events))

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Run the agent loop for a user message.

        Returns TurnResult with response text and compression decision.
        """
        ctx = context or {}
        run_id = str(ctx.get("chat_id", id(asyncio.current_task())))
        cancel_event = asyncio.Event()
        self._cancel_events[run_id] = cancel_event

        try:
            return await self._run_loop(user_message, ctx, cancel_event, history)
        finally:
            self._cancel_events.pop(run_id, None)

    async def _run_loop(
        self,
        user_message: str,
        ctx: dict[str, Any],
        cancel_event: asyncio.Event,
        history: list[Message] | None,
    ) -> TurnResult:
        """Inner agent loop, isolated with its own *cancel_event*."""
        messages = list(history) if history else []
        messages.append(Message(role="user", content=user_message))

        tool_schemas = self._tools.get_schemas() or None
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        set_context(ctx)
        round_num = 0

        while True:
            round_num += 1
            effective_max = self._max_rounds if self._max_rounds > 0 else HARD_MAX_ROUNDS
            if round_num > effective_max:
                break
            if cancel_event.is_set():
                return TurnResult(response="(cancelled by user)", usage=total_usage)

            logger.info("Agent round %d", round_num)
            response = await self._chat_with_fallback(messages, tool_schemas)

            for k, v in response.usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

            if not response.has_tool_calls:
                # End of turn — decide on compression
                response_text = response.text or ""
                decision = await self._end_of_turn_decision(messages, total_usage)
                logger.info(
                    "Agent done in %d rounds. topic_changed=%s, compress=%s",
                    round_num,
                    decision.get("topic_changed"),
                    decision.get("should_compress"),
                )
                return TurnResult(
                    response=response_text,
                    topic_changed=decision.get("topic_changed", False),
                    current_topic=decision.get("current_topic", ""),
                    should_compress=decision.get("should_compress", False),
                    compress_reason=decision.get("reason", ""),
                    usage=total_usage,
                )

            # Record assistant turn with tool calls
            messages.append(Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls))

            # Execute tool calls with authorization hooks
            for tc in response.tool_calls:
                if cancel_event.is_set():
                    return TurnResult(response="(cancelled during tool execution)", usage=total_usage)

                # Authorization check
                approved, reason = await self._hooks.authorize(tc.name, tc.arguments, ctx)
                if not approved:
                    logger.warning("Tool %s denied: %s", tc.name, reason)
                    messages.append(
                        Message(
                            role="tool_result",
                            content=f"Authorization denied: {reason}",
                            tool_call_id=tc.id,
                        )
                    )
                    continue

                logger.info("Tool call: %s(id=%s)", tc.name, tc.id)
                result = await self._tools.execute(tc.name, tc.arguments)
                messages.append(Message(role="tool_result", content=result, tool_call_id=tc.id))

        # Max rounds exceeded
        logger.warning("Max rounds (%d) exceeded, forcing final answer", self._max_rounds)
        response = await self._chat_with_fallback(messages, tools=None)
        return TurnResult(response=response.text or "(max rounds reached)", usage=total_usage)

    async def _end_of_turn_decision(self, messages: list[Message], usage: dict[str, int]) -> dict[str, Any]:
        """Ask LLM whether to compress memory at end of turn.

        Uses a lightweight call to decide if the topic changed and if old context
        should be compressed.
        """
        # Only worth checking if there's enough history
        total_messages = len(messages)
        estimated_tokens = sum(len(m.content) // 3 for m in messages if m.content)

        if total_messages < 10 or estimated_tokens < 5000:
            return {"topic_changed": False, "should_compress": False, "reason": "too short"}

        # Get last 3 turns for context
        recent = messages[-6:] if len(messages) >= 6 else messages
        recent_text = "\n".join(f"[{m.role}]: {m.content[:200]}" for m in recent if m.content)

        prompt = COMPRESS_DECISION_PROMPT.format(
            recent_turns=recent_text,
            total_messages=total_messages,
            estimated_tokens=estimated_tokens,
        )

        try:
            resp = await self._llm.chat(
                messages=[Message(role="user", content=prompt)],
                system="You are a conversation analyzer. Respond only with valid JSON.",
                max_tokens=100,
            )
            if resp.text:
                # Parse JSON from response
                text = resp.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                return json.loads(text)
        except Exception:
            logger.debug("Compression decision call failed, skipping", exc_info=True)

        return {"topic_changed": False, "should_compress": False, "reason": "decision_error"}

    async def _chat_with_fallback(
        self,
        messages: list[Message],
        tools: list[dict] | None,  # type: ignore[type-arg]
    ) -> LLMResponse:
        """Try primary LLM, fall back on failure."""
        providers = [self._llm, *self._fallback_llms]
        last_error: Exception | None = None

        for i, llm in enumerate(providers):
            try:
                return await llm.chat(
                    messages=messages,
                    system=self._system_prompt,
                    tools=tools,
                )
            except Exception as e:
                last_error = e
                if i < len(providers) - 1:
                    logger.warning("LLM %s failed (%s), trying fallback", llm.model_name, e)
                else:
                    logger.error("All LLM providers failed")

        raise last_error or RuntimeError("No LLM providers available")
