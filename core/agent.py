"""Core agent with perception-decision-action-reflection loop.

The agent is the application's central entry point.
All messages flow through the agent, which orchestrates LLM, tools, memory, and hooks.

Final output is a structured JSON response (enforced by LLM output_config),
eliminating the need for extra compression-decision calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from core.config import AgentConfig
from core.hooks import HookRegistry
from core.tools import ToolRegistry, set_context
from models import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)

# JSON schema for the agent's final structured response
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "Reply to the user. Empty string if nothing to say (NO_OP).",
        },
        "memory_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Facts, preferences, or decisions worth remembering for future conversations.",
        },
        "current_topic": {
            "type": "string",
            "description": "Concise 3-10 word description of the current conversation topic.",
        },
        "should_compress": {
            "type": "boolean",
            "description": "True if older messages are no longer relevant and should be summarized.",
        },
        "did_success": {
            "type": "boolean",
            "description": "True if task completed successfully. False if any error occurred.",
        },
    },
    "required": ["reply", "memory_notes", "current_topic", "should_compress", "did_success"],
    "additionalProperties": False,
}


@dataclass
class TurnResult:
    """Structured result from a single agent turn."""

    response: str
    memory_notes: list[str] = field(default_factory=list)
    current_topic: str = ""
    should_compress: bool = False
    did_success: bool = True
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def is_noop(self) -> bool:
        return not self.response.strip()

    @classmethod
    def from_json(cls, data: dict[str, Any], usage: dict[str, int]) -> TurnResult:
        return cls(
            response=data.get("reply", ""),
            memory_notes=data.get("memory_notes", []),
            current_topic=data.get("current_topic", ""),
            should_compress=data.get("should_compress", False),
            did_success=data.get("did_success", True),
            usage=usage,
        )

    @classmethod
    def fallback(cls, text: str, usage: dict[str, int]) -> TurnResult:
        """Create a TurnResult from plain text when JSON parsing fails."""
        return cls(response=text, usage=usage)


class Agent:
    """Agentic loop: the central entry point for all interactions.

    The LLM's final text output is constrained to RESPONSE_SCHEMA via output_config.
    Tool calls proceed normally as intermediate steps.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str = "You are Memoo, a helpful AI assistant.",
        max_rounds: int = 0,
        fallback_llms: list[LLMProvider] | None = None,
        hooks: HookRegistry | None = None,
        compressor_llm: LLMProvider | None = None,
        memory: Any = None,
        gateway: Any = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_rounds = max_rounds
        self._fallback_llms = fallback_llms or []
        self._hooks = hooks or HookRegistry()
        # Cheap/fast LLM for context compression (haiku/mini). Falls back to main LLM.
        self._compressor = compressor_llm or (fallback_llms[-1] if fallback_llms else llm)
        self._memory = memory
        self._gateway = gateway  # For streaming tool events to clients
        self._cfg = agent_config or AgentConfig()
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._inboxes: dict[str, asyncio.Queue[str]] = {}
        # Cumulative token tracking across all runs
        self.total_tokens: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_runs": 0}

    def cancel(self, run_id: str | None = None) -> None:
        if run_id is not None:
            ev = self._cancel_events.get(run_id)
            if ev:
                ev.set()
                logger.info("Agent run cancelled (run_id=%s)", run_id)
        else:
            for ev in self._cancel_events.values():
                ev.set()
            if self._cancel_events:
                logger.info("All agent runs cancelled (%d)", len(self._cancel_events))

    def inject(self, run_id: str, text: str) -> bool:
        """Inject a user message into an active agent turn.

        The message will be appended to the conversation between the current
        tool execution and the next LLM call, so the LLM sees the correction
        without restarting the entire turn.

        Returns True if the message was queued, False if no active run exists.
        """
        q = self._inboxes.get(run_id)
        if q is None:
            return False
        q.put_nowait(text)
        logger.info("Message injected into run_id=%s (%d chars)", run_id, len(text))
        return True

    async def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
        context: dict[str, Any] | None = None,
    ) -> TurnResult:
        ctx = context or {}
        run_id = str(ctx.get("chat_id", id(asyncio.current_task())))
        cancel_event = asyncio.Event()
        self._cancel_events[run_id] = cancel_event
        inbox: asyncio.Queue[str] = asyncio.Queue()
        self._inboxes[run_id] = inbox

        try:
            result = await self._run_loop(user_message, ctx, cancel_event, inbox, history)
            # Accumulate token usage
            for k, v in result.usage.items():
                self.total_tokens[k] = self.total_tokens.get(k, 0) + v
            self.total_tokens["total_runs"] = self.total_tokens.get("total_runs", 0) + 1
            return result
        finally:
            self._cancel_events.pop(run_id, None)
            self._inboxes.pop(run_id, None)

    async def _run_loop(
        self,
        user_message: str,
        ctx: dict[str, Any],
        cancel_event: asyncio.Event,
        inbox: asyncio.Queue[str],
        history: list[Message] | None,
    ) -> TurnResult:
        messages = list(history) if history else []

        # Inject source context
        source = ctx.get("source", "")
        if source:
            prefix = f"[System: this message is from {source}"
            task_name = ctx.get("task_name", "")
            if task_name:
                prefix += f" ({task_name})"
            prefix += ", not from the human user.]\n\n"
            messages.append(Message(role="user", content=prefix + user_message))
        else:
            messages.append(Message(role="user", content=user_message))

        # Hard cap: compress if context exceeds token window
        messages = await self._enforce_context_window(messages)

        tool_schemas = self._tools.get_schemas() or None
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        # Expose state for sub-agent spawning and cancel propagation
        ctx["_messages"] = messages
        ctx.setdefault("_agent_depth", 0)
        ctx["_system_prompt"] = self._system_prompt
        ctx["_cancel_event"] = cancel_event
        set_context(ctx)
        round_num = 0

        while True:
            round_num += 1
            effective_max = self._max_rounds if self._max_rounds > 0 else self._cfg.hard_max_rounds
            if round_num > effective_max:
                break
            if cancel_event.is_set():
                return TurnResult(response="(cancelled by user)", usage=total_usage)

            logger.info("Agent round %d", round_num)
            response = await self._chat_with_fallback(messages, tool_schemas)

            for k, v in response.usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

            # Log cache status per round
            cache_read = response.usage.get("cache_read_tokens", 0)
            cache_create = response.usage.get("cache_creation_tokens", 0)
            if cache_read or cache_create:
                logger.info("Cache: read=%d, created=%d", cache_read, cache_create)

            if not response.has_tool_calls:
                result = self._parse_structured_response(response.text or "", total_usage)
                logger.info(
                    "Agent done in %d rounds. topic=%s, success=%s, noop=%s",
                    round_num,
                    result.current_topic,
                    result.did_success,
                    result.is_noop,
                )
                return result

            # Record assistant turn with tool calls
            messages.append(Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls))

            # Execute tool calls — parallel if multiple, sequential if single
            if cancel_event.is_set():
                return TurnResult(response="(cancelled during tool execution)", usage=total_usage)

            if len(response.tool_calls) > 1:
                logger.info("Parallel tool execution: %d calls", len(response.tool_calls))
                results = await asyncio.gather(*[self._execute_one_tool(tc, ctx) for tc in response.tool_calls])
                for tc, result in zip(response.tool_calls, results):
                    messages.append(Message(role="tool_result", content=result, tool_call_id=tc.id))
            else:
                tc = response.tool_calls[0]
                result = await self._execute_one_tool(tc, ctx)
                messages.append(Message(role="tool_result", content=result, tool_call_id=tc.id))

            # Drain inbox: inject user messages between tool execution and next LLM call
            injected_count = 0
            while not inbox.empty():
                injected_text = inbox.get_nowait()
                messages.append(Message(role="user", content=injected_text))
                injected_count += 1
            if injected_count:
                chat_id = ctx.get("chat_id", "")
                logger.info("Injected %d user message(s) mid-turn (chat_id=%s)", injected_count, chat_id)
                if self._gateway and chat_id:
                    self._gateway.send_event(chat_id, {"event": "message_injected", "count": injected_count})

        # Max rounds exceeded
        logger.warning("Max rounds (%d) exceeded, forcing final answer", effective_max)
        response = await self._chat_with_fallback(messages, tools=None)
        return self._parse_structured_response(response.text or "(max rounds reached)", total_usage)

    async def _execute_one_tool(self, tc: Any, ctx: dict[str, Any]) -> str:
        """Execute a single tool call with auth check and gateway events."""
        approved, reason = await self._hooks.authorize(tc.name, tc.arguments, ctx)
        if not approved:
            logger.warning("Tool %s denied: %s", tc.name, reason)
            return f"Authorization denied: {reason}"

        logger.info("Tool call: %s(id=%s)", tc.name, tc.id)

        chat_id = ctx.get("chat_id", "")
        if self._gateway and chat_id:
            args_preview = ", ".join(f"{k}={repr(v)[:50]}" for k, v in tc.arguments.items())
            self._gateway.send_event(chat_id, {"event": "tool_start", "name": tc.name, "args": args_preview})

        result = await self._tools.execute(tc.name, tc.arguments)

        if self._gateway and chat_id:
            ok = "error" not in result[:20].lower()
            preview = result.replace("\n", " ")[:120]
            self._gateway.send_event(chat_id, {"event": "tool_done", "name": tc.name, "ok": ok, "result": preview})

        return result

    async def _enforce_context_window(self, messages: list[Message]) -> list[Message]:
        """Hard cap: if context exceeds self._cfg.context_window_tokens, compress in two phases.

        Phase 1: Replace old messages with archived memory summaries (if available).
        Phase 2: If still over budget, strip middle and summarize with cheap LLM.
        """
        total_tokens = self._estimate_tokens(messages)
        if total_tokens <= self._cfg.context_window_tokens:
            return messages

        logger.warning("Context: %d tokens > %d limit", total_tokens, self._cfg.context_window_tokens)

        # --- Phase 1: Replace old messages with archived memory summaries ---
        if self._memory:
            from core.tools import get_context

            chat_id = get_context().get("chat_id", "")
            if chat_id:
                messages = await self._replace_with_memory(messages, chat_id)

        # --- Phase 2: Strip middle, compress with cheap LLM ---
        total_tokens = self._estimate_tokens(messages)
        if total_tokens <= self._cfg.context_window_tokens:
            return messages

        keep_start = min(2, len(messages))
        keep_end = min(10, len(messages) - keep_start)

        if len(messages) <= keep_start + keep_end:
            return messages

        head = messages[:keep_start]
        middle = messages[keep_start : len(messages) - keep_end]
        tail = messages[len(messages) - keep_end :]

        middle_text = "\n".join(f"[{m.role}]: {m.content[:500]}" for m in middle if m.content)
        try:
            summary_resp = await self._compressor.chat(
                messages=[Message(role="user", content=f"Summarize this conversation concisely:\n\n{middle_text}")],
                system="Output a concise summary preserving key facts, decisions, and context.",
                max_tokens=1000,
            )
            summary = summary_resp.text or ""
        except Exception:
            logger.exception("Context compression failed, truncating")
            summary = f"[{len(middle)} messages omitted]"

        logger.info("Phase 2: %d middle msgs -> %d char summary", len(middle), len(summary))
        return head + [Message(role="system", content=f"[Compressed context]: {summary}")] + tail

    def _estimate_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content) for m in messages if m.content) // self._cfg.chars_per_token

    async def _replace_with_memory(self, messages: list[Message], chat_id: str) -> list[Message]:
        """Phase 1: Replace old verbose messages with archived memory summaries.

        Scans messages from oldest to newest. If a chunk of old messages has
        already been archived (we have a summary for that period), replace
        those messages with the compact summary.
        """
        try:
            archives = await self._memory.list_archive(chat_id=chat_id, limit=10)
        except Exception:
            return messages

        if not archives:
            return messages

        # Build a single summary from all archived entries
        summary_parts = [f"- [{a.get('date', '?')}] {a['topic']}: {a.get('summary', '')}" for a in archives]
        memory_summary = "Archived conversation history:\n" + "\n".join(summary_parts)

        # Keep only recent messages (last N that fit in budget), prepend memory summary
        target_tokens = int(self._cfg.context_window_tokens * 0.7)
        kept: list[Message] = []
        token_count = len(memory_summary) // self._cfg.chars_per_token

        # Walk from newest to oldest, keep messages until we hit budget
        for msg in reversed(messages):
            msg_tokens = len(msg.content) // self._cfg.chars_per_token if msg.content else 0
            if token_count + msg_tokens > target_tokens:
                break
            kept.insert(0, msg)
            token_count += msg_tokens

        result = [Message(role="system", content=f"[Memory]: {memory_summary}")] + kept
        logger.info(
            "Phase 1: replaced %d old msgs with %d archives, kept %d recent msgs",
            len(messages) - len(kept),
            len(archives),
            len(kept),
        )
        return result

    @staticmethod
    def _parse_structured_response(text: str, usage: dict[str, int]) -> TurnResult:
        """Parse the LLM's JSON-constrained text output into a TurnResult."""
        if not text.strip():
            return TurnResult.fallback("", usage)
        try:
            data = json.loads(text)
            return TurnResult.from_json(data, usage)
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat raw text as reply (e.g. from OpenAI fallback without schema)
            logger.debug("Failed to parse structured response, using raw text")
            return TurnResult.fallback(text, usage)

    async def _chat_with_fallback(
        self,
        messages: list[Message],
        tools: list[dict] | None,  # type: ignore[type-arg]
    ) -> LLMResponse:
        providers = [self._llm, *self._fallback_llms]
        last_error: Exception | None = None

        for i, llm in enumerate(providers):
            try:
                return await llm.chat(
                    messages=messages,
                    system=self._system_prompt,
                    tools=tools,
                    output_schema=RESPONSE_SCHEMA,
                )
            except Exception as e:
                from core.crash import report_crash

                last_error = e
                if i < len(providers) - 1:
                    logger.warning("LLM %s failed (%s), trying fallback", llm.model_name, e)
                else:
                    report_crash(e, context={"providers_tried": len(providers)}, component="agent.llm")

        raise last_error or RuntimeError("No LLM providers available")
