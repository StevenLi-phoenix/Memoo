"""Sub-agent spawning tool — delegate tasks to independent agent instances.

Features:
- Model selection — use any configured LLM provider
- Context sharing — full (expensive, cache-reusable), summary (compressed), none (clean)
- Sandbox restrictions — readonly (deny file writes), no_network (deny network) via SBPL profile
- Background mode — block (wait) or bg (return run_id immediately)
- Timeout → background — auto-convert to bg on timeout instead of killing
- Cancel propagation — parent cancellation propagates to sub-agents
- Inter-agent communication — read background agent output/status
- Cost tracking — audit log to .logs/subagent_audit.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.tools import ToolRegistry, get_context

logger = logging.getLogger(__name__)

MAX_TRACKED_RUNS = 50


@dataclass
class SubagentRun:
    """Tracks a running or completed sub-agent."""

    run_id: str
    task: asyncio.Task[Any]
    agent: Any = None  # Agent instance, for cancellation
    model: str = ""
    prompt: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    result: Any = None  # TurnResult when done
    events: list[dict[str, Any]] = field(default_factory=list)
    _cancel_watcher: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def is_done(self) -> bool:
        return self.task.done()

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)

    @property
    def status(self) -> str:
        if self.task.cancelled():
            return "cancelled"
        if self.task.done():
            return "failed" if self.task.exception() else "completed"
        return "running"


class SubagentEventCollector:
    """Captures sub-agent gateway events for inter-agent communication."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def send_event(self, chat_id: str, event: dict[str, Any]) -> None:
        self._events.append(event)

    def set_reply_extra(self, chat_id: str, extra: dict[str, Any]) -> None:
        pass  # Not needed for sub-agents


# Module-level registry of active/completed sub-agent runs
_active_runs: dict[str, SubagentRun] = {}


def _cleanup_old_runs() -> None:
    """Remove oldest completed runs if over limit."""
    if len(_active_runs) <= MAX_TRACKED_RUNS:
        return
    completed = sorted(
        [(k, v) for k, v in _active_runs.items() if v.is_done],
        key=lambda x: x[1].finished_at,
    )
    while len(_active_runs) > MAX_TRACKED_RUNS and completed:
        key, _ = completed.pop(0)
        del _active_runs[key]


def _on_task_done(task: asyncio.Task[Any], run: SubagentRun) -> None:
    """Callback when sub-agent task finishes — stores result and writes audit log."""
    run.finished_at = time.time()
    try:
        run.result = task.result()
    except (asyncio.CancelledError, Exception):
        pass
    if run._cancel_watcher:
        run._cancel_watcher.cancel()
    _audit_log(run)
    _cleanup_old_runs()


def _audit_log(run: SubagentRun) -> None:
    """Write sub-agent run to audit log."""
    try:
        log_dir = Path(".logs")
        log_dir.mkdir(exist_ok=True)

        usage = run.result.usage if run.result else {}
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": run.run_id,
            "model": run.model,
            "prompt": run.prompt[:200],
            "status": run.status,
            "elapsed_s": run.elapsed,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "tool_calls": len([e for e in run.events if e.get("event") == "tool_done"]),
        }

        with open(log_dir / "subagent_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to write subagent audit log: %s", e)


def _format_result(run: SubagentRun) -> str:
    """Format a completed sub-agent run as JSON with full metadata."""
    result = run.result
    if not result:
        return json.dumps({"error": "No result available", "run_id": run.run_id, "status": run.status})

    return json.dumps(
        {
            "reply": result.response,
            "topic": result.current_topic,
            "success": result.did_success,
            "memory_notes": result.memory_notes,
            "usage": result.usage,
            "run_id": run.run_id,
            "elapsed_s": run.elapsed,
        },
        ensure_ascii=False,
    )


async def _summarize_messages(app: Any, messages: list[Any]) -> str:
    """Summarize parent messages using the compressor LLM (cheapest fallback)."""
    from models import Message

    text_parts = []
    for m in messages[-50:]:
        if m.content:
            text_parts.append(f"[{m.role}]: {m.content[:500]}")

    combined = "\n".join(text_parts)

    compressor = None
    if hasattr(app, "agent") and app.agent:
        compressor = app.agent._compressor
    if not compressor:
        compressor = app.llm

    if not compressor:
        return combined[:2000]

    try:
        resp = await compressor.chat(
            messages=[Message(role="user", content=f"Summarize this conversation concisely:\n\n{combined}")],
            system="Output a concise summary preserving key facts, decisions, and context. Max 500 words.",
            max_tokens=1000,
        )
        return resp.text or combined[:2000]
    except Exception as e:
        logger.warning("Failed to summarize parent context: %s", e)
        return combined[:2000]


def register(registry: ToolRegistry, **deps: Any) -> None:
    """Register sub-agent tools. Auto-discovered by tools/__init__.py."""
    app = deps.get("app")
    config = deps.get("config")

    if app is None:
        logger.warning("App not provided, skipping subagent tools")
        return

    @registry.tool
    async def spawn_agent(
        prompt: str,
        model: str = "",
        context_mode: str = "none",
        max_rounds: int = 0,
        system_prompt: str = "",
        readonly: bool = False,
        network_access: bool = True,
        background: str = "block",
        timeout: int = 0,
        timeout_action: str = "background",
    ) -> str:
        """Spawn a sub-agent to handle a task. Result injected back into session.

        Args:
            prompt: Task description for the sub-agent.
            model: Provider name (e.g. 'anthropic', 'openai'). Empty = current provider.
            context_mode: Context sharing — 'full', 'summary', or 'none'.
            max_rounds: Max tool rounds. 0 = use config default.
            system_prompt: Custom system prompt. Empty = inherit parent's.
            readonly: If true, sandbox denies file writes (commands can still run).
            network_access: If false, sandbox denies network access.
            background: 'block' = wait for result, 'bg' = run in background (returns run_id).
            timeout: Timeout in seconds. 0 = no limit.
            timeout_action: On timeout — 'background' (default, move to bg) or 'kill' (cancel).
        """
        from core.agent import Agent
        from models import Message

        ctx = get_context()

        # --- Depth check ---
        depth = ctx.get("_agent_depth", 0)
        max_depth = config.subagent.max_depth if config else 3
        if depth >= max_depth:
            return json.dumps({"error": f"Max sub-agent depth ({max_depth}) reached"})

        # --- Resolve LLM provider ---
        if model:
            providers: dict[str, Any] = getattr(app, "_providers", {})
            llm = providers.get(model)
            if not llm:
                available = list(providers.keys())
                return json.dumps({"error": f"Provider '{model}' not found", "available": available})
        else:
            llm = app.llm

        if not llm:
            return json.dumps({"error": "No LLM provider available"})

        # --- Resolve parameters ---
        effective_rounds = max_rounds or (config.subagent.default_max_rounds if config else 10)
        effective_system = system_prompt or ctx.get("_system_prompt", "You are a helpful sub-agent.")

        # --- Build context based on context_mode ---
        history: list[Message] | None = None
        parent_messages: list[Message] = ctx.get("_messages", [])

        if context_mode == "full":
            history = list(parent_messages)
            logger.info("Sub-agent: full context (%d messages)", len(history))
        elif context_mode == "summary":
            if parent_messages:
                summary = await _summarize_messages(app, parent_messages)
                history = [Message(role="system", content=f"[Parent context]: {summary}")]
                logger.info("Sub-agent: summary context (%d chars)", len(summary))
        else:
            logger.info("Sub-agent: no context sharing")

        # --- Create sub-agent with event collector ---
        run_id = uuid.uuid4().hex[:8]
        events: list[dict[str, Any]] = []
        collector = SubagentEventCollector(events)

        sub_agent = Agent(
            llm=llm,
            tools=app.tools,
            system_prompt=effective_system,
            max_rounds=effective_rounds,
            fallback_llms=app.fallback_llms,
            hooks=app.hooks,
            memory=app.memory,
            gateway=collector,
        )

        # Sandbox restrictions via context flags (enforced by core.sandbox)
        sub_ctx: dict[str, Any] = {
            "chat_id": ctx.get("chat_id", ""),
            "sandbox_dir": ctx.get("sandbox_dir", "./sandbox"),
            "source": "subagent",
            "_agent_depth": depth + 1,
            "_sandbox_readonly": readonly,
            "_sandbox_no_network": not network_access,
        }

        logger.info(
            "Spawning sub-agent %s: model=%s, depth=%d, context=%s, rounds=%d, readonly=%s, no_net=%s, bg=%s",
            run_id,
            llm.model_name,
            depth + 1,
            context_mode,
            effective_rounds,
            readonly,
            not network_access,
            background,
        )

        # Start as asyncio.Task for context isolation (ContextVar copy)
        task = asyncio.create_task(sub_agent.run(prompt, history=history, context=sub_ctx))

        run = SubagentRun(
            run_id=run_id,
            task=task,
            agent=sub_agent,
            model=llm.model_name,
            prompt=prompt[:200],
            events=events,
        )

        # --- Cancel propagation: parent cancel → sub-agent cancel ---
        parent_cancel = ctx.get("_cancel_event")
        if parent_cancel:

            async def _propagate_cancel() -> None:
                await parent_cancel.wait()
                sub_agent.cancel()

            run._cancel_watcher = asyncio.create_task(_propagate_cancel())

        # Register done callback for cleanup + audit
        task.add_done_callback(lambda t: _on_task_done(t, run))
        _active_runs[run_id] = run

        # --- Background mode: return immediately ---
        if background == "bg":
            return json.dumps(
                {
                    "run_id": run_id,
                    "status": "running",
                    "model": llm.model_name,
                    "message": f"Use read_agent_output('{run_id}') to check progress.",
                }
            )

        # --- Blocking mode with optional timeout ---
        if timeout > 0:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if not done:
                if timeout_action == "kill":
                    logger.info("Sub-agent %s timed out (%ds), killing", run_id, timeout)
                    if run.agent:
                        run.agent.cancel()
                    run.task.cancel()
                    return json.dumps(
                        {
                            "run_id": run_id,
                            "status": "killed",
                            "elapsed_s": run.elapsed,
                            "message": f"Sub-agent killed after {timeout}s timeout.",
                        }
                    )
                # Default: move to background
                logger.info("Sub-agent %s timed out (%ds), moved to background", run_id, timeout)
                return json.dumps(
                    {
                        "run_id": run_id,
                        "status": "moved_to_background",
                        "elapsed_s": run.elapsed,
                        "message": f"Timed out after {timeout}s. Use read_agent_output('{run_id}') to check later.",
                    }
                )

        # --- Blocking: await completion ---
        try:
            await task
        except asyncio.CancelledError:
            return json.dumps({"run_id": run_id, "status": "cancelled", "elapsed_s": run.elapsed})
        except Exception as e:
            return json.dumps({"run_id": run_id, "status": "failed", "error": str(e), "elapsed_s": run.elapsed})

        return _format_result(run)

    @registry.tool
    async def read_agent_output(run_id: str) -> str:
        """Read the output/status of a background sub-agent.

        Args:
            run_id: The run ID returned by spawn_agent in background mode.
        """
        run = _active_runs.get(run_id)
        if not run:
            available = list(_active_runs.keys())
            return json.dumps({"error": f"Unknown run_id: {run_id}", "available_runs": available})

        output: dict[str, Any] = {
            "run_id": run_id,
            "status": run.status,
            "model": run.model,
            "elapsed_s": run.elapsed,
            "events_count": len(run.events),
        }

        if run.is_done and run.result:
            output["reply"] = run.result.response
            output["topic"] = run.result.current_topic
            output["success"] = run.result.did_success
            output["memory_notes"] = run.result.memory_notes
            output["usage"] = run.result.usage
        else:
            # Show recent events so the parent can see progress
            output["recent_events"] = run.events[-10:]

        return json.dumps(output, ensure_ascii=False)

    @registry.tool
    async def cancel_agent(run_id: str) -> str:
        """Cancel a running background sub-agent.

        Args:
            run_id: The run ID to cancel.
        """
        run = _active_runs.get(run_id)
        if not run:
            return json.dumps({"error": f"Unknown run_id: {run_id}"})

        if run.is_done:
            return json.dumps({"error": "Agent already finished", "status": run.status})

        # Graceful cancel via agent + hard cancel via task
        if run.agent:
            run.agent.cancel()
        run.task.cancel()

        return json.dumps({"run_id": run_id, "status": "cancelled", "elapsed_s": run.elapsed})

    @registry.tool
    def list_agents() -> str:
        """List all active and recently completed sub-agent runs."""
        runs = [
            {
                "run_id": r.run_id,
                "status": r.status,
                "model": r.model,
                "prompt": r.prompt[:100],
                "elapsed_s": r.elapsed,
            }
            for r in _active_runs.values()
        ]
        return json.dumps(runs, ensure_ascii=False)
