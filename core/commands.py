"""Slash commands — shared logic for all channels.

Commands are handled at the channel level (TUI, Telegram) without an LLM call.
Channels call these functions directly. handle_message has a fallback for
gateway clients that don't implement local command parsing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# All registered commands
COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/new": "Start a new conversation (TUI: new session, others: clear + reset)",
    "/clear": "Clear conversation memory for this chat",
    "/compact": "Compact conversation history (archive old messages, keep recent)",
    "/config": "Show current configuration",
    "/model": "Show or switch the active LLM model",
    "/memory": "Show archived memory summaries",
    "/schedule": "List scheduled tasks",
    "/status": "Show agent status (tokens, model, uptime)",
    "/dream": "Consolidate archived memories into long-term knowledge",
    "/quit": "Exit (TUI only)",
}


async def handle_command(cmd: str, chat_id: str, deps: dict[str, Any]) -> str | None:
    """Execute a slash command. Returns response string, or None if not a command.

    deps should contain: memory, config, agent, scheduler, app (all optional).
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        return _cmd_help(deps)

    if command == "/new":
        memory = deps.get("memory")
        if memory:
            await memory.clear(chat_id)
        app = deps.get("app")
        if app:
            app._current_topics.pop(chat_id, None)
        return "New conversation started."

    if command == "/clear":
        memory = deps.get("memory")
        if memory:
            await memory.clear(chat_id)
        return "Memory cleared."

    if command == "/compact":
        return await _cmd_compact(chat_id, deps)

    if command == "/config":
        config = deps.get("config")
        if config:
            return "```\n" + json.dumps(config.to_display_dict(), indent=2, ensure_ascii=False) + "\n```"
        return "Config not available."

    if command == "/model":
        return _cmd_model(arg, deps)

    if command == "/memory":
        return await _cmd_memory(chat_id, deps)

    if command == "/schedule":
        return await _cmd_schedule(deps)

    if command == "/dream":
        return await _cmd_dream(deps)

    if command == "/status":
        return await _cmd_status(deps)

    # Check if it's a skill trigger — let it fall through to agent
    if command.startswith("/"):
        app = deps.get("app")
        if app:
            skill_registry = getattr(app, "skill_registry", None)
            if skill_registry and skill_registry.get_meta(command[1:]):
                return None  # Handled by main.py as skill trigger

        return _suggest_command(command, deps)

    return None  # Not a command at all


def _suggest_command(typed: str, deps: dict[str, Any]) -> str:
    """Suggest commands or skills matching a partial input."""
    prefix = typed.lower()

    # Collect all known names: built-in commands + skill names
    all_names = list(COMMANDS.keys())
    app = deps.get("app")
    if app:
        skill_registry = getattr(app, "skill_registry", None)
        if skill_registry:
            all_names.extend(f"/{name}" for name in skill_registry.skill_names)

    matches = [n for n in all_names if n.startswith(prefix)]
    if not matches:
        slug = prefix.lstrip("/")
        matches = [n for n in all_names if slug and slug[:2] in n]
    if matches:
        suggestions = ", ".join(f"`{c}`" for c in sorted(matches)[:8])
        return f"Unknown command `{typed}`. Did you mean: {suggestions}?"
    return f"Unknown command `{typed}`. Type `/help` for available commands."


def _cmd_help(deps: dict[str, Any]) -> str:
    lines = ["**Available Commands**", ""]
    for cmd, desc in COMMANDS.items():
        lines.append(f"  `{cmd}` — {desc}")

    # List available skills
    app = deps.get("app")
    if app:
        skill_registry = getattr(app, "skill_registry", None)
        if skill_registry and skill_registry.skill_names:
            lines.append("")
            lines.append("**Skills** (use `/{name}` to activate):")
            for name in skill_registry.skill_names:
                meta = skill_registry.get_meta(name)
                desc = meta.description if meta else ""
                lines.append(f"  `/{name}` — {desc}")

    lines.append("")
    lines.append("All other messages are sent to the AI agent.")
    return "\n".join(lines)


def _cmd_model(arg: str, deps: dict[str, Any]) -> str:
    app = deps.get("app")
    config = deps.get("config")

    if not app or not config:
        return "Not available."

    if not arg:
        # Show current model
        llm = getattr(app, "llm", None)
        model_name = llm.model_name if llm else "unknown"
        current_alias = config.llm.default or "unknown"
        lines = [f"**Current model**: `{current_alias}` (`{model_name}`)", ""]

        lines.append("**Configured models**:")
        for model in config.llm.models:
            marker = " ← current" if model.name == current_alias else ""
            lines.append(f"  `{model.name}` -> `{model.model}` via `{model.provider}`{marker}")

        discovered = getattr(app, "discovered_models", {}) or {}
        if discovered:
            lines.append("")
            lines.append("**Discovered models**:")
            for provider in config.llm.providers:
                if not provider.allow_model_discovery:
                    continue
                models = discovered.get(provider.name, [])
                if not models:
                    continue
                lines.append(f"  `{provider.name}`:")
                for info in models:
                    lines.append(f"    `{info.id}`")
        lines.append("")
        lines.append("Switch: `/model <name>` (e.g. `/model haiku`)")
        return "\n".join(lines)

    # Switch model — delegate to config_tools._update_model
    from tools.config_tools import _update_model

    result = _update_model(arg, config, app)
    logger.info("Model switched via /model: %s", result)
    return result


async def _cmd_compact(chat_id: str, deps: dict[str, Any]) -> str:
    app = deps.get("app")
    if not app:
        return "Compact not available."

    memory = deps.get("memory")
    if not memory:
        return "Memory not available."

    history = await memory.get_history(chat_id)
    if len(history) < 6:
        return f"Nothing to compact ({len(history)} messages, need at least 6)."

    try:
        await app._compact_memory(chat_id)
        remaining = await memory.get_history(chat_id)
        archived = len(history) - len(remaining)
        return f"Compacted: {archived} messages archived, {len(remaining)} kept."
    except Exception as e:
        logger.exception("Compact failed for chat_id=%s", chat_id)
        return f"Compact failed: {e}"


async def _cmd_memory(chat_id: str, deps: dict[str, Any]) -> str:
    memory = deps.get("memory")
    if not memory:
        return "Memory not available."

    entries = await memory.list_archive(chat_id=chat_id, limit=10)
    if not entries:
        return "No archived memories."

    lines = ["**Archived Memories**", ""]
    for e in entries:
        lines.append(f"  `#{e['id']}` [{e.get('date', '?')}] {e['topic']} — {e.get('summary', '')[:60]}")
    return "\n".join(lines)


async def _cmd_dream(deps: dict[str, Any]) -> str:
    memory = deps.get("memory")
    app = deps.get("app")
    config = deps.get("config")
    if not memory or not app or not config:
        return "Dream not available — missing dependencies."
    llm = getattr(app, "llm", None)
    if not llm:
        return "Dream not available — no LLM provider."

    from core.dream import run_dream

    return await run_dream(memory, llm, config)


async def _cmd_schedule(deps: dict[str, Any]) -> str:
    scheduler = deps.get("scheduler")
    if not scheduler:
        return "Scheduler not available."

    result = await scheduler.list_schedules()
    if result == "No scheduled tasks.":
        return result
    return f"**Scheduled Tasks**\n\n{result}"


async def _cmd_status(deps: dict[str, Any]) -> str:
    import time

    app = deps.get("app")
    agent = deps.get("agent")

    lines = ["**Memoo Status**", ""]

    # Uptime
    if app and hasattr(app, "_start_time"):
        elapsed = time.monotonic() - app._start_time
        lines.append(f"  Uptime: {_format_duration(elapsed)}")

    # Model info
    if app and hasattr(app, "llm") and app.llm:
        lines.append(f"  Model: `{app.llm.model_name}`")
        if agent:
            compressor_name = getattr(agent.compressor, "model_name", "same as primary")
            if compressor_name != app.llm.model_name:
                lines.append(f"  Compressor: `{compressor_name}`")

    # Token usage
    if agent:
        tokens = getattr(agent, "total_tokens", {})
        total_runs = tokens.get("total_runs", 0)
        input_t = tokens.get("input_tokens", 0)
        output_t = tokens.get("output_tokens", 0)
        lines.append(f"  Runs: {total_runs}")
        lines.append(f"  Tokens: {input_t + output_t:,} (in={input_t:,}, out={output_t:,})")
        cache_read = tokens.get("cache_read_tokens", 0)
        cache_create = tokens.get("cache_creation_tokens", 0)
        if cache_read or cache_create:
            hit_rate = cache_read / max(input_t, 1) * 100
            lines.append(f"  Cache: read={cache_read:,}, created={cache_create:,} ({hit_rate:.0f}% hit rate)")

    # Memory stats
    memory = deps.get("memory")
    if memory:
        lines.append("")
        lines.append("**Memory**")
        try:
            # Count active messages across known chats
            active_chats: dict[str, int] = {}
            if app and hasattr(app, "_current_topics"):
                for cid in app._current_topics:
                    count = await memory.message_count(cid)
                    if count:
                        active_chats[cid] = count
            total_active = sum(active_chats.values())
            lines.append(f"  Active messages: {total_active} across {len(active_chats)} chat(s)")

            archive_entries = await memory.list_archive(limit=100)
            lines.append(f"  Archive entries: {len(archive_entries)}")
        except Exception:
            lines.append("  (stats unavailable)")

    # Infrastructure
    if app:
        lines.append("")
        lines.append("**Infrastructure**")
        channels = list(getattr(app, "_channel_map", {}).keys())
        gateway = getattr(app, "gateway", None)
        gw_count = 0
        if gateway:
            gw_count = len(getattr(gateway, "_all_writers", set()))
        lines.append(f"  Channels: {', '.join(channels) if channels else 'none'}")
        lines.append(f"  Gateway clients: {gw_count}")
        tools = getattr(app, "tools", None)
        lines.append(f"  Tools: {len(tools.tool_names) if tools else '?'}")

        scheduler = deps.get("scheduler")
        if scheduler:
            sched_result = await scheduler.list_schedules()
            sched_count = sched_result.count("\n") if sched_result != "No scheduled tasks." else 0
            lines.append(f"  Scheduled tasks: {sched_count}")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    if h < 24:
        return f"{h}h {m}m"
    d = h // 24
    h = h % 24
    return f"{d}d {h}h {m}m"
