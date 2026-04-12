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
    "/clear": "Clear conversation memory for this chat",
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

    if command == "/clear":
        memory = deps.get("memory")
        if memory:
            await memory.clear(chat_id)
        return "Memory cleared."

    if command == "/config":
        config = deps.get("config")
        if config:
            return "```\n" + json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n```"
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
        return _cmd_status(deps)

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
        lines = [f"**Current model**: `{model_name}`", ""]

        # List available from cache
        from models import ModelCache

        cache = ModelCache()
        for p in config.llm.providers:
            cached = cache.get(p.provider)
            if cached:
                model_ids = [m.id for m in sorted(cached, key=lambda x: x.created, reverse=True)[:8]]
                lines.append(f"**{p.name}** ({p.provider}):")
                for mid in model_ids:
                    marker = " ← current" if mid == model_name else ""
                    lines.append(f"  `{mid}`{marker}")
        lines.append("")
        lines.append("Switch: `/model <name>` (e.g. `/model haiku`)")
        return "\n".join(lines)

    # Switch model

    # Use the config tool's set_model logic directly
    from models import ModelCache

    for p in config.llm.providers:
        if p.name == config.llm.default:
            cache = ModelCache()
            cached = cache.get(p.provider)
            if cached:
                exact = [m for m in cached if m.id == arg]
                matches = exact or [m for m in cached if arg in m.id]
                if not matches:
                    model_ids = [m.id for m in sorted(cached, key=lambda x: x.created, reverse=True)[:8]]
                    return f"Model `{arg}` not found. Available:\n" + "\n".join(f"  `{m}`" for m in model_ids)
                resolved = exact[0].id if exact else sorted(matches, key=lambda x: x.created, reverse=True)[0].id
            else:
                resolved = arg

            old = getattr(app.llm, "model_name", "?")
            app.llm.model_name = resolved  # type: ignore[union-attr]
            p.model = resolved
            config.save()
            logger.info("Model switched via /model: %s -> %s", old, resolved)
            return f"Model switched: `{old}` → `{resolved}`"

    return "No default provider configured."


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


def _cmd_status(deps: dict[str, Any]) -> str:
    app = deps.get("app")
    agent = deps.get("agent")

    lines = ["**Memoo Status**", ""]

    if app and hasattr(app, "llm") and app.llm:
        lines.append(f"  Model: `{app.llm.model_name}`")

    if agent:
        tokens = getattr(agent, "total_tokens", {})
        lines.append(f"  Total runs: {tokens.get('total_runs', 0)}")
        lines.append(f"  Input tokens: {tokens.get('input_tokens', 0):,}")
        lines.append(f"  Output tokens: {tokens.get('output_tokens', 0):,}")
        cache_read = tokens.get("cache_read_tokens", 0)
        cache_create = tokens.get("cache_creation_tokens", 0)
        if cache_read or cache_create:
            lines.append(f"  Cache: read={cache_read:,}, created={cache_create:,}")

    if app:
        channels = list(getattr(app, "_channel_map", {}).keys())
        # Include TUI/gateway clients
        gateway = getattr(app, "gateway", None)
        if gateway:
            gw_clients = getattr(gateway, "_clients", {})
            for gw_id in gw_clients:
                if gw_id not in channels:
                    channels.append(f"tui:{gw_id}")
        lines.append(f"  Channels: {channels}")
        lines.append(f"  Tools: {len(getattr(app, 'tools', None).tool_names) if hasattr(app, 'tools') else '?'}")

    return "\n".join(lines)
