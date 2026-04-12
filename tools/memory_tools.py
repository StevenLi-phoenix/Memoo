"""Memory tools — agent can search and read archived conversations."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.tools import ToolRegistry, get_context

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, **deps: Any) -> None:
    """Register memory tools. Auto-discovered by tools/__init__.py."""
    memory = deps.get("memory")
    if memory is None:
        logger.warning("Memory not provided, skipping memory tools")
        return

    @registry.tool
    async def search_memory(query: str, limit: int = 5) -> str:
        """Search past conversations by keyword. Returns summaries of matching archived conversations.

        Args:
            query: Search keywords to find relevant past conversations.
            limit: Maximum number of results to return.
        """
        chat_id = get_context().get("chat_id", "")
        logger.info("search_memory: query=%s, limit=%d, chat_id=%s", query, limit, chat_id)
        results = await memory.search_archive(query, chat_id=chat_id, limit=limit)

        if not results:
            return "No matching memories found."

        lines: list[str] = []
        for r in results:
            lines.append(
                f"[Memory #{r['id']}] ({r['created_at']}) Topic: {r['topic']}\n"
                f"  Summary: {r['summary'][:200]}\n"
                f"  Use `read_memory({r['id']})` to see full conversation."
            )
        return "\n\n".join(lines)

    @registry.tool
    async def read_memory(memory_id: int) -> str:
        """Read the full archived conversation for a specific memory entry.

        Args:
            memory_id: The ID of the archived memory to read (from search_memory results).
        """
        if memory_id < 0:
            return "Error: invalid memory_id"
        chat_id = get_context().get("chat_id", "")
        logger.info("read_memory: id=%d, chat_id=%s", memory_id, chat_id)
        entry = await memory.get_archive_entry(memory_id)

        if not entry or entry.get("chat_id") != chat_id:
            return f"Memory #{memory_id} not found."

        try:
            messages = json.loads(entry["full_messages"])
        except (json.JSONDecodeError, TypeError):
            return f"Memory #{memory_id} data corrupted."
        lines: list[str] = [
            f"**Memory #{entry['id']}** — Topic: {entry['topic']}",
            f"Date: {entry['created_at']}",
            f"Summary: {entry['summary']}",
            "",
            "--- Full Conversation ---",
        ]
        for msg in messages:
            lines.append(f"[{msg['role']}]: {msg['content']}")

        output = "\n".join(lines)
        if len(output) > 10000:
            output = output[:10000] + "\n...(truncated)"
        return output

    @registry.tool
    async def list_memories(limit: int = 10) -> str:
        """List recent archived conversation summaries (progressive disclosure).

        Args:
            limit: Maximum number of entries to list.
        """
        chat_id = get_context().get("chat_id", "")
        logger.info("list_memories: limit=%d, chat_id=%s", limit, chat_id)
        entries = await memory.list_archive(chat_id=chat_id, limit=limit)

        if not entries:
            return "No archived memories."

        lines: list[str] = []
        for e in entries:
            lines.append(f"#{e['id']} [{e.get('date', '')}] {e['topic']} — {e.get('summary', '')}")
        return "\n".join(lines)
