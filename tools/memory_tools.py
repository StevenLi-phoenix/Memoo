"""Memory tools — agent can search and read archived conversations."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.tools import ToolRegistry

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
        logger.info("search_memory: query=%s, limit=%d", query, limit)
        results = await memory.search_archive(query, limit=limit)

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
        logger.info("read_memory: id=%d", memory_id)
        entry = await memory.get_archive_entry(memory_id)

        if not entry:
            return f"Memory #{memory_id} not found."

        messages = json.loads(entry["full_messages"])
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
        logger.info("list_memories: limit=%d", limit)
        entries = await memory.list_archive(limit=limit)

        if not entries:
            return "No archived memories."

        lines: list[str] = []
        for e in entries:
            lines.append(f"#{e['id']} [{e.get('date', '')}] {e['topic']} — {e.get('summary', '')}")
        return "\n".join(lines)
