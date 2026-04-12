"""Schedule tools — agent can dynamically create/manage cron jobs."""

from __future__ import annotations

import logging
from typing import Any

from core.tools import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, **deps: Any) -> None:
    """Register schedule tools. Auto-discovered by tools/__init__.py."""
    scheduler = deps.get("scheduler")
    if scheduler is None:
        logger.warning("Scheduler not provided, skipping schedule tools")
        return
    default_channel = deps.get("default_channel", "telegram")

    @registry.tool
    async def create_schedule(name: str, cron: str, prompt: str, chat_id: str = "", channel: str = "") -> str:
        """Create a recurring scheduled task with a cron expression.

        Args:
            name: Unique name for this schedule (e.g. 'morning_briefing').
            cron: Cron expression with 5 fields: minute hour day month weekday.
                Examples: '0 8 * * *' (daily 8am), '*/30 * * * *' (every 30min).
            prompt: The message/prompt to execute when the schedule fires.
            chat_id: Target chat ID to send results to. Defaults to current chat.
            channel: Channel to use (telegram, wechat). Defaults to configured default.
        """
        ch = channel or default_channel
        return await scheduler.create_schedule(name, cron, chat_id, ch, prompt)

    @registry.tool
    async def delete_schedule(name: str) -> str:
        """Delete a scheduled task by name.

        Args:
            name: Name of the schedule to delete.
        """
        return await scheduler.delete_schedule(name)

    @registry.tool
    async def list_schedules() -> str:
        """List all scheduled tasks with their cron expressions and status."""
        return await scheduler.list_schedules()

    @registry.tool
    async def toggle_schedule(name: str, enabled: bool = True) -> str:
        """Enable or disable a scheduled task.

        Args:
            name: Name of the schedule to toggle.
            enabled: True to enable, False to disable.
        """
        return await scheduler.toggle_schedule(name, enabled)
