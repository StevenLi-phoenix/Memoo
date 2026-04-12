"""Crontab-style scheduled task runner with wait-sleep model.

Agent creates/manages schedules via tools at runtime.
Persisted to SQLite. Uses precise wait-until-next-fire instead of busy polling.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiosqlite

logger = logging.getLogger(__name__)

CRON_FIELDS = 5

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    cron TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'telegram',
    prompt TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _match_cron_field(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        return value % int(field[2:]) == 0
    if "-" in field and "," not in field:
        low, high = field.split("-", 1)
        return int(low) <= value <= int(high)
    if "," in field:
        return value in {int(v) for v in field.split(",")}
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check if cron expression matches a datetime. Format: minute hour day month weekday."""
    parts = cron_expr.strip().split()
    if len(parts) != CRON_FIELDS:
        return False
    minute, hour, day, month, weekday = parts
    return (
        _match_cron_field(minute, dt.minute)
        and _match_cron_field(hour, dt.hour)
        and _match_cron_field(day, dt.day)
        and _match_cron_field(month, dt.month)
        and _match_cron_field(weekday, dt.isoweekday() % 7)
    )


def next_cron_fire(cron_expr: str, after: datetime) -> datetime:
    """Calculate the next time a cron expression will fire after a given datetime.

    Scans minute-by-minute up to 366 days ahead.
    """
    # Start from the next minute boundary
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    max_iterations = 366 * 24 * 60  # 1 year of minutes

    for _ in range(max_iterations):
        if cron_matches(cron_expr, candidate):
            return candidate
        candidate += timedelta(minutes=1)

    # Fallback: 1 hour
    return after + timedelta(hours=1)


ScheduleHandler = Callable[[str, str, str], Awaitable[str]]


class Scheduler:
    """Dynamic crontab scheduler with SQLite persistence and wait-sleep model."""

    def __init__(self, db_path: str = "./data/schedules.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()  # signal to recalculate next fire time

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_INIT_SQL)
        await self._db.commit()
        logger.info("Scheduler initialized: db=%s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # --- CRUD (called by agent tools) ---

    async def create_schedule(self, name: str, cron: str, chat_id: str, channel: str, prompt: str) -> str:
        assert self._db is not None
        parts = cron.strip().split()
        if len(parts) != CRON_FIELDS:
            return f"Error: invalid cron '{cron}'. Need 5 fields: minute hour day month weekday"

        try:
            await self._db.execute(
                "INSERT INTO schedules (name, cron, chat_id, channel, prompt) VALUES (?, ?, ?, ?, ?)",
                (name, cron, chat_id, channel, prompt),
            )
            await self._db.commit()
            nxt = next_cron_fire(cron, datetime.now())
            logger.info("Schedule created: name=%s, cron=%s, next=%s", name, cron, nxt)
            self._wake_event.set()  # recalculate sleep
            return f"Schedule '{name}' created. Cron: {cron}. Next fire: {nxt:%Y-%m-%d %H:%M}"
        except aiosqlite.IntegrityError:
            return f"Error: schedule '{name}' already exists."

    async def delete_schedule(self, name: str) -> str:
        assert self._db is not None
        cursor = await self._db.execute("DELETE FROM schedules WHERE name = ?", (name,))
        await self._db.commit()
        if cursor.rowcount > 0:
            self._wake_event.set()
            return f"Schedule '{name}' deleted."
        return f"Schedule '{name}' not found."

    async def list_schedules(self) -> str:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT name, cron, chat_id, channel, prompt, enabled FROM schedules ORDER BY name"
        )
        rows = await cursor.fetchall()
        if not rows:
            return "No scheduled tasks."

        lines: list[str] = []
        now = datetime.now()
        for name, cron, chat_id, channel, prompt, enabled in rows:
            status = "ON" if enabled else "OFF"
            nxt = next_cron_fire(cron, now) if enabled else None
            next_str = f" next={nxt:%H:%M}" if nxt else ""
            lines.append(f"- [{status}] {name}: {cron}{next_str} ({channel}) -> {prompt[:50]}")
        return "\n".join(lines)

    async def toggle_schedule(self, name: str, enabled: bool) -> str:
        assert self._db is not None
        cursor = await self._db.execute("UPDATE schedules SET enabled = ? WHERE name = ?", (1 if enabled else 0, name))
        await self._db.commit()
        if cursor.rowcount > 0:
            self._wake_event.set()
            return f"Schedule '{name}' {'enabled' if enabled else 'disabled'}."
        return f"Schedule '{name}' not found."

    # --- Runtime: wait-sleep model ---

    async def start(self, handler: ScheduleHandler) -> None:
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop(handler))
        logger.info("Scheduler started (wait-sleep model)")

    async def stop(self) -> None:
        self._running = False
        self._wake_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        await self.close()

    async def _get_next_fire(self) -> tuple[float, list[dict[str, Any]]]:
        """Calculate seconds until next fire and which tasks fire then."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT name, cron, chat_id, channel, prompt FROM schedules WHERE enabled = 1")
        rows = await cursor.fetchall()

        if not rows:
            return 3600.0, []  # no tasks, sleep 1 hour

        now = datetime.now()
        earliest = now + timedelta(days=366)
        earliest_tasks: list[dict[str, Any]] = []

        for name, cron, chat_id, channel, prompt in rows:
            nxt = next_cron_fire(cron, now)
            if nxt < earliest:
                earliest = nxt
                earliest_tasks = [{"name": name, "chat_id": chat_id, "channel": channel, "prompt": prompt}]
            elif nxt == earliest:
                earliest_tasks.append({"name": name, "chat_id": chat_id, "channel": channel, "prompt": prompt})

        delay = max(0.0, (earliest - now).total_seconds())
        return delay, earliest_tasks

    async def _run_loop(self, handler: ScheduleHandler) -> None:
        while self._running:
            delay, tasks = await self._get_next_fire()

            if delay > 0:
                logger.debug("Scheduler sleeping %.0fs until next fire", delay)
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=delay)
                    # Woken early (schedule changed), recalculate
                    continue
                except asyncio.TimeoutError:
                    pass  # timer expired, fire the tasks

            # Execute due tasks
            for task in tasks:
                logger.info("Firing scheduled task: %s", task["name"])
                try:
                    response = await handler(task["chat_id"], task["prompt"], task["channel"])
                    logger.info("Scheduled task %s done: %s", task["name"], response[:100])
                except Exception:
                    logger.exception("Scheduled task %s failed", task["name"])

            # Brief sleep to avoid re-firing the same minute
            await asyncio.sleep(60)
