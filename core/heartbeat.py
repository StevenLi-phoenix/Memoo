"""Heartbeat — periodically wakes the agent for background tasks.

Tasks are defined as markdown files in the heartbeat/ directory.
Each .md file is a heartbeat task with optional frontmatter for interval config.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 3600  # 1 hour

HeartbeatHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-like frontmatter from markdown. Returns (metadata, body)."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    meta: dict[str, Any] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Parse int values
            try:
                meta[key] = int(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    meta[key] = val.lower() == "true"
                else:
                    meta[key] = val

    body = content[match.end() :]
    return meta, body


class HeartbeatTask:
    """A single heartbeat task loaded from a markdown file."""

    def __init__(self, name: str, prompt: str, interval: int, enabled: bool = True) -> None:
        self.name = name
        self.prompt = prompt
        self.interval = interval
        self.enabled = enabled
        self.last_run: float = 0.0

    @classmethod
    def from_file(cls, path: Path) -> HeartbeatTask:
        content = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(content)
        return cls(
            name=meta.get("name", path.stem),
            prompt=body.strip(),
            interval=meta.get("interval", DEFAULT_INTERVAL),
            enabled=meta.get("enabled", True),
        )

    def is_due(self, now: float) -> bool:
        return self.enabled and (now - self.last_run) >= self.interval


class Heartbeat:
    """Periodic agent wake-up. Tasks defined as markdown files in heartbeat/ directory."""

    def __init__(self, heartbeat_dir: str = "./heartbeat", default_interval: int = DEFAULT_INTERVAL) -> None:
        self._dir = Path(heartbeat_dir)
        self._default_interval = default_interval
        self._tasks: list[HeartbeatTask] = []
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

    def load_tasks(self) -> None:
        """Load all .md files from the heartbeat directory."""
        if not self._dir.exists():
            logger.warning("Heartbeat directory not found: %s", self._dir)
            return

        self._tasks.clear()
        for md_file in sorted(self._dir.glob("*.md")):
            task = HeartbeatTask.from_file(md_file)
            self._tasks.append(task)
            logger.info(
                "Heartbeat task loaded: %s (interval=%ds, enabled=%s)",
                task.name,
                task.interval,
                task.enabled,
            )

        logger.info("Loaded %d heartbeat tasks from %s", len(self._tasks), self._dir)

    async def start(self, handler: HeartbeatHandler) -> None:
        self.load_tasks()
        if not self._tasks:
            logger.info("No heartbeat tasks, skipping")
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop(handler))
        logger.info("Heartbeat started")

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat stopped")

    async def _run_loop(self, handler: HeartbeatHandler) -> None:
        # Wait for app to fully initialize
        await asyncio.sleep(10)

        while self._running:
            now = time.time()

            for task in self._tasks:
                if not task.is_due(now):
                    continue

                task.last_run = now
                context = {
                    "source": "heartbeat",
                    "task_name": task.name,
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info("Heartbeat firing: %s", task.name)
                try:
                    response = await handler(task.prompt, context)
                    if response.strip().lower() != "all clear":
                        logger.info("Heartbeat %s: %s", task.name, response[:200])
                except Exception:
                    logger.exception("Heartbeat task %s failed", task.name)

            # Sleep until the soonest next-due task
            sleep_time = self._next_sleep(now)
            await asyncio.sleep(sleep_time)

    def _next_sleep(self, now: float) -> float:
        """Calculate optimal sleep time until next task is due."""
        if not self._tasks:
            return self._default_interval

        soonest = float("inf")
        for task in self._tasks:
            if not task.enabled:
                continue
            remaining = task.interval - (now - task.last_run)
            soonest = min(soonest, max(remaining, 1.0))

        return min(soonest, self._default_interval)
