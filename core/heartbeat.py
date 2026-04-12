"""Heartbeat — periodically wakes the agent for background tasks.

Tasks are defined as markdown files in the heartbeat/ directory.
Each .md file is a heartbeat task with optional frontmatter for interval config.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.utils import parse_frontmatter

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 3600  # 1 hour

HeartbeatHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


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
        meta, body = parse_frontmatter(content)
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
        from core.crash import report_crash

        # Initialize last_run so first fire waits a full interval
        now = time.time()
        for task in self._tasks:
            if task.last_run == 0.0:
                task.last_run = now

        # Sleep until first task is due
        sleep_time = self._next_sleep(now)
        await asyncio.sleep(sleep_time)

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
                except Exception as e:
                    report_crash(e, context={"task": task.name}, component="heartbeat")

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
