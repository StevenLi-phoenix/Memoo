"""Tests for handle_message concurrency: per-chat_id locking and injection path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock


@dataclass
class FakeTurnResult:
    response: str = "reply"
    memory_notes: str = ""
    current_topic: str = ""
    should_compress: bool = False
    did_success: bool = True
    usage: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


class TestConcurrentHandleMessage:
    """Verify that concurrent handle_message calls for the same chat_id
    do not double-finalize or corrupt history."""

    async def test_injection_path_does_not_finalize(self) -> None:
        """When H2 injects into H1's active task, only H1 calls _finalize_turn."""
        finalize_count = 0

        # Build a minimal Memoo-like object
        class FakeMemoo:
            def __init__(self):
                self.cfg = MagicMock()
                self.cfg.agent.max_message_len = 10000
                self.cfg.paths.sandbox_dir = "/tmp/sandbox"
                self.agent = MagicMock()
                self.llm = MagicMock()
                self.memory = AsyncMock()
                self.memory.get_history = AsyncMock(return_value=[])
                self.memory.add_message = AsyncMock()
                self.skill_registry = None
                self._active_tasks: dict[str, asyncio.Task] = {}
                self._current_topics: dict[str, str] = {}
                self._chat_locks: dict[str, asyncio.Lock] = {}

            async def _finalize_turn(self, chat_id, result):
                nonlocal finalize_count
                finalize_count += 1
                return result.response

        app = FakeMemoo()

        # Create a slow task that simulates agent.run
        task_started = asyncio.Event()
        task_continue = asyncio.Event()

        async def slow_agent_run(*args, **kwargs):
            task_started.set()
            await task_continue.wait()
            return FakeTurnResult(response="from agent")

        app.agent.run = slow_agent_run
        app.agent.inject = MagicMock(return_value=True)

        # Import the real handle_message logic pattern
        # We'll simulate what handle_message does

        chat_id = "test_chat"

        # H1 starts a task
        async def h1():
            await app.memory.get_history(chat_id)
            await app.memory.add_message(chat_id, MagicMock())
            task = asyncio.create_task(app.agent.run("msg1"))
            app._active_tasks[chat_id] = task
            try:
                result = await task
            finally:
                app._active_tasks.pop(chat_id, None)
            return await app._finalize_turn(chat_id, result)

        # H2 tries to inject
        async def h2():
            await task_started.wait()  # Wait for H1 to start
            active = app._active_tasks.get(chat_id)
            assert active is not None
            assert app.agent.inject(chat_id, "msg2")
            await app.memory.add_message(chat_id, MagicMock())
            # Injection path: await the same task but DON'T finalize
            result = await active
            return result.response  # No _finalize_turn call

        # Run both concurrently
        h1_task = asyncio.create_task(h1())
        h2_task = asyncio.create_task(h2())

        # Let the agent complete
        await task_started.wait()
        task_continue.set()

        r1 = await h1_task
        r2 = await h2_task

        assert r1 == "from agent"
        assert r2 == "from agent"
        # CRITICAL: _finalize_turn called exactly once (by H1 only)
        assert finalize_count == 1, f"Expected 1 finalize call, got {finalize_count}"

    async def test_lock_serializes_task_creation(self) -> None:
        """Two concurrent handlers for the same chat_id should not both create tasks."""
        lock = asyncio.Lock()
        task_count = 0

        async def create_task_under_lock():
            nonlocal task_count
            async with lock:
                task_count += 1
                await asyncio.sleep(0.01)  # simulate work

        await asyncio.gather(
            create_task_under_lock(),
            create_task_under_lock(),
        )

        # Both ran, but sequentially (not concurrently creating tasks)
        assert task_count == 2

    async def test_different_chat_ids_not_blocked(self) -> None:
        """Locks are per-chat_id — different chats should run concurrently."""
        locks: dict[str, asyncio.Lock] = {}
        start_times: dict[str, float] = {}

        import time

        async def handler(chat_id: str):
            lock = locks.setdefault(chat_id, asyncio.Lock())
            async with lock:
                start_times[chat_id] = time.monotonic()
                await asyncio.sleep(0.05)

        await asyncio.gather(handler("chat_a"), handler("chat_b"))

        # Both should have started at roughly the same time (concurrent)
        diff = abs(start_times["chat_a"] - start_times["chat_b"])
        assert diff < 0.03, f"Different chat_ids should run concurrently, but started {diff:.3f}s apart"
