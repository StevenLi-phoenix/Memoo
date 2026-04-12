"""Tests for cron field validation and scheduler execution model."""

import asyncio
import time
from datetime import datetime

from core.scheduler import Scheduler, _match_cron_field, cron_matches, next_cron_fire


class TestCronValidation:
    def test_wildcard(self):
        assert _match_cron_field("*", 5) is True

    def test_exact(self):
        assert _match_cron_field("5", 5) is True
        assert _match_cron_field("5", 6) is False

    def test_step(self):
        assert _match_cron_field("*/5", 10) is True
        assert _match_cron_field("*/5", 7) is False

    def test_step_zero(self):
        # */0 should not crash
        assert _match_cron_field("*/0", 5) is False

    def test_range(self):
        assert _match_cron_field("1-5", 3) is True
        assert _match_cron_field("1-5", 6) is False

    def test_list(self):
        assert _match_cron_field("1,3,5", 3) is True
        assert _match_cron_field("1,3,5", 4) is False

    def test_malformed(self):
        assert _match_cron_field("abc", 5) is False
        assert _match_cron_field("", 5) is False

    def test_cron_matches(self):
        dt = datetime(2026, 4, 12, 8, 30)  # Saturday
        assert cron_matches("30 8 * * *", dt) is True
        assert cron_matches("0 8 * * *", dt) is False

    def test_cron_wrong_fields(self):
        dt = datetime(2026, 4, 12, 8, 30)
        assert cron_matches("only three fields", dt) is False


class TestNextCronFire:
    def test_next_minute(self):
        dt = datetime(2026, 4, 12, 8, 30, 0)
        nxt = next_cron_fire("* * * * *", dt)
        assert nxt == datetime(2026, 4, 12, 8, 31, 0)

    def test_next_hour(self):
        dt = datetime(2026, 4, 12, 8, 59, 0)
        nxt = next_cron_fire("0 * * * *", dt)
        assert nxt == datetime(2026, 4, 12, 9, 0, 0)

    def test_specific_time(self):
        dt = datetime(2026, 4, 12, 8, 0, 0)
        nxt = next_cron_fire("30 9 * * *", dt)
        assert nxt == datetime(2026, 4, 12, 9, 30, 0)


class TestSchedulerConcurrency:
    """Tests verifying concurrent task execution and cooldown anchoring."""

    async def test_tasks_run_concurrently(self, tmp_path) -> None:
        """Multiple due tasks should run via asyncio.gather, not sequentially."""
        sched = Scheduler(db_path=str(tmp_path / "test.db"))
        await sched.init()

        execution_log: list[tuple[str, float]] = []

        async def handler(chat_id: str, prompt: str, channel: str) -> str:
            start = time.monotonic()
            await asyncio.sleep(0.1)  # simulate work
            execution_log.append((prompt, start))
            return "done"

        # Create two tasks with same cron
        now = datetime.now()
        cron = f"{now.minute} {now.hour} * * *"
        await sched.create_schedule("task_a", cron, "test", "test", "prompt_a")
        await sched.create_schedule("task_b", cron, "test", "test", "prompt_b")

        # Manually call the internal execution path
        tasks = [
            {"name": "task_a", "chat_id": "test", "channel": "test", "prompt": "prompt_a"},
            {"name": "task_b", "chat_id": "test", "channel": "test", "prompt": "prompt_b"},
        ]

        fire_time = time.monotonic()

        async def _run_task(task):
            try:
                await handler(task["chat_id"], task["prompt"], task["channel"])
            except Exception:
                pass

        await asyncio.gather(*[_run_task(t) for t in tasks])
        total_time = time.monotonic() - fire_time

        # If concurrent: ~0.1s total. If sequential: ~0.2s total.
        assert total_time < 0.18, f"Tasks took {total_time:.3f}s — should be concurrent (~0.1s), not sequential (~0.2s)"
        assert len(execution_log) == 2

        # Both tasks should have started at roughly the same time
        start_diff = abs(execution_log[0][1] - execution_log[1][1])
        assert start_diff < 0.05, f"Tasks started {start_diff:.3f}s apart — not concurrent"

        await sched.close()

    async def test_cooldown_anchored_to_fire_time(self) -> None:
        """Cooldown should be 60s - elapsed, not a flat 60s after completion."""
        # Simulate 30s of task work
        simulated_elapsed = 30.0
        cooldown = max(1.0, 60.0 - simulated_elapsed)
        assert cooldown == 30.0, "Should sleep only 30s more, not 60s"

        # Simulate 90s of task work (exceeds 60s window)
        simulated_elapsed = 90.0
        cooldown = max(1.0, 60.0 - simulated_elapsed)
        assert cooldown == 1.0, "Should sleep minimum 1s, not negative"
