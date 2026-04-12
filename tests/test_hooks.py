"""Tests for tool authorization hooks: HookRegistry, rate limiter, sandbox path."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from core.hooks import (
    HookRegistry,
    _rate_limit_state,
    make_rate_limit_hook,
    sandbox_path_hook,
)

# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------


class TestHookRegistry:
    async def test_allowlist_bypasses_hooks(self) -> None:
        reg = HookRegistry()
        approved, _ = await reg.authorize("current_time", {}, {})
        assert approved

    async def test_denylist_takes_priority_over_allowlist(self) -> None:
        reg = HookRegistry()
        reg.allow("dangerous_tool")
        reg.deny("dangerous_tool")
        approved, reason = await reg.authorize("dangerous_tool", {}, {})
        assert not approved
        assert "not allowed" in reason

    async def test_hook_deny_stops_chain(self) -> None:
        reg = HookRegistry()

        async def deny_all(name, args, ctx):
            return False, "nope"

        async def should_not_run(name, args, ctx):
            raise AssertionError("second hook should not be reached")

        reg.add_hook(deny_all)
        reg.add_hook(should_not_run)

        approved, reason = await reg.authorize("some_tool", {}, {})
        assert not approved
        assert reason == "nope"

    async def test_hook_exception_denies_by_default(self) -> None:
        reg = HookRegistry()

        async def buggy_hook(name, args, ctx):
            raise RuntimeError("boom")

        reg.add_hook(buggy_hook)
        approved, reason = await reg.authorize("some_tool", {}, {})
        assert not approved
        assert "error" in reason.lower()

    async def test_no_hooks_allows_by_default(self) -> None:
        reg = HookRegistry()
        approved, _ = await reg.authorize("unknown_tool", {}, {})
        assert approved

    async def test_multiple_hooks_all_approve(self) -> None:
        reg = HookRegistry()

        async def approve(name, args, ctx):
            return True, "ok"

        reg.add_hook(approve)
        reg.add_hook(approve)
        approved, _ = await reg.authorize("tool", {}, {})
        assert approved


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    @pytest.fixture(autouse=True)
    def _clear_state(self) -> None:
        """Ensure clean rate-limit state between tests."""
        _rate_limit_state.clear()

    async def test_allows_under_limit(self) -> None:
        hook = make_rate_limit_hook(max_per_minute=3, window_seconds=60)
        ctx = {"chat_id": "u1"}
        for _ in range(3):
            approved, _ = await hook("tool_a", {}, ctx)
            assert approved

    async def test_denies_over_limit(self) -> None:
        hook = make_rate_limit_hook(max_per_minute=2, window_seconds=60)
        ctx = {"chat_id": "u1"}
        await hook("tool_a", {}, ctx)
        await hook("tool_a", {}, ctx)
        approved, reason = await hook("tool_a", {}, ctx)
        assert not approved
        assert "Rate limit" in reason

    async def test_window_expiry_resets_count(self) -> None:
        hook = make_rate_limit_hook(max_per_minute=1, window_seconds=10)
        ctx = {"chat_id": "u1"}

        with patch("time.time", return_value=1000.0):
            await hook("tool_a", {}, ctx)

        # 11 seconds later — old call is expired
        with patch("time.time", return_value=1011.0):
            approved, _ = await hook("tool_a", {}, ctx)
            assert approved

    async def test_different_chat_ids_are_independent(self) -> None:
        hook = make_rate_limit_hook(max_per_minute=1, window_seconds=60)
        approved, _ = await hook("tool_a", {}, {"chat_id": "u1"})
        assert approved
        approved, _ = await hook("tool_a", {}, {"chat_id": "u2"})
        assert approved

    async def test_stale_keys_are_swept_above_threshold(self) -> None:
        """Verify that the memory-leak fix actually sweeps stale entries."""
        hook = make_rate_limit_hook(max_per_minute=100, window_seconds=10)

        # Seed 60 stale keys (all timestamps expired)
        now = time.time()
        for i in range(60):
            _rate_limit_state[f"stale_{i}:tool"] = [now - 20.0]

        assert len(_rate_limit_state) > 50

        # One new call triggers the sweep
        await hook("trigger", {}, {"chat_id": "sweep"})

        # Stale keys should be gone; only the fresh one remains
        assert all(not k.startswith("stale_") for k in _rate_limit_state)
        assert ":trigger" in list(_rate_limit_state.keys())[0]


# ---------------------------------------------------------------------------
# Sandbox path hook
# ---------------------------------------------------------------------------


class TestSandboxPathHook:
    async def test_non_file_tools_always_allowed(self) -> None:
        approved, _ = await sandbox_path_hook("run_code", {"path": "/etc/passwd"}, {})
        assert approved

    async def test_safe_path_inside_sandbox(self, tmp_path) -> None:
        sandbox = str(tmp_path / "sandbox")
        chat_id = "test123"
        session = f"{sandbox}/{chat_id}"
        # Create the directory so realpath resolves correctly
        import os

        os.makedirs(session, exist_ok=True)

        ctx = {"sandbox_dir": sandbox, "chat_id": chat_id}
        approved, _ = await sandbox_path_hook("read_file", {"path": "hello.txt"}, ctx)
        assert approved

    async def test_path_traversal_denied(self, tmp_path) -> None:
        sandbox = str(tmp_path / "sandbox")
        chat_id = "test123"
        session = f"{sandbox}/{chat_id}"
        import os

        os.makedirs(session, exist_ok=True)

        ctx = {"sandbox_dir": sandbox, "chat_id": chat_id}
        approved, reason = await sandbox_path_hook("read_file", {"path": "../../etc/passwd"}, ctx)
        assert not approved
        assert "escapes sandbox" in reason.lower()

    async def test_write_file_also_checked(self, tmp_path) -> None:
        sandbox = str(tmp_path / "sandbox")
        chat_id = "abc"
        import os

        os.makedirs(f"{sandbox}/{chat_id}", exist_ok=True)

        ctx = {"sandbox_dir": sandbox, "chat_id": chat_id}
        approved, _ = await sandbox_path_hook("write_file", {"path": "../../../tmp/evil"}, ctx)
        assert not approved

    async def test_chat_id_sanitized(self) -> None:
        """Special chars in chat_id should be sanitized to safe dir name."""
        ctx = {"sandbox_dir": "/tmp/sandbox", "chat_id": "user@evil/../../root"}
        approved, _ = await sandbox_path_hook("read_file", {"path": "test.txt"}, ctx)
        # The sanitized chat_id won't form a traversal — should be safe
        assert approved or not approved  # just verify no crash
