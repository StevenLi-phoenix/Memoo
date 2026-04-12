"""Tool calling authorization hooks.

Before any tool is executed, hooks can approve, deny, or modify the call.
This enables per-user permissions, dangerous-tool confirmation, rate limiting, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Hook signature: (tool_name, arguments, context) -> (approved, reason)
# context contains chat_id, user_id, etc.
AuthHook = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[tuple[bool, str]]]


class HookRegistry:
    """Registry for tool authorization hooks.

    Hooks are checked in order. First deny wins.
    """

    def __init__(self) -> None:
        self._hooks: list[AuthHook] = []
        # Tools that are always allowed without hook checks
        self._allowlist: set[str] = {"current_time"}
        # Tools that always require explicit approval
        self._denylist: set[str] = set()

    def add_hook(self, hook: AuthHook) -> None:
        """Register an authorization hook."""
        self._hooks.append(hook)

    def allow(self, *tool_names: str) -> None:
        """Add tools to the allowlist (skip hook checks)."""
        self._allowlist.update(tool_names)

    def deny(self, *tool_names: str) -> None:
        """Add tools to the denylist (always denied)."""
        self._denylist.update(tool_names)

    async def authorize(self, tool_name: str, arguments: dict[str, Any], context: dict[str, Any]) -> tuple[bool, str]:
        """Check if a tool call is authorized.

        Returns (approved, reason).
        """
        # Denylist takes priority
        if tool_name in self._denylist:
            logger.warning("Tool %s denied by denylist", tool_name)
            return False, f"Tool '{tool_name}' is not allowed"

        # Allowlist bypasses hooks
        if tool_name in self._allowlist:
            return True, "allowed"

        # Run hooks — first deny wins
        for hook in self._hooks:
            try:
                approved, reason = await hook(tool_name, arguments, context)
                if not approved:
                    logger.warning("Tool %s denied by hook: %s", tool_name, reason)
                    return False, reason
            except Exception:
                logger.exception("Hook error for tool %s, denying by default", tool_name)
                return False, "Authorization hook error"

        return True, "allowed"


# --- Built-in hooks ---


# Module-level state for rate limiter (persists across requests)
_rate_limit_state: dict[str, list[float]] = {}


async def rate_limit_hook(tool_name: str, arguments: dict[str, Any], context: dict[str, Any]) -> tuple[bool, str]:
    """Simple rate limiting hook. Tracks calls per chat per minute."""
    import time

    key = f"{context.get('chat_id', '')}:{tool_name}"
    now = time.time()

    calls = _rate_limit_state.setdefault(key, [])
    calls[:] = [t for t in calls if now - t < 60]

    max_per_minute = 30
    if len(calls) >= max_per_minute:
        return False, f"Rate limit exceeded: {tool_name} ({max_per_minute}/min)"

    calls.append(now)
    return True, "allowed"


async def sandbox_path_hook(tool_name: str, arguments: dict[str, Any], context: dict[str, Any]) -> tuple[bool, str]:
    """Deny file tools that access paths outside the session's sandbox."""
    import os

    if tool_name not in ("read_file", "write_file"):
        return True, "allowed"

    path = arguments.get("path", "")
    sandbox_dir = context.get("sandbox_dir", "./sandbox")
    chat_id = context.get("chat_id", "")
    # Check against the per-session sandbox directory
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(chat_id)) if chat_id else ""
    session_dir = os.path.join(sandbox_dir, safe_id) if safe_id else sandbox_dir
    abs_sandbox = os.path.realpath(session_dir)
    abs_path = os.path.realpath(os.path.join(abs_sandbox, path))

    if not abs_path.startswith(abs_sandbox + os.sep) and abs_path != abs_sandbox:
        return False, f"Path escapes sandbox: {path}"

    return True, "allowed"
