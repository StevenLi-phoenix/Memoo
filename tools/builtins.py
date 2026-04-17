"""Built-in tools for the Memoo agent.

Delegates OS-level sandboxing to core.sandbox.Sandbox.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.sandbox import Sandbox
from core.tools import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, **deps: Any) -> None:
    """Register built-in tools."""
    from core.tools import get_context

    cfg = deps.get("config")
    timeout = cfg.sandbox.timeout if cfg else 300
    max_output = cfg.sandbox.max_output if cfg else 10_000
    env = cfg.sandbox.env if cfg else {}
    env_from_cmd = cfg.sandbox.env_from_cmd if cfg else {}
    env_passthrough = cfg.sandbox.env_passthrough if cfg else []

    sb = Sandbox(
        base_dir=deps.get("sandbox_dir", "./sandbox"),
        timeout=timeout,
        max_output=max_output,
        env=env,
        env_from_cmd=env_from_cmd,
        env_passthrough=env_passthrough,
    )

    def _session_id() -> str:
        import uuid

        return get_context().get("chat_id", uuid.uuid4().hex[:12])

    @registry.tool
    async def run_code(code: str, language: str = "python", timeout: int = 0) -> str:
        """Execute code in an OS-level sandbox. Full power inside, no write access outside.

        Each session has its own isolated directory — files persist across calls.

        Args:
            code: The code to execute.
            language: Programming language (python, bash).
            timeout: Max execution time in seconds. 0 uses default (from config).
        """
        ctx = get_context()
        sid = _session_id()

        logger.info(
            "run_code: lang=%s, backend=%s, session=%s, timeout=%d, len=%d",
            language,
            sb.backend,
            sid,
            timeout or sb._timeout,
            len(code),
        )

        return await sb.exec_code(
            code,
            language,
            session_id=sid,
            readonly=ctx.get("_sandbox_readonly", False),
            no_network=ctx.get("_sandbox_no_network", False),
            timeout=timeout,
        )

    @registry.tool
    async def read_file(path: str) -> str:
        """Read a file from this session's sandbox.

        Args:
            path: Path relative to session sandbox.
        """
        try:
            abs_path = sb.check_path(path, session_id=_session_id())
        except PermissionError as e:
            return f"Error: {e}"

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
                if len(content) > max_output:
                    content = content[:max_output] + "\n...(truncated)"
                return content or "(empty file)"
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as e:
            return f"Error: {e}"

    @registry.tool
    async def write_file(path: str, content: str) -> str:
        """Write a file to this session's sandbox.

        Args:
            path: Path relative to session sandbox.
            content: Content to write.
        """
        if get_context().get("_sandbox_readonly"):
            return "Error: write denied — sandbox is in readonly mode"

        try:
            abs_path = sb.check_path(path, session_id=_session_id())
        except PermissionError as e:
            return f"Error: {e}"

        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    @registry.tool
    def current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
