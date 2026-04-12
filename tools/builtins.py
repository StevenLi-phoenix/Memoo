"""Built-in tools for the Memoo agent.

Sandbox principle: FULL power inside sandbox/{uuid}/, ZERO write access outside.
Uses macOS sandbox-exec for OS-level enforcement. No fallback — if sandbox-exec
is unavailable, the application refuses to start.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import sys
import uuid
from typing import Any

from core.tools import ToolRegistry

logger = logging.getLogger(__name__)

MAX_OUTPUT = 10000
DEFAULT_EXEC_TIMEOUT = 300  # 5 min default, agent can override per-call

# macOS sandbox-exec profile (SBPL):
# - Read: anywhere (system libs, binaries, data)
# - Write: ONLY inside {sandbox_dir} and /dev
# - Execute: any program
# - Network: allowed
_SANDBOX_PROFILE = """
(version 1)
(deny default)
(allow file-read*)
(allow file-write* (subpath "{sandbox_dir}") (subpath "/dev"))
(allow process-exec*)
(allow process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow mach-register)
(allow ipc-posix*)
(allow signal)
(allow network*)
"""


def _check_sandbox_exec() -> None:
    """Verify sandbox-exec is available. Exit if not."""
    if platform.system() != "Darwin":
        logger.error("OS-level sandbox requires macOS (sandbox-exec). Current: %s", platform.system())
        sys.exit(1)
    if not shutil.which("sandbox-exec"):
        logger.error("sandbox-exec not found in PATH. Cannot start without OS-level sandbox.")
        sys.exit(1)

    # Smoke test
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", "(version 1)(allow default)", "echo", "ok"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.error("sandbox-exec smoke test failed: %s", result.stderr)
            sys.exit(1)
    except Exception as e:
        logger.error("sandbox-exec smoke test error: %s", e)
        sys.exit(1)

    logger.info("OS-level sandbox (sandbox-exec) verified")


def _check_sandbox_path(path: str, sandbox_dir: str) -> str:
    """Resolve and validate path stays within sandbox."""
    abs_sandbox = os.path.realpath(sandbox_dir)
    abs_path = os.path.realpath(os.path.join(abs_sandbox, path))
    if not abs_path.startswith(abs_sandbox + os.sep) and abs_path != abs_sandbox:
        raise PermissionError(f"Path escapes sandbox: {path}")
    return abs_path


def _safe_env(exec_dir: str) -> dict[str, str]:
    """Clean environment: no secrets, sandbox as HOME."""
    safe = {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "HOME": exec_dir,
        "TMPDIR": exec_dir,
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": exec_dir,
        "TERM": "dumb",
    }
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(key)
        if val:
            safe[key] = val
    return safe


def _make_profile(exec_dir: str) -> str:
    """Build sandbox profile with the real (resolved) sandbox path."""
    real_dir = os.path.realpath(exec_dir)
    return _SANDBOX_PROFILE.format(sandbox_dir=real_dir)


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n...(truncated)"
    return text


def register(registry: ToolRegistry, **deps: Any) -> None:
    """Register built-in tools. Requires macOS sandbox-exec."""
    from core.tools import get_context

    base_sandbox = os.path.realpath(deps.get("sandbox_dir", "./sandbox"))
    os.makedirs(base_sandbox, exist_ok=True)

    _check_sandbox_exec()

    def _get_session_dir() -> str:
        """Get or create the sandbox directory for the current agent session.

        Each session (chat_id) gets a persistent sandbox/{session_id}/ directory.
        Files persist across tool calls within the same session.
        Sessions are isolated from each other at the OS level.
        """
        ctx = get_context()
        session_id = ctx.get("chat_id", uuid.uuid4().hex[:12])
        # Sanitize session_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id))
        session_dir = os.path.join(base_sandbox, safe_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    @registry.tool
    async def run_code(code: str, language: str = "python", timeout: int = 0) -> str:
        """Execute code in an OS-level sandbox. Full power inside, no write access outside.

        Each session has its own isolated directory — files persist across calls.

        Args:
            code: The code to execute.
            language: Programming language (python, bash).
            timeout: Max execution time in seconds. 0 uses default (from config).
        """
        if language not in ("python", "bash"):
            return f"Unsupported language: {language}. Supported: python, bash"

        session_dir = _get_session_dir()
        profile = _make_profile(session_dir)

        exec_timeout = timeout if timeout > 0 else DEFAULT_EXEC_TIMEOUT
        logger.info(
            "run_code: lang=%s, session_dir=%s, timeout=%ds, len=%d", language, session_dir, exec_timeout, len(code)
        )

        if language == "python":
            return await _run_sandboxed(
                ["sandbox-exec", "-p", profile, "python3", "-u", "-c", code],
                session_dir,
                exec_timeout,
            )
        else:
            return await _run_sandboxed(
                ["sandbox-exec", "-p", profile, "bash", "-c", f"set -euo pipefail\n{code}"],
                session_dir,
                exec_timeout,
            )

    async def _run_sandboxed(cmd: list[str], exec_dir: str, timeout_s: int = DEFAULT_EXEC_TIMEOUT) -> str:
        """Execute a command inside sandbox-exec."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=exec_dir,
                env=_safe_env(exec_dir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            output = stdout.decode(errors="replace")
            if stderr:
                err_text = stderr.decode(errors="replace")
                if err_text.strip():
                    output += "\n[stderr]\n" + err_text
            return _truncate(output) or "(no output)"
        except asyncio.TimeoutError:
            proc.kill()  # type: ignore[possibly-undefined]
            return f"Error: timed out ({timeout_s}s)"
        except Exception as e:
            return f"Error: {e}"

    @registry.tool
    async def read_file(path: str) -> str:
        """Read a file from this session's sandbox.

        Args:
            path: Path relative to session sandbox.
        """
        session_dir = _get_session_dir()
        try:
            abs_path = _check_sandbox_path(path, session_dir)
        except PermissionError as e:
            return f"Error: {e}"

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                return _truncate(f.read()) or "(empty file)"
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
        session_dir = _get_session_dir()
        try:
            abs_path = _check_sandbox_path(path, session_dir)
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
