"""Cross-platform OS-level sandbox for code execution.

Backends:
  - macOS: sandbox-exec with dynamically generated SBPL profiles
  - Linux: bubblewrap (bwrap) with namespace isolation

Usage:
    sb = Sandbox(base_dir="./sandbox", timeout=300, max_output=10000)
    result = await sb.exec_code("print(1+1)", "python", session_id="tui")
    abs_path = sb.check_path("file.txt", session_id="tui")
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


class Sandbox:
    """Cross-platform OS-level sandbox.

    Wraps macOS sandbox-exec and Linux bubblewrap behind a unified API.
    Each session gets an isolated directory under base_dir.
    """

    def __init__(
        self,
        base_dir: str = "./sandbox",
        timeout: int = 300,
        max_output: int = 10_000,
        env: dict[str, str] | None = None,
        env_from_cmd: dict[str, str] | None = None,
        env_passthrough: list[str] | None = None,
    ) -> None:
        self._base_dir = os.path.realpath(base_dir)
        self._timeout = timeout
        self._max_output = max_output
        self._backend = ""
        self._extra_env_literal = dict(env or {})
        self._extra_env_cmd = dict(env_from_cmd or {})
        self._extra_env_passthrough = list(env_passthrough or [])
        self._extra_env_cache: dict[str, str] = {}

        os.makedirs(self._base_dir, exist_ok=True)
        self._detect_backend()
        self._resolve_extra_env()

    @property
    def backend(self) -> str:
        return self._backend

    # ── Public API ──────────────────────────────────────────────────

    async def exec_code(
        self,
        code: str,
        language: str,
        session_id: str,
        *,
        readonly: bool = False,
        no_network: bool = False,
        timeout: int = 0,
    ) -> str:
        """Execute code in the sandbox. High-level convenience method.

        Args:
            code: Source code to execute.
            language: "python" or "bash".
            session_id: Session identifier (determines sandbox directory).
            readonly: Deny file writes.
            no_network: Deny network access.
            timeout: Override default timeout (0 = use default).
        """
        if language == "python":
            cmd = ["python3", "-u", "-c", code]
        elif language == "bash":
            cmd = ["bash", "-c", f"set -euo pipefail\n{code}"]
        else:
            return f"Unsupported language: {language}. Supported: python, bash"

        return await self.run(
            cmd,
            session_id=session_id,
            readonly=readonly,
            no_network=no_network,
            timeout=timeout or self._timeout,
        )

    async def run(
        self,
        inner_cmd: list[str],
        session_id: str,
        *,
        readonly: bool = False,
        no_network: bool = False,
        timeout: int = 0,
    ) -> str:
        """Execute an arbitrary command inside the OS sandbox.

        Args:
            inner_cmd: Command + args (e.g. ["python3", "-c", "print(1)"]).
            session_id: Session identifier (determines sandbox directory).
            readonly: Deny file writes.
            no_network: Deny network access.
            timeout: Override default timeout (0 = use default).
        """
        exec_dir = self.session_dir(session_id)
        effective_timeout = timeout or self._timeout
        cmd = self._build_cmd(exec_dir, inner_cmd, readonly=readonly, no_network=no_network)

        # bwrap sets its own env/cwd via flags; darwin needs them via subprocess
        env = _safe_env(exec_dir, self._extra_env_cache) if self._backend == "darwin" else None
        cwd = exec_dir if self._backend == "darwin" else None

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
            output = stdout.decode(errors="replace")
            if stderr:
                err_text = stderr.decode(errors="replace")
                if err_text.strip():
                    output += "\n[stderr]\n" + err_text
            return self._truncate(output) or "(no output)"
        except asyncio.TimeoutError:
            proc.kill()  # type: ignore[possibly-undefined]
            return f"Error: timed out ({effective_timeout}s)"
        except Exception as e:
            return f"Error: {e}"

    def session_dir(self, session_id: str) -> str:
        """Get or create the sandbox directory for a session. Sanitizes the ID."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id))
        d = os.path.join(self._base_dir, safe_id)
        os.makedirs(d, exist_ok=True)
        return d

    def check_path(self, path: str, session_id: str) -> str:
        """Resolve and validate that path stays within the session sandbox.

        Returns the absolute resolved path. Raises PermissionError on escape.
        """
        sandbox_dir = self.session_dir(session_id)
        abs_sandbox = os.path.realpath(sandbox_dir)
        abs_path = os.path.realpath(os.path.join(abs_sandbox, path))
        if not abs_path.startswith(abs_sandbox + os.sep) and abs_path != abs_sandbox:
            raise PermissionError(f"Path escapes sandbox: {path}")
        return abs_path

    # ── Backend detection ───────────────────────────────────────────

    def _detect_backend(self) -> None:
        system = platform.system()

        if system == "Darwin":
            self._verify_darwin()
            self._backend = "darwin"
        elif system == "Linux":
            self._verify_bwrap()
            self._backend = "bwrap"
        else:
            logger.error("No sandbox backend for platform: %s", system)
            sys.exit(1)

    @staticmethod
    def _verify_darwin() -> None:
        if not shutil.which("sandbox-exec"):
            logger.error("sandbox-exec not found in PATH.")
            sys.exit(1)
        try:
            result = subprocess.run(
                ["sandbox-exec", "-p", "(version 1)(allow default)", "echo", "ok"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                logger.error("sandbox-exec smoke test failed: %s", result.stderr)
                sys.exit(1)
        except Exception as e:
            logger.error("sandbox-exec smoke test error: %s", e)
            sys.exit(1)
        logger.info("Sandbox backend: sandbox-exec (macOS)")

    @staticmethod
    def _verify_bwrap() -> None:
        if not shutil.which("bwrap"):
            logger.error("bwrap not found. Install: apt install bubblewrap / dnf install bubblewrap")
            sys.exit(1)
        try:
            result = subprocess.run(
                ["bwrap", "--ro-bind", "/", "/", "echo", "ok"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                logger.error("bwrap smoke test failed: %s", result.stderr)
                sys.exit(1)
        except Exception as e:
            logger.error("bwrap smoke test error: %s", e)
            sys.exit(1)
        logger.info("Sandbox backend: bubblewrap (Linux)")

    # ── Command building ────────────────────────────────────────────

    def _build_cmd(
        self,
        exec_dir: str,
        inner_cmd: list[str],
        *,
        readonly: bool = False,
        no_network: bool = False,
    ) -> list[str]:
        if self._backend == "darwin":
            profile = _make_sbpl(exec_dir, readonly=readonly, no_network=no_network)
            return ["sandbox-exec", "-p", profile, *inner_cmd]
        elif self._backend == "bwrap":
            return _make_bwrap(
                exec_dir, inner_cmd,
                readonly=readonly, no_network=no_network,
                extra_env=self._extra_env_cache,
            )
        else:
            raise RuntimeError(f"Sandbox not initialized (backend={self._backend!r})")

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_output:
            return text[: self._max_output] + "\n...(truncated)"
        return text

    def _resolve_extra_env(self) -> None:
        """Materialize env values from config once at startup.

        - `env`  literal values (expanduser applied)
        - `env_from_cmd`  shell command, stdout becomes the value
        - `env_passthrough`  copy from the parent env if set

        Empty results are dropped. Failures log and skip.
        """
        resolved: dict[str, str] = {}
        for k, v in self._extra_env_literal.items():
            expanded = os.path.expanduser(str(v))
            if expanded:
                resolved[k] = expanded
        for k, cmd in self._extra_env_cmd.items():
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    resolved[k] = result.stdout.strip()
                    logger.info("Sandbox env: resolved %s via command", k)
                else:
                    logger.warning(
                        "Sandbox env_from_cmd[%s] failed (rc=%d): %s",
                        k, result.returncode, result.stderr.strip()[:200],
                    )
            except Exception as e:
                logger.warning("Sandbox env_from_cmd[%s] error: %s", k, e)
        for k in self._extra_env_passthrough:
            val = os.environ.get(k)
            if val:
                resolved[k] = val
        self._extra_env_cache = resolved
        if resolved:
            logger.info("Sandbox: %d extra env vars loaded: %s", len(resolved), ", ".join(sorted(resolved)))


# ── macOS: sandbox-exec (SBPL) ──────────────────────────────────────


def _make_sbpl(exec_dir: str, *, readonly: bool = False, no_network: bool = False) -> str:
    real_dir = os.path.realpath(exec_dir)
    parts = [
        "(version 1)",
        "(deny default)",
        "(allow file-read*)",
    ]
    if readonly:
        parts.append('(allow file-write* (subpath "/dev"))')
    else:
        parts.append(f'(allow file-write* (subpath "{real_dir}") (subpath "/dev"))')
    parts.extend([
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow mach-register)",
        "(allow ipc-posix*)",
        "(allow signal)",
    ])
    if not no_network:
        parts.append("(allow network*)")
    return "\n".join(parts)


# ── Linux: bubblewrap ───────────────────────────────────────────────

_BWRAP_RO_PATHS = [
    "/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/opt",
    "/etc/alternatives", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf",
    "/etc/ssl", "/etc/pki", "/etc/ca-certificates",
    "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d",
    "/etc/passwd", "/etc/group",
]


def _make_bwrap(
    exec_dir: str,
    inner_cmd: list[str],
    *,
    readonly: bool = False,
    no_network: bool = False,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    real_dir = os.path.realpath(exec_dir)
    args: list[str] = ["bwrap"]

    for p in _BWRAP_RO_PATHS:
        if os.path.exists(p):
            args.extend(["--ro-bind", p, p])

    # Any extra env value that looks like an existing filesystem path gets
    # bind-mounted read-only at the same path inside the sandbox. This keeps
    # the config generic: users declare `GH_CONFIG_DIR: ~/.config/gh` and the
    # sandbox exposes the directory automatically.
    resolved_extra: dict[str, str] = {}
    for k, v in (extra_env or {}).items():
        if v and os.path.exists(v):
            args.extend(["--ro-bind", v, v])
        resolved_extra[k] = v

    args.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])

    if readonly:
        args.extend(["--ro-bind", real_dir, "/workspace"])
    else:
        args.extend(["--bind", real_dir, "/workspace"])
    args.extend(["--chdir", "/workspace"])

    args.extend(["--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup"])
    if no_network:
        args.append("--unshare-net")

    args.extend(["--die-with-parent", "--new-session"])
    args.extend(["--setenv", "HOME", "/workspace"])
    args.extend(["--setenv", "TMPDIR", "/tmp"])
    args.extend(["--setenv", "LANG", "en_US.UTF-8"])
    args.extend(["--setenv", "PYTHONDONTWRITEBYTECODE", "1"])
    args.extend(["--setenv", "TERM", "dumb"])
    args.extend(["--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"])
    for k, v in resolved_extra.items():
        args.extend(["--setenv", k, v])

    args.extend(inner_cmd)
    return args


# ── Helpers ─────────────────────────────────────────────────────────


def _safe_env(exec_dir: str, extra: dict[str, str] | None = None) -> dict[str, str]:
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
    if extra:
        safe.update(extra)
    return safe
