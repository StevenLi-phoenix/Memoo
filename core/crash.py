"""Crash handler middleware — structured error logging, reporting, and auto-repair hooks.

All unhandled exceptions are captured, logged to .logs/ with full context,
and optionally reported via webhook or queued for `claude -p` auto-fix.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

LOGS_DIR = Path(".logs")
CRASH_REPORTS_DIR = LOGS_DIR / "crashes"
AUTOFIX_QUEUE = LOGS_DIR / "autofix_queue.jsonl"

# Optional webhook for remote reporting (set via config or env)
_webhook_url: str = ""
_webhook_headers: dict[str, str] = {}


def init(
    logs_dir: str = ".logs",
    webhook_url: str = "",
    webhook_headers: dict[str, str] | None = None,
) -> None:
    """Initialize crash handler — set up directories and file logging."""
    global LOGS_DIR, CRASH_REPORTS_DIR, AUTOFIX_QUEUE, _webhook_url, _webhook_headers

    LOGS_DIR = Path(logs_dir)
    CRASH_REPORTS_DIR = LOGS_DIR / "crashes"
    AUTOFIX_QUEUE = LOGS_DIR / "autofix_queue.jsonl"
    _webhook_url = webhook_url
    _webhook_headers = webhook_headers or {}

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CRASH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add file handler for all ERROR+ logs
    file_handler = logging.FileHandler(LOGS_DIR / "error.log", encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)

    # Also log everything to a rotating app log
    app_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(app_handler)

    # Install global exception hook
    sys.excepthook = _global_excepthook

    logger.info("Crash handler initialized: logs_dir=%s, webhook=%s", logs_dir, bool(webhook_url))


def _global_excepthook(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
    """Global unhandled exception hook."""
    report_crash(exc_value, context={"source": "unhandled_exception"})
    # Still print to stderr
    traceback.print_exception(exc_type, exc_value, exc_tb)


def _collect_environment() -> dict[str, Any]:
    """Collect runtime environment snapshot — only called on crash."""
    import os
    import platform
    import resource

    env: dict[str, Any] = {}

    # System
    env["platform"] = platform.platform()
    env["python"] = sys.version
    env["pid"] = os.getpid()
    env["cwd"] = os.getcwd()

    # Memory usage
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        env["memory_mb"] = round(rusage.ru_maxrss / (1024 * 1024), 1)  # macOS reports bytes
        env["user_time_s"] = round(rusage.ru_utime, 2)
        env["sys_time_s"] = round(rusage.ru_stime, 2)
    except Exception:
        pass

    # Async tasks
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        all_tasks = asyncio.all_tasks(loop)
        env["async_tasks"] = len(all_tasks)
        env["async_tasks_names"] = [t.get_name() for t in list(all_tasks)[:20]]
    except Exception:
        pass

    # Threads
    try:
        import threading

        env["threads"] = threading.active_count()
        env["thread_names"] = [t.name for t in threading.enumerate()[:20]]
    except Exception:
        pass

    # Open file descriptors
    try:
        fd_dir = f"/proc/{os.getpid()}/fd"
        if os.path.isdir(fd_dir):
            env["open_fds"] = len(os.listdir(fd_dir))
        else:
            # macOS fallback
            import subprocess

            result = subprocess.run(
                ["lsof", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
                timeout=3,
            )
            env["open_fds"] = result.stdout.count("\n") - 1 if result.returncode == 0 else -1
    except Exception:
        pass

    # Key package versions
    for pkg in ("anthropic", "openai", "httpx", "aiosqlite"):
        try:
            mod = __import__(pkg)
            env[f"pkg_{pkg}"] = getattr(mod, "__version__", "?")
        except ImportError:
            pass

    # Disk space on data dir
    try:
        stat = os.statvfs("./data")
        env["disk_free_mb"] = round(stat.f_bavail * stat.f_frsize / (1024 * 1024), 0)
    except Exception:
        pass

    return env


def report_crash(
    error: BaseException,
    context: dict[str, Any] | None = None,
    component: str = "",
) -> str:
    """Create a structured crash report.

    Returns the crash report ID.
    """
    crash_id = uuid.uuid4().hex[:12]
    now = datetime.now()

    report = {
        "id": crash_id,
        "timestamp": now.isoformat(),
        "error_type": type(error).__qualname__,
        "error_message": str(error),
        "traceback": traceback.format_exception(error),
        "component": component,
        "context": context or {},
        "environment": _collect_environment(),
    }

    # Write crash report to file
    report_path = CRASH_REPORTS_DIR / f"{now:%Y%m%d_%H%M%S}_{crash_id}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Log it
    logger.error(
        "CRASH [%s] %s.%s: %s (report: %s)",
        crash_id,
        type(error).__module__,
        type(error).__qualname__,
        error,
        report_path,
    )

    # Queue for auto-fix (claude -p)
    _queue_autofix(report)

    # Async webhook — fire and forget
    _try_webhook(report)

    return crash_id


def _queue_autofix(report: dict[str, Any]) -> None:
    """Append crash to autofix queue for `claude -p` processing.

    Usage: claude -p "$(cat .logs/autofix_queue.jsonl | tail -1)"
    Or a cron job that processes the queue.
    """
    entry = {
        "id": report["id"],
        "timestamp": report["timestamp"],
        "error_type": report["error_type"],
        "error_message": report["error_message"],
        "traceback": "".join(report["traceback"][-5:]),  # last 5 frames
        "component": report["component"],
        "status": "pending",
    }
    with open(AUTOFIX_QUEUE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _try_webhook(report: dict[str, Any]) -> None:
    """Best-effort webhook notification — non-blocking in async contexts.

    When called from an async context (scheduler, agent, heartbeat), the HTTP
    request is dispatched to a thread pool via run_in_executor so it doesn't
    block the event loop. Falls back to synchronous when no loop is running
    (e.g., from sys.excepthook).
    """
    if not _webhook_url:
        return

    payload = {
        "id": report["id"],
        "error_type": report["error_type"],
        "error_message": report["error_message"],
        "component": report["component"],
        "timestamp": report["timestamp"],
    }

    try:
        loop = asyncio.get_running_loop()
        # Event loop is running — offload to thread pool (fire-and-forget)
        loop.run_in_executor(None, _sync_webhook_post, payload)
    except RuntimeError:
        # No event loop — we're in a sync context (e.g., sys.excepthook), call directly
        _sync_webhook_post(payload)


def _sync_webhook_post(payload: dict[str, Any]) -> None:
    """Synchronous HTTP POST — runs in thread pool when called from async contexts."""
    try:
        import httpx

        with httpx.Client(timeout=5) as client:
            client.post(_webhook_url, json=payload, headers=_webhook_headers)
    except Exception:
        logger.debug("Webhook notification failed", exc_info=True)


# --- Middleware decorator ---


def crash_boundary(component: str = "") -> Callable[..., Any]:
    """Decorator that catches exceptions, reports them, and re-raises.

    Usage:
        @crash_boundary("agent")
        async def run(...):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    report_crash(
                        e,
                        context={"args_summary": _summarize_args(args, kwargs)},
                        component=component or func.__qualname__,
                    )
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    report_crash(
                        e,
                        context={"args_summary": _summarize_args(args, kwargs)},
                        component=component or func.__qualname__,
                    )
                    raise

            return sync_wrapper

    return decorator


def _summarize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Summarize function args without leaking sensitive data."""
    parts: list[str] = []
    for a in args[:3]:
        parts.append(f"{type(a).__name__}")
    for k in list(kwargs)[:3]:
        parts.append(f"{k}=...")
    return ", ".join(parts)
