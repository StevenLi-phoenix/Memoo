#!/usr/bin/env python3
"""Memoo TUI — thin chat client with tool call display and markdown rendering.

Usage:
  1. Start Memoo: python main.py
  2. Open TUI:    python tui.py [--host localhost] [--port 8000] [--timeout 600]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import signal
import ssl
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# ─── ANSI helpers ───────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GREY = "\033[90m"

BG_CODE = "\033[48;5;236m"  # dark grey background for inline code

ERASE_LINE = "\033[2K"

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _tw() -> int:
    """Terminal width, fallback 80."""
    return shutil.get_terminal_size((80, 24)).columns


# ─── Slash Command Completion ──────────────────────────────────────

# Populated at startup from core.commands + skills
KNOWN_COMMANDS: dict[str, str] = {}


def _init_commands() -> None:
    """Load command definitions and skill names for completion."""
    global KNOWN_COMMANDS
    try:
        from core.commands import COMMANDS

        KNOWN_COMMANDS = dict(COMMANDS)
    except ImportError:
        KNOWN_COMMANDS = {
            "/help": "Show available commands",
            "/clear": "Clear conversation memory",
            "/config": "Show configuration",
            "/model": "Show or switch model",
            "/memory": "Show archived memories",
            "/schedule": "List scheduled tasks",
            "/status": "Show agent status",
            "/quit": "Exit TUI",
            "/exit": "Exit TUI",
        }

    # Discover skills and add as /{skill_name}
    try:
        from core.skills import SkillRegistry

        registry = SkillRegistry(skills_dir=Path("skills"))
        registry.discover()
        for name in registry.skill_names:
            meta = registry.get_meta(name)
            if meta:
                KNOWN_COMMANDS[f"/{name}"] = f"Skill: {meta.description}"
    except (ImportError, OSError):
        pass


class SlashCompleter(Completer):
    """Autocomplete slash commands and skills with descriptions."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in sorted(KNOWN_COMMANDS.items()):
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=FormattedText([("class:completion-cmd", cmd)]),
                    display_meta=desc,
                )


# prompt_toolkit style
PT_STYLE = Style.from_dict(
    {
        "prompt": "#e5c07b bold",
        "completion-menu.completion": "bg:#3e4452 #abb2bf",
        "completion-menu.completion.current": "bg:#528bff #ffffff",
        "completion-menu.meta.completion": "bg:#3e4452 #5c6370 italic",
        "completion-menu.meta.completion.current": "bg:#528bff #c8ccd4 italic",
        "completion-cmd": "#61afef",
    }
)


# ─── Markdown Renderer ─────────────────────────────────────────────


def render_markdown(text: str) -> str:
    """Render markdown to terminal with ANSI colors.

    Supports: headings, bold, italic, inline code, fenced code blocks,
    unordered/ordered lists, blockquotes, horizontal rules, links, strikethrough.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_code_block = False
    code_lang = ""
    code_lines: list[str] = []

    for line in lines:
        # ── fenced code blocks ──
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = line[3:].strip()
                code_lines = []
                continue
            else:
                _render_code_block(out, code_lines, code_lang)
                in_code_block = False
                code_lang = ""
                code_lines = []
                continue

        if in_code_block:
            code_lines.append(line)
            continue

        # ── horizontal rule ──
        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", line):
            out.append(f"  {DIM}{'─' * (_tw() - 4)}{RESET}")
            continue

        # ── headings ──
        if line.startswith("### "):
            out.append(f"  {BOLD}{YELLOW}{line[4:]}{RESET}")
            continue
        if line.startswith("## "):
            out.append(f"  {BOLD}{CYAN}{line[3:]}{RESET}")
            continue
        if line.startswith("# "):
            out.append(f"  {BOLD}{MAGENTA}{line[2:]}{RESET}")
            continue

        # ── blockquote ──
        if line.startswith("> "):
            content = _render_inline(line[2:])
            out.append(f"  {DIM}▎{RESET} {ITALIC}{content}{RESET}")
            continue

        # ── unordered list ──
        m = re.match(r"^(\s*)[-*+] (.+)", line)
        if m:
            depth = len(m.group(1)) // 2
            content = _render_inline(m.group(2))
            out.append(f"  {'  ' * depth}{GREY}•{RESET} {content}")
            continue

        # ── ordered list ──
        m = re.match(r"^(\s*)(\d+)\. (.+)", line)
        if m:
            depth = len(m.group(1)) // 2
            num = m.group(2)
            content = _render_inline(m.group(3))
            out.append(f"  {'  ' * depth}{DIM}{num}.{RESET} {content}")
            continue

        # ── normal line ──
        out.append(_render_inline(line))

    # unclosed code block — render what we have
    if in_code_block and code_lines:
        _render_code_block(out, code_lines, code_lang)

    return "\n".join(out)


def _render_code_block(out: list[str], code_lines: list[str], lang: str) -> None:
    """Render a fenced code block with box-drawing characters."""
    width = _tw() - 4
    inner = width - 2  # space inside the box borders

    # top border
    if lang:
        label = f" {lang} "
        out.append(f"  {DIM}┌─{BOLD}{label}{RESET}{DIM}{'─' * max(0, inner - len(label) - 1)}┐{RESET}")
    else:
        out.append(f"  {DIM}┌{'─' * inner}┐{RESET}")

    # code lines
    for cl in code_lines:
        visible = cl[:inner]
        pad = inner - len(visible)
        out.append(f"  {DIM}│{RESET}{YELLOW}{visible}{RESET}{' ' * pad}{DIM}│{RESET}")

    # bottom border
    out.append(f"  {DIM}└{'─' * inner}┘{RESET}")


def _render_inline(line: str) -> str:
    """Apply inline formatting: bold, italic, code, links, strikethrough."""
    # links: [text](url) → underlined text + dimmed url
    line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", rf"{UNDERLINE}\1{RESET}{DIM} (\2){RESET}", line)
    # bold + italic
    line = re.sub(r"\*\*\*(.+?)\*\*\*", rf"{BOLD}{ITALIC}\1{RESET}", line)
    # bold
    line = re.sub(r"\*\*(.+?)\*\*", rf"{BOLD}\1{RESET}", line)
    # italic (single * not preceded/followed by *)
    line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", rf"{ITALIC}\1{RESET}", line)
    # inline code
    line = re.sub(r"`([^`]+)`", rf"{BG_CODE}{YELLOW} \1 {RESET}", line)
    # strikethrough
    line = re.sub(r"~~(.+?)~~", rf"{DIM}\1{RESET}", line)
    return line


# ─── Tool Display ──────────────────────────────────────────────────


class ToolTracker:
    """Track active tool calls with elapsed time and animated spinner."""

    def __init__(self) -> None:
        self._active: dict[str, float] = {}  # name -> start_time
        self._count = 0
        self._spinner_idx = 0

    def start(self, name: str, args: str) -> None:
        self._active[name] = time.monotonic()
        self._count += 1
        self._render_active()

    def done(self, name: str, ok: bool, result: str) -> None:
        elapsed = time.monotonic() - self._active.pop(name, time.monotonic())
        status = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        elapsed_str = f" {DIM}({elapsed:.1f}s){RESET}" if elapsed >= 0.1 else ""
        preview = result.replace("\n", " ")[:100]
        result_str = f" {DIM}→ {preview}{RESET}" if preview else ""
        sys.stdout.write(f"{ERASE_LINE}\r  {status} {GREY}{name}{RESET}{elapsed_str}{result_str}\n")
        sys.stdout.flush()

    def tick(self) -> None:
        """Advance spinner animation for active tools."""
        if self._active:
            self._render_active()

    def _render_active(self) -> None:
        name = next(iter(self._active))
        elapsed = time.monotonic() - self._active[name]
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        self._spinner_idx += 1
        elapsed_str = f" {DIM}{elapsed:.0f}s{RESET}" if elapsed >= 1.0 else ""
        sys.stdout.write(f"{ERASE_LINE}\r  {CYAN}{frame}{RESET} {GREY}{name} ...{RESET}{elapsed_str}")
        sys.stdout.flush()

    @property
    def total(self) -> int:
        return self._count


# ─── Main ──────────────────────────────────────────────────────────

_running = True


async def main() -> None:
    global _running

    _init_commands()

    parser = argparse.ArgumentParser(description="Memoo TUI Client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chat-id", default="tui")
    parser.add_argument("--timeout", type=int, default=600, help="Response timeout in seconds")
    args = parser.parse_args()

    # mTLS if certs exist
    ssl_ctx = None
    try:
        from core.gateway import create_client_ssl

        ssl_ctx = create_client_ssl()
    except ImportError:
        pass

    try:
        reader, writer = await asyncio.open_connection(args.host, args.port, ssl=ssl_ctx)
    except ConnectionRefusedError:
        print(f"\n{RED}Cannot connect to Memoo at {args.host}:{args.port}{RESET}")
        print(f"Start Memoo first: {BOLD}python main.py{RESET}")
        return
    except ssl.SSLError as e:
        print(f"\n{RED}TLS handshake failed: {e}{RESET}")
        print(f"Generate certs: {BOLD}python -m core.gateway --generate-certs{RESET}")
        return

    proto = "mTLS" if ssl_ctx else "TCP"
    _print_banner(args.host, args.port, proto, args.chat_id)

    loop = asyncio.get_running_loop()

    def _stop() -> None:
        global _running
        _running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    session = PromptSession(
        completer=SlashCompleter(),
        history=InMemoryHistory(),
        style=PT_STYLE,
        complete_while_typing=True,
    )

    while _running:
        try:
            with patch_stdout():
                text = await session.prompt_async(
                    [("class:prompt", "You: ")],
                )
        except (EOFError, KeyboardInterrupt):
            break

        text = text.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break

        # All /commands go to server (which handles them without LLM)
        msg = json.dumps({"chat_id": args.chat_id, "text": text})
        writer.write(msg.encode() + b"\n")
        await writer.drain()

        await _read_events(reader, args.timeout)
        # visual separator between exchanges
        print(f"  {DIM}{'·' * min(40, _tw() - 4)}{RESET}\n")

    print(f"\n{DIM}Bye!{RESET}")
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


async def _read_events(reader: asyncio.StreamReader, timeout: int = 600) -> None:
    """Read streaming events from gateway with spinner animation."""
    tracker = ToolTracker()

    async def _spin() -> None:
        try:
            while True:
                await asyncio.sleep(0.08)
                tracker.tick()
        except asyncio.CancelledError:
            pass

    spinner_task = asyncio.create_task(_spin())

    try:
        while True:
            try:
                resp_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                sys.stdout.write(f"{ERASE_LINE}\r")
                print(f"{RED}Timeout ({timeout}s) waiting for response.{RESET}")
                break

            if not resp_line:
                print(f"\n{RED}Connection closed by server.{RESET}")
                break

            try:
                event = json.loads(resp_line.decode())
            except json.JSONDecodeError:
                continue

            ev_type = event.get("event", "")

            if ev_type == "tool_start":
                tracker.start(event.get("name", "?"), event.get("args", ""))

            elif ev_type == "tool_done":
                tracker.done(
                    event.get("name", "?"),
                    event.get("ok", True),
                    event.get("result", ""),
                )

            elif ev_type == "reply":
                sys.stdout.write(f"{ERASE_LINE}\r")
                reply = event.get("reply", "")
                if reply.strip():
                    print(f"\n{CYAN}{BOLD}Memoo:{RESET} {render_markdown(reply)}\n")
                # Footer: tool count + token usage
                footer_parts: list[str] = []
                if tracker.total:
                    footer_parts.append(f"{tracker.total} tool{'s' if tracker.total != 1 else ''}")
                usage = event.get("usage")
                if usage:
                    footer_parts.append(_format_usage(usage))
                if footer_parts:
                    print(f"  {DIM}{' · '.join(footer_parts)}{RESET}")
                break

            elif ev_type == "error":
                sys.stdout.write(f"{ERASE_LINE}\r")
                print(f"{RED}Error: {event.get('error', '?')}{RESET}")
                break

            # unknown events: ignore silently, don't break the loop
    finally:
        spinner_task.cancel()
        try:
            await spinner_task
        except asyncio.CancelledError:
            pass


def _format_usage(usage: dict[str, int]) -> str:
    """Format token usage dict into a compact display string."""
    parts: list[str] = []
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    if inp:
        parts.append(f"in:{_fmt_tokens(inp)}")
    if out:
        parts.append(f"out:{_fmt_tokens(out)}")
    cache_read = usage.get("cache_read_tokens", 0)
    cache_create = usage.get("cache_creation_tokens", 0)
    if cache_read:
        parts.append(f"cache↑{_fmt_tokens(cache_read)}")
    if cache_create:
        parts.append(f"cache+{_fmt_tokens(cache_create)}")
    return " ".join(parts) if parts else ""


def _fmt_tokens(n: int) -> str:
    """Format token count: 1234 -> '1.2k', 12345 -> '12k'."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _print_banner(host: str, port: int, proto: str, chat_id: str) -> None:
    w = _tw()
    inner = w - 6
    print()
    print(f"  {BOLD}{CYAN}╔{'═' * inner}╗{RESET}")
    title = "Memoo TUI"
    pad = inner - len(title) - 1
    print(f"  {BOLD}{CYAN}║{RESET} {BOLD}{title}{RESET}{' ' * pad}{BOLD}{CYAN}║{RESET}")
    info = f"{host}:{port} ({proto}) · chat: {chat_id}"
    pad = inner - len(info) - 1
    print(f"  {BOLD}{CYAN}║{RESET} {DIM}{info}{RESET}{' ' * pad}{BOLD}{CYAN}║{RESET}")
    print(f"  {BOLD}{CYAN}╚{'═' * inner}╝{RESET}")
    print(f"  {DIM}Type /help for commands, Tab to complete{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
