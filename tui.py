#!/usr/bin/env python3
"""Memoo TUI — thin chat client with tool call display and markdown rendering.

Usage:
  1. Start Memoo: python main.py
  2. Open TUI:    python tui.py [--host localhost] [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import select
import signal
import sys

# --- Simple terminal markdown renderer ---


def render_markdown(text: str) -> str:
    """Render markdown to terminal with ANSI colors."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if line.startswith("### "):
            out.append(f"\033[1;33m{line[4:]}\033[0m")
        elif line.startswith("## "):
            out.append(f"\033[1;36m{line[3:]}\033[0m")
        elif line.startswith("# "):
            out.append(f"\033[1;35m{line[2:]}\033[0m")
        else:
            line = re.sub(r"\*\*(.+?)\*\*", r"\033[1m\1\033[0m", line)
            line = re.sub(r"`([^`]+)`", r"\033[33m\1\033[0m", line)
            if line.startswith("- "):
                line = f"  \033[90m•\033[0m {line[2:]}"
            out.append(line)
    return "\n".join(out)


_running = True


async def main() -> None:
    global _running

    parser = argparse.ArgumentParser(description="Memoo TUI Client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chat-id", default="tui")
    args = parser.parse_args()

    try:
        reader, writer = await asyncio.open_connection(args.host, args.port)
    except ConnectionRefusedError:
        print(f"Cannot connect to Memoo at {args.host}:{args.port}")
        print("Start Memoo first: python main.py")
        return

    print(f"\n--- Memoo TUI (connected to {args.host}:{args.port}) ---")
    print("Type your message and press Enter. Type /quit to exit.\n")

    loop = asyncio.get_running_loop()

    def _stop() -> None:
        global _running
        _running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    while _running:
        line = await loop.run_in_executor(None, _read_input)
        if line is None:
            if not _running:
                break
            continue

        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break

        msg = json.dumps({"chat_id": args.chat_id, "text": text})
        writer.write(msg.encode() + b"\n")
        await writer.drain()

        await _read_events(reader)

    print("Bye!")
    writer.close()
    os._exit(0)


async def _read_events(reader: asyncio.StreamReader) -> None:
    """Read streaming events from gateway."""
    while True:
        try:
            resp_line = await asyncio.wait_for(reader.readline(), timeout=300)
        except asyncio.TimeoutError:
            print("\033[31mTimeout waiting for response.\033[0m")
            break

        if not resp_line:
            print("Connection closed by server.")
            break

        try:
            event = json.loads(resp_line.decode())
        except json.JSONDecodeError:
            continue

        ev_type = event.get("event", "")

        if ev_type == "tool_start":
            name = event.get("name", "?")
            args = event.get("args", "")
            sys.stdout.write(f"\033[90m  ▶ {name}({args[:80]}) ...\033[0m\r")
            sys.stdout.flush()

        elif ev_type == "tool_done":
            name = event.get("name", "?")
            ok = event.get("ok", True)
            result = event.get("result", "")[:100]
            status = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
            sys.stdout.write(f"\033[2K\033[90m  {status} {name} → {result}\033[0m\n")
            sys.stdout.flush()

        elif ev_type == "reply":
            reply = event.get("reply", "")
            if reply.strip():
                print(f"\n\033[36mMemoo:\033[0m {render_markdown(reply)}\n")
            break

        elif ev_type == "error":
            print(f"\033[31mError: {event.get('error', '?')}\033[0m")
            break

        else:
            break


def _read_input() -> str | None:
    """Non-blocking stdin read with 0.5s poll."""
    try:
        sys.stdout.write("\033[33mYou:\033[0m ")
        sys.stdout.flush()
        while _running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if ready:
                line = sys.stdin.readline()
                return line if line else None
        return None
    except (EOFError, OSError):
        return None


if __name__ == "__main__":
    asyncio.run(main())
