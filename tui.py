#!/usr/bin/env python3
"""Memoo TUI — thin chat client that connects to a running Memoo instance.

Usage:
  1. Start Memoo: python main.py
  2. Open TUI:    python tui.py [--host localhost] [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import select
import sys


async def main() -> None:
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

    try:
        while True:
            line = await loop.run_in_executor(None, _read_input)
            if line is None:
                break

            text = line.strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                print("Bye!")
                break

            # Send to Memoo
            msg = json.dumps({"chat_id": args.chat_id, "text": text})
            writer.write(msg.encode() + b"\n")
            await writer.drain()

            # Read response
            print("\033[90m(thinking...)\033[0m", end="\r")
            resp_line = await reader.readline()
            if not resp_line:
                print("Connection closed by server.")
                break

            resp = json.loads(resp_line.decode())
            if "error" in resp:
                print(f"\033[31mError: {resp['error']}\033[0m")
            elif resp.get("reply", "").strip():
                print(f"\n\033[36mMemoo:\033[0m {resp['reply']}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    finally:
        writer.close()


def _read_input() -> str | None:
    try:
        sys.stdout.write("\033[33mYou:\033[0m ")
        sys.stdout.flush()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if ready:
                line = sys.stdin.readline()
                return line if line else None
    except (EOFError, OSError):
        return None


if __name__ == "__main__":
    asyncio.run(main())
