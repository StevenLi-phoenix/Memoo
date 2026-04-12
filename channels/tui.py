"""TUI (Terminal UI) channel — fallback interactive channel via stdin/stdout."""

from __future__ import annotations

import asyncio
import logging
import select
import sys

from channels import MessageHandler

logger = logging.getLogger(__name__)

DEFAULT_CHAT_ID = "tui"


class TUIChannel:
    """Interactive terminal channel. Uses select() for non-blocking stdin reads."""

    def __init__(self, chat_id: str = DEFAULT_CHAT_ID) -> None:
        self._chat_id = chat_id
        self._handler: MessageHandler | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._running = True
        self._task = asyncio.create_task(self._input_loop())
        logger.info("TUI channel started (chat_id=%s)", self._chat_id)
        print("\n--- Memoo TUI ---")
        print("Type your message and press Enter. Type /quit to exit.\n")

    async def send(self, chat_id: str, text: str) -> None:
        print(f"\n\033[36mMemoo:\033[0m {text}\n")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TUI channel stopped")

    async def _input_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                # Non-blocking stdin check via run_in_executor + select
                line = await loop.run_in_executor(None, self._read_line_nonblocking)

                if line is None:
                    if not self._running:
                        break
                    await asyncio.sleep(0.1)
                    continue

                text = line.strip()
                if not text:
                    continue

                if text in ("/quit", "/exit"):
                    print("Bye!")
                    break

                if text == "/clear":
                    if self._handler is not None:
                        await self._handler(self._chat_id, text, {"command": "clear"})
                        print("Memory cleared.")
                    continue

                if self._handler is None:
                    continue

                try:
                    print("\033[90m(thinking...)\033[0m", end="\r")
                    response = await self._handler(self._chat_id, text, {"platform": "tui"})
                    if response.strip() != "NO_OP":
                        await self.send(self._chat_id, response)
                except Exception:
                    logger.exception("Error handling TUI message")
                    print("\033[31mError processing message.\033[0m")
        except asyncio.CancelledError:
            pass

    def _read_line_nonblocking(self) -> str | None:
        """Read a line from stdin with 0.5s timeout using select().

        Returns None if no input available (allows cancellation check).
        """
        try:
            sys.stdout.write("\033[33mYou:\033[0m ")
            sys.stdout.flush()
            # Wait up to 0.5s for stdin to become readable
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if ready:
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        return None
                    return line
            return None
        except (EOFError, OSError):
            return None
