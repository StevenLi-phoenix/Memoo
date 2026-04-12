"""TUI (Terminal UI) channel — fallback interactive channel via stdin/stdout."""

from __future__ import annotations

import asyncio
import logging
import sys

from channels import MessageHandler

logger = logging.getLogger(__name__)

DEFAULT_CHAT_ID = "tui"


class TUIChannel:
    """Interactive terminal channel. Used as fallback when no other channel is enabled."""

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

        while self._running:
            try:
                line = await loop.run_in_executor(None, self._read_line)
            except (EOFError, KeyboardInterrupt):
                break

            if line is None:
                break

            text = line.strip()
            if not text:
                continue

            if text in ("/quit", "/exit"):
                print("Bye!")
                # Signal shutdown
                loop.call_soon(lambda: sys.exit(0))
                break

            if text == "/clear":
                assert self._handler is not None
                await self._handler(self._chat_id, text, {"command": "clear"})
                print("Memory cleared.")
                continue

            assert self._handler is not None
            try:
                response = await self._handler(self._chat_id, text, {"platform": "tui"})
                await self.send(self._chat_id, response)
            except Exception:
                logger.exception("Error handling TUI message")
                print("\033[31mError processing message.\033[0m")

    @staticmethod
    def _read_line() -> str | None:
        try:
            return input("\033[33mYou:\033[0m ")
        except EOFError:
            return None
