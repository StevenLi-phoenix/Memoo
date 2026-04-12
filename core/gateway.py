"""Lightweight JSON-over-TCP gateway for external clients (TUI, etc).

Protocol: line-delimited JSON over TCP.
  Client -> {"chat_id": "tui", "text": "hello"}
  Server -> {"reply": "hi", "topic": "greeting"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

MessageHandler = Callable[[str, str, dict[str, Any]], Awaitable[str]]


class Gateway:
    """TCP server that accepts JSON messages and routes them to the agent."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._host = host
        self._port = port
        self._handler: MessageHandler | None = None
        self._server: asyncio.Server | None = None

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._server = await asyncio.start_server(self._on_connect, self._host, self._port)
        logger.info("Gateway listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Gateway stopped")

    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        logger.info("Gateway client connected: %s", addr)

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    self._write(writer, {"error": "invalid JSON"})
                    continue

                chat_id = msg.get("chat_id", "gateway")
                text = msg.get("text", "")
                metadata = msg.get("metadata", {})
                metadata["platform"] = "gateway"

                if not text:
                    self._write(writer, {"error": "empty text"})
                    continue

                if self._handler is None:
                    self._write(writer, {"error": "not ready"})
                    continue

                try:
                    response = await self._handler(chat_id, text, metadata)
                    self._write(writer, {"reply": response, "chat_id": chat_id})
                except Exception as e:
                    logger.exception("Gateway handler error")
                    self._write(writer, {"error": str(e)})
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            logger.info("Gateway client disconnected: %s", addr)

    @staticmethod
    def _write(writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        writer.write(json.dumps(data, ensure_ascii=False).encode() + b"\n")
