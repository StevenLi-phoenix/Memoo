"""Lightweight JSON-over-TCP gateway with streaming events.

Protocol: line-delimited JSON over TCP.
  Client -> {"chat_id": "tui", "text": "hello"}
  Server -> {"event": "tool_start", "name": "run_code", "args": "code='print(1)'"}
  Server -> {"event": "tool_done", "name": "run_code", "ok": true, "result": "1"}
  Server -> {"event": "reply", "reply": "The answer is 1", "topic": "math"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

MessageHandler = Callable[[str, str, dict[str, Any]], Awaitable[str]]


class Gateway:
    """TCP server that accepts JSON messages, streams tool events, and returns replies."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._host = host
        self._port = port
        self._handler: MessageHandler | None = None
        self._server: asyncio.Server | None = None
        # Active client writers keyed by chat_id for streaming events
        self._clients: dict[str, list[asyncio.StreamWriter]] = {}

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._server = await asyncio.start_server(self._on_connect, self._host, self._port)
        logger.info("Gateway listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Gateway stopped")

    def send_event(self, chat_id: str, event: dict[str, Any]) -> None:
        """Send a streaming event to all clients subscribed to this chat_id."""
        writers = self._clients.get(chat_id, [])
        line = json.dumps(event, ensure_ascii=False).encode() + b"\n"
        for w in writers:
            try:
                w.write(line)
            except Exception:
                pass

    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        logger.info("Gateway client connected: %s", addr)

        current_chat_id: str | None = None

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    self._write(writer, {"event": "error", "error": "invalid JSON"})
                    continue

                chat_id = msg.get("chat_id", "gateway")
                text = msg.get("text", "")
                metadata = msg.get("metadata", {})
                metadata["platform"] = "gateway"

                if not text:
                    self._write(writer, {"event": "error", "error": "empty text"})
                    continue

                if self._handler is None:
                    self._write(writer, {"event": "error", "error": "not ready"})
                    continue

                # Register client for streaming events
                current_chat_id = chat_id
                self._clients.setdefault(chat_id, []).append(writer)

                try:
                    response = await self._handler(chat_id, text, metadata)
                    self._write(writer, {"event": "reply", "reply": response, "chat_id": chat_id})
                except Exception as e:
                    logger.exception("Gateway handler error")
                    self._write(writer, {"event": "error", "error": str(e)})
                finally:
                    # Unregister client
                    clients = self._clients.get(chat_id, [])
                    if writer in clients:
                        clients.remove(writer)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            # Cleanup
            if current_chat_id:
                clients = self._clients.get(current_chat_id, [])
                if writer in clients:
                    clients.remove(writer)
            writer.close()
            logger.info("Gateway client disconnected: %s", addr)

    @staticmethod
    def _write(writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        writer.write(json.dumps(data, ensure_ascii=False).encode() + b"\n")
