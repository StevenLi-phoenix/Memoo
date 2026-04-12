"""Tests for Gateway: auth handshake, chat_id binding, send_event drain."""

from __future__ import annotations

import asyncio
import json

from core.gateway import Gateway


async def _readline_json(reader: asyncio.StreamReader) -> dict:
    line = await asyncio.wait_for(reader.readline(), timeout=5)
    return json.loads(line.decode())


async def _write_json(writer: asyncio.StreamWriter, data: dict) -> None:
    writer.write(json.dumps(data).encode() + b"\n")
    await writer.drain()


class TestGatewayAuth:
    async def test_valid_token_authenticates(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        handler_called = asyncio.Event()

        async def handler(chat_id: str, text: str, metadata: dict) -> str:
            handler_called.set()
            return f"echo: {text}"

        await gw.start(handler)
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await _write_json(writer, {"auth": gw.token})
            resp = await _readline_json(reader)
            assert resp["event"] == "auth_ok"
        finally:
            writer.close()
            await gw.stop()

    async def test_invalid_token_rejected(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        await gw.start(lambda *a: "ok")
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await _write_json(writer, {"auth": "wrong-token"})
            resp = await _readline_json(reader)
            assert resp["event"] == "auth_fail"
        finally:
            writer.close()
            await gw.stop()

    async def test_invalid_json_rejected(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        await gw.start(lambda *a: "ok")
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(b"not json\n")
            await writer.drain()
            resp = await _readline_json(reader)
            assert resp["event"] == "auth_fail"
        finally:
            writer.close()
            await gw.stop()


class TestGatewayMessaging:
    async def test_message_roundtrip(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")

        async def handler(chat_id: str, text: str, metadata: dict) -> str:
            return f"reply:{text}"

        await gw.start(handler)
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            # Auth
            await _write_json(writer, {"auth": gw.token})
            resp = await _readline_json(reader)
            assert resp["event"] == "auth_ok"

            # Send message
            await _write_json(writer, {"chat_id": "test", "text": "hello"})
            resp = await _readline_json(reader)
            assert resp["event"] == "reply"
            assert resp["reply"] == "reply:hello"
            assert resp["chat_id"] == "test"
        finally:
            writer.close()
            await gw.stop()

    async def test_chat_id_binding_rejects_mismatch(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")

        async def handler(chat_id: str, text: str, metadata: dict) -> str:
            return "ok"

        await gw.start(handler)
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await _write_json(writer, {"auth": gw.token})
            await _readline_json(reader)

            # First message binds chat_id
            await _write_json(writer, {"chat_id": "user1", "text": "hi"})
            await _readline_json(reader)

            # Second with different chat_id should error
            await _write_json(writer, {"chat_id": "user2", "text": "hi"})
            resp = await _readline_json(reader)
            assert resp["event"] == "error"
            assert "mismatch" in resp["error"]
        finally:
            writer.close()
            await gw.stop()

    async def test_empty_text_rejected(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        await gw.start(lambda *a: "ok")
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await _write_json(writer, {"auth": gw.token})
            await _readline_json(reader)

            await _write_json(writer, {"chat_id": "t", "text": ""})
            resp = await _readline_json(reader)
            assert resp["event"] == "error"
            assert "empty" in resp["error"]
        finally:
            writer.close()
            await gw.stop()


class TestSendEvent:
    async def test_send_event_delivers_to_client(self, tmp_path) -> None:
        """Verify send_event actually delivers and drains data to the client."""
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        event_received = asyncio.Event()

        async def handler(chat_id: str, text: str, metadata: dict) -> str:
            # While handling, send a streaming event
            await gw.send_event(chat_id, {"event": "tool_start", "name": "test_tool", "args": ""})
            event_received.set()
            return "done"

        await gw.start(handler)
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await _write_json(writer, {"auth": gw.token})
            await _readline_json(reader)

            await _write_json(writer, {"chat_id": "stream", "text": "go"})

            # We should receive both the tool_start event AND the reply
            msgs = []
            for _ in range(2):
                msg = await _readline_json(reader)
                msgs.append(msg)

            events = [m["event"] for m in msgs]
            assert "tool_start" in events
            assert "reply" in events
        finally:
            writer.close()
            await gw.stop()

    async def test_send_event_no_clients_is_safe(self, tmp_path) -> None:
        """send_event with no connected clients should not raise."""
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        await gw.start(lambda *a: "ok")
        # No clients connected — should be a no-op
        await gw.send_event("nonexistent", {"event": "test"})
        await gw.stop()


class TestShutdown:
    async def test_shutdown_broadcasts(self, tmp_path) -> None:
        gw = Gateway(host="127.0.0.1", port=0, token_file=tmp_path / ".token")
        await gw.start(lambda *a: "ok")
        port = gw._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await _write_json(writer, {"auth": gw.token})
            await _readline_json(reader)

            # Register the writer as a known client
            gw._all_writers.add(writer)

            await gw.stop()

            # Client should receive shutdown event
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if line:
                msg = json.loads(line.decode())
                assert msg["event"] == "shutdown"
        finally:
            writer.close()

    async def test_token_file_cleaned_up(self, tmp_path) -> None:
        token_file = tmp_path / ".token"
        gw = Gateway(host="127.0.0.1", port=0, token_file=token_file)
        await gw.start(lambda *a: "ok")
        assert token_file.exists()
        await gw.stop()
        assert not token_file.exists()
