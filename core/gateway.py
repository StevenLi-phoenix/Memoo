"""JSON-over-TCP gateway with optional mTLS and streaming events.

Protocol: line-delimited JSON over TCP (optionally TLS).
  Client -> {"chat_id": "tui", "text": "hello"}
  Server -> {"event": "tool_start", "name": "run_code", "args": "..."}
  Server -> {"event": "tool_done", "name": "run_code", "ok": true, "result": "..."}
  Server -> {"event": "reply", "reply": "...", "chat_id": "tui"}

mTLS: Both server and client present certificates signed by the same CA.
  Generate certs: python -m core.gateway --generate-certs
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from pathlib import Path
from typing import Any

from channels import MessageHandler

logger = logging.getLogger(__name__)

CERTS_DIR = Path("certs")

_LOCALHOST_ADDRS = {"localhost", "127.0.0.1", "::1"}


def _is_localhost(host: str) -> bool:
    return host in _LOCALHOST_ADDRS


def create_server_ssl(host: str = "localhost", certs_dir: Path = CERTS_DIR) -> ssl.SSLContext | None:
    """Create server-side mTLS context.

    - localhost: returns None (plain TCP allowed), logs a warning
    - non-localhost: auto-generates certs if missing, always returns a context
    """
    ca = certs_dir / "ca.pem"
    cert = certs_dir / "server.pem"
    key = certs_dir / "server-key.pem"
    has_certs = all(p.exists() for p in (ca, cert, key))

    if not has_certs:
        if _is_localhost(host):
            logger.warning("Gateway on %s without mTLS — plain TCP (localhost only)", host)
            return None
        # Non-localhost: auto-generate certs
        logger.info("Non-localhost bind (%s) without certs — auto-generating mTLS certificates", host)
        generate_certs(certs_dir)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    ctx.load_verify_locations(str(ca))
    ctx.verify_mode = ssl.CERT_REQUIRED  # mTLS: require client cert
    ctx.check_hostname = False
    logger.info("mTLS enabled (server): ca=%s, cert=%s", ca, cert)
    return ctx


def create_client_ssl(certs_dir: Path = CERTS_DIR) -> ssl.SSLContext | None:
    """Create client-side mTLS context. Returns None if certs not found."""
    ca = certs_dir / "ca.pem"
    cert = certs_dir / "client.pem"
    key = certs_dir / "client-key.pem"

    if not all(p.exists() for p in (ca, cert, key)):
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(str(cert), str(key))
    ctx.load_verify_locations(str(ca))
    ctx.check_hostname = False
    logger.info("mTLS enabled (client): ca=%s, cert=%s", ca, cert)
    return ctx


class Gateway:
    """TCP/TLS server with streaming tool events."""

    def __init__(self, host: str = "localhost", port: int = 8000, ssl_ctx: ssl.SSLContext | None = None) -> None:
        self._host = host
        self._port = port
        self._ssl = ssl_ctx
        self._handler: MessageHandler | None = None
        self._server: asyncio.Server | None = None
        self._clients: dict[str, list[asyncio.StreamWriter]] = {}
        self._reply_extra: dict[str, dict[str, Any]] = {}

    def set_reply_extra(self, chat_id: str, extra: dict[str, Any]) -> None:
        """Attach extra fields to the next reply event for this chat_id."""
        self._reply_extra[chat_id] = extra

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._server = await asyncio.start_server(self._on_connect, self._host, self._port, ssl=self._ssl)
        proto = "mTLS" if self._ssl else "TCP (no auth)"
        logger.info("Gateway listening on %s:%d (%s)", self._host, self._port, proto)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Gateway stopped")

    def send_event(self, chat_id: str, event: dict[str, Any]) -> None:
        writers = self._clients.get(chat_id, [])
        line = json.dumps(event, ensure_ascii=False).encode() + b"\n"
        for w in writers:
            try:
                w.write(line)
            except Exception:
                pass

    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        # Log client cert CN if mTLS
        ssl_obj = writer.get_extra_info("ssl_object")
        if ssl_obj:
            try:
                cert = ssl_obj.getpeercert()
                cn = dict(x[0] for x in cert.get("subject", ())).get("commonName", "?")
                logger.info("Gateway mTLS client: %s (CN=%s)", addr, cn)
            except Exception:
                logger.info("Gateway client connected: %s (mTLS)", addr)
        else:
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

                current_chat_id = chat_id
                self._clients.setdefault(chat_id, []).append(writer)

                try:
                    response = await self._handler(chat_id, text, metadata)
                    reply_event: dict[str, Any] = {"event": "reply", "reply": response, "chat_id": chat_id}
                    reply_event.update(self._reply_extra.pop(chat_id, {}))
                    self._write(writer, reply_event)
                except Exception as e:
                    logger.exception("Gateway handler error")
                    self._write(writer, {"event": "error", "error": str(e)})
                finally:
                    clients = self._clients.get(chat_id, [])
                    if writer in clients:
                        clients.remove(writer)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            if current_chat_id:
                clients = self._clients.get(current_chat_id, [])
                if writer in clients:
                    clients.remove(writer)
            writer.close()
            logger.info("Gateway client disconnected: %s", addr)

    @staticmethod
    def _write(writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        writer.write(json.dumps(data, ensure_ascii=False).encode() + b"\n")


def generate_certs(certs_dir: Path = CERTS_DIR) -> None:
    """Generate self-signed CA + server + client certs for mTLS."""
    import subprocess

    certs_dir.mkdir(parents=True, exist_ok=True)

    # CA
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-days",
            "3650",
            "-nodes",
            "-subj",
            "/CN=Memoo CA",
            "-keyout",
            str(certs_dir / "ca-key.pem"),
            "-out",
            str(certs_dir / "ca.pem"),
        ],
        check=True,
        capture_output=True,
    )

    # Server cert
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-nodes",
            "-subj",
            "/CN=memoo-server",
            "-keyout",
            str(certs_dir / "server-key.pem"),
            "-out",
            str(certs_dir / "server.csr"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(certs_dir / "server.csr"),
            "-CA",
            str(certs_dir / "ca.pem"),
            "-CAkey",
            str(certs_dir / "ca-key.pem"),
            "-CAcreateserial",
            "-days",
            "3650",
            "-out",
            str(certs_dir / "server.pem"),
        ],
        check=True,
        capture_output=True,
    )

    # Client cert
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-nodes",
            "-subj",
            "/CN=memoo-client",
            "-keyout",
            str(certs_dir / "client-key.pem"),
            "-out",
            str(certs_dir / "client.csr"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(certs_dir / "client.csr"),
            "-CA",
            str(certs_dir / "ca.pem"),
            "-CAkey",
            str(certs_dir / "ca-key.pem"),
            "-CAcreateserial",
            "-days",
            "3650",
            "-out",
            str(certs_dir / "client.pem"),
        ],
        check=True,
        capture_output=True,
    )

    # Cleanup CSRs
    for f in certs_dir.glob("*.csr"):
        f.unlink()
    for f in certs_dir.glob("*.srl"):
        f.unlink()

    print(f"Certificates generated in {certs_dir}/")
    print("  ca.pem          — CA certificate (shared)")
    print("  server.pem/key  — for main.py")
    print("  client.pem/key  — for tui.py")


if __name__ == "__main__":
    generate_certs()
