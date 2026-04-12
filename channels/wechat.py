"""WeChat channel via iLink Bot API (openclaw-weixin protocol).

Reverse-engineered from @tencent-weixin/openclaw-weixin.
Uses iLink Bot API (ilinkai.weixin.qq.com) with long-polling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from channels import MessageHandler

logger = logging.getLogger(__name__)

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
POLL_TIMEOUT = 35  # seconds, matches openclaw-weixin default


class WeChatChannel:
    """WeChat channel using iLink Bot API with long-polling.

    Config:
        token: iLink Bot API token (obtained via QR code login)
        uin: WeChat UIN for replay protection
    """

    def __init__(self, token: str, uin: str = "") -> None:
        self._token = token
        self._uin = uin
        self._handler: MessageHandler | None = None
        self._running = False
        self._cursor: str = ""  # get_updates_buf for dedup
        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task[None] | None = None

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._running = True
        self._client = httpx.AsyncClient(
            base_url=ILINK_BASE_URL,
            headers=self._build_headers(),
            timeout=httpx.Timeout(POLL_TIMEOUT + 10, connect=10),
        )

        logger.info("WeChat channel starting: uin=%s", self._uin[:6] + "***" if self._uin else "none")
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def send(self, chat_id: str, text: str, context_token: str = "") -> None:
        """Send a text message. context_token must echo from the inbound message."""
        assert self._client is not None

        payload: dict[str, Any] = {
            "to_user": chat_id,
            "item_list": [{"type": 1, "content": text}],
        }
        if context_token:
            payload["context_token"] = context_token

        try:
            resp = await self._client.post("/sendmessage", json=payload)
            resp.raise_for_status()
            logger.debug("WeChat sent message to %s: %s", chat_id, text[:50])
        except httpx.HTTPError:
            logger.exception("Failed to send WeChat message to %s", chat_id)

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator."""
        assert self._client is not None
        try:
            await self._client.post("/sendtyping", json={"to_user": chat_id})
        except httpx.HTTPError:
            logger.debug("Failed to send typing indicator to %s", chat_id)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        logger.info("WeChat channel stopped")

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if self._uin:
            headers["X-WECHAT-UIN"] = self._uin
        return headers

    async def _poll_loop(self) -> None:
        """Long-polling loop for incoming messages."""
        while self._running:
            try:
                messages = await self._get_updates()
                for msg in messages:
                    await self._dispatch(msg)
            except asyncio.CancelledError:
                break
            except httpx.HTTPError:
                logger.exception("WeChat poll error, retrying in 5s")
                await asyncio.sleep(5)
            except Exception:
                logger.exception("Unexpected error in WeChat poll loop, retrying in 5s")
                await asyncio.sleep(5)

    async def _get_updates(self) -> list[dict[str, Any]]:
        """Fetch new messages via long-polling."""
        assert self._client is not None
        payload: dict[str, Any] = {"timeout": POLL_TIMEOUT}
        if self._cursor:
            payload["get_updates_buf"] = self._cursor

        resp = await self._client.post("/getupdates", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Update cursor for dedup
        if "get_updates_buf" in data:
            self._cursor = data["get_updates_buf"]

        messages: list[dict[str, Any]] = data.get("message_list", [])
        if messages:
            logger.info("WeChat poll: received %d messages", len(messages))
        return messages

    async def _dispatch(self, raw_msg: dict[str, Any]) -> None:
        """Parse and dispatch a single inbound message."""
        if not self._handler:
            return

        from_user = raw_msg.get("from_user", "")
        context_token = raw_msg.get("context_token", "")
        item_list: list[dict[str, Any]] = raw_msg.get("item_list", [])

        # Extract text content (type 1 = text)
        text_parts: list[str] = []
        for item in item_list:
            if item.get("type") == 1:
                text_parts.append(item.get("content", ""))

        if not text_parts:
            logger.debug("WeChat: skipping non-text message from %s", from_user)
            return

        user_text = "\n".join(text_parts)
        metadata: dict[str, Any] = {
            "context_token": context_token,
            "platform": "wechat",
        }

        logger.info("WeChat message from %s: %s", from_user, user_text[:100])

        try:
            await self.send_typing(from_user)
            response = await self._handler(from_user, user_text, metadata)
            await self.send(from_user, response, context_token=context_token)
        except Exception:
            logger.exception("Error handling WeChat message from %s", from_user)
            await self.send(from_user, "Sorry, something went wrong.", context_token=context_token)
