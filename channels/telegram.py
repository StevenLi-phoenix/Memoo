"""Telegram channel using python-telegram-bot."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, filters
from telegram.ext import MessageHandler as TGMessageHandler

from channels import MessageHandler

logger = logging.getLogger(__name__)


class TelegramChannel:
    """Telegram bot channel with polling or webhook support."""

    def __init__(self, token: str, mode: str = "polling") -> None:
        self._token = token
        self._mode = mode
        self._app: Application | None = None  # type: ignore[type-arg]
        self._handler: MessageHandler | None = None

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._app = Application.builder().token(self._token).build()

        # Register handlers — slash commands go through handle_message (server-side)
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("help", self._on_command))
        self._app.add_handler(CommandHandler("clear", self._on_command))
        self._app.add_handler(CommandHandler("config", self._on_command))
        self._app.add_handler(CommandHandler("model", self._on_command))
        self._app.add_handler(CommandHandler("memory", self._on_command))
        self._app.add_handler(CommandHandler("schedule", self._on_command))
        self._app.add_handler(CommandHandler("status", self._on_command))
        self._app.add_handler(TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))

        logger.info("Telegram channel starting: mode=%s", self._mode)

        if self._mode == "polling":
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling()  # type: ignore[union-attr]
        else:
            raise NotImplementedError(f"Telegram mode '{self._mode}' not implemented yet")

    async def send(self, chat_id: str, text: str) -> None:
        if self._app is None:
            return
        # Telegram has a 4096 char limit per message
        for i in range(0, len(text), 4000):
            chunk = text[i : i + 4000]
            await self._app.bot.send_message(chat_id=int(chat_id), text=chunk)

    async def stop(self) -> None:
        if self._app:
            logger.info("Telegram channel stopping")
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def _on_start(self, update: Update, _context: Any) -> None:
        """Handle /start command."""
        if update.effective_chat:
            await self.send(
                str(update.effective_chat.id), "Hi! I'm Memoo. Send me a message.\nType /help for commands."
            )

    async def _on_command(self, update: Update, _context: Any) -> None:
        """Handle all slash commands — forward to handle_message which routes to core/commands.py."""
        if not update.message or not update.effective_chat or not self._handler:
            return
        chat_id = str(update.effective_chat.id)
        text = update.message.text or ""
        response = await self._handler(chat_id, text, {"platform": "telegram"})
        if response.strip():
            await self.send(chat_id, response)

    async def _on_message(self, update: Update, _context: Any) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text or not update.effective_chat:
            return

        chat_id = str(update.effective_chat.id)
        user_text = update.message.text
        metadata: dict[str, Any] = {}

        if update.effective_user:
            metadata["username"] = update.effective_user.username or ""
            metadata["user_id"] = update.effective_user.id

        logger.info("Telegram message from chat_id=%s: %s", chat_id, user_text[:100])

        try:
            if self._handler is None:
                return
            response = await self._handler(chat_id, user_text, metadata)
            if response.strip():
                await self.send(chat_id, response)
        except Exception:
            logger.exception("Error handling message from chat_id=%s", chat_id)
            await self.send(chat_id, "Sorry, something went wrong.")
