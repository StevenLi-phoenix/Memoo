"""Telegram channel using python-telegram-bot with bind-code auth.

Security model:
  - On startup, a one-time bind code is generated and printed to the terminal.
  - Send /bind <code> in Telegram to register as an allowed user.
  - Once at least one user is bound, only bound users can interact (fail-close).
  - Bound user IDs are persisted to config via the on_bind callback.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Callable

from telegram import Update
from telegram.ext import Application, CommandHandler, filters
from telegram.ext import MessageHandler as TGMessageHandler

from channels import MessageHandler

logger = logging.getLogger(__name__)


class TelegramChannel:
    """Telegram bot channel with polling or webhook support and bind-code auth."""

    def __init__(
        self,
        token: str,
        mode: str = "polling",
        allowed_users: list[str] | None = None,
        on_bind: Callable[[str], None] | None = None,
    ) -> None:
        self._token = token
        self._mode = mode
        self._allowed_users: set[str] = set(allowed_users) if allowed_users else set()
        self._on_bind = on_bind  # callback to persist newly bound user
        self._bind_code: str = secrets.token_hex(4)  # 8-char hex code
        self._app: Application | None = None  # type: ignore[type-arg]
        self._handler: MessageHandler | None = None

    @property
    def bind_code(self) -> str:
        return self._bind_code

    def _is_allowed(self, update: Update) -> bool:
        """Check if user is in the allowlist. Fail-close: no users = reject all."""
        if not self._allowed_users:
            return False
        user = update.effective_user
        if not user:
            return False
        return str(user.id) in self._allowed_users

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._app = Application.builder().token(self._token).build()

        # /bind is always available (handles its own auth via bind code)
        self._app.add_handler(CommandHandler("bind", self._on_bind_command))

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

        logger.info("Telegram channel starting: mode=%s, bound_users=%d", self._mode, len(self._allowed_users))
        if not self._allowed_users:
            logger.warning("Telegram: no bound users — send /bind %s to the bot to register", self._bind_code)

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

    async def _on_bind_command(self, update: Update, _context: Any) -> None:
        """Handle /bind <code> — register the sender as an allowed user."""
        if not update.message or not update.effective_chat or not update.effective_user:
            return

        chat_id = str(update.effective_chat.id)
        user_id = str(update.effective_user.id)
        username = update.effective_user.username or ""
        text = (update.message.text or "").strip()
        parts = text.split(maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ""

        if not secrets.compare_digest(code, self._bind_code):
            logger.warning("Telegram bind failed: user=%s (%s), wrong code", user_id, username)
            await self.send(chat_id, "Invalid bind code.")
            return

        if user_id in self._allowed_users:
            await self.send(chat_id, "You are already registered.")
            return

        self._allowed_users.add(user_id)
        # Regenerate bind code after successful bind (one-time use)
        self._bind_code = secrets.token_hex(4)
        logger.info("Telegram user bound: id=%s username=%s", user_id, username)

        # Persist via callback
        if self._on_bind:
            self._on_bind(user_id)

        await self.send(chat_id, f"Registered! User {user_id} ({username}) is now allowed.")

    async def _on_start(self, update: Update, _context: Any) -> None:
        """Handle /start command."""
        if not self._is_allowed(update):
            return
        if update.effective_chat:
            await self.send(
                str(update.effective_chat.id), "Hi! I'm Memoo. Send me a message.\nType /help for commands."
            )

    async def _on_command(self, update: Update, _context: Any) -> None:
        """Handle all slash commands — forward to handle_message which routes to core/commands.py."""
        if not self._is_allowed(update):
            return
        if not update.message or not update.effective_chat or not self._handler:
            return
        chat_id = str(update.effective_chat.id)
        text = update.message.text or ""
        response = await self._handler(chat_id, text, {"platform": "telegram"})
        if response.strip():
            await self.send(chat_id, response)

    async def _on_message(self, update: Update, _context: Any) -> None:
        """Handle incoming text messages."""
        if not self._is_allowed(update):
            return
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
