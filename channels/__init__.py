"""Channel abstraction for messaging platforms with dynamic registration."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

MessageHandler = Callable[[str, str, dict[str, Any]], Awaitable[str]]


class Channel(Protocol):
    """Protocol for messaging channel implementations."""

    async def start(self, handler: MessageHandler) -> None: ...
    async def send(self, chat_id: str, text: str) -> None: ...
    async def stop(self) -> None: ...


# Maps channel type name -> (module_path, class_name)
_CHANNEL_REGISTRY: dict[str, tuple[str, str]] = {
    "telegram": ("channels.telegram", "TelegramChannel"),
    "wechat": ("channels.wechat", "WeChatChannel"),
    "tui": ("channels.tui", "TUIChannel"),
}


def register_channel(name: str, module_path: str, class_name: str) -> None:
    """Register a new channel type for dynamic loading."""
    _CHANNEL_REGISTRY[name] = (module_path, class_name)
    logger.info("Registered channel type: %s -> %s.%s", name, module_path, class_name)


def create_channel(channel_type: str, **kwargs: Any) -> Channel:
    """Dynamically create a channel instance by type name."""
    if channel_type not in _CHANNEL_REGISTRY:
        available = ", ".join(sorted(_CHANNEL_REGISTRY.keys()))
        raise ValueError(f"Unknown channel type: {channel_type}. Available: {available}")

    module_path, class_name = _CHANNEL_REGISTRY[channel_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
