"""LLM provider abstraction layer with dynamic registration and model discovery."""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

CACHE_PATH = Path("./data/model_cache.json")
CACHE_TTL = 86400  # 24 hours


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    """Discovered model metadata."""

    id: str
    provider: str
    display_name: str = ""
    created: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "provider": self.provider, "display_name": self.display_name, "created": self.created}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelInfo:
        return cls(
            id=d["id"], provider=d["provider"], display_name=d.get("display_name", ""), created=d.get("created", 0)
        )


class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128000,
        output_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    @property
    def model_name(self) -> str: ...


@runtime_checkable
class DiscoverableProvider(Protocol):
    """Provider that supports model discovery."""

    async def discover_models(self) -> list[ModelInfo]: ...


# --- Provider Registry ---

_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("models.anthropic", "AnthropicProvider"),
    "openai": ("models.openai", "OpenAIProvider"),
}


def create_provider(provider_type: str, **kwargs: Any) -> LLMProvider:
    if provider_type not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider: {provider_type}. Available: {available}")

    module_path, class_name = _PROVIDER_REGISTRY[provider_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


# --- Model Cache ---


class ModelCache:
    """Disk-cached model discovery results."""

    def __init__(self, path: Path = CACHE_PATH, ttl: int = CACHE_TTL) -> None:
        self._path = path
        self._ttl = ttl

    def get(self, provider: str) -> list[ModelInfo] | None:
        """Return cached models for a provider, or None if stale/missing."""
        data = self._load()
        entry = data.get(provider)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self._ttl:
            return None
        return [ModelInfo.from_dict(m) for m in entry.get("models", [])]

    def put(self, provider: str, models: list[ModelInfo]) -> None:
        data = self._load()
        data[provider] = {"ts": time.time(), "models": [m.to_dict() for m in models]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
