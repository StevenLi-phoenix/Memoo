"""Memoo — Lightweight personal AI agent bot.

Inspired by NanoClaw. Built on Claude API.
The agent is the central entry point for all interactions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from channels import create_channel
from core.agent import Agent, TurnResult
from core.crash import crash_boundary
from core.crash import init as init_crash_handler
from core.heartbeat import Heartbeat
from core.hooks import HookRegistry, rate_limit_hook, sandbox_path_hook
from core.memory import Memory
from core.scheduler import Scheduler
from core.tools import ToolRegistry
from models import DiscoverableProvider, LLMProvider, Message, ModelCache, ModelInfo, create_provider
from tools import auto_discover_tools

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("memoo")

# Initialize crash handler — logs to .logs/, queues for auto-fix
init_crash_handler(
    logs_dir=".logs",
    webhook_url=os.environ.get("MEMOO_CRASH_WEBHOOK", ""),
)


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    logger.info("Config loaded from %s", path)
    return config


def load_system_prompt(prompt_path: str) -> str:
    p = Path(prompt_path)
    if p.exists() and p.is_file():
        content = p.read_text(encoding="utf-8").strip()
        logger.info("System prompt loaded from %s (%d chars)", prompt_path, len(content))
        return content
    logger.warning("System prompt not found: %s", prompt_path)
    return "You are Memoo, a helpful AI assistant."


_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _get_api_key(provider_type: str) -> str:
    env_var = _API_KEY_ENV.get(provider_type, f"{provider_type.upper()}_API_KEY")
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(f"{env_var} not set")
    return key


async def _discover_and_cache(provider_type: str, provider: Any, cache: ModelCache) -> list[ModelInfo]:
    """Discover models from provider, using cache if fresh."""
    cached = cache.get(provider_type)
    if cached:
        logger.debug("%s: using %d cached models", provider_type, len(cached))
        return cached

    if isinstance(provider, DiscoverableProvider):
        models = await provider.discover_models()
        if models:
            cache.put(provider_type, models)
            return models

    logger.warning("%s: no models discovered", provider_type)
    return []


def _pick_model(models: list[ModelInfo], preference: str = "") -> str | None:
    """Pick the best model from discovered list.

    If preference is set, try exact match first.
    Otherwise pick the newest model.
    """
    if not models:
        return None

    if preference:
        # Exact match
        for m in models:
            if m.id == preference:
                return m.id
        # Substring match
        for m in models:
            if preference in m.id:
                return m.id

    # Default: newest by created timestamp
    by_created = sorted(models, key=lambda m: m.created, reverse=True)
    return by_created[0].id


async def build_llm_registry(
    llm_config: dict[str, Any], tools_config: dict[str, Any]
) -> tuple[LLMProvider, list[LLMProvider], dict[str, list[ModelInfo]]]:
    """Build LLM providers with model discovery.

    Returns (default_provider, fallback_chain, discovered_models_by_provider).
    """
    web_search = tools_config.get("web_search", True)
    cache = ModelCache()
    providers_conf: list[dict[str, Any]] = llm_config.get("providers", [])
    default_provider_name = llm_config.get("default", "")
    fallback_names: list[str] = llm_config.get("fallback", [])

    built: dict[str, LLMProvider] = {}
    all_discovered: dict[str, list[ModelInfo]] = {}

    for p_conf in providers_conf:
        p_type = p_conf["provider"] if isinstance(p_conf, dict) else p_conf
        p_name = p_conf.get("name", p_type) if isinstance(p_conf, dict) else p_conf
        preference = p_conf.get("model", "") if isinstance(p_conf, dict) else ""

        try:
            api_key = _get_api_key(p_type)
        except ValueError as e:
            logger.warning("Skipping %s: %s", p_name, e)
            continue

        # Create provider without model (for discovery)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if p_type == "anthropic":
            kwargs["web_search"] = web_search
        provider = create_provider(p_type, **kwargs)

        # Discover models
        models = await _discover_and_cache(p_type, provider, cache)
        all_discovered[p_name] = models

        # Pick executor model
        model_id = _pick_model(models, preference)
        if not model_id:
            logger.warning("No models available for %s, skipping", p_name)
            continue

        # Set the chosen model
        provider.model_name = model_id  # type: ignore[union-attr]

        # Resolve advisor model from discovered models (Anthropic only)
        if p_type == "anthropic":
            advisor_pref = p_conf.get("advisor", "") if isinstance(p_conf, dict) else ""
            if advisor_pref:
                advisor_id = _pick_model(models, advisor_pref)
                if advisor_id and advisor_id != model_id:
                    provider._advisor_model = advisor_id  # type: ignore[union-attr]
                    logger.info("%s: advisor model %s", p_name, advisor_id)
                else:
                    logger.warning("%s: advisor model '%s' not found or same as executor", p_name, advisor_pref)

        built[p_name] = provider
        logger.info("%s: selected model %s (from %d available)", p_name, model_id, len(models))

    if not built:
        logger.error("No LLM providers available")
        sys.exit(1)

    # Resolve default
    if default_provider_name not in built:
        default_provider_name = next(iter(built))

    default_llm = built[default_provider_name]
    fallback_chain = [built[n] for n in fallback_names if n in built and n != default_provider_name]

    logger.info(
        "LLM: default=%s (%s), fallbacks=%d", default_provider_name, default_llm.model_name, len(fallback_chain)
    )
    return default_llm, fallback_chain, all_discovered


class Memoo:
    """Main application orchestrator. Agent is the central hub."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.llm: LLMProvider | None = None
        self.fallback_llms: list[LLMProvider] = []
        self.discovered_models: dict[str, list[ModelInfo]] = {}

        mem_config = config.get("memory", {})
        self.memory = Memory(
            db_path=mem_config.get("db_path", "./data/memory.db"),
            max_context=mem_config.get("max_context_messages", 200),
            token_window=mem_config.get("token_window", 100_000),
        )

        db_dir = str(Path(mem_config.get("db_path", "./data/memory.db")).parent)
        self.scheduler = Scheduler(db_path=str(Path(db_dir) / "schedules.db"))

        self.heartbeat = Heartbeat(heartbeat_dir="./heartbeat")
        self.tools = ToolRegistry()
        self.hooks = HookRegistry()
        self.hooks.add_hook(sandbox_path_hook)
        self.hooks.add_hook(rate_limit_hook)
        self.hooks.allow("current_time", "list_schedules", "list_memories")

        self.agent: Agent | None = None
        self.channels: list[Any] = []
        self._channel_map: dict[str, Any] = {}
        self._active_tasks: dict[str, asyncio.Task[TurnResult]] = {}
        self._current_topics: dict[str, str] = {}

    async def start(self) -> None:
        # Discover models and build LLM providers
        llm_config = self.config.get("llm", {})
        tools_config = self.config.get("tools", {})
        self.llm, self.fallback_llms, self.discovered_models = await build_llm_registry(llm_config, tools_config)

        # Build agent
        prompt_path = self.config.get("agent", {}).get("system_prompt", "systemprompt/default.md")
        system_prompt = load_system_prompt(prompt_path)
        max_rounds = self.config.get("agent", {}).get("max_tool_rounds", 0)

        self.agent = Agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            max_rounds=max_rounds,
            fallback_llms=self.fallback_llms,
            hooks=self.hooks,
        )

        await self.memory.init()
        await self.scheduler.init()

        # Auto-discover tools
        auto_discover_tools(
            self.tools, deps={"memory": self.memory, "scheduler": self.scheduler, "sandbox_dir": "./sandbox"}
        )

        # Channels (fallback to TUI)
        await self._start_channels(self.config.get("channels", {}))
        if not self.channels:
            logger.info("No remote channels started, falling back to TUI")
            tui = create_channel("tui")
            await tui.start(self.handle_message)
            self.channels.append(tui)
            self._channel_map["tui"] = tui

        await self.scheduler.start(self._handle_scheduled)
        await self.heartbeat.start(self._handle_heartbeat)
        logger.info("Memoo is running. Press Ctrl+C to stop.")

    async def _start_channels(self, channels_config: dict[str, Any]) -> None:
        for channel_type, ch_config in channels_config.items():
            if not isinstance(ch_config, dict) or not ch_config.get("enabled", False):
                continue
            try:
                kwargs = self._resolve_channel_kwargs(channel_type, ch_config)
                ch = create_channel(channel_type, **kwargs)
                await ch.start(self.handle_message)
                self.channels.append(ch)
                self._channel_map[channel_type] = ch
                logger.info("Channel started: %s", channel_type)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping channel %s: %s", channel_type, e)

    @staticmethod
    def _resolve_channel_kwargs(channel_type: str, ch_config: dict[str, Any]) -> dict[str, Any]:
        env_prefix = channel_type.upper()
        if channel_type == "telegram":
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            if not token:
                raise ValueError("TELEGRAM_BOT_TOKEN not set")
            return {"token": token, "mode": ch_config.get("mode", "polling")}
        elif channel_type == "wechat":
            token = os.environ.get("WECHAT_ILINK_TOKEN", "")
            if not token:
                raise ValueError("WECHAT_ILINK_TOKEN not set")
            return {"token": token, "uin": os.environ.get("WECHAT_UIN", "")}
        else:
            token = os.environ.get(f"{env_prefix}_TOKEN", "")
            kwargs: dict[str, Any] = {}
            if token:
                kwargs["token"] = token
            kwargs.update({k: v for k, v in ch_config.items() if k != "enabled"})
            return kwargs

    @crash_boundary("Memoo.handle_message")
    async def handle_message(self, chat_id: str, text: str, metadata: dict[str, Any]) -> str:
        if self.agent is None or self.llm is None:
            return "Error: Memoo not initialized"

        # Truncate excessively long messages
        MAX_MSG_LEN = 50_000
        if len(text) > MAX_MSG_LEN:
            text = text[:MAX_MSG_LEN] + "\n...(truncated)"
            logger.warning("Message from chat_id=%s truncated: %d -> %d chars", chat_id, len(text), MAX_MSG_LEN)

        if metadata.get("command") == "clear":
            await self.memory.clear(chat_id)
            return "Memory cleared."

        active = self._active_tasks.get(chat_id)
        if active and not active.done():
            logger.info("Interrupting active agent for chat_id=%s", chat_id)
            self.agent.cancel(chat_id)
            active.cancel()
            try:
                await active
            except asyncio.CancelledError:
                pass
            await self.memory.add_message(chat_id, Message(role="assistant", content="(interrupted by new message)"))

        await self.memory.add_message(chat_id, Message(role="user", content=text, metadata=metadata))

        history = await self.memory.get_history(chat_id)
        context = {
            "chat_id": chat_id,
            "sandbox_dir": "./sandbox",
            "current_topic": self._current_topics.get(chat_id, ""),
            **metadata,
        }

        task = asyncio.create_task(self.agent.run(text, history=history, context=context))
        self._active_tasks[chat_id] = task

        try:
            result = await task
        except asyncio.CancelledError:
            return "(processing interrupted)"
        finally:
            self._active_tasks.pop(chat_id, None)

        await self.memory.add_message(
            chat_id,
            Message(
                role="assistant",
                content=result.response,
                metadata={"topic": result.current_topic, "topic_changed": result.topic_changed, "usage": result.usage},
            ),
        )

        if result.current_topic:
            self._current_topics[chat_id] = result.current_topic
        if result.should_compress:
            logger.info("Agent requested compression: %s", result.compress_reason)
            await self._compact_memory(chat_id)

        return result.response

    @crash_boundary("Memoo._handle_scheduled")
    async def _handle_scheduled(self, chat_id: str, prompt: str, channel_name: str) -> str:
        """Handle scheduled task — run agent and deliver result to the target chat."""
        response = await self.handle_message(chat_id, prompt, {"source": "scheduler"})

        # Deliver to target channel
        ch = self._channel_map.get(channel_name)
        if ch:
            await ch.send(chat_id, f"[Scheduled Task]\n{response}")
        return response

    async def _handle_heartbeat(self, prompt: str, context: dict[str, Any]) -> str:
        """Handle heartbeat — run in system chat, forward actionable results to all active channels."""
        heartbeat_chat = f"__heartbeat__{context.get('task_name', 'default')}"
        response = await self.handle_message(heartbeat_chat, prompt, {"source": "heartbeat", **context})

        # Forward non-trivial results to all active channels
        if response.strip().lower() != "all clear":
            task_name = context.get("task_name", "heartbeat")
            notification = f"[Heartbeat: {task_name}]\n{response}"
            for ch in self.channels:
                try:
                    # Send to a default/broadcast chat — each channel picks its own
                    await ch.send("__broadcast__", notification)
                except Exception:
                    logger.debug("Failed to broadcast heartbeat to channel", exc_info=True)

        return response

    async def _compact_memory(self, chat_id: str) -> None:
        if self.llm is None:
            return
        history = await self.memory.get_history(chat_id)
        if len(history) < 6:
            return

        current_topic = self._current_topics.get(chat_id, "general conversation")
        token_count = await self.memory.get_token_count(chat_id)
        target_tokens = int(self.memory._token_window * 0.6)

        tokens_so_far = 0
        split_idx = 0
        for i, msg in enumerate(history):
            tokens_so_far += len(msg.content) // 3
            if token_count - tokens_so_far <= target_tokens:
                split_idx = i + 1
                break
        else:
            split_idx = len(history) // 2

        if split_idx < 2:
            return

        old_messages = history[:split_idx]
        old_text = "\n".join(f"[{m.role}]: {m.content}" for m in old_messages if m.content)

        summary_response = await self.llm.chat(
            messages=[
                Message(
                    role="user",
                    content=f"Summarize this conversation. Current topic: {current_topic}. "
                    f"Preserve key facts and context.\n\n{old_text}",
                )
            ],
            system="Output a concise conversation summary preserving key context.",
            max_tokens=500,
        )
        summary = summary_response.text or ""

        await self.memory.archive_messages(chat_id=chat_id, messages=old_messages, topic=current_topic, summary=summary)
        logger.info("Compacted chat_id=%s: archived %d msgs, %d tokens freed", chat_id, split_idx, tokens_so_far)

        await self.memory.clear(chat_id)
        await self.memory.add_message(
            chat_id,
            Message(
                role="system",
                content=(
                    f"[Conversation summary — topic: {current_topic}]: {summary}\n\n"
                    "(Use search_memory to retrieve full archived conversations.)"
                ),
            ),
        )
        for msg in history[split_idx:]:
            await self.memory.add_message(chat_id, msg)

    async def stop(self) -> None:
        logger.info("Shutting down Memoo...")
        for task in self._active_tasks.values():
            task.cancel()
        await self.heartbeat.stop()
        await self.scheduler.stop()
        for ch in self.channels:
            await ch.stop()
        await self.memory.close()
        logger.info("Memoo stopped.")


async def main() -> None:
    config = load_config()
    app = Memoo(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await app.start()
    await stop_event.wait()
    await app.stop()

    # Force exit — kill any lingering executor threads
    import os

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
