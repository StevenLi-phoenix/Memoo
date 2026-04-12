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

from dotenv import load_dotenv

from channels import create_channel
from core.agent import Agent, TurnResult
from core.config import AppConfig
from core.crash import crash_boundary
from core.crash import init as init_crash_handler
from core.gateway import Gateway
from core.heartbeat import Heartbeat
from core.hooks import HookRegistry, make_rate_limit_hook, sandbox_path_hook
from core.memory import Memory
from core.scheduler import Scheduler
from core.tools import ToolRegistry
from models import (
    DiscoverableProvider,
    LLMProvider,
    Message,
    ModelCache,
    ModelInfo,
    configure_model_cache,
    create_provider,
)
from tools import auto_discover_tools

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("memoo")

init_crash_handler(logs_dir=".logs", webhook_url=os.environ.get("MEMOO_CRASH_WEBHOOK", ""))


def load_memory_files(memory_dir: Path) -> str:
    """Load all .md files from the memory/ directory for system prompt injection.

    memory/ is a general-purpose knowledge store. Any component (dream, agent,
    tools, or the user) can write .md files here. All are concatenated and
    injected into the system prompt at startup.
    """
    if not memory_dir.is_dir():
        return ""

    parts: list[str] = []
    for md_file in sorted(memory_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"## {md_file.stem}\n\n{content}")

    if not parts:
        return ""
    return "\n\n---\n\n# Persistent Memory\n\n" + "\n\n".join(parts)


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
    if not models:
        return None
    if preference:
        for m in models:
            if m.id == preference:
                return m.id
        for m in models:
            if preference in m.id:
                return m.id
    by_created = sorted(models, key=lambda m: m.created, reverse=True)
    return by_created[0].id


async def build_llm_registry(
    cfg: AppConfig,
) -> tuple[LLMProvider, list[LLMProvider], dict[str, list[ModelInfo]], dict[str, LLMProvider]]:
    """Build LLM providers with model discovery from AppConfig."""
    configure_model_cache(ttl=cfg.llm.model_cache_ttl)
    web_search = cfg.tools.web_search
    cache = ModelCache()

    built: dict[str, LLMProvider] = {}
    all_discovered: dict[str, list[ModelInfo]] = {}

    for p_conf in cfg.llm.providers:
        try:
            api_key = _get_api_key(p_conf.provider)
        except ValueError as e:
            logger.warning("Skipping %s: %s", p_conf.name, e)
            continue

        kwargs: dict[str, Any] = {"api_key": api_key}
        if p_conf.provider == "anthropic":
            kwargs["web_search"] = web_search
        provider = create_provider(p_conf.provider, **kwargs)

        models = await _discover_and_cache(p_conf.provider, provider, cache)
        all_discovered[p_conf.name] = models

        model_id = _pick_model(models, p_conf.model)
        if not model_id:
            logger.warning("No models available for %s, skipping", p_conf.name)
            continue

        provider.model_name = model_id  # type: ignore[union-attr]

        if p_conf.provider == "anthropic" and p_conf.advisor:
            advisor_id = _pick_model(models, p_conf.advisor)
            if advisor_id and advisor_id != model_id:
                provider._advisor_model = advisor_id  # type: ignore[union-attr]
                logger.info("%s: advisor model %s", p_conf.name, advisor_id)

        built[p_conf.name] = provider
        logger.info("%s: selected model %s (from %d available)", p_conf.name, model_id, len(models))

    if not built:
        logger.error("No LLM providers available")
        sys.exit(1)

    default_name = cfg.llm.default
    if default_name not in built:
        default_name = next(iter(built))

    default_llm = built[default_name]
    fallback_chain = [built[n] for n in cfg.llm.fallback if n in built and n != default_name]

    logger.info("LLM: default=%s (%s), fallbacks=%d", default_name, default_llm.model_name, len(fallback_chain))
    return default_llm, fallback_chain, all_discovered, built


class Memoo:
    """Main application orchestrator. Agent is the central hub."""

    def __init__(self, cfg: AppConfig) -> None:
        import time

        self.cfg = cfg
        self._start_time = time.monotonic()
        self.llm: LLMProvider | None = None
        self.fallback_llms: list[LLMProvider] = []
        self.discovered_models: dict[str, list[ModelInfo]] = {}
        self._providers: dict[str, LLMProvider] = {}

        self.memory = Memory(
            db_path=cfg.memory.db_path,
            max_context=cfg.memory.max_context_messages,
            token_window=cfg.memory.token_window,
            chars_per_token=cfg.agent.chars_per_token,
        )

        db_dir = str(Path(cfg.memory.db_path).parent)
        self.scheduler = Scheduler(db_path=str(Path(db_dir) / "schedules.db"))

        self.heartbeat = Heartbeat(
            heartbeat_dir=cfg.paths.heartbeat_dir,
            default_interval=cfg.heartbeat.default_interval,
        )
        from core.gateway import create_server_ssl

        self.gateway = Gateway(
            host=cfg.host,
            port=cfg.port,
            ssl_ctx=create_server_ssl(host=cfg.host, certs_dir=Path(cfg.paths.certs_dir)),
        )
        self.tools = ToolRegistry()
        self.hooks = HookRegistry()
        self.hooks.add_hook(sandbox_path_hook)
        self.hooks.add_hook(make_rate_limit_hook(cfg.hooks.rate_limit_per_minute, cfg.hooks.rate_limit_window))
        self.hooks.allow("current_time", "list_schedules", "list_memories", "get_config")

        self.agent: Agent | None = None
        self.channels: list[Any] = []
        self._channel_map: dict[str, Any] = {}
        self._active_tasks: dict[str, asyncio.Task[TurnResult]] = {}
        self._current_topics: dict[str, str] = {}
        self._chat_locks: dict[str, asyncio.Lock] = {}  # per-chat_id serialization

    async def start(self) -> None:
        self.llm, self.fallback_llms, self.discovered_models, self._providers = await build_llm_registry(self.cfg)

        # Configure embedding provider
        from core.embeddings import configure as configure_embeddings

        emb = self.cfg.embedding
        if not emb.api_key:
            emb.api_key = os.environ.get("EMBEDDING_API_KEY", "")
        configure_embeddings(emb)

        await self.memory.init()
        await self.scheduler.init()

        # Discover skills and inject metadata into system prompt
        from core.skills import SkillRegistry

        self.skill_registry = SkillRegistry(skills_dir=Path(self.cfg.paths.skills_dir))
        self.skill_registry.discover()

        system_prompt = load_system_prompt(self.cfg.agent.system_prompt)
        skills_section = self.skill_registry.build_system_prompt_section()
        if skills_section:
            system_prompt += "\n" + skills_section

        # Inject knowledge files from memory/ directory
        memory_context = load_memory_files(Path(self.cfg.paths.memory_dir))
        if memory_context:
            system_prompt += "\n" + memory_context
            logger.info("Memory files injected into system prompt (%d chars)", len(memory_context))

        self.agent = Agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            max_rounds=self.cfg.agent.max_tool_rounds,
            fallback_llms=self.fallback_llms,
            hooks=self.hooks,
            memory=self.memory,
            gateway=self.gateway,
            agent_config=self.cfg.agent,
        )

        auto_discover_tools(
            self.tools,
            deps={
                "memory": self.memory,
                "scheduler": self.scheduler,
                "sandbox_dir": self.cfg.paths.sandbox_dir,
                "config": self.cfg,
                "skill_registry": self.skill_registry,
                "app": self,
            },
        )

        # Start gateway (TCP API for TUI and external clients)
        await self.gateway.start(self.handle_message)

        # Start messaging channels
        await self._start_channels()

        await self.scheduler.start(self._handle_scheduled)
        await self.heartbeat.start(self._handle_heartbeat)
        logger.info("Memoo is running. Press Ctrl+C to stop.")

    async def _start_channels(self) -> None:
        for channel_type, ch_config in self.cfg.channels.items():
            if not ch_config.enabled:
                continue
            try:
                kwargs = self._resolve_channel_kwargs(channel_type, ch_config)
                ch = create_channel(channel_type, **kwargs)
                await ch.start(self.handle_message)
                self.channels.append(ch)
                self._channel_map[channel_type] = ch
                logger.info("Channel started: %s", channel_type)

                # Log bind code for Telegram if no users bound yet
                if channel_type == "telegram" and hasattr(ch, "bind_code") and not ch_config.allowed_users:
                    logger.info("Telegram bind code: %s  (send /bind %s to the bot)", ch.bind_code, ch.bind_code)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping channel %s: %s", channel_type, e)

    def _resolve_channel_kwargs(self, channel_type: str, ch_config: Any) -> dict[str, Any]:
        env_prefix = channel_type.upper()
        if channel_type == "telegram":
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            if not token:
                raise ValueError("TELEGRAM_BOT_TOKEN not set")

            def _on_bind(user_id: str) -> None:
                """Persist newly bound Telegram user to config.yaml."""
                tg_cfg = self.cfg.channels.get("telegram")
                if tg_cfg and user_id not in tg_cfg.allowed_users:
                    tg_cfg.allowed_users.append(user_id)
                    self.cfg.save()
                    logger.info("Telegram user %s persisted to config.yaml", user_id)

            return {
                "token": token,
                "mode": ch_config.mode,
                "allowed_users": ch_config.allowed_users,
                "on_bind": _on_bind,
            }
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
            return kwargs

    @crash_boundary("Memoo.handle_message")
    async def handle_message(self, chat_id: str, text: str, metadata: dict[str, Any]) -> str:
        if self.agent is None or self.llm is None:
            return "Error: Memoo not initialized"

        max_msg_len = self.cfg.agent.max_message_len
        if len(text) > max_msg_len:
            text = text[:max_msg_len] + "\n...(truncated)"
            logger.warning("Message truncated: chat_id=%s", chat_id)

        # Slash commands and skill triggers — handled without LLM call
        if text.startswith("/"):
            from core.commands import handle_command

            cmd_result = await handle_command(
                text,
                chat_id,
                deps={
                    "memory": self.memory,
                    "config": self.cfg,
                    "agent": self.agent,
                    "scheduler": self.scheduler,
                    "app": self,
                },
            )
            if cmd_result is not None:
                return cmd_result

            # Not a built-in command — check if it's a skill trigger
            if self.skill_registry:
                parts = text.split(maxsplit=1)
                skill_name = parts[0][1:].lower()
                meta = self.skill_registry.get_meta(skill_name)
                if meta:
                    instructions = self.skill_registry.load_instructions(skill_name) or ""
                    user_msg = parts[1] if len(parts) > 1 else ""
                    text = f"[Skill activated: {meta.name}]\n\n{instructions}\n\nUser request: {user_msg}"
                    logger.info("Skill triggered: %s (chat_id=%s)", skill_name, chat_id)

        # Fast path: inject into an already-running turn (no lock needed).
        # The injector does NOT finalize — the task owner handles that.
        active = self._active_tasks.get(chat_id)
        if active and not active.done():
            if self.agent.inject(chat_id, text):
                logger.info("Injected message into active turn for chat_id=%s", chat_id)
                await self.memory.add_message(chat_id, Message(role="user", content=text, metadata=metadata))
                try:
                    result = await active
                except asyncio.CancelledError:
                    return "(processing interrupted)"
                # Return response only — the task creator will call _finalize_turn
                return result.response

        # Serialize the create-task path per chat_id to prevent two handlers
        # from both creating tasks and both calling _finalize_turn.
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            # Re-check: another handler may have created a task while we waited
            active = self._active_tasks.get(chat_id)
            if active and not active.done():
                # Inject (now under lock, safe) or cancel
                if self.agent.inject(chat_id, text):
                    logger.info("Injected message (post-lock) for chat_id=%s", chat_id)
                    await self.memory.add_message(chat_id, Message(role="user", content=text, metadata=metadata))
                    try:
                        result = await active
                    except asyncio.CancelledError:
                        return "(processing interrupted)"
                    return result.response

                logger.info("Inject failed, cancelling active agent for chat_id=%s", chat_id)
                self.agent.cancel(chat_id)
                active.cancel()
                try:
                    await active
                except asyncio.CancelledError:
                    pass
                await self.memory.add_message(
                    chat_id, Message(role="assistant", content="(interrupted by new message)")
                )

            history = await self.memory.get_history(chat_id)
            await self.memory.add_message(chat_id, Message(role="user", content=text, metadata=metadata))

            context = {
                "chat_id": chat_id,
                "sandbox_dir": self.cfg.paths.sandbox_dir,
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

            return await self._finalize_turn(chat_id, result)

    async def _finalize_turn(self, chat_id: str, result: TurnResult) -> str:
        """Post-turn bookkeeping: persist reply, update topic, compress if needed."""
        await self.memory.add_message(
            chat_id,
            Message(
                role="assistant",
                content=result.response,
                metadata={"topic": result.current_topic, "memory_notes": result.memory_notes, "usage": result.usage},
            ),
        )

        if not result.did_success:
            logger.warning("Agent reported failure for chat_id=%s: %s", chat_id, result.response[:200])

        if result.memory_notes:
            logger.info("Memory notes for chat_id=%s: %s", chat_id, result.memory_notes)
        if result.current_topic:
            self._current_topics[chat_id] = result.current_topic
        if result.should_compress:
            await self._compact_memory(chat_id)

        if result.usage:
            self.gateway.set_reply_extra(chat_id, {"usage": result.usage})

        return result.response

    @crash_boundary("Memoo._handle_scheduled")
    async def _handle_scheduled(self, chat_id: str, prompt: str, channel_name: str) -> str:
        response = await self.handle_message(chat_id, prompt, {"source": "scheduler"})
        if response.strip():
            ch = self._channel_map.get(channel_name)
            if ch:
                await ch.send(chat_id, f"[Scheduled Task]\n{response}")
        return response

    async def _handle_heartbeat(self, prompt: str, context: dict[str, Any]) -> str:
        heartbeat_chat = f"__heartbeat__{context.get('task_name', 'default')}"
        response = await self.handle_message(heartbeat_chat, prompt, {"source": "heartbeat", **context})
        if response.strip():
            task_name = context.get("task_name", "heartbeat")
            notification = f"[Heartbeat: {task_name}]\n{response}"
            for ch in self.channels:
                try:
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
            tokens_so_far += len(msg.content) // self.cfg.agent.chars_per_token
            if token_count - tokens_so_far <= target_tokens:
                split_idx = i + 1
                break
        else:
            split_idx = len(history) // 2

        if split_idx < 2:
            return

        old_messages = history[:split_idx]
        old_text = "\n".join(f"[{m.role}]: {m.content}" for m in old_messages if m.content)

        # Use the cheap compressor model (e.g. Haiku) for summarization — same as
        # Agent._enforce_context_window. Falls back to primary model if unavailable.
        compressor = self.agent.compressor if self.agent else self.llm
        summary_response = await compressor.chat(
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

        summary_msg = Message(
            role="system",
            content=(
                f"[Conversation summary — topic: {current_topic}]: {summary}\n\n"
                "(Use search_memory to retrieve full archived conversations.)"
            ),
        )
        await self.memory.compact_replace(chat_id, [summary_msg] + history[split_idx:])

    async def stop(self) -> None:
        logger.info("Shutting down Memoo...")
        for task in self._active_tasks.values():
            task.cancel()
        await self.heartbeat.stop()
        await self.scheduler.stop()
        for ch in self.channels:
            await ch.stop()
        await self.gateway.stop()
        await self.memory.close()
        logger.info("Memoo stopped.")


async def main() -> None:
    cfg = AppConfig.load()
    app = Memoo(cfg)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await app.start()
    await stop_event.wait()
    await app.stop()

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
