"""Configuration data structure — mirrors config.yaml as in-memory dataclass.

Single source of truth for runtime config. Agent can update via tools.
Changes persist back to config.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    name: str = ""
    provider: str = ""
    model: str = ""
    advisor: str = ""


@dataclass
class LLMConfig:
    default: str = "anthropic"
    providers: list[ProviderConfig] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)
    model_cache_ttl: int = 86400  # seconds (24h)


@dataclass
class AgentConfig:
    system_prompt: str = "systemprompt/default.md"
    max_tool_rounds: int = 0  # 0 = use hard_max_rounds
    hard_max_rounds: int = 200
    context_window_tokens: int = 128_000
    max_message_len: int = 50_000
    chars_per_token: int = 3


@dataclass
class MemoryConfig:
    db_path: str = "./data/memory.db"
    max_context_messages: int = 200
    token_window: int = 100_000


@dataclass
class ChannelConfig:
    enabled: bool = False
    mode: str = "polling"
    allowed_users: list[str] = field(default_factory=list)


@dataclass
class ToolsConfig:
    web_search: bool = True
    run_code: bool = True
    read_file: bool = True


@dataclass
class SandboxConfig:
    timeout: int = 300  # seconds — per-execution timeout
    max_output: int = 10_000  # chars


@dataclass
class EmbeddingConfig:
    provider: str = "off"  # local | openai | off
    base_url: str = "http://localhost:1234/v1"
    model: str = ""
    api_key: str = ""


@dataclass
class SubagentConfig:
    max_depth: int = 3
    default_max_rounds: int = 10


@dataclass
class PathsConfig:
    sandbox_dir: str = "./sandbox"
    heartbeat_dir: str = "./heartbeat"
    skills_dir: str = "./skills"
    memory_dir: str = "./memory"
    logs_dir: str = ".logs"
    certs_dir: str = "./certs"


@dataclass
class DreamConfig:
    batch_size: int = 30


@dataclass
class HooksConfig:
    rate_limit_per_minute: int = 30
    rate_limit_window: int = 60  # seconds


@dataclass
class HeartbeatConfig:
    default_interval: int = 3600  # seconds (1h)


@dataclass
class SchedulerConfig:
    default_channel: str = "telegram"


@dataclass
class AppConfig:
    """Root config — mirrors config.yaml."""

    host: str = "localhost"
    port: int = 8000
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    channels: dict[str, ChannelConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    subagent: SubagentConfig = field(default_factory=SubagentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dream: DreamConfig = field(default_factory=DreamConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    _path: str = field(default="config.yaml", repr=False)

    @classmethod
    def load(cls, path: str = "config.yaml") -> AppConfig:
        """Load from YAML file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Config file not found: %s, using defaults", path)
            return cls(_path=path)

        with open(p, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        cfg = cls(_path=path)
        cfg.host = raw.get("host", cfg.host)
        cfg.port = raw.get("port", cfg.port)

        # LLM
        llm_raw = raw.get("llm", {})
        cfg.llm.default = llm_raw.get("default", cfg.llm.default)
        cfg.llm.fallback = llm_raw.get("fallback", [])
        cfg.llm.model_cache_ttl = llm_raw.get("model_cache_ttl", cfg.llm.model_cache_ttl)
        cfg.llm.providers = [
            ProviderConfig(
                name=p.get("name", p.get("provider", "")),
                provider=p.get("provider", ""),
                model=p.get("model", ""),
                advisor=p.get("advisor", ""),
            )
            for p in llm_raw.get("providers", [])
        ]

        # Agent
        agent_raw = raw.get("agent", {})
        cfg.agent.system_prompt = agent_raw.get("system_prompt", cfg.agent.system_prompt)
        cfg.agent.max_tool_rounds = agent_raw.get("max_tool_rounds", cfg.agent.max_tool_rounds)
        cfg.agent.hard_max_rounds = agent_raw.get("hard_max_rounds", cfg.agent.hard_max_rounds)
        cfg.agent.context_window_tokens = agent_raw.get("context_window_tokens", cfg.agent.context_window_tokens)
        cfg.agent.max_message_len = agent_raw.get("max_message_len", cfg.agent.max_message_len)
        cfg.agent.chars_per_token = agent_raw.get("chars_per_token", cfg.agent.chars_per_token)

        # Memory
        mem_raw = raw.get("memory", {})
        cfg.memory.db_path = mem_raw.get("db_path", cfg.memory.db_path)
        cfg.memory.max_context_messages = mem_raw.get("max_context_messages", cfg.memory.max_context_messages)
        cfg.memory.token_window = mem_raw.get("token_window", cfg.memory.token_window)

        # Channels
        for ch_name, ch_raw in raw.get("channels", {}).items():
            if isinstance(ch_raw, dict):
                cfg.channels[ch_name] = ChannelConfig(
                    enabled=ch_raw.get("enabled", False),
                    mode=ch_raw.get("mode", "polling"),
                    allowed_users=ch_raw.get("allowed_users", []),
                )

        # Tools
        tools_raw = raw.get("tools", {})
        cfg.tools.web_search = tools_raw.get("web_search", cfg.tools.web_search)
        cfg.tools.run_code = tools_raw.get("run_code", cfg.tools.run_code)
        cfg.tools.read_file = tools_raw.get("read_file", cfg.tools.read_file)

        # Sandbox
        sb_raw = raw.get("sandbox", {})
        cfg.sandbox.timeout = sb_raw.get("timeout", cfg.sandbox.timeout)
        cfg.sandbox.max_output = sb_raw.get("max_output", cfg.sandbox.max_output)

        # Embedding
        emb_raw = raw.get("embedding", {})
        cfg.embedding.provider = emb_raw.get("provider", cfg.embedding.provider)
        cfg.embedding.base_url = emb_raw.get("base_url", cfg.embedding.base_url)
        cfg.embedding.model = emb_raw.get("model", cfg.embedding.model)
        cfg.embedding.api_key = emb_raw.get("api_key", cfg.embedding.api_key)

        # Subagent
        sa_raw = raw.get("subagent", {})
        cfg.subagent.max_depth = sa_raw.get("max_depth", cfg.subagent.max_depth)
        cfg.subagent.default_max_rounds = sa_raw.get("default_max_rounds", cfg.subagent.default_max_rounds)

        # Paths
        paths_raw = raw.get("paths", {})
        for fld in ("sandbox_dir", "heartbeat_dir", "skills_dir", "memory_dir", "logs_dir", "certs_dir"):
            setattr(cfg.paths, fld, paths_raw.get(fld, getattr(cfg.paths, fld)))

        # Dream
        dream_raw = raw.get("dream", {})
        cfg.dream.batch_size = dream_raw.get("batch_size", cfg.dream.batch_size)

        # Hooks
        hooks_raw = raw.get("hooks", {})
        cfg.hooks.rate_limit_per_minute = hooks_raw.get("rate_limit_per_minute", cfg.hooks.rate_limit_per_minute)
        cfg.hooks.rate_limit_window = hooks_raw.get("rate_limit_window", cfg.hooks.rate_limit_window)

        # Heartbeat
        hb_raw = raw.get("heartbeat", {})
        cfg.heartbeat.default_interval = hb_raw.get("default_interval", cfg.heartbeat.default_interval)

        # Scheduler
        sched_raw = raw.get("scheduler", {})
        cfg.scheduler.default_channel = sched_raw.get("default_channel", cfg.scheduler.default_channel)

        logger.info("Config loaded from %s", path)
        return cfg

    def save(self) -> None:
        """Persist current config back to YAML file."""
        data = self.to_dict()
        # Strip falsy optional fields from providers for clean YAML
        for p in data.get("llm", {}).get("providers", []):
            for k in list(p):
                if not p[k]:
                    del p[k]
        Path(self._path).write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        logger.info("Config saved to %s", self._path)

    def to_display_dict(self) -> dict[str, Any]:
        """Serialize for LLM/user display — sensitive fields redacted."""
        d = self.to_dict()
        # Redact secrets from display
        if d.get("embedding", {}).get("api_key"):
            d["embedding"]["api_key"] = "***"
        return d

    def to_dict(self) -> dict[str, Any]:
        """Full serialization including sensitive fields — used by save()."""
        return {
            "host": self.host,
            "port": self.port,
            "llm": {
                "default": self.llm.default,
                "providers": [
                    {"name": p.name, "provider": p.provider, "model": p.model, "advisor": p.advisor}
                    for p in self.llm.providers
                ],
                "fallback": self.llm.fallback,
                "model_cache_ttl": self.llm.model_cache_ttl,
            },
            "agent": {
                "system_prompt": self.agent.system_prompt,
                "max_tool_rounds": self.agent.max_tool_rounds,
                "hard_max_rounds": self.agent.hard_max_rounds,
                "context_window_tokens": self.agent.context_window_tokens,
                "max_message_len": self.agent.max_message_len,
                "chars_per_token": self.agent.chars_per_token,
            },
            "memory": {
                "db_path": self.memory.db_path,
                "max_context_messages": self.memory.max_context_messages,
                "token_window": self.memory.token_window,
            },
            "channels": {
                n: {"enabled": c.enabled, "mode": c.mode, "allowed_users": c.allowed_users}
                for n, c in self.channels.items()
            },
            "tools": {
                "web_search": self.tools.web_search,
                "run_code": self.tools.run_code,
                "read_file": self.tools.read_file,
            },
            "sandbox": {
                "timeout": self.sandbox.timeout,
                "max_output": self.sandbox.max_output,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "base_url": self.embedding.base_url,
                "model": self.embedding.model,
                "api_key": self.embedding.api_key,
            },
            "subagent": {
                "max_depth": self.subagent.max_depth,
                "default_max_rounds": self.subagent.default_max_rounds,
            },
            "paths": {
                "sandbox_dir": self.paths.sandbox_dir,
                "heartbeat_dir": self.paths.heartbeat_dir,
                "skills_dir": self.paths.skills_dir,
                "memory_dir": self.paths.memory_dir,
                "logs_dir": self.paths.logs_dir,
                "certs_dir": self.paths.certs_dir,
            },
            "dream": {"batch_size": self.dream.batch_size},
            "hooks": {
                "rate_limit_per_minute": self.hooks.rate_limit_per_minute,
                "rate_limit_window": self.hooks.rate_limit_window,
            },
            "heartbeat": {"default_interval": self.heartbeat.default_interval},
            "scheduler": {"default_channel": self.scheduler.default_channel},
        }
