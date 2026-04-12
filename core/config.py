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


@dataclass
class AgentConfig:
    system_prompt: str = "systemprompt/default.md"
    max_tool_rounds: int = 0


@dataclass
class MemoryConfig:
    db_path: str = "./data/memory.db"
    max_context_messages: int = 200
    token_window: int = 100_000


@dataclass
class ChannelConfig:
    enabled: bool = False
    mode: str = "polling"


@dataclass
class ToolsConfig:
    web_search: bool = True
    run_code: bool = True
    read_file: bool = True


@dataclass
class SandboxConfig:
    timeout: int = 30
    max_output: int = 10000


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

        logger.info("Config loaded from %s", path)
        return cfg

    def save(self) -> None:
        """Persist current config back to YAML file."""
        data: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "llm": {
                "default": self.llm.default,
                "providers": [
                    {
                        k: v
                        for k, v in {
                            "name": p.name,
                            "provider": p.provider,
                            "model": p.model,
                            "advisor": p.advisor,
                        }.items()
                        if v
                    }
                    for p in self.llm.providers
                ],
                "fallback": self.llm.fallback,
            },
            "agent": {
                "system_prompt": self.agent.system_prompt,
                "max_tool_rounds": self.agent.max_tool_rounds,
            },
            "memory": {
                "db_path": self.memory.db_path,
                "max_context_messages": self.memory.max_context_messages,
                "token_window": self.memory.token_window,
            },
            "channels": {name: {"enabled": ch.enabled, "mode": ch.mode} for name, ch in self.channels.items()},
            "tools": {
                "web_search": self.tools.web_search,
                "run_code": self.tools.run_code,
                "read_file": self.tools.read_file,
            },
            "sandbox": {
                "timeout": self.sandbox.timeout,
                "max_output": self.sandbox.max_output,
            },
        }
        Path(self._path).write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        logger.info("Config saved to %s", self._path)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for display."""
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
            },
            "agent": {
                "system_prompt": self.agent.system_prompt,
                "max_tool_rounds": self.agent.max_tool_rounds,
            },
            "memory": {"token_window": self.memory.token_window},
            "channels": {n: c.enabled for n, c in self.channels.items()},
            "tools": {"web_search": self.tools.web_search, "run_code": self.tools.run_code},
            "sandbox": {"timeout": self.sandbox.timeout},
        }
