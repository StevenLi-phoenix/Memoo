"""Tests for config tool model help and hot-reload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import AppConfig, ModelConfig, ProviderConfig
from core.tools import ToolRegistry
from models import ModelInfo
from tools import config_tools
from tools.config_tools import register


class TestConfigHelp:
    @pytest.mark.asyncio
    async def test_model_help_lists_discovered_models_for_enabled_provider(self) -> None:
        cfg = AppConfig()
        cfg.llm.default = "anthropic/claude-sonnet-4-6"
        cfg.llm.providers = [
            ProviderConfig(name="anthropic", provider="anthropic", allow_model_discovery=True),
            ProviderConfig(name="openai", provider="openai", allow_model_discovery=False),
        ]
        cfg.llm.models = [
            ModelConfig(name="anthropic/claude-sonnet-4-6", provider="anthropic", model="claude-sonnet-4-6"),
            ModelConfig(name="openai/gpt5.4-mini", provider="openai", model="gpt5.4-mini"),
        ]

        app = type(
            "MockApp",
            (),
            {
                "discovered_models": {
                    "anthropic": [
                        ModelInfo(id="claude-haiku-4-5", provider="anthropic"),
                        ModelInfo(id="claude-opus-4-6", provider="anthropic"),
                    ],
                    "openai": [
                        ModelInfo(id="gpt-4.1", provider="openai"),
                    ],
                }
            },
        )()

        registry = ToolRegistry()
        register(registry, config=cfg, app=app)

        result = await registry.execute("config_help", {"key": "llm.model"})
        assert isinstance(result, str)
        assert "Configured models:" in result
        assert "Discovered models:" in result
        assert "anthropic:" in result
        assert "claude-opus-4-6" in result
        assert "gpt-4.1" not in result

    @pytest.mark.asyncio
    async def test_update_model_promotes_discovered_model_into_config(self, tmp_path: Path, monkeypatch) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg = AppConfig.load(str(cfg_file))
        cfg.llm.default = "anthropic/claude-sonnet-4-6"
        cfg.llm.providers = [
            ProviderConfig(name="anthropic", provider="anthropic", allow_model_discovery=True),
        ]
        cfg.llm.models = [
            ModelConfig(name="anthropic/claude-sonnet-4-6", provider="anthropic", model="claude-sonnet-4-6"),
        ]
        cfg.save()

        fake_provider = type("FakeProvider", (), {"model_name": "claude-opus-4-6"})()
        monkeypatch.setattr(config_tools, "_build_live_provider", lambda config, model: fake_provider)

        app = type(
            "MockApp",
            (),
            {
                "discovered_models": {
                    "anthropic": [
                        ModelInfo(id="claude-opus-4-6", provider="anthropic"),
                    ]
                },
                "_providers": {
                    "anthropic/claude-sonnet-4-6": type("DefaultProvider", (), {"model_name": "claude-sonnet-4-6"})(),
                },
                "llm": None,
                "fallback_llms": [],
                "agent": type(
                    "MockAgent",
                    (),
                    {
                        "_llm": None,
                        "_fallback_llms": [],
                    },
                )(),
            },
        )()

        result = config_tools._update_model("claude-opus-4-6", cfg, app)

        assert "added to config" in result
        assert cfg.llm.default == "anthropic/claude-opus-4-6"
        assert any(model.name == "anthropic/claude-opus-4-6" for model in cfg.llm.models)
        assert app._providers["anthropic/claude-opus-4-6"] is fake_provider

        reloaded = AppConfig.load(str(cfg_file))
        assert any(model.name == "anthropic/claude-opus-4-6" for model in reloaded.llm.models)

    @pytest.mark.asyncio
    async def test_update_model_matches_shorthand_discovered_name(self, tmp_path: Path, monkeypatch) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg = AppConfig.load(str(cfg_file))
        cfg.llm.default = "anthropic/claude-sonnet-4-6"
        cfg.llm.providers = [
            ProviderConfig(name="anthropic", provider="anthropic", allow_model_discovery=True),
        ]
        cfg.llm.models = [
            ModelConfig(name="anthropic/claude-sonnet-4-6", provider="anthropic", model="claude-sonnet-4-6"),
        ]
        cfg.save()

        fake_provider = type("FakeProvider", (), {"model_name": "claude-opus-4-7"})()
        monkeypatch.setattr(config_tools, "_build_live_provider", lambda config, model: fake_provider)

        app = type(
            "MockApp",
            (),
            {
                "discovered_models": {
                    "anthropic": [
                        ModelInfo(id="claude-opus-4-7", provider="anthropic"),
                    ]
                },
                "_providers": {
                    "anthropic/claude-sonnet-4-6": type("DefaultProvider", (), {"model_name": "claude-sonnet-4-6"})(),
                },
                "llm": None,
                "fallback_llms": [],
                "agent": type(
                    "MockAgent",
                    (),
                    {
                        "_llm": None,
                        "_fallback_llms": [],
                    },
                )(),
            },
        )()

        result = config_tools._update_model("opus-4-7", cfg, app)

        assert "added to config" in result
        assert cfg.llm.default == "anthropic/claude-opus-4-7"
        assert any(model.name == "anthropic/claude-opus-4-7" for model in cfg.llm.models)

    def test_not_found_lists_discovered_models(self) -> None:
        cfg = AppConfig()
        cfg.llm.providers = [
            ProviderConfig(name="anthropic", provider="anthropic", allow_model_discovery=True),
        ]
        cfg.llm.models = [
            ModelConfig(name="anthropic/claude-sonnet-4-6", provider="anthropic", model="claude-sonnet-4-6"),
        ]

        app = type(
            "MockApp",
            (),
            {
                "discovered_models": {
                    "anthropic": [
                        ModelInfo(id="claude-opus-4-7", provider="anthropic"),
                    ]
                }
            },
        )()

        result = config_tools._update_model("does-not-exist", cfg, app)

        assert "claude-opus-4-7" in result
