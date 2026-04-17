"""Tests for slash command routing."""

import pytest

from core.commands import COMMANDS, _format_duration, handle_command
from core.config import AppConfig, ModelConfig, ProviderConfig
from models import ModelInfo


class TestCommands:
    async def test_help(self):
        result = await handle_command("/help", "test", {})
        assert result is not None
        assert "Available Commands" in result

    async def test_help_lists_all(self):
        result = await handle_command("/help", "test", {})
        for cmd in COMMANDS:
            assert cmd in result

    async def test_unknown_command_suggests(self):
        result = await handle_command("/nonexistent", "test", {})
        assert result is not None
        assert "Unknown command" in result
        assert "/help" in result

    async def test_clear(self):
        cleared = False

        class MockMemory:
            async def clear(self, chat_id: str) -> None:
                nonlocal cleared
                cleared = True

        result = await handle_command("/clear", "test", {"memory": MockMemory()})
        assert result == "Memory cleared."
        assert cleared

    async def test_status_no_deps(self):
        result = await handle_command("/status", "test", {})
        assert result is not None
        assert "Memoo Status" in result

    async def test_status_with_uptime(self):
        import time

        class MockApp:
            _start_time = time.monotonic() - 3661  # 1h 1m 1s ago
            llm = None

        result = await handle_command("/status", "test", {"app": MockApp()})
        assert "Uptime:" in result
        assert "1h 1m" in result

    async def test_status_with_agent_tokens(self):
        class MockAgent:
            total_tokens = {"total_runs": 5, "input_tokens": 1000, "output_tokens": 500}

            @property
            def compressor(self):
                return type("M", (), {"model_name": "haiku"})()

        result = await handle_command("/status", "test", {"agent": MockAgent()})
        assert "Runs: 5" in result
        assert "1,500" in result  # total tokens

    async def test_config_no_deps(self):
        result = await handle_command("/config", "test", {})
        assert result == "Config not available."

    async def test_case_insensitive(self):
        result = await handle_command("/HELP", "test", {})
        assert result is not None

    async def test_model_no_arg(self):
        result = await handle_command("/model", "test", {})
        assert result == "Not available."

    async def test_model_lists_discovered_models_for_enabled_provider(self):
        class MockLLM:
            model_name = "claude-sonnet-4-6"

        class MockApp:
            llm = MockLLM()
            discovered_models = {
                "anthropic": [
                    ModelInfo(id="claude-haiku-4-5", provider="anthropic"),
                    ModelInfo(id="claude-sonnet-4-6", provider="anthropic"),
                    ModelInfo(id="claude-opus-4-6", provider="anthropic"),
                ]
            }

        cfg = AppConfig()
        cfg.llm.default = "anthropic/claude-sonnet-4-6"
        cfg.llm.providers = [
            ProviderConfig(name="anthropic", provider="anthropic", allow_model_discovery=True),
        ]
        cfg.llm.models = [
            ModelConfig(name="anthropic/claude-sonnet-4-6", provider="anthropic", model="claude-sonnet-4-6"),
        ]

        result = await handle_command("/model", "test", {"app": MockApp(), "config": cfg})
        assert "Configured models" in result
        assert "Discovered models" in result
        assert "claude-opus-4-6" in result

    async def test_not_a_command(self):
        result = await handle_command("hello", "test", {})
        # Doesn't start with / so handle_command should not match
        # Actually it does start processing... let me check
        assert result is None


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (30, "30s"),
            (90, "1m 30s"),
            (3600, "1h 0m"),
            (3661, "1h 1m"),
            (86400, "1d 0h 0m"),
            (90061, "1d 1h 1m"),
        ],
    )
    def test_format_duration(self, seconds: float, expected: str) -> None:
        assert _format_duration(seconds) == expected
