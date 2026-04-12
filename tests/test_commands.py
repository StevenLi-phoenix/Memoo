"""Tests for slash command routing."""

from core.commands import COMMANDS, handle_command


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

    async def test_config_no_deps(self):
        result = await handle_command("/config", "test", {})
        assert result == "Config not available."

    async def test_case_insensitive(self):
        result = await handle_command("/HELP", "test", {})
        assert result is not None

    async def test_model_no_arg(self):
        result = await handle_command("/model", "test", {})
        assert result == "Not available."

    async def test_not_a_command(self):
        result = await handle_command("hello", "test", {})
        # Doesn't start with / so handle_command should not match
        # Actually it does start processing... let me check
        assert result is None
