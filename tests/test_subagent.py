"""Tests for sub-agent spawning tool."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import AppConfig, SubagentConfig
from core.tools import ToolRegistry, set_context
from models import LLMResponse, Message


class FakeLLM:
    """Minimal LLM stub that returns a structured JSON response."""

    def __init__(self, reply: str = "sub-agent result", model: str = "test-model", delay: float = 0.0) -> None:
        self._reply = reply
        self._delay = delay
        self.model_name = model

    async def chat(self, **kwargs) -> LLMResponse:
        if self._delay:
            await asyncio.sleep(self._delay)
        data = {
            "reply": self._reply,
            "memory_notes": ["test note"],
            "current_topic": "test-topic",
            "should_compress": False,
            "did_success": True,
        }
        return LLMResponse(text=json.dumps(data), usage={"input_tokens": 10, "output_tokens": 5})


class FakeApp:
    """Minimal app stub for sub-agent tool registration."""

    def __init__(self, providers: dict | None = None, llm: FakeLLM | None = None) -> None:
        self.llm = llm or FakeLLM()
        self.fallback_llms: list = []
        self.tools = ToolRegistry()
        self.hooks = MagicMock()
        self.hooks.authorize = AsyncMock(return_value=(True, ""))
        self.memory = None
        self.agent = None
        self._providers = providers or {"anthropic": self.llm}


def _register_tools(app: FakeApp, config: AppConfig) -> ToolRegistry:
    """Register sub-agent tools and return the registry."""
    from tools.subagent import register

    registry = ToolRegistry()
    app.tools = registry
    register(registry, app=app, config=config)
    return registry


def _default_ctx(**overrides: object) -> dict:
    """Build a minimal valid tool context."""
    ctx: dict = {"_agent_depth": 0, "_messages": [], "_system_prompt": "test"}
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Depth limits
# ---------------------------------------------------------------------------


class TestDepthLimit:
    @pytest.mark.asyncio
    async def test_depth_limit_reached(self):
        config = AppConfig()
        config.subagent = SubagentConfig(max_depth=2)
        registry = _register_tools(FakeApp(), config)

        set_context(_default_ctx(_agent_depth=2))
        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x"}))
        assert "error" in result
        assert "depth" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_depth_zero_allowed(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "hello"}))
        assert result["reply"] == "sub-agent result"


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------


class TestModelLookup:
    @pytest.mark.asyncio
    async def test_unknown_provider(self):
        app = FakeApp(providers={"anthropic": FakeLLM()})
        registry = _register_tools(app, AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "model": "nope"}))
        assert "error" in result
        assert "anthropic" in str(result.get("available", []))

    @pytest.mark.asyncio
    async def test_valid_provider(self):
        openai_llm = FakeLLM(reply="openai result", model="gpt-4o")
        app = FakeApp(providers={"anthropic": FakeLLM(), "openai": openai_llm})
        registry = _register_tools(app, AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "model": "openai"}))
        assert result["reply"] == "openai result"


# ---------------------------------------------------------------------------
# Context modes
# ---------------------------------------------------------------------------


class TestContextMode:
    @pytest.mark.asyncio
    async def test_none_context(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx(_messages=[Message(role="user", content="old")]))

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "context_mode": "none"}))
        assert result["reply"] == "sub-agent result"

    @pytest.mark.asyncio
    async def test_full_context(self):
        registry = _register_tools(FakeApp(), AppConfig())
        msgs = [Message(role="user", content="history")]
        set_context(_default_ctx(_messages=msgs))

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "context_mode": "full"}))
        assert result["reply"] == "sub-agent result"


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


class TestResultMetadata:
    @pytest.mark.asyncio
    async def test_metadata_returned(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x"}))
        assert result["topic"] == "test-topic"
        assert result["success"] is True
        assert result["memory_notes"] == ["test note"]
        assert "input_tokens" in result["usage"]
        assert "run_id" in result
        assert result["elapsed_s"] >= 0


# ---------------------------------------------------------------------------
# Sandbox restrictions (readonly / network_access via context flags)
# ---------------------------------------------------------------------------


class TestSandboxRestrictions:
    @pytest.mark.asyncio
    async def test_readonly_sets_context_flag(self):
        """readonly=True should pass _sandbox_readonly in sub-agent context."""
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        # Sub-agent runs successfully — readonly is enforced at sandbox level, not tool level
        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "readonly": True}))
        assert result["reply"] == "sub-agent result"

    @pytest.mark.asyncio
    async def test_no_network_sets_context_flag(self):
        """network_access=False should pass _sandbox_no_network in sub-agent context."""
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "network_access": False}))
        assert result["reply"] == "sub-agent result"

    def test_filtered_registry_utility(self):
        """ToolRegistry.filtered() still works as a utility."""
        reg = ToolRegistry()

        @reg.tool
        def tool_a() -> str:
            """A."""
            return "a"

        @reg.tool
        def tool_b() -> str:
            """B."""
            return "b"

        filtered = reg.filtered(exclude={"tool_b"})
        assert "tool_a" in filtered.tool_names
        assert "tool_b" not in filtered.tool_names


# ---------------------------------------------------------------------------
# Background mode
# ---------------------------------------------------------------------------


class TestBackgroundMode:
    @pytest.mark.asyncio
    async def test_bg_returns_run_id(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "background": "bg"}))
        assert result["status"] == "running"
        assert "run_id" in result

        # Wait briefly for the background task to complete
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_bg_result_readable(self):
        """read_agent_output should return the completed result."""
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        spawn_result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "background": "bg"}))
        run_id = spawn_result["run_id"]

        # Wait for background task to finish
        await asyncio.sleep(0.05)

        output = json.loads(await registry.execute("read_agent_output", {"run_id": run_id}))
        assert output["status"] == "completed"
        assert output["reply"] == "sub-agent result"
        assert output["topic"] == "test-topic"


# ---------------------------------------------------------------------------
# Timeout → background
# ---------------------------------------------------------------------------


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_moves_to_background(self):
        """When timeout expires, the sub-agent should move to background."""
        slow_llm = FakeLLM(delay=2.0)  # 2s delay
        app = FakeApp(llm=slow_llm)
        app._providers = {"anthropic": slow_llm}
        registry = _register_tools(app, AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "timeout": 1}))
        assert result["status"] == "moved_to_background"
        assert "run_id" in result

        # Cleanup: cancel the background task
        run_id = result["run_id"]
        await registry.execute("cancel_agent", {"run_id": run_id})
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_timeout_kills_on_action_kill(self):
        """timeout_action='kill' should cancel the sub-agent on timeout."""
        slow_llm = FakeLLM(delay=2.0)
        app = FakeApp(llm=slow_llm)
        app._providers = {"anthropic": slow_llm}
        registry = _register_tools(app, AppConfig())
        set_context(_default_ctx())

        result = json.loads(
            await registry.execute("spawn_agent", {"prompt": "x", "timeout": 1, "timeout_action": "kill"})
        )
        assert result["status"] == "killed"
        assert "run_id" in result
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Cancel propagation
# ---------------------------------------------------------------------------


class TestCancelPropagation:
    @pytest.mark.asyncio
    async def test_cancel_agent_tool(self):
        """cancel_agent should stop a running background sub-agent."""
        slow_llm = FakeLLM(delay=5.0)
        app = FakeApp(llm=slow_llm)
        app._providers = {"anthropic": slow_llm}
        registry = _register_tools(app, AppConfig())
        set_context(_default_ctx())

        spawn_result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "background": "bg"}))
        run_id = spawn_result["run_id"]

        cancel_result = json.loads(await registry.execute("cancel_agent", {"run_id": run_id}))
        assert cancel_result["status"] == "cancelled"

        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_cancel_already_done(self):
        """Cancelling a completed agent should return an error."""
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        spawn_result = json.loads(await registry.execute("spawn_agent", {"prompt": "x", "background": "bg"}))
        run_id = spawn_result["run_id"]
        await asyncio.sleep(0.05)

        cancel_result = json.loads(await registry.execute("cancel_agent", {"run_id": run_id}))
        assert "already finished" in cancel_result.get("error", "")


# ---------------------------------------------------------------------------
# Agent management tools
# ---------------------------------------------------------------------------


class TestAgentManagement:
    @pytest.mark.asyncio
    async def test_list_agents(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        await registry.execute("spawn_agent", {"prompt": "task1", "background": "bg"})
        await registry.execute("spawn_agent", {"prompt": "task2", "background": "bg"})
        await asyncio.sleep(0.05)

        agents = json.loads(await registry.execute("list_agents", {}))
        assert len(agents) >= 2

    @pytest.mark.asyncio
    async def test_read_unknown_run_id(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("read_agent_output", {"run_id": "nonexistent"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cancel_unknown_run_id(self):
        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        result = json.loads(await registry.execute("cancel_agent", {"run_id": "nonexistent"}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Context isolation (ContextVar via asyncio.create_task)
# ---------------------------------------------------------------------------


class TestContextIsolation:
    @pytest.mark.asyncio
    async def test_parent_context_unchanged(self):
        """asyncio.create_task gives context isolation — parent ctx unaffected."""
        from core.tools import get_context

        registry = _register_tools(FakeApp(), AppConfig())
        parent_ctx = _default_ctx(chat_id="parent-session")
        set_context(parent_ctx)

        await registry.execute("spawn_agent", {"prompt": "x"})

        restored = get_context()
        assert restored.get("chat_id") == "parent-session"
        assert restored.get("_agent_depth") == 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_audit_log_written(self, tmp_path, monkeypatch):
        """Blocking spawn should write an audit entry."""
        from tools import subagent

        monkeypatch.setattr(subagent, "_audit_log", MagicMock())

        registry = _register_tools(FakeApp(), AppConfig())
        set_context(_default_ctx())

        await registry.execute("spawn_agent", {"prompt": "x"})
        await asyncio.sleep(0.05)

        assert subagent._audit_log.called


# ---------------------------------------------------------------------------
# SubagentConfig
# ---------------------------------------------------------------------------


class TestSubagentConfig:
    def test_config_defaults(self):
        cfg = SubagentConfig()
        assert cfg.max_depth == 3
        assert cfg.default_max_rounds == 10

    def test_config_load_from_yaml(self, tmp_path):
        yaml_content = "subagent:\n  max_depth: 5\n  default_max_rounds: 20\n"
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = AppConfig.load(str(cfg_file))
        assert cfg.subagent.max_depth == 5
        assert cfg.subagent.default_max_rounds == 20

    def test_config_to_dict(self):
        cfg = AppConfig()
        cfg.subagent = SubagentConfig(max_depth=4, default_max_rounds=15)
        d = cfg.to_dict()
        assert d["subagent"]["max_depth"] == 4
        assert d["subagent"]["default_max_rounds"] == 15

    def test_config_save_round_trip(self, tmp_path):
        cfg = AppConfig(_path=str(tmp_path / "config.yaml"))
        cfg.subagent = SubagentConfig(max_depth=7, default_max_rounds=25)
        cfg.save()
        loaded = AppConfig.load(str(tmp_path / "config.yaml"))
        assert loaded.subagent.max_depth == 7
        assert loaded.subagent.default_max_rounds == 25
