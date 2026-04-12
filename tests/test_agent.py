"""Tests for agent loop, structured output parsing, and TurnResult."""

import json

from core.agent import RESPONSE_SCHEMA, TurnResult


class TestTurnResult:
    def test_from_json_full(self):
        data = {
            "reply": "Hello!",
            "memory_notes": ["user greeted"],
            "current_topic": "greeting",
            "should_compress": False,
            "did_success": True,
        }
        result = TurnResult.from_json(data, {"input_tokens": 100})
        assert result.response == "Hello!"
        assert result.memory_notes == ["user greeted"]
        assert result.current_topic == "greeting"
        assert result.did_success is True
        assert not result.is_noop

    def test_from_json_noop(self):
        data = {
            "reply": "",
            "memory_notes": [],
            "current_topic": "idle",
            "should_compress": False,
            "did_success": True,
        }
        result = TurnResult.from_json(data, {})
        assert result.is_noop

    def test_from_json_missing_fields(self):
        data = {"reply": "hi"}
        result = TurnResult.from_json(data, {})
        assert result.response == "hi"
        assert result.memory_notes == []
        assert result.did_success is True  # default

    def test_fallback(self):
        result = TurnResult.fallback("raw text", {"input_tokens": 50})
        assert result.response == "raw text"
        assert result.memory_notes == []
        assert result.did_success is True

    def test_response_schema_has_required_fields(self):
        required = RESPONSE_SCHEMA["required"]
        assert "reply" in required
        assert "memory_notes" in required
        assert "current_topic" in required
        assert "should_compress" in required
        assert "did_success" in required


class TestParseStructuredResponse:
    def test_valid_json(self):
        from core.agent import Agent

        text = json.dumps(
            {"reply": "42", "memory_notes": [], "current_topic": "math", "should_compress": False, "did_success": True}
        )
        result = Agent._parse_structured_response(text, {"input_tokens": 10})
        assert result.response == "42"
        assert result.current_topic == "math"

    def test_invalid_json_fallback(self):
        from core.agent import Agent

        result = Agent._parse_structured_response("not json at all", {})
        assert result.response == "not json at all"

    def test_empty_string(self):
        from core.agent import Agent

        result = Agent._parse_structured_response("", {})
        assert result.is_noop


class TestImportanceScoring:
    def test_basic_importance(self):
        from core.memory import Memory
        from models import Message

        msgs = [Message(role="user", content="hello")]
        score = Memory._score_importance(msgs)
        assert 0.0 <= score <= 1.0

    def test_tool_usage_boosts(self):
        from core.memory import Memory
        from models import Message

        no_tools = [Message(role="user", content="hello")]
        with_tools = [
            Message(role="user", content="run code"),
            Message(role="tool_result", content="output"),
            Message(role="tool_result", content="output2"),
        ]
        assert Memory._score_importance(with_tools) > Memory._score_importance(no_tools)

    def test_empty_messages(self):
        from core.memory import Memory

        assert Memory._score_importance([]) == 0.5
