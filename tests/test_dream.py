"""Tests for dream cycle: _parse_json and _apply_dream_results."""

from __future__ import annotations

from pathlib import Path

from core.dream import _apply_dream_results, _parse_json, _read_cursor, _write_cursor


class TestParseJson:
    """Test the JSON extractor used by dream Phase 2."""

    def test_pure_json(self) -> None:
        assert _parse_json('{"memory": "hello"}') == {"memory": "hello"}

    def test_empty_input(self) -> None:
        assert _parse_json("") is None
        assert _parse_json(None) is None  # type: ignore[arg-type]

    def test_no_json(self) -> None:
        assert _parse_json("No changes needed.") is None

    def test_leading_prose(self) -> None:
        text = 'Here is the updated JSON:\n{"memory": "new facts", "user": "preferences"}'
        result = _parse_json(text)
        assert result == {"memory": "new facts", "user": "preferences"}

    def test_markdown_fenced(self) -> None:
        text = 'Here you go:\n```json\n{"memory": "fenced content"}\n```'
        result = _parse_json(text)
        assert result == {"memory": "fenced content"}

    def test_markdown_fenced_no_lang(self) -> None:
        text = '```\n{"user": "plain fence"}\n```'
        result = _parse_json(text)
        assert result == {"user": "plain fence"}

    def test_braces_inside_string_values(self) -> None:
        """The critical bug: braces inside JSON string values should not confuse the parser."""
        text = 'Analysis: {"memory": "User opened {a template} and closed it", "user": "ok"}'
        result = _parse_json(text)
        assert result is not None
        assert result["memory"] == "User opened {a template} and closed it"
        assert result["user"] == "ok"

    def test_nested_objects(self) -> None:
        text = '{"outer": {"inner": "value"}}'
        result = _parse_json(text)
        assert result == {"outer": {"inner": "value"}}

    def test_escaped_quotes_in_strings(self) -> None:
        text = r'{"key": "value with \"escaped\" quotes"}'
        result = _parse_json(text)
        assert result is not None
        assert "escaped" in result["key"]

    def test_mixed_braces_and_escapes(self) -> None:
        text = r'Result: {"msg": "func() { return \"hello\"; }", "ok": true}'
        result = _parse_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_trailing_text_after_json(self) -> None:
        text = '{"memory": "updated"}\n\nHope this helps!'
        result = _parse_json(text)
        assert result == {"memory": "updated"}

    def test_returns_none_for_array(self) -> None:
        """Dream expects a dict, not an array."""
        assert _parse_json("[1, 2, 3]") is None

    def test_multiline_json(self) -> None:
        text = """{
  "memory": "line1\\nline2",
  "user": "preferences"
}"""
        result = _parse_json(text)
        assert result is not None
        assert result["user"] == "preferences"


class TestApplyDreamResults:
    """Test _apply_dream_results file writing and cursor management."""

    def test_writes_memory_file(self, tmp_path: Path) -> None:
        cursor_file = tmp_path / ".dream_cursor"
        memory_file = tmp_path / "MEMORY.md"
        user_file = tmp_path / "USER.md"

        entries = [{"id": 10}]
        raw = '{"memory": "# Facts\\nNew knowledge"}'

        result = _apply_dream_results(raw, entries, cursor_file, memory_file, user_file)
        assert "MEMORY.md" in result
        assert memory_file.read_text() == "# Facts\nNew knowledge"
        assert not user_file.exists()
        assert _read_cursor(cursor_file) == 10

    def test_writes_both_files(self, tmp_path: Path) -> None:
        cursor_file = tmp_path / ".dream_cursor"
        memory_file = tmp_path / "MEMORY.md"
        user_file = tmp_path / "USER.md"

        entries = [{"id": 5}, {"id": 8}]
        raw = '{"memory": "updated memory", "user": "updated user"}'

        result = _apply_dream_results(raw, entries, cursor_file, memory_file, user_file)
        assert "MEMORY.md" in result
        assert "USER.md" in result
        assert memory_file.read_text() == "updated memory"
        assert user_file.read_text() == "updated user"
        assert _read_cursor(cursor_file) == 8  # last entry ID

    def test_no_valid_json_still_advances_cursor(self, tmp_path: Path) -> None:
        cursor_file = tmp_path / ".dream_cursor"
        memory_file = tmp_path / "MEMORY.md"
        user_file = tmp_path / "USER.md"

        entries = [{"id": 15}]
        result = _apply_dream_results("not json", entries, cursor_file, memory_file, user_file)
        assert "no memory updates" in result.lower()
        assert _read_cursor(cursor_file) == 15

    def test_empty_values_not_written(self, tmp_path: Path) -> None:
        cursor_file = tmp_path / ".dream_cursor"
        memory_file = tmp_path / "MEMORY.md"
        user_file = tmp_path / "USER.md"

        entries = [{"id": 20}]
        raw = '{"memory": "", "user": ""}'

        result = _apply_dream_results(raw, entries, cursor_file, memory_file, user_file)
        assert "no changes" in result.lower()
        assert not memory_file.exists()
        assert not user_file.exists()


class TestCursor:
    def test_read_write_roundtrip(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor"
        _write_cursor(f, 42)
        assert _read_cursor(f) == 42

    def test_read_missing_file(self, tmp_path: Path) -> None:
        assert _read_cursor(tmp_path / "nope") == 0

    def test_read_corrupt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "cursor"
        f.write_text("not a number")
        assert _read_cursor(f) == 0
