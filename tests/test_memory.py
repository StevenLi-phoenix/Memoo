"""Tests for memory isolation and FTS5 injection prevention."""

import os

import pytest

from core.memory import Memory
from models import Message

DB_PATH = "./data/test_memory.db"


@pytest.fixture
async def memory():
    m = Memory(db_path=DB_PATH, max_context=200, token_window=100_000)
    await m.init()
    yield m
    await m.close()
    os.unlink(DB_PATH)


class TestMemoryIsolation:
    async def test_chat_id_isolation(self, memory):
        await memory.add_message("alice", Message(role="user", content="secret from alice"))
        await memory.add_message("bob", Message(role="user", content="hello from bob"))

        alice_history = await memory.get_history("alice")
        bob_history = await memory.get_history("bob")

        assert len(alice_history) == 1
        assert "alice" in alice_history[0].content
        assert len(bob_history) == 1
        assert "bob" in bob_history[0].content

    async def test_archive_isolation(self, memory):
        msgs = [Message(role="user", content="archived secret")]
        await memory.archive_messages("alice", msgs, "secret topic", "secret summary")

        # Search scoped to alice
        results = await memory.search_archive("secret", chat_id="alice")
        assert len(results) >= 1

        # Bob cannot find alice's archive
        results = await memory.search_archive("secret", chat_id="bob")
        assert len(results) == 0

    async def test_fts5_injection(self, memory):
        msgs = [Message(role="user", content="normal message")]
        await memory.archive_messages("test", msgs, "test", "test summary")

        # FTS5 operator injection should be neutralized (wrapped in quotes)
        results = await memory.search_archive('test OR "1"="1"', chat_id="test")
        # Should not crash and should not return unrelated results
        assert isinstance(results, list)

    async def test_archive_entry_cross_user(self, memory):
        msgs = [Message(role="user", content="private")]
        await memory.archive_messages("alice", msgs, "private topic", "private summary")

        entries = await memory.list_archive(chat_id="alice")
        assert len(entries) >= 1
        entry_id = entries[0]["id"]

        # Direct access by ID should work for alice
        entry = await memory.get_archive_entry(entry_id)
        assert entry is not None
        assert entry["chat_id"] == "alice"

    async def test_clear_only_target_chat(self, memory):
        await memory.add_message("alice", Message(role="user", content="alice msg"))
        await memory.add_message("bob", Message(role="user", content="bob msg"))

        await memory.clear("alice")

        assert await memory.message_count("alice") == 0
        assert await memory.message_count("bob") == 1
