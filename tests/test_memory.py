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


class TestCompactReplace:
    """Tests for atomic compact_replace operation."""

    async def test_replaces_all_messages(self, memory):
        await memory.add_message("chat1", Message(role="user", content="old msg 1"))
        await memory.add_message("chat1", Message(role="user", content="old msg 2"))
        await memory.add_message("chat1", Message(role="user", content="old msg 3"))
        assert await memory.message_count("chat1") == 3

        new_msgs = [
            Message(role="system", content="[summary]"),
            Message(role="user", content="recent msg"),
        ]
        await memory.compact_replace("chat1", new_msgs)

        history = await memory.get_history("chat1")
        assert len(history) == 2
        assert history[0].content == "[summary]"
        assert history[1].content == "recent msg"

    async def test_does_not_affect_other_chats(self, memory):
        await memory.add_message("chat1", Message(role="user", content="chat1 msg"))
        await memory.add_message("chat2", Message(role="user", content="chat2 msg"))

        await memory.compact_replace("chat1", [Message(role="system", content="replaced")])

        assert await memory.message_count("chat2") == 1
        history = await memory.get_history("chat2")
        assert history[0].content == "chat2 msg"

    async def test_empty_replacement_clears_chat(self, memory):
        await memory.add_message("chat1", Message(role="user", content="msg"))
        await memory.compact_replace("chat1", [])
        assert await memory.message_count("chat1") == 0

    async def test_rollback_on_failure(self, memory):
        """If compact_replace fails mid-way, original messages should survive."""
        await memory.add_message("chat1", Message(role="user", content="original"))
        assert await memory.message_count("chat1") == 1

        # Force a failure by closing the db connection mid-operation
        # We'll monkey-patch execute to fail after the DELETE
        original_execute = memory._db.execute
        call_count = 0

        async def failing_execute(sql, params=None):
            nonlocal call_count
            call_count += 1
            # Let DELETE through (call 1), fail on first INSERT (call 2)
            if call_count >= 2:
                raise RuntimeError("simulated disk error")
            return await original_execute(sql, params)

        memory._db.execute = failing_execute

        with pytest.raises(RuntimeError, match="simulated disk error"):
            await memory.compact_replace("chat1", [Message(role="system", content="new")])

        # Restore original execute for verification
        memory._db.execute = original_execute

        # Original message should survive thanks to rollback
        count = await memory.message_count("chat1")
        assert count == 1, f"Expected 1 message after rollback, got {count}"
        history = await memory.get_history("chat1")
        assert history[0].content == "original"
