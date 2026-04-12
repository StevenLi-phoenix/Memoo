"""Conversation memory with SQLite persistence, archival, and RAG retrieval.

Active messages: current conversation context.
Archive: full compacted messages preserved for future retrieval.
Uses SQLite FTS5 for keyword-based RAG search on archived conversations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from models import Message, ToolCall

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_call_id TEXT,
    tool_calls TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);

-- Archive: full compacted messages for long-term retrieval
CREATE TABLE IF NOT EXISTS archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL,
    full_messages TEXT NOT NULL,  -- JSON: original messages before compaction
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_archive_chat_id ON archive(chat_id);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
    topic, summary, full_messages,
    content='archive',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS archive_ai AFTER INSERT ON archive BEGIN
    INSERT INTO archive_fts(rowid, topic, summary, full_messages)
    VALUES (new.id, new.topic, new.summary, new.full_messages);
END;
CREATE TRIGGER IF NOT EXISTS archive_ad AFTER DELETE ON archive BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, topic, summary, full_messages)
    VALUES ('delete', old.id, old.topic, old.summary, old.full_messages);
END;
"""


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


class Memory:
    """Per-chat conversation memory with archive and RAG retrieval.

    Active messages: working context for the current conversation.
    Archive: compacted messages searchable via FTS5 for progressive disclosure.
    """

    def __init__(
        self,
        db_path: str = "./data/memory.db",
        max_context: int = 200,
        token_window: int = 100_000,
    ) -> None:
        self._db_path = db_path
        self._max_context = max_context
        self._token_window = token_window
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_INIT_SQL)
        # FTS5 setup (may fail if already exists with different config, that's OK)
        try:
            await self._db.executescript(_FTS_SQL)
        except Exception:
            logger.debug("FTS5 tables already exist or not supported")
        await self._db.commit()
        logger.info("Memory initialized: db=%s, token_window=%d", self._db_path, self._token_window)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # --- Active messages ---

    async def add_message(self, chat_id: str, message: Message) -> None:
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        tool_calls_json = (
            json.dumps([{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in message.tool_calls])
            if message.tool_calls
            else None
        )
        metadata_json = json.dumps(message.metadata) if message.metadata else None

        await self._db.execute(
            "INSERT INTO messages (chat_id, role, content, tool_call_id, tool_calls, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                chat_id,
                message.role,
                message.content,
                message.tool_call_id,
                tool_calls_json,
                metadata_json,
            ),
        )
        await self._db.commit()

    async def get_history(self, chat_id: str) -> list[Message]:
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        cursor = await self._db.execute(
            "SELECT role, content, tool_call_id, tool_calls, metadata"
            " FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, self._max_context),
        )
        rows = await cursor.fetchall()
        rows.reverse()
        return [self._row_to_message(row) for row in rows]

    async def get_token_count(self, chat_id: str) -> int:
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        cursor = await self._db.execute("SELECT SUM(LENGTH(content)) FROM messages WHERE chat_id = ?", (chat_id,))
        row = await cursor.fetchone()
        return (row[0] if row and row[0] else 0) // CHARS_PER_TOKEN

    def needs_compaction(self, token_count: int) -> bool:
        return token_count > self._token_window

    async def clear(self, chat_id: str) -> None:
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        await self._db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await self._db.commit()
        logger.info("Cleared active memory for chat_id=%s", chat_id)

    async def message_count(self, chat_id: str) -> int:
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        cursor = await self._db.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
        row = await cursor.fetchone()
        return row[0] if row else 0

    # --- Archive (long-term memory) ---

    async def archive_messages(self, chat_id: str, messages: list[Message], topic: str, summary: str) -> None:
        """Archive compacted messages for future RAG retrieval."""
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        full_json = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages if m.content],
            ensure_ascii=False,
        )
        token_count = sum(estimate_tokens(m.content) for m in messages if m.content)

        await self._db.execute(
            "INSERT INTO archive (chat_id, topic, summary, full_messages, token_count) VALUES (?, ?, ?, ?, ?)",
            (chat_id, topic, summary, full_json, token_count),
        )
        await self._db.commit()
        logger.info("Archived %d messages for chat_id=%s, topic=%s", len(messages), chat_id, topic)

    async def search_archive(self, query: str, chat_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """RAG search: find relevant archived conversations by keyword.

        Uses FTS5 for full-text search with BM25 ranking.
        """
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")

        # Sanitize FTS5 query: escape special chars to prevent query injection
        # FTS5 operators: AND OR NOT NEAR * ^ "
        sanitized = query.replace('"', '""')
        sanitized = f'"{sanitized}"'  # phrase query — disables operator parsing

        try:
            if chat_id:
                cursor = await self._db.execute(
                    """SELECT a.id, a.chat_id, a.topic, a.summary, a.full_messages, a.created_at,
                              bm25(archive_fts) as rank
                       FROM archive_fts
                       JOIN archive a ON a.id = archive_fts.rowid
                       WHERE archive_fts MATCH ? AND a.chat_id = ?
                       ORDER BY rank
                       LIMIT ?""",
                    (sanitized, chat_id, limit),
                )
            else:
                cursor = await self._db.execute(
                    """SELECT a.id, a.chat_id, a.topic, a.summary, a.full_messages, a.created_at,
                              bm25(archive_fts) as rank
                       FROM archive_fts
                       JOIN archive a ON a.id = archive_fts.rowid
                       WHERE archive_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (sanitized, limit),
                )
            rows = await cursor.fetchall()
        except Exception:
            # Fallback: LIKE-based search if FTS5 not available
            logger.debug("FTS5 search failed, falling back to LIKE", exc_info=True)
            like_query = f"%{query}%"
            _cols = "id, chat_id, topic, summary, full_messages, created_at"
            if chat_id:
                cursor = await self._db.execute(
                    f"SELECT {_cols} FROM archive"
                    " WHERE chat_id = ? AND (summary LIKE ? OR topic LIKE ?)"
                    " ORDER BY created_at DESC LIMIT ?",
                    (chat_id, like_query, like_query, limit),
                )
            else:
                cursor = await self._db.execute(
                    f"SELECT {_cols} FROM archive"
                    " WHERE summary LIKE ? OR topic LIKE ?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (like_query, like_query, limit),
                )
            rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "chat_id": row[1],
                    "topic": row[2],
                    "summary": row[3],
                    "full_messages": row[4],  # JSON string
                    "created_at": row[5],
                }
            )
        return results

    async def get_archive_entry(self, archive_id: int) -> dict[str, Any] | None:
        """Get a specific archive entry by ID (for agent to read full context)."""
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        cursor = await self._db.execute(
            "SELECT id, chat_id, topic, summary, full_messages, created_at FROM archive WHERE id = ?",
            (archive_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "chat_id": row[1],
            "topic": row[2],
            "summary": row[3],
            "full_messages": row[4],
            "created_at": row[5],
        }

    async def list_archive(self, chat_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """List archived conversation summaries (progressive disclosure — summaries only)."""
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        if chat_id:
            cursor = await self._db.execute(
                "SELECT id, topic, summary, token_count, created_at FROM archive"
                " WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT id, chat_id, topic, summary, token_count, created_at"
                " FROM archive ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            entry: dict[str, Any] = {"id": row[0]}
            if chat_id:
                entry.update({"topic": row[1], "summary": row[2][:100], "tokens": row[3], "date": row[4]})
            else:
                entry.update(
                    {
                        "chat_id": row[1],
                        "topic": row[2],
                        "summary": row[3][:100],
                        "tokens": row[4],
                        "date": row[5],
                    }
                )
            results.append(entry)
        return results

    # --- Helpers ---

    @staticmethod
    def _row_to_message(row: tuple[Any, ...]) -> Message:
        role, content, tool_call_id, tool_calls_json, metadata_json = row
        tool_calls: list[ToolCall] = []
        if tool_calls_json:
            for tc in json.loads(tool_calls_json):
                tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
        metadata: dict[str, Any] = json.loads(metadata_json) if metadata_json else {}
        return Message(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            metadata=metadata,
        )
