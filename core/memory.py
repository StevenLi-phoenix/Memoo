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
    importance REAL NOT NULL DEFAULT 0.5,  -- 0.0-1.0 importance score
    embedding TEXT,  -- JSON-serialized float vector
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
    """Per-chat conversation memory with archive and semantic RAG retrieval.

    Active messages: working context for the current conversation.
    Archive: compacted messages with embeddings for semantic search + FTS5 keyword fallback.
    Importance scoring: based on message density, tool usage, and topic shifts.
    """

    def __init__(
        self,
        db_path: str = "./data/memory.db",
        max_context: int = 200,
        token_window: int = 100_000,
        embedding_provider: Any = None,
    ) -> None:
        self._db_path = db_path
        self._max_context = max_context
        self._token_window = token_window
        self._db: aiosqlite.Connection | None = None
        self._embed_provider = embedding_provider

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_INIT_SQL)
        # Migrate: add columns that may be missing in older databases
        for col, defn in (
            ("importance", "REAL NOT NULL DEFAULT 0.5"),
            ("embedding", "TEXT"),
        ):
            try:
                await self._db.execute(f"ALTER TABLE archive ADD COLUMN {col} {defn}")
                logger.info("Migration: added archive.%s", col)
            except Exception:
                pass  # column already exists
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
        """Archive compacted messages with embedding and importance scoring."""
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        full_json = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages if m.content],
            ensure_ascii=False,
        )
        token_count = sum(estimate_tokens(m.content) for m in messages if m.content)
        importance = self._score_importance(messages)

        # Compute embedding for semantic search
        from core.embeddings import embed_text, serialize_embedding

        embed_input = f"{topic}\n{summary}"
        vec = await embed_text(embed_input, self._embed_provider)
        embedding_json = serialize_embedding(vec)

        await self._db.execute(
            "INSERT INTO archive (chat_id, topic, summary, full_messages, token_count, importance, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, topic, summary, full_json, token_count, importance, embedding_json),
        )
        await self._db.commit()
        logger.info(
            "Archived %d msgs for chat_id=%s, topic=%s, importance=%.2f", len(messages), chat_id, topic, importance
        )

    @staticmethod
    def _score_importance(messages: list[Message]) -> float:
        """Score importance 0.0-1.0 based on message content signals."""
        if not messages:
            return 0.5

        score = 0.5
        total_content = " ".join(m.content for m in messages if m.content)

        # More messages = more important
        score += min(0.15, len(messages) * 0.01)

        # Tool usage signals active work
        tool_msgs = sum(1 for m in messages if m.role == "tool_result")
        score += min(0.15, tool_msgs * 0.05)

        # Longer conversations = more context worth keeping
        score += min(0.1, len(total_content) / 50000)

        # Decision/action keywords boost importance
        action_words = ["decided", "created", "fixed", "installed", "configured", "remember", "important"]
        hits = sum(1 for w in action_words if w in total_content.lower())
        score += min(0.1, hits * 0.03)

        return min(1.0, score)

    async def search_archive(self, query: str, chat_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Hybrid RAG search: semantic (embedding) + keyword (FTS5) + importance scoring.

        Results are ranked by: 0.5 * semantic_sim + 0.3 * keyword_match + 0.2 * importance
        """
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")

        # Load all candidate entries for this chat
        _cols = "id, chat_id, topic, summary, full_messages, created_at, importance, embedding"
        if chat_id:
            cursor = await self._db.execute(
                f"SELECT {_cols} FROM archive WHERE chat_id = ? ORDER BY created_at DESC LIMIT 100",
                (chat_id,),
            )
        else:
            cursor = await self._db.execute(f"SELECT {_cols} FROM archive ORDER BY created_at DESC LIMIT 100")
        rows = await cursor.fetchall()

        if not rows:
            return []

        # Compute query embedding
        from core.embeddings import cosine_similarity, deserialize_embedding, embed_text

        query_vec = await embed_text(query, self._embed_provider)
        query_lower = query.lower()

        # Score each entry
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            entry = {
                "id": row[0],
                "chat_id": row[1],
                "topic": row[2],
                "summary": row[3],
                "full_messages": row[4],
                "created_at": row[5],
            }
            importance = row[6] or 0.5
            embedding_json = row[7]

            # Semantic similarity (0.0-1.0)
            if embedding_json:
                entry_vec = deserialize_embedding(embedding_json)
                semantic_score = max(0.0, cosine_similarity(query_vec, entry_vec))
            else:
                semantic_score = 0.0

            # Keyword match (0.0 or 1.0)
            text = f"{entry['topic']} {entry['summary']}".lower()
            keyword_score = 1.0 if query_lower in text else 0.0
            # Partial word match
            if not keyword_score:
                words = query_lower.split()
                hits = sum(1 for w in words if w in text)
                keyword_score = hits / max(len(words), 1)

            # Combined score
            final = 0.5 * semantic_score + 0.3 * keyword_score + 0.2 * importance
            scored.append((final, entry))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

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
        _cols = "id, chat_id, topic, summary, token_count, created_at"
        if chat_id:
            cursor = await self._db.execute(
                f"SELECT {_cols} FROM archive WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit),
            )
        else:
            cursor = await self._db.execute(
                f"SELECT {_cols} FROM archive ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            entry: dict[str, Any] = {
                "id": row[0],
                "topic": row[2],
                "summary": row[3][:100],
                "tokens": row[4],
                "date": row[5],
            }
            if not chat_id:
                entry["chat_id"] = row[1]
            results.append(entry)
        return results

    async def get_archive_since(self, cursor_id: int = 0, limit: int = 30) -> list[dict[str, Any]]:
        """Get archive entries with id > cursor_id (for dream processing)."""
        if self._db is None:
            raise RuntimeError("Memory not initialized — call init() first")
        rows = await self._db.execute_fetchall(
            "SELECT id, chat_id, topic, summary, token_count, importance, created_at"
            " FROM archive WHERE id > ? ORDER BY id ASC LIMIT ?",
            (cursor_id, limit),
        )
        return [
            {
                "id": row[0],
                "chat_id": row[1],
                "topic": row[2],
                "summary": row[3],
                "token_count": row[4],
                "importance": row[5],
                "date": row[6],
            }
            for row in rows
        ]

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
