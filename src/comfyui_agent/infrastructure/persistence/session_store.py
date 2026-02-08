"""SQLite-based session persistence."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

_CURRENT_VERSION = 2


class SessionStore:
    """SQLite-backed session and message storage."""

    def __init__(self, db_path: str = "data/sessions.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.executescript(SCHEMA_V1)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.commit()
            await self._migrate(self._db)
        return self._db

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        """Run schema migrations using PRAGMA user_version."""
        cursor = await db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        version = row[0] if row else 0

        if version < 2:
            # V2: add columns for sub-agent, summary, token tracking
            for stmt in [
                "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT DEFAULT NULL",
                "ALTER TABLE sessions ADD COLUMN summary_message_id INTEGER DEFAULT NULL",
                "ALTER TABLE sessions ADD COLUMN total_input_tokens INTEGER DEFAULT 0",
                "ALTER TABLE sessions ADD COLUMN total_output_tokens INTEGER DEFAULT 0",
                "ALTER TABLE messages ADD COLUMN ordinal INTEGER DEFAULT 0",
            ]:
                try:
                    await db.execute(stmt)
                except Exception:
                    pass  # column already exists
            await db.execute(f"PRAGMA user_version = {_CURRENT_VERSION}")
            await db.commit()
            logger.info("DB migrated to version %d", _CURRENT_VERSION)

    # ------------------------------------------------------------------
    # Original methods (backward compatible)
    # ------------------------------------------------------------------

    async def create_session(self, title: str = "") -> str:
        db = await self._get_db()
        session_id = str(uuid.uuid4())
        now = time.time()
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        await db.commit()
        return session_id

    async def list_sessions(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE parent_session_id IS NULL ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def delete_session(self, session_id: str) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()

    async def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Replace all messages for a session (backward compatible)."""
        db = await self._get_db()
        now = time.time()

        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        for i, msg in enumerate(messages):
            content = json.dumps(msg.get("content", "")) if not isinstance(msg.get("content"), str) else msg["content"]
            await db.execute(
                "INSERT INTO messages (session_id, role, content, created_at, ordinal) VALUES (?, ?, ?, ?, ?)",
                (session_id, msg["role"], content, now, i),
            )
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        await db.commit()

    async def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = await cursor.fetchall()
        messages = []
        for row in rows:
            content = row["content"]
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                pass
            messages.append({"role": row["role"], "content": content})
        return messages

    # ------------------------------------------------------------------
    # New methods for incremental persistence
    # ------------------------------------------------------------------

    async def append_message(self, session_id: str, role: str, content: Any) -> int:
        """Append a single message, return its row ID."""
        db = await self._get_db()
        now = time.time()
        if not isinstance(content, str):
            content = json.dumps(content)

        # Get next ordinal
        cursor = await db.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        ordinal = row[0] if row else 0

        cursor = await db.execute(
            "INSERT INTO messages (session_id, role, content, created_at, ordinal) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, now, ordinal),
        )
        msg_id = cursor.lastrowid
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        await db.commit()
        return msg_id  # type: ignore[return-value]

    async def load_messages_from(self, session_id: str, from_id: int = 0) -> list[dict[str, Any]]:
        """Load messages starting from a given message ID (for summary checkpoint)."""
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND id >= ? ORDER BY id",
            (session_id, from_id),
        )
        rows = await cursor.fetchall()
        messages = []
        for row in rows:
            content = row["content"]
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                pass
            messages.append({"role": row["role"], "content": content})
        return messages

    async def get_session_meta(self, session_id: str) -> dict[str, Any]:
        """Get session metadata."""
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return {}
        return dict(row)

    async def update_session_meta(self, session_id: str, **kwargs: Any) -> None:
        """Update session metadata fields."""
        db = await self._get_db()
        allowed = {"title", "summary_message_id", "total_input_tokens", "total_output_tokens"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(time.time())
        values.append(session_id)
        await db.execute(
            f"UPDATE sessions SET {set_clause}, updated_at = ? WHERE id = ?",
            values,
        )
        await db.commit()

    async def create_child_session(self, parent_id: str, title: str) -> str:
        """Create a child session (for sub-agent)."""
        db = await self._get_db()
        session_id = str(uuid.uuid4())
        now = time.time()
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, parent_session_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, now, now, parent_id),
        )
        await db.commit()
        return session_id

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
