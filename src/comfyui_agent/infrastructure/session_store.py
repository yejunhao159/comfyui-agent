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

SCHEMA = """
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
            await self._db.executescript(SCHEMA)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.commit()
        return self._db

    async def create_session(self, title: str = "") -> str:
        db = await self._get_db()
        session_id = str(uuid.uuid4())
        import time
        now = time.time()
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        await db.commit()
        return session_id

    async def list_sessions(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def delete_session(self, session_id: str) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()

    async def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Replace all messages for a session."""
        db = await self._get_db()
        import time
        now = time.time()

        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        for msg in messages:
            content = json.dumps(msg.get("content", "")) if not isinstance(msg.get("content"), str) else msg["content"]
            await db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, msg["role"], content, now),
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

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
