"""Tests for SessionStore — SQLite persistence."""

from __future__ import annotations

import os
import tempfile

import pytest

from comfyui_agent.infrastructure.persistence.session_store import SessionStore


@pytest.fixture
async def store():
    """Create a temporary SessionStore."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = SessionStore(db_path=path)
    yield s
    await s.close()
    os.unlink(path)


class TestSessionStore:
    @pytest.mark.asyncio
    async def test_create_session(self, store: SessionStore):
        sid = await store.create_session("Test Session")
        assert sid
        assert len(sid) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_list_sessions(self, store: SessionStore):
        await store.create_session("Session 1")
        await store.create_session("Session 2")
        sessions = await store.list_sessions()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self, store: SessionStore):
        sid = await store.create_session("To Delete")
        await store.delete_session(sid)
        sessions = await store.list_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_save_and_load_messages(self, store: SessionStore):
        sid = await store.create_session("Chat")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        await store.save_messages(sid, messages)
        loaded = await store.load_messages(sid)
        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[0]["content"] == "Hello"
        assert loaded[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_save_complex_content(self, store: SessionStore):
        """Tool results have list content, not string."""
        sid = await store.create_session("Tool Chat")
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_use", "id": "t1", "name": "test", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
                ],
            },
        ]
        await store.save_messages(sid, messages)
        loaded = await store.load_messages(sid)
        assert len(loaded) == 3
        # Complex content should be preserved
        assert isinstance(loaded[1]["content"], list)
        assert loaded[1]["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_empty_session_messages(self, store: SessionStore):
        sid = await store.create_session("Empty")
        loaded = await store.load_messages(sid)
        assert loaded == []
