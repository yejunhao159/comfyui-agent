"""Tests for SessionStore v2 — incremental persistence methods."""

from __future__ import annotations

import pytest

from comfyui_agent.infrastructure.persistence.session_store import SessionStore


@pytest.fixture
async def store(tmp_path):
    """Create a SessionStore with a temp DB."""
    db_path = str(tmp_path / "test.db")
    s = SessionStore(db_path=db_path)
    yield s
    await s.close()


class TestAppendMessage:
    @pytest.mark.asyncio
    async def test_append_returns_id(self, store):
        session_id = await store.create_session("test")
        msg_id = await store.append_message(session_id, "user", "hello")
        assert isinstance(msg_id, int)
        assert msg_id > 0

    @pytest.mark.asyncio
    async def test_append_preserves_order(self, store):
        session_id = await store.create_session("test")
        await store.append_message(session_id, "user", "first")
        await store.append_message(session_id, "assistant", "second")
        await store.append_message(session_id, "user", "third")

        messages = await store.load_messages(session_id)
        assert len(messages) == 3
        assert messages[0]["content"] == "first"
        assert messages[1]["content"] == "second"
        assert messages[2]["content"] == "third"

    @pytest.mark.asyncio
    async def test_append_complex_content(self, store):
        session_id = await store.create_session("test")
        content = [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
        ]
        await store.append_message(session_id, "assistant", content)

        messages = await store.load_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["content"] == content


class TestLoadMessagesFrom:
    @pytest.mark.asyncio
    async def test_load_from_beginning(self, store):
        session_id = await store.create_session("test")
        await store.append_message(session_id, "user", "a")
        await store.append_message(session_id, "assistant", "b")

        messages = await store.load_messages_from(session_id, from_id=0)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_load_from_checkpoint(self, store):
        session_id = await store.create_session("test")
        await store.append_message(session_id, "user", "old")
        msg_id = await store.append_message(session_id, "user", "summary")
        await store.append_message(session_id, "assistant", "new")

        messages = await store.load_messages_from(session_id, from_id=msg_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "summary"
        assert messages[1]["content"] == "new"


class TestSessionMeta:
    @pytest.mark.asyncio
    async def test_get_session_meta(self, store):
        session_id = await store.create_session("my session")
        meta = await store.get_session_meta(session_id)
        assert meta["id"] == session_id
        assert meta["title"] == "my session"
        assert meta["summary_message_id"] is None
        assert meta["total_input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_update_session_meta(self, store):
        session_id = await store.create_session("test")
        await store.update_session_meta(
            session_id,
            summary_message_id=42,
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        meta = await store.get_session_meta(session_id)
        assert meta["summary_message_id"] == 42
        assert meta["total_input_tokens"] == 1000
        assert meta["total_output_tokens"] == 500

    @pytest.mark.asyncio
    async def test_update_ignores_unknown_fields(self, store):
        session_id = await store.create_session("test")
        # Should not raise
        await store.update_session_meta(session_id, unknown_field="value")

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_empty(self, store):
        meta = await store.get_session_meta("nonexistent")
        assert meta == {}


class TestChildSession:
    @pytest.mark.asyncio
    async def test_create_child_session(self, store):
        parent_id = await store.create_session("parent")
        child_id = await store.create_child_session(parent_id, "child task")

        meta = await store.get_session_meta(child_id)
        assert meta["parent_session_id"] == parent_id
        assert meta["title"] == "child task"

    @pytest.mark.asyncio
    async def test_child_sessions_hidden_from_list(self, store):
        parent_id = await store.create_session("parent")
        await store.create_child_session(parent_id, "child")

        sessions = await store.list_sessions()
        # Only parent should appear
        assert len(sessions) == 1
        assert sessions[0]["id"] == parent_id


class TestMigration:
    @pytest.mark.asyncio
    async def test_migration_idempotent(self, store):
        """Running migration twice should not fail."""
        db = await store._get_db()
        # Force re-migration
        await store._migrate(db)
        # Should still work
        session_id = await store.create_session("test")
        assert session_id
