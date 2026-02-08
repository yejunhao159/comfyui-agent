"""Tests for Summarizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from comfyui_agent.application.summarizer import Summarizer


def _make_messages(count: int, chars_per_msg: int = 1000) -> list[dict]:
    """Generate test messages with predictable token counts."""
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * chars_per_msg
        messages.append({"role": role, "content": content})
    return messages


@pytest.fixture
def summarizer():
    """Create a Summarizer with low threshold for testing."""
    llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Summary: user asked about nodes, agent found KSampler."
    llm.chat = AsyncMock(return_value=mock_response)

    session_store = AsyncMock()
    session_store.append_message = AsyncMock(return_value=999)
    session_store.update_session_meta = AsyncMock()

    event_bus = AsyncMock()
    event_bus.emit = AsyncMock()

    return Summarizer(
        llm=llm,
        session_store=session_store,
        event_bus=event_bus,
        threshold=500,  # Very low for testing
        keep_recent=4,
    )


class TestSummarizerTrigger:
    @pytest.mark.asyncio
    async def test_no_summarize_under_threshold(self, summarizer):
        """Should not summarize when under threshold."""
        messages = [{"role": "user", "content": "hi"}]
        result = await summarizer.maybe_summarize("s1", messages)
        assert result == messages
        summarizer._llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_over_threshold(self, summarizer):
        """Should summarize when over threshold."""
        messages = _make_messages(20, chars_per_msg=200)
        result = await summarizer.maybe_summarize("s1", messages)

        # Should have summary + keep_recent messages
        assert len(result) < len(messages)
        assert "[Previous conversation summary]" in result[0]["content"]
        summarizer._llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_too_few_messages_skips(self, summarizer):
        """Should not summarize if too few messages to be meaningful."""
        # With keep_recent=4, need at least 7 messages
        messages = _make_messages(5, chars_per_msg=200)
        result = await summarizer.maybe_summarize("s1", messages)
        assert result == messages


class TestSummarizerOutput:
    @pytest.mark.asyncio
    async def test_summary_persisted(self, summarizer):
        """Summary should be saved to DB."""
        messages = _make_messages(20, chars_per_msg=200)
        await summarizer.maybe_summarize("s1", messages)

        summarizer._session_store.append_message.assert_called_once()
        summarizer._session_store.update_session_meta.assert_called_once_with(
            "s1", summary_message_id=999
        )

    @pytest.mark.asyncio
    async def test_emits_event(self, summarizer):
        """Should emit CONTEXT_SUMMARIZED event."""
        messages = _make_messages(20, chars_per_msg=200)
        await summarizer.maybe_summarize("s1", messages)

        summarizer._event_bus.emit.assert_called_once()
        event = summarizer._event_bus.emit.call_args[0][0]
        assert event.type.value == "context.summarized"
        assert "original_tokens" in event.data
        assert "summary_tokens" in event.data

    @pytest.mark.asyncio
    async def test_keeps_recent_messages(self, summarizer):
        """Recent messages should be preserved after summarization."""
        messages = _make_messages(20, chars_per_msg=200)
        result = await summarizer.maybe_summarize("s1", messages)

        # Last keep_recent messages should be preserved
        for i, msg in enumerate(result[1:]):  # Skip summary
            original_idx = len(messages) - summarizer._keep_recent + i
            assert msg["content"] == messages[original_idx]["content"]


class TestCondenseForSummary:
    def test_plain_text_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = Summarizer._condense_for_summary(messages)
        assert "user: hello" in result
        assert "assistant: hi there" in result

    def test_tool_use_blocks(self):
        messages = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me search"},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
            ]},
        ]
        result = Summarizer._condense_for_summary(messages)
        assert "search" in result

    def test_truncates_long_messages(self):
        messages = [
            {"role": "user", "content": "x" * 1000},
        ]
        result = Summarizer._condense_for_summary(messages)
        assert len(result) < 600  # 500 + role prefix + "..."
