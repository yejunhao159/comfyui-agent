"""Tests for LLM retry logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfyui_agent.infrastructure.clients.llm_client import LLMClient, LLMResponse


class FakeRateLimitError(Exception):
    """Simulates anthropic.RateLimitError."""
    def __init__(self, message="rate limited"):
        super().__init__(message)
        self.response = MagicMock()
        self.response.headers = {}


class FakeInternalServerError(Exception):
    """Simulates anthropic.InternalServerError."""
    def __init__(self, message="internal error"):
        super().__init__(message)
        self.response = MagicMock()
        self.response.headers = {}


@pytest.fixture
def llm_client():
    """Create an LLMClient with fast retry settings for testing."""
    client = LLMClient(
        api_key="test-key",
        max_retries=3,
        retry_base_delay_ms=10,  # Very fast for tests
        retry_max_delay_ms=50,
    )
    return client


class TestRetryDelay:
    def test_exponential_backoff(self, llm_client):
        """Delay should increase exponentially with attempts."""
        d1 = llm_client._calc_delay(1, FakeRateLimitError())
        d2 = llm_client._calc_delay(2, FakeRateLimitError())
        d3 = llm_client._calc_delay(3, FakeRateLimitError())
        # With jitter, d2 should be roughly 2x d1, d3 roughly 4x d1
        assert d2 > d1 * 1.2  # At least somewhat larger
        assert d3 > d2 * 1.2

    def test_respects_max_delay(self, llm_client):
        """Delay should never exceed max_delay_ms."""
        delay = llm_client._calc_delay(10, FakeRateLimitError())
        assert delay <= llm_client.retry_max_delay_ms

    def test_respects_retry_after_header(self, llm_client):
        """Should use Retry-After header when present."""
        error = FakeRateLimitError()
        error.response.headers = {"retry-after": "5"}
        delay = llm_client._calc_delay(1, error)
        assert delay == 5000

    def test_jitter_varies(self, llm_client):
        """Jitter should produce different delays for same attempt."""
        delays = set()
        for _ in range(20):
            delays.add(llm_client._calc_delay(1, FakeRateLimitError()))
        # With 20 samples and 0.8-1.2 jitter, we should get multiple values
        assert len(delays) > 1


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self, llm_client):
        """No retry needed when first attempt succeeds."""
        expected = LLMResponse(text="hello", stop_reason="end_turn")
        llm_client._do_chat = AsyncMock(return_value=expected)

        result = await llm_client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.text == "hello"
        assert llm_client._do_chat.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self, llm_client):
        """Should retry on RateLimitError and succeed."""
        import anthropic

        expected = LLMResponse(text="ok", stop_reason="end_turn")
        llm_client._do_chat = AsyncMock(
            side_effect=[
                anthropic.RateLimitError.__new__(anthropic.RateLimitError),
                expected,
            ]
        )
        # Patch the isinstance check to work with our mock
        with patch.object(llm_client, '_do_chat', side_effect=[
            FakeRateLimitError("429"),
            expected,
        ]) as mock_chat:
            # We need to patch anthropic error types for isinstance checks
            with patch('comfyui_agent.infrastructure.clients.llm_client.anthropic') as mock_anthropic:
                mock_anthropic.RateLimitError = FakeRateLimitError
                mock_anthropic.InternalServerError = FakeInternalServerError

                result = await llm_client.chat(
                    messages=[{"role": "user", "content": "hi"}]
                )

                assert result.text == "ok"
                assert mock_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries(self, llm_client):
        """Should raise after exhausting all retries."""
        with patch.object(llm_client, '_do_chat', side_effect=FakeRateLimitError("429")):
            with patch('comfyui_agent.infrastructure.clients.llm_client.anthropic') as mock_anthropic:
                mock_anthropic.RateLimitError = FakeRateLimitError
                mock_anthropic.InternalServerError = FakeInternalServerError

                with pytest.raises(FakeRateLimitError):
                    await llm_client.chat(
                        messages=[{"role": "user", "content": "hi"}]
                    )

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self, llm_client):
        """Non-retryable errors should not be retried."""
        llm_client._do_chat = AsyncMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            await llm_client.chat(messages=[{"role": "user", "content": "hi"}])

        assert llm_client._do_chat.call_count == 1

    @pytest.mark.asyncio
    async def test_emits_retry_event(self, llm_client):
        """Should emit LLM_RETRY event on each retry."""
        from comfyui_agent.infrastructure.event_bus import EventBus

        event_bus = EventBus()
        llm_client.event_bus = event_bus
        events_received = []
        event_bus.on_all(lambda e: events_received.append(e))

        expected = LLMResponse(text="ok", stop_reason="end_turn")
        with patch.object(llm_client, '_do_chat', side_effect=[
            FakeRateLimitError("429"),
            expected,
        ]):
            with patch('comfyui_agent.infrastructure.clients.llm_client.anthropic') as mock_anthropic:
                mock_anthropic.RateLimitError = FakeRateLimitError
                mock_anthropic.InternalServerError = FakeInternalServerError

                await llm_client.chat(
                    messages=[{"role": "user", "content": "hi"}]
                )

        retry_events = [e for e in events_received if e.type.value == "llm.retry"]
        assert len(retry_events) == 1
        assert retry_events[0].data["attempt"] == 1
