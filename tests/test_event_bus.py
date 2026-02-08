"""Tests for EventBus — pub/sub event system."""

from __future__ import annotations

import asyncio

import pytest

from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.infrastructure.event_bus import EventBus


@pytest.fixture
def bus() -> EventBus:
    return EventBus(history_size=10)


class TestEventBusSubscription:
    @pytest.mark.asyncio
    async def test_exact_topic(self, bus: EventBus):
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.on(EventType.STATE_THINKING, handler)
        await bus.emit(Event(type=EventType.STATE_THINKING, data={"x": 1}))
        await bus.emit(Event(type=EventType.STATE_RESPONDING))  # should not match

        assert len(received) == 1
        assert received[0].data == {"x": 1}

    @pytest.mark.asyncio
    async def test_prefix_subscription(self, bus: EventBus):
        received = []

        async def handler(event: Event) -> None:
            received.append(event.type)

        bus.on_prefix("state.", handler)
        await bus.emit(Event(type=EventType.STATE_THINKING))
        await bus.emit(Event(type=EventType.STATE_RESPONDING))
        await bus.emit(Event(type=EventType.STREAM_TEXT_DELTA))  # should not match

        assert len(received) == 2
        assert EventType.STATE_THINKING in received
        assert EventType.STATE_RESPONDING in received

    @pytest.mark.asyncio
    async def test_all_subscription(self, bus: EventBus):
        received = []

        async def handler(event: Event) -> None:
            received.append(event.type)

        bus.on_all(handler)
        await bus.emit(Event(type=EventType.STATE_THINKING))
        await bus.emit(Event(type=EventType.STREAM_TEXT_DELTA))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus: EventBus):
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        unsub = bus.on(EventType.STATE_THINKING, handler)
        await bus.emit(Event(type=EventType.STATE_THINKING))
        assert len(received) == 1

        unsub()
        await bus.emit(Event(type=EventType.STATE_THINKING))
        assert len(received) == 1  # no new events after unsubscribe


class TestEventBusHistory:
    @pytest.mark.asyncio
    async def test_history_recorded(self, bus: EventBus):
        await bus.emit(Event(type=EventType.STATE_THINKING))
        await bus.emit(Event(type=EventType.STATE_RESPONDING))

        history = bus.get_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_history_filtered(self, bus: EventBus):
        await bus.emit(Event(type=EventType.STATE_THINKING))
        await bus.emit(Event(type=EventType.STATE_RESPONDING))
        await bus.emit(Event(type=EventType.STATE_THINKING))

        history = bus.get_history(EventType.STATE_THINKING)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_history_size_limit(self, bus: EventBus):
        for i in range(20):
            await bus.emit(Event(type=EventType.TURN_START, data={"i": i}))

        history = bus.get_history()
        assert len(history) == 10  # history_size=10

    @pytest.mark.asyncio
    async def test_clear(self, bus: EventBus):
        received = []
        bus.on(EventType.STATE_THINKING, lambda e: received.append(e))
        await bus.emit(Event(type=EventType.STATE_THINKING))
        assert len(received) == 1

        bus.clear()
        assert len(bus.get_history()) == 0  # history cleared

        await bus.emit(Event(type=EventType.STATE_THINKING))
        assert len(received) == 1  # handler was cleared, no new events received
        assert len(bus.get_history()) == 1  # but new emit still records to history


class TestEventBusErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break_others(self, bus: EventBus):
        received = []

        async def bad_handler(event: Event) -> None:
            raise ValueError("boom")

        async def good_handler(event: Event) -> None:
            received.append(event)

        bus.on(EventType.STATE_THINKING, bad_handler)
        bus.on(EventType.STATE_THINKING, good_handler)

        await bus.emit(Event(type=EventType.STATE_THINKING))
        assert len(received) == 1  # good_handler still ran


class TestEventBusSyncHandler:
    @pytest.mark.asyncio
    async def test_sync_handler(self, bus: EventBus):
        received = []

        def sync_handler(event: Event) -> None:
            received.append(event.type)

        bus.on(EventType.STATE_THINKING, sync_handler)
        await bus.emit(Event(type=EventType.STATE_THINKING))
        assert len(received) == 1
