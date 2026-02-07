"""Event bus for internal pub/sub communication.

Inspired by AgentX's EventQueue but simplified for our use case:
- In-memory pub/sub using asyncio
- Topic-based subscription with wildcard support
- Event history for debugging
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Awaitable

from comfyui_agent.domain.models.events import Event, EventType

logger = logging.getLogger(__name__)

EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Central event bus for agent internal communication.

    Supports:
    - Exact topic subscription: bus.on(EventType.STATE_THINKING, handler)
    - Prefix subscription: bus.on_prefix("state.", handler)
    - All events: bus.on_all(handler)
    """

    def __init__(self, history_size: int = 100) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._prefix_handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._all_handlers: list[EventHandler] = []
        self._history: list[Event] = []
        self._history_size = history_size

    def on(self, event_type: EventType, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to a specific event type. Returns unsubscribe function."""
        self._handlers[event_type].append(handler)
        return lambda: self._handlers[event_type].remove(handler)

    def on_prefix(self, prefix: str, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to all events matching a prefix (e.g., 'state.', 'comfyui.')."""
        self._prefix_handlers[prefix].append(handler)
        return lambda: self._prefix_handlers[prefix].remove(handler)

    def on_all(self, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to all events."""
        self._all_handlers.append(handler)
        return lambda: self._all_handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all matching subscribers."""
        # Record history
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]

        # Collect all matching handlers
        handlers: list[EventHandler] = []

        # Exact match
        if event.type in self._handlers:
            handlers.extend(self._handlers[event.type])

        # Prefix match
        event_str = event.type.value
        for prefix, prefix_handlers in self._prefix_handlers.items():
            if event_str.startswith(prefix):
                handlers.extend(prefix_handlers)

        # All handlers
        handlers.extend(self._all_handlers)

        # Execute handlers
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Event handler error for %s", event.type.value)

    def emit_sync(self, event: Event) -> None:
        """Emit an event synchronously (schedules async handlers)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self.emit(event))
        else:
            loop.run_until_complete(self.emit(event))

    def get_history(self, event_type: EventType | None = None) -> list[Event]:
        """Get event history, optionally filtered by type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.type == event_type]

    def clear(self) -> None:
        """Clear all handlers and history."""
        self._handlers.clear()
        self._prefix_handlers.clear()
        self._all_handlers.clear()
        self._history.clear()
