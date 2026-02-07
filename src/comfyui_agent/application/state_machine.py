"""Agent state machine.

Manages agent state transitions driven by events.
Inspired by AgentX's Mealy Machine pattern: (state, input) → (state, outputs)
"""

from __future__ import annotations

import logging
from typing import Callable

from comfyui_agent.domain.models.events import AgentState, Event, EventType, StateChange

logger = logging.getLogger(__name__)

StateChangeHandler = Callable[[StateChange], None]

# State transition table: (current_state, event_type) → new_state
_TRANSITIONS: dict[tuple[AgentState, EventType], AgentState] = {
    # Conversation start
    (AgentState.IDLE, EventType.STATE_CONVERSATION_START): AgentState.THINKING,

    # LLM thinking → responding
    (AgentState.THINKING, EventType.STATE_RESPONDING): AgentState.RESPONDING,

    # LLM decides to call a tool
    (AgentState.THINKING, EventType.STATE_TOOL_PLANNED): AgentState.PLANNING_TOOL,
    (AgentState.RESPONDING, EventType.STATE_TOOL_PLANNED): AgentState.PLANNING_TOOL,

    # Tool execution
    (AgentState.PLANNING_TOOL, EventType.STATE_TOOL_EXECUTING): AgentState.AWAITING_TOOL_RESULT,

    # Tool completed → back to thinking (LLM processes result)
    (AgentState.AWAITING_TOOL_RESULT, EventType.STATE_TOOL_COMPLETED): AgentState.THINKING,
    (AgentState.AWAITING_TOOL_RESULT, EventType.STATE_TOOL_FAILED): AgentState.THINKING,

    # Conversation end
    (AgentState.RESPONDING, EventType.STATE_CONVERSATION_END): AgentState.IDLE,
    (AgentState.THINKING, EventType.STATE_CONVERSATION_END): AgentState.IDLE,

    # Error from any active state
    (AgentState.THINKING, EventType.STATE_ERROR): AgentState.ERROR,
    (AgentState.RESPONDING, EventType.STATE_ERROR): AgentState.ERROR,
    (AgentState.PLANNING_TOOL, EventType.STATE_ERROR): AgentState.ERROR,
    (AgentState.AWAITING_TOOL_RESULT, EventType.STATE_ERROR): AgentState.ERROR,

    # Recovery from error
    (AgentState.ERROR, EventType.STATE_CONVERSATION_END): AgentState.IDLE,
}


class AgentStateMachine:
    """Manages agent state transitions.

    State flow:
        idle → thinking → responding → idle
                   ↓
          planning_tool → awaiting_tool_result → thinking → ...

    Any active state can transition to error → idle
    """

    def __init__(self) -> None:
        self._state = AgentState.IDLE
        self._handlers: list[StateChangeHandler] = []

    @property
    def state(self) -> AgentState:
        return self._state

    def process(self, event: Event) -> AgentState:
        """Process an event and transition state if applicable.

        Returns the new state (may be unchanged).
        """
        key = (self._state, event.type)
        new_state = _TRANSITIONS.get(key)

        if new_state is not None and new_state != self._state:
            prev = self._state
            self._state = new_state
            logger.debug("State: %s → %s (event: %s)", prev.value, new_state.value, event.type.value)
            change = StateChange(prev=prev, current=new_state)
            self._notify(change)

        return self._state

    def on_state_change(self, handler: StateChangeHandler) -> Callable[[], None]:
        """Subscribe to state changes. Returns unsubscribe function."""
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    def reset(self) -> None:
        """Reset to idle state."""
        if self._state != AgentState.IDLE:
            prev = self._state
            self._state = AgentState.IDLE
            self._notify(StateChange(prev=prev, current=AgentState.IDLE))

    def _notify(self, change: StateChange) -> None:
        for handler in self._handlers:
            try:
                handler(change)
            except Exception:
                logger.exception("State change handler error: %s → %s", change.prev, change.current)
