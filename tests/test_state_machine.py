"""Tests for AgentStateMachine — Mealy Machine state transitions."""

from __future__ import annotations

import pytest

from comfyui_agent.application.state_machine import AgentStateMachine
from comfyui_agent.domain.models.events import AgentState, Event, EventType


@pytest.fixture
def sm() -> AgentStateMachine:
    return AgentStateMachine()


class TestStateMachineBasic:
    def test_initial_state(self, sm: AgentStateMachine):
        assert sm.state == AgentState.IDLE

    def test_reset(self, sm: AgentStateMachine):
        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        assert sm.state != AgentState.IDLE
        sm.reset()
        assert sm.state == AgentState.IDLE


class TestStateMachineTransitions:
    def test_conversation_flow(self, sm: AgentStateMachine):
        """idle → thinking → responding → idle"""
        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        assert sm.state == AgentState.THINKING

        sm.process(Event(type=EventType.STATE_RESPONDING))
        assert sm.state == AgentState.RESPONDING

        sm.process(Event(type=EventType.STATE_CONVERSATION_END))
        assert sm.state == AgentState.IDLE

    def test_tool_flow(self, sm: AgentStateMachine):
        """idle → thinking → planning_tool → awaiting → thinking"""
        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        assert sm.state == AgentState.THINKING

        sm.process(Event(type=EventType.STATE_TOOL_PLANNED))
        assert sm.state == AgentState.PLANNING_TOOL

        sm.process(Event(type=EventType.STATE_TOOL_EXECUTING))
        assert sm.state == AgentState.AWAITING_TOOL_RESULT

        sm.process(Event(type=EventType.STATE_TOOL_COMPLETED))
        assert sm.state == AgentState.THINKING

    def test_tool_failure_returns_to_thinking(self, sm: AgentStateMachine):
        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        sm.process(Event(type=EventType.STATE_TOOL_PLANNED))
        sm.process(Event(type=EventType.STATE_TOOL_EXECUTING))
        sm.process(Event(type=EventType.STATE_TOOL_FAILED))
        assert sm.state == AgentState.THINKING

    def test_error_from_thinking(self, sm: AgentStateMachine):
        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        assert sm.state == AgentState.THINKING

        sm.process(Event(type=EventType.STATE_ERROR))
        assert sm.state == AgentState.ERROR

    def test_error_recovery(self, sm: AgentStateMachine):
        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        sm.process(Event(type=EventType.STATE_ERROR))
        assert sm.state == AgentState.ERROR

        sm.process(Event(type=EventType.STATE_CONVERSATION_END))
        assert sm.state == AgentState.IDLE

    def test_invalid_transition_stays(self, sm: AgentStateMachine):
        """Invalid transitions should not change state."""
        assert sm.state == AgentState.IDLE
        sm.process(Event(type=EventType.STATE_RESPONDING))  # invalid from IDLE
        assert sm.state == AgentState.IDLE


class TestStateMachineObserver:
    def test_state_change_handler(self, sm: AgentStateMachine):
        changes = []
        sm.on_state_change(lambda c: changes.append((c.prev, c.current)))

        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        sm.process(Event(type=EventType.STATE_RESPONDING))

        assert len(changes) == 2
        assert changes[0] == (AgentState.IDLE, AgentState.THINKING)
        assert changes[1] == (AgentState.THINKING, AgentState.RESPONDING)

    def test_unsubscribe_handler(self, sm: AgentStateMachine):
        changes = []
        unsub = sm.on_state_change(lambda c: changes.append(c))

        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        assert len(changes) == 1

        unsub()
        sm.process(Event(type=EventType.STATE_RESPONDING))
        assert len(changes) == 1  # no new changes

    def test_reset_notifies(self, sm: AgentStateMachine):
        changes = []
        sm.on_state_change(lambda c: changes.append((c.prev, c.current)))

        sm.process(Event(type=EventType.STATE_CONVERSATION_START))
        sm.reset()

        assert len(changes) == 2
        assert changes[1] == (AgentState.THINKING, AgentState.IDLE)

    def test_reset_from_idle_no_notify(self, sm: AgentStateMachine):
        changes = []
        sm.on_state_change(lambda c: changes.append(c))
        sm.reset()  # already idle
        assert len(changes) == 0
