"""Event types for the agent system.

Four-layer event system inspired by AgentX:
- Stream: Raw LLM streaming events (text_delta, tool_call, etc.)
- State: Agent state transitions (thinking, responding, executing)
- Message: Complete messages (user, assistant, tool_result)
- Turn: Analytics (cost, duration, tokens)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# Agent States (Mealy Machine)
# ============================================================

class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    RESPONDING = "responding"
    PLANNING_TOOL = "planning_tool"
    AWAITING_TOOL_RESULT = "awaiting_tool_result"
    ERROR = "error"


# ============================================================
# Event Types
# ============================================================

class EventType(str, Enum):
    # Stream layer - raw LLM events
    STREAM_TEXT_DELTA = "stream.text_delta"
    STREAM_TOOL_CALL_START = "stream.tool_call_start"
    STREAM_TOOL_CALL_DELTA = "stream.tool_call_delta"
    STREAM_MESSAGE_STOP = "stream.message_stop"

    # State layer - agent state transitions
    STATE_CONVERSATION_START = "state.conversation_start"
    STATE_THINKING = "state.thinking"
    STATE_RESPONDING = "state.responding"
    STATE_TOOL_PLANNED = "state.tool_planned"
    STATE_TOOL_EXECUTING = "state.tool_executing"
    STATE_TOOL_COMPLETED = "state.tool_completed"
    STATE_TOOL_FAILED = "state.tool_failed"
    STATE_CONVERSATION_END = "state.conversation_end"
    STATE_ERROR = "state.error"

    # Message layer - complete messages
    MESSAGE_USER = "message.user"
    MESSAGE_ASSISTANT = "message.assistant"
    MESSAGE_TOOL_CALL = "message.tool_call"
    MESSAGE_TOOL_RESULT = "message.tool_result"
    MESSAGE_ERROR = "message.error"

    # Turn layer - analytics
    TURN_START = "turn.start"
    TURN_END = "turn.end"

    # ComfyUI specific events
    COMFYUI_PROGRESS = "comfyui.progress"
    COMFYUI_EXECUTING = "comfyui.executing"
    COMFYUI_EXECUTED = "comfyui.executed"
    COMFYUI_PREVIEW = "comfyui.preview"
    COMFYUI_ERROR = "comfyui.error"
    COMFYUI_QUEUE_UPDATE = "comfyui.queue_update"

    # Workflow events — sent to plugin for canvas integration
    WORKFLOW_SUBMITTED = "workflow.submitted"


# ============================================================
# Event Data Classes
# ============================================================

@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()


@dataclass
class StateChange:
    prev: AgentState
    current: AgentState
