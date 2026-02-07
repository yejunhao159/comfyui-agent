"""LLM client for Claude API with tool_use support.

Wraps the Anthropic SDK to provide streaming responses and tool calling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import anthropic

from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class ToolSchema:
    """Tool definition for the LLM."""
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMClient:
    """Client for Anthropic Claude API with streaming and tool_use."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        event_bus: EventBus | None = None,
    ) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.event_bus = event_bus

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        system: str = "",
    ) -> LLMResponse:
        """Send messages to Claude and get a complete response.

        Uses streaming internally for real-time event emission.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        response = LLMResponse()

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                await self._handle_stream_event(event, response)

            # get_final_message() must be called inside the async with block
            final = await stream.get_final_message()

        response.stop_reason = final.stop_reason or ""
        response.usage = {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
        }

        # Extract tool calls from the final message content blocks
        for block in final.content:
            if block.type == "tool_use":
                response.tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        logger.info(
            "LLM response: stop=%s, tools=%d, tokens=%s",
            response.stop_reason,
            len(response.tool_calls),
            response.usage,
        )
        return response

    async def _handle_stream_event(self, event: Any, response: LLMResponse) -> None:
        """Process a streaming event from Claude."""
        event_type = type(event).__name__

        if event_type == "TextDelta":
            response.text += event.text
            if self.event_bus:
                await self.event_bus.emit(Event(
                    type=EventType.STREAM_TEXT_DELTA,
                    data={"text": event.text},
                ))

        elif event_type == "InputJsonDelta":
            if self.event_bus:
                await self.event_bus.emit(Event(
                    type=EventType.STREAM_TOOL_CALL_DELTA,
                    data={"partial_json": event.partial_json},
                ))

        elif event_type == "ContentBlockStart":
            block = event.content_block
            if hasattr(block, "type") and block.type == "tool_use":
                if self.event_bus:
                    await self.event_bus.emit(Event(
                        type=EventType.STREAM_TOOL_CALL_START,
                        data={"tool_name": block.name, "tool_id": block.id},
                    ))

        elif event_type == "MessageStop":
            if self.event_bus:
                await self.event_bus.emit(Event(
                    type=EventType.STREAM_MESSAGE_STOP,
                    data={"stop_reason": response.stop_reason},
                ))

    async def close(self) -> None:
        """Close the client."""
        await self.client.close()
