"""Tests for SubAgentTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from comfyui_agent.domain.tools.subagent import SubAgentTool
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult


class FakeReadOnlyTool(Tool):
    """A minimal read-only tool for testing."""

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="fake_search",
            description="Fake search tool",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, params):
        return ToolResult.success("found 3 nodes")


class TestSubAgentToolInfo:
    def test_tool_name(self):
        tool = SubAgentTool(
            llm=MagicMock(),
            session_store=MagicMock(),
            event_bus=MagicMock(),
            read_only_tools=[],
        )
        info = tool.info()
        assert info.name == "delegate_task"
        assert "task" in info.parameters["properties"]

    def test_requires_task_param(self):
        tool = SubAgentTool(
            llm=MagicMock(),
            session_store=MagicMock(),
            event_bus=MagicMock(),
            read_only_tools=[],
        )
        info = tool.info()
        assert "task" in info.parameters["required"]


class TestSubAgentExecution:
    @pytest.mark.asyncio
    async def test_empty_task_returns_error(self):
        tool = SubAgentTool(
            llm=MagicMock(),
            session_store=MagicMock(),
            event_bus=MagicMock(),
            read_only_tools=[],
        )
        result = await tool.run({"task": ""})
        assert result.is_error
        assert "required" in result.text

    @pytest.mark.asyncio
    async def test_creates_child_session(self):
        session_store = AsyncMock()
        session_store.create_child_session = AsyncMock(return_value="child-123")
        session_store.get_session_meta = AsyncMock(return_value={})
        session_store.load_messages_from = AsyncMock(return_value=[])
        session_store.append_message = AsyncMock(return_value=1)
        session_store.update_session_meta = AsyncMock()

        event_bus = AsyncMock()
        event_bus.emit = AsyncMock()

        # Mock LLM to return a final answer immediately
        llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Found: KSampler has 8 inputs"
        mock_response.tool_calls = []
        mock_response.has_tool_calls.return_value = False
        mock_response.stop_reason = "end_turn"
        mock_response.usage = {"input_tokens": 100, "output_tokens": 50}
        llm.chat = AsyncMock(return_value=mock_response)

        tool = SubAgentTool(
            llm=llm,
            session_store=session_store,
            event_bus=event_bus,
            read_only_tools=[FakeReadOnlyTool()],
        )

        result = await tool.run({"task": "What inputs does KSampler have?"})

        assert not result.is_error
        assert "KSampler" in result.text
        session_store.create_child_session.assert_called_once()
