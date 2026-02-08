"""Sub-agent tool — delegates exploration tasks to a child agent.

The sub-agent runs with a limited set of read-only tools and a smaller
iteration budget, returning its findings as a tool result to the parent.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult

if TYPE_CHECKING:
    from comfyui_agent.application.prompt_manager import get_default_prompt
    from comfyui_agent.domain.ports import EventBusPort, LLMPort, SessionPort

logger = logging.getLogger(__name__)

_SUBAGENT_SYSTEM_PROMPT = """\
You are a ComfyUI research assistant. Your job is to investigate a specific \
question about ComfyUI nodes, models, or workflows and return a clear, \
concise answer.

You have access to read-only ComfyUI tools. Use them to gather information, \
then provide your findings as a final text response.

Rules:
- Be concise — your output will be fed back to the main agent as context
- Do NOT attempt to queue prompts or modify anything
- Focus on answering the specific question asked
- If you can't find the answer, say so clearly
"""


class SubAgentTool(Tool):
    """Delegates exploration tasks to a child agent with read-only tools."""

    def __init__(
        self,
        llm: LLMPort,
        session_store: SessionPort,
        event_bus: EventBusPort,
        read_only_tools: list[Tool],
        max_iterations: int = 10,
    ) -> None:
        self._llm = llm
        self._session_store = session_store
        self._event_bus = event_bus
        self._read_only_tools = read_only_tools
        self._max_iterations = max_iterations

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="delegate_task",
            description=(
                "Delegate a research or exploration task to a sub-agent. "
                "The sub-agent has read-only access to ComfyUI tools "
                "(search_nodes, get_node_detail, get_connectable, list_models, system_stats). "
                "Use this for complex investigations that require multiple tool calls, "
                "so you can continue focusing on the main task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A clear description of what to investigate.",
                    },
                },
                "required": ["task"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        task = params.get("task", "")
        if not task:
            return ToolResult.error("task parameter is required")

        from comfyui_agent.application.agent_loop import AgentLoop
        from comfyui_agent.domain.models.events import Event, EventType

        # Create child session
        # Use a dummy parent_id — the caller should provide session context
        # but the tool doesn't have direct access to it. We use a fixed prefix.
        child_session_id = await self._session_store.create_child_session(
            parent_id="subagent", title=f"Sub-agent: {task[:50]}"
        )

        await self._event_bus.emit(Event(
            type=EventType.SUBAGENT_START,
            data={"task": task, "child_session_id": child_session_id},
        ))

        try:
            sub_agent = AgentLoop(
                llm=self._llm,
                tools=self._read_only_tools,
                session_store=self._session_store,
                event_bus=self._event_bus,
                max_iterations=self._max_iterations,
                system_prompt=_SUBAGENT_SYSTEM_PROMPT,
            )
            result_text = await sub_agent.run(child_session_id, task)

            await self._event_bus.emit(Event(
                type=EventType.SUBAGENT_END,
                data={"result_preview": result_text[:200]},
            ))

            return ToolResult.success(result_text)

        except Exception as e:
            logger.exception("Sub-agent failed for task: %s", task)
            await self._event_bus.emit(Event(
                type=EventType.SUBAGENT_END,
                data={"result_preview": f"Error: {e}"},
            ))
            return ToolResult.error(f"Sub-agent failed: {e}")
