"""Tool registration, lookup, and execution.

Handles tool lifecycle: registration, schema generation, execution with
timeout and output truncation. Isolates tool errors from the agent loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from comfyui_agent.domain.tools.base import Tool, ToolResult
from comfyui_agent.infrastructure.clients.llm_client import ToolSchema

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 15000


class ToolExecutor:
    """Manages tool registration and execution."""

    def __init__(self, tools: list[Tool], timeout: float = 60.0) -> None:
        self._tools: dict[str, Tool] = {t.info().name: t for t in tools}
        self._timeout = timeout
        self.schemas: list[ToolSchema] = [
            ToolSchema(
                name=t.info().name,
                description=t.info().description,
                input_schema=t.info().parameters,
            )
            for t in tools
        ]

    def get(self, tool_name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(tool_name)

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool with error isolation, timeout, and output truncation."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {tool_name}")

        try:
            result = await asyncio.wait_for(
                tool.run(params), timeout=self._timeout
            )
            result.text = _truncate(result.text)
            logger.info(
                "Tool %s completed: %s",
                tool_name,
                "error" if result.is_error else "ok",
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Tool %s timed out", tool_name)
            return ToolResult.error(
                f"Tool '{tool_name}' timed out after {self._timeout:.0f} seconds"
            )
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return ToolResult.error(f"Tool '{tool_name}' failed: {e}")


def _truncate(text: str, max_len: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate large tool output, keeping first and last portions."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    mid_lines = text[half:-half].count("\n")
    return (
        f"{text[:half]}\n\n"
        f"... [{mid_lines} lines truncated] ...\n\n"
        f"{text[-half:]}"
    )
