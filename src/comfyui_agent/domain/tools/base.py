"""Base tool interface for the agent.

Every tool the agent can use must implement this interface.
Inspired by OpenCode's BaseTool pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInfo:
    """Tool metadata exposed to the LLM."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    images: list[str] = field(default_factory=list)  # URLs or base64

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(text=message, is_error=True)

    @classmethod
    def success(cls, text: str, **kwargs: Any) -> ToolResult:
        return cls(text=text, **kwargs)


class Tool(ABC):
    """Abstract base class for all agent tools.

    Each tool represents a capability the agent can invoke.
    The LLM sees info() to decide when to call it,
    and run() executes the actual operation.
    """

    @abstractmethod
    def info(self) -> ToolInfo:
        """Return tool metadata (name, description, parameter schema).

        This is sent to the LLM as part of the tools list.
        """
        ...

    @abstractmethod
    async def run(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            params: Parameters from the LLM's tool_use call

        Returns:
            ToolResult with text output and optional data/images
        """
        ...

    def to_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema format."""
        info = self.info()
        return {
            "name": info.name,
            "description": info.description,
            "input_schema": info.parameters,
        }
