"""Message construction helpers for the Anthropic API format.

Builds assistant messages (text + tool_use blocks) and tool result
messages in the format expected by the Claude API.
"""

from __future__ import annotations

from typing import Any

from comfyui_agent.infrastructure.clients.llm_client import LLMResponse


def build_assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Build an assistant message with text and tool_use blocks."""
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for tc in response.tool_calls:
        content.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.input,
        })
    return {"role": "assistant", "content": content}


def build_tool_results_message(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a user message containing tool_result content blocks.

    Anthropic API expects tool results as role=user with tool_result blocks.
    """
    return {"role": "user", "content": results}


def build_tool_result_block(
    tool_use_id: str, text: str, is_error: bool
) -> dict[str, Any]:
    """Build a single tool_result content block."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": is_error,
    }
