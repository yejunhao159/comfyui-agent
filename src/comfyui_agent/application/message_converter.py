"""Convert Anthropic API messages to frontend ChatItem format.

The backend stores messages in Anthropic API format (tool_use, tool_result blocks).
The frontend needs ChatItem[] with AgentMessage objects containing ContentBlock[].
This module bridges the two formats for session history loading.
"""

from __future__ import annotations

import uuid
from typing import Any


def api_messages_to_chat_items(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic API format messages to frontend ChatItem list.

    Returns a list of dicts matching the frontend ChatItem type:
      { kind: "message", data: AgentMessage }
    """
    items: list[dict[str, Any]] = []
    current_agent: dict[str, Any] | None = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # Plain text user message
            if isinstance(content, str):
                # Flush any pending agent message
                if current_agent:
                    items.append({"kind": "message", "data": current_agent})
                    current_agent = None

                items.append({
                    "kind": "message",
                    "data": _make_user_message(content),
                })
            elif isinstance(content, list):
                # Tool result message — attach results to current agent
                if current_agent:
                    _attach_tool_results(current_agent, content)

        elif role == "assistant":
            # Flush previous agent message if any
            if current_agent:
                items.append({"kind": "message", "data": current_agent})

            current_agent = _make_agent_message(content)

    # Flush last agent message
    if current_agent:
        items.append({"kind": "message", "data": current_agent})

    return items


def _make_user_message(text: str) -> dict[str, Any]:
    return {
        "id": _uid(),
        "role": "user",
        "content": text,
        "toolCalls": [],
        "blocks": [{"kind": "text", "text": text}],
        "timestamp": 0,
    }


def _make_agent_message(content: Any) -> dict[str, Any]:
    """Build an agent message from assistant content (str or list of blocks)."""
    blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []

    if isinstance(content, str):
        blocks.append({"kind": "text", "text": content})
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                blocks.append({"kind": "text", "text": text})
                text_parts.append(text)
            elif btype == "tool_use":
                tc = {
                    "id": block.get("id", _uid()),
                    "name": _resolve_tool_name(block),
                    "status": "completed",
                }
                tool_calls.append(tc)
                blocks.append({"kind": "tool", "tool": tc})

    return {
        "id": _uid(),
        "role": "agent",
        "content": "\n".join(text_parts),
        "toolCalls": tool_calls,
        "blocks": blocks,
        "timestamp": 0,
    }


def _resolve_tool_name(block: dict[str, Any]) -> str:
    """Resolve display name for a tool_use block.

    For dispatcher tools (action+params pattern), use the action name.
    """
    inp = block.get("input", {})
    if isinstance(inp, dict) and "action" in inp:
        return str(inp["action"])
    return block.get("name", "unknown")


def _attach_tool_results(
    agent_msg: dict[str, Any],
    content_blocks: list[Any],
) -> None:
    """Attach tool_result content to matching tools in the agent message."""
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue

        tool_use_id = block.get("tool_use_id", "")
        result_text = block.get("content", "")
        if isinstance(result_text, list):
            # content can be a list of text blocks
            result_text = " ".join(
                b.get("text", "") for b in result_text if isinstance(b, dict)
            )

        # Find matching tool call by id
        for tc in agent_msg.get("toolCalls", []):
            if tc.get("id") == tool_use_id:
                tc["result"] = str(result_text)[:500]
                if block.get("is_error"):
                    tc["status"] = "failed"
                    tc["error"] = str(result_text)[:500]
                break

        # Also update in blocks
        for b in agent_msg.get("blocks", []):
            if b.get("kind") == "tool" and b.get("tool", {}).get("id") == tool_use_id:
                b["tool"]["result"] = str(result_text)[:500]
                if block.get("is_error"):
                    b["tool"]["status"] = "failed"
                    b["tool"]["error"] = str(result_text)[:500]
                break


def _uid() -> str:
    return str(uuid.uuid4())[:8]
