"""Context window manager for the Agent loop.

Prevents token overflow by compacting messages before each LLM call.
Uses local heuristic token estimation (~4 chars/token) — no API calls.

Compression strategy (by increasing aggressiveness):
1. Truncate old tool results (>500 chars → keep first 200 + "[truncated]")
2. Emergency trim: keep only the last turn (user + assistant)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Model name → max context window tokens
_MODEL_CONTEXT_SIZES: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
}
_DEFAULT_CONTEXT_SIZE = 200_000


def _resolve_context_size(model: str) -> int:
    """Resolve max context window for a model name."""
    if model in _MODEL_CONTEXT_SIZES:
        return _MODEL_CONTEXT_SIZES[model]
    # Fuzzy match: check if any known key is a prefix
    for key, size in _MODEL_CONTEXT_SIZES.items():
        if model.startswith(key):
            return size
    return _DEFAULT_CONTEXT_SIZE


# ------------------------------------------------------------------
# Token estimation
# ------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (~4 chars/token heuristic)."""
    return max(1, len(text) // 4)


def _content_text(content: Any) -> str:
    """Extract text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # text block, tool_result, tool_use — all have useful text
                parts.append(
                    block.get("text", "")
                    or block.get("content", "")
                    or json.dumps(block.get("input", ""))
                )
        return " ".join(parts)
    return str(content)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of Anthropic-format messages."""
    total = 0
    for msg in messages:
        # Role overhead (~4 tokens)
        total += 4
        total += estimate_tokens(_content_text(msg.get("content", "")))
    return total


# ------------------------------------------------------------------
# ContextManager
# ------------------------------------------------------------------

# Overhead tokens reserved for system prompt, tool schemas, output, safety
_SYSTEM_OVERHEAD = 2_000
_TOOL_SCHEMA_OVERHEAD = 3_000
_SAFETY_BUFFER = 5_000


class ContextManager:
    """Manages context window budget for the agent loop.

    Before each LLM call, call ``prepare_messages()`` to ensure the
    conversation history fits within the model's context window.

    Args:
        model: Model name (used to auto-resolve context window size).
        max_output_tokens: Max tokens reserved for LLM output.
        context_budget: Override auto-resolved context size (0 = auto).
    """

    def __init__(
        self,
        model: str = "",
        max_output_tokens: int = 8192,
        context_budget: int = 0,
    ) -> None:
        if context_budget > 0:
            self._context_size = context_budget
        else:
            self._context_size = _resolve_context_size(model)

        self._max_output = max_output_tokens
        self._history_budget = (
            self._context_size
            - _SYSTEM_OVERHEAD
            - _TOOL_SCHEMA_OVERHEAD
            - self._max_output
            - _SAFETY_BUFFER
        )
        logger.info(
            "ContextManager: context=%d, history_budget=%d (model=%s)",
            self._context_size, self._history_budget, model or "default",
        )

    @property
    def history_budget(self) -> int:
        return self._history_budget

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compact messages to fit within the history budget.

        Returns a new list (does not mutate the original).
        """
        tokens = estimate_messages_tokens(messages)
        if tokens <= self._history_budget:
            return messages

        logger.info(
            "Context compaction needed: %d tokens > %d budget",
            tokens, self._history_budget,
        )

        # Strategy 1: truncate old tool results
        compacted = self._compact_tool_results(messages, keep_recent=6)
        tokens = estimate_messages_tokens(compacted)
        if tokens <= self._history_budget:
            logger.info("After tool-result truncation: %d tokens", tokens)
            return compacted

        # Strategy 2: emergency — keep only last turn
        logger.warning(
            "Emergency trim: %d tokens still over budget %d",
            tokens, self._history_budget,
        )
        compacted = self._emergency_trim(compacted)
        tokens = estimate_messages_tokens(compacted)
        logger.info("After emergency trim: %d tokens", tokens)
        return compacted

    @staticmethod
    def _compact_tool_results(
        messages: list[dict[str, Any]],
        keep_recent: int = 6,
        max_result_chars: int = 500,
    ) -> list[dict[str, Any]]:
        """Truncate tool_result content in older messages.

        Messages in the last ``keep_recent`` positions are left intact.
        """
        out: list[dict[str, Any]] = []
        cutoff = max(0, len(messages) - keep_recent)

        for i, msg in enumerate(messages):
            if i >= cutoff:
                out.append(msg)
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue

            new_blocks: list[Any] = []
            changed = False
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > max_result_chars
                ):
                    new_blocks.append({
                        **block,
                        "content": (
                            block["content"][:200]
                            + "\n\n... [truncated, was "
                            + str(len(block["content"]))
                            + " chars]"
                        ),
                    })
                    changed = True
                else:
                    new_blocks.append(block)

            if changed:
                out.append({**msg, "content": new_blocks})
            else:
                out.append(msg)

        return out

    @staticmethod
    def _emergency_trim(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep only the last user message and everything after it.

        This preserves the current turn's context while dropping all
        prior history.
        """
        # Find the last user message that is a plain text message
        # (not a tool_result message)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                content = msg.get("content")
                # Plain user message (string or text blocks, not tool_result)
                if isinstance(content, str):
                    return messages[i:]
                if isinstance(content, list) and content:
                    if content[0].get("type") != "tool_result":
                        return messages[i:]

        # Fallback: keep last 2 messages
        return messages[-2:] if len(messages) >= 2 else messages
