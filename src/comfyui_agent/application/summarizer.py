"""Context summarizer — compresses conversation history using LLM.

When the conversation exceeds a token threshold, generates a summary
of older messages and replaces them with the summary. This preserves
semantic information that simple truncation would lose.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from comfyui_agent.application.context_manager import estimate_messages_tokens
from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.domain.ports import EventBusPort, LLMPort, SessionPort

logger = logging.getLogger(__name__)

# Trigger summarization when messages exceed this token count
_SUMMARIZE_THRESHOLD = 80_000
# Keep the most recent N messages unsummarized
_KEEP_RECENT = 10

_SUMMARIZE_PROMPT = """\
You are a conversation summarizer. Summarize the following conversation \
between a user and a ComfyUI assistant. Focus on:

1. What the user wanted to accomplish
2. Key decisions made (node types chosen, model names, parameters)
3. Workflows that were built or submitted (include prompt_ids)
4. Any errors encountered and how they were resolved
5. Current state of the conversation

Be concise but preserve all technical details that would be needed to \
continue the conversation. Output a single summary paragraph.

Conversation to summarize:
"""


class Summarizer:
    """Summarizes conversation history to reduce context window usage."""

    def __init__(
        self,
        llm: LLMPort,
        session_store: SessionPort,
        event_bus: EventBusPort,
        threshold: int = _SUMMARIZE_THRESHOLD,
        keep_recent: int = _KEEP_RECENT,
    ) -> None:
        self._llm = llm
        self._session_store = session_store
        self._event_bus = event_bus
        self._threshold = threshold
        self._keep_recent = keep_recent

    async def maybe_summarize(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check token count and summarize if over threshold.

        Returns the (possibly compressed) message list.
        """
        total_tokens = estimate_messages_tokens(messages)
        if total_tokens <= self._threshold:
            return messages

        if len(messages) <= self._keep_recent + 2:
            # Not enough messages to summarize meaningfully
            return messages

        logger.info(
            "Summarization triggered: %d tokens > %d threshold (%d messages)",
            total_tokens, self._threshold, len(messages),
        )

        # Split: old messages to summarize, recent messages to keep
        cutoff = len(messages) - self._keep_recent
        old_messages = messages[:cutoff]
        recent_messages = messages[cutoff:]

        summary_text = await self._generate_summary(old_messages)

        # Build new message list: summary + recent
        summary_message = {
            "role": "user",
            "content": f"[Previous conversation summary]\n{summary_text}",
        }
        new_messages = [summary_message] + recent_messages

        # Persist the summary as a checkpoint
        summary_msg_id = await self._session_store.append_message(
            session_id, "user", summary_message["content"]
        )
        await self._session_store.update_session_meta(
            session_id, summary_message_id=summary_msg_id
        )

        new_tokens = estimate_messages_tokens(new_messages)
        await self._event_bus.emit(Event(
            type=EventType.CONTEXT_SUMMARIZED,
            data={
                "original_tokens": total_tokens,
                "summary_tokens": new_tokens,
                "messages_summarized": len(old_messages),
            },
        ))

        logger.info(
            "Summarized %d messages: %d → %d tokens",
            len(old_messages), total_tokens, new_tokens,
        )
        return new_messages

    async def _generate_summary(self, messages: list[dict[str, Any]]) -> str:
        """Use LLM to generate a summary of the messages."""
        condensed = self._condense_for_summary(messages)
        prompt = _SUMMARIZE_PROMPT + condensed

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            system="You are a concise summarizer. Output only the summary.",
        )
        return response.text

    @staticmethod
    def _condense_for_summary(messages: list[dict[str, Any]]) -> str:
        """Convert messages to a readable text format for summarization."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Extract text from content blocks
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            texts.append(
                                f"[Tool: {block.get('name', '?')}({json.dumps(block.get('input', {}))[:200]})]"
                            )
                        elif block.get("type") == "tool_result":
                            result_text = str(block.get("content", ""))[:300]
                            texts.append(f"[Result: {result_text}]")
                text = " ".join(texts)
            else:
                text = str(content)

            # Truncate very long messages
            if len(text) > 500:
                text = text[:500] + "..."

            parts.append(f"{role}: {text}")

        return "\n".join(parts)
