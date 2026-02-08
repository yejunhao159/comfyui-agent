"""Core Agent loop.

The central orchestrator that coordinates LLM, tools, and state management.
Inspired by OpenCode's agent.go:276-311 pattern.

Responsibilities are split across:
- agent_loop.py     — ReAct loop orchestration (this file)
- tool_executor.py  — Tool registration, lookup, execution
- message_builder.py — Anthropic API message formatting
- prompt_manager.py — System prompt loading
- state_machine.py  — Mealy state machine
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from comfyui_agent.application.context_manager import ContextManager
from comfyui_agent.application.message_builder import (
    build_assistant_message,
    build_tool_result_block,
    build_tool_results_message,
)
from comfyui_agent.application.prompt_manager import get_default_prompt
from comfyui_agent.application.state_machine import AgentStateMachine
from comfyui_agent.application.tool_executor import ToolExecutor
from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.domain.ports import EventBusPort, LLMPort, SessionPort
from comfyui_agent.domain.tools.base import Tool

logger = logging.getLogger(__name__)


class AgentLoop:
    """Core agent loop: user input → LLM → tool calls → repeat → response.

    Implements the ReAct (Reason + Act) pattern:
    1. User sends a message
    2. LLM reasons about what to do
    3. If LLM wants to use a tool → execute it → feed result back → goto 2
    4. If LLM has a final answer → return it to user
    """

    def __init__(
        self,
        llm: LLMPort,
        tools: list[Tool],
        session_store: SessionPort,
        event_bus: EventBusPort,
        max_iterations: int = 20,
        system_prompt: str | None = None,
        context_manager: ContextManager | None = None,
        summarizer: Any | None = None,
    ) -> None:
        self.llm = llm
        self.tool_executor = ToolExecutor(tools)
        self.session_store = session_store
        self.event_bus = event_bus
        self.state_machine = AgentStateMachine()
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or get_default_prompt()
        self.context_manager = context_manager
        self.summarizer = summarizer
        self._cancel_flags: dict[str, bool] = {}

    async def run(self, session_id: str, user_input: str) -> str:
        """Run the agent loop for a user message.

        Args:
            session_id: Session identifier
            user_input: User's natural language input

        Returns:
            Agent's final text response
        """
        await self._emit_state(EventType.STATE_CONVERSATION_START, session_id)
        self.state_machine.process(Event(type=EventType.STATE_CONVERSATION_START))

        # Load from summary checkpoint if available
        meta = await self.session_store.get_session_meta(session_id)
        from_id = meta.get("summary_message_id") or 0
        messages = await self.session_store.load_messages_from(session_id, from_id=from_id)

        # Persist user message immediately
        messages.append({"role": "user", "content": user_input})
        await self.session_store.append_message(session_id, "user", user_input)

        await self._emit(EventType.MESSAGE_USER, session_id, {"content": user_input})
        await self._emit(EventType.TURN_START, session_id)

        turn_start = time.time()
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        self._cancel_flags[session_id] = False
        iteration = -1
        recent_tools: list[str] = []  # Track recent tool names for loop detection

        try:
            for iteration in range(self.max_iterations):
                if self._cancel_flags.get(session_id, False):
                    logger.info("Agent cancelled for session %s", session_id)
                    break

                logger.info(
                    "Iteration %d/%d for session %s",
                    iteration + 1, self.max_iterations, session_id,
                )
                self.state_machine.process(Event(type=EventType.STATE_THINKING))
                # Emit thinking event so frontend can reset streaming text
                await self._emit_state(EventType.STATE_THINKING, session_id)

                # Summarize if needed (semantic compression)
                if self.summarizer:
                    messages = await self.summarizer.maybe_summarize(
                        session_id, messages
                    )

                # Compact context if needed (safety fallback)
                if self.context_manager:
                    messages = self.context_manager.prepare_messages(messages)

                # Build system prompt with dynamic warnings
                system = self.system_prompt
                loop_warning = self._check_tool_loop(recent_tools)
                if loop_warning:
                    system += "\n\n" + loop_warning

                response = await self.llm.chat(
                    messages=messages,
                    tools=self.tool_executor.schemas or None,
                    system=system,
                )

                # Accumulate usage
                for k in total_usage:
                    total_usage[k] += response.usage.get(k, 0)

                # Tool calls → execute and loop
                if response.has_tool_calls():
                    # Track tool names for loop detection
                    for tc in response.tool_calls:
                        recent_tools.append(self._tool_display_name(tc))

                    assistant_msg = build_assistant_message(response)
                    messages.append(assistant_msg)
                    await self.session_store.append_message(
                        session_id, "assistant", assistant_msg["content"]
                    )
                    await self._emit(
                        EventType.MESSAGE_ASSISTANT, session_id,
                        {"content": response.text, "tool_calls": len(response.tool_calls)},
                    )

                    tool_results = await self._execute_tools(
                        response.tool_calls, session_id
                    )
                    tool_results_msg = build_tool_results_message(tool_results)
                    messages.append(tool_results_msg)
                    await self.session_store.append_message(
                        session_id, "user", tool_results_msg["content"]
                    )

                    continue

                # Final answer — persist immediately
                self.state_machine.process(Event(type=EventType.STATE_RESPONDING))
                messages.append({"role": "assistant", "content": response.text})
                await self.session_store.append_message(
                    session_id, "assistant", response.text
                )

                await self._emit(
                    EventType.MESSAGE_ASSISTANT, session_id,
                    {"content": response.text, "tool_calls": 0},
                )

                # Update token totals
                await self.session_store.update_session_meta(
                    session_id,
                    total_input_tokens=total_usage["input_tokens"],
                    total_output_tokens=total_usage["output_tokens"],
                )

                await self._finish_turn(
                    session_id, turn_start, iteration + 1, total_usage
                )
                return response.text

            # Loop exited: cancelled or max iterations
            if self._cancel_flags.get(session_id, False):
                final_text = "Request cancelled."
                logger.info("Agent cancelled for session %s", session_id)
            else:
                final_text = (
                    "I've reached the maximum number of steps. "
                    "Here's what I've done so far."
                )
                logger.warning(
                    "Max iterations reached for session %s", session_id
                )

            messages.append({"role": "assistant", "content": final_text})
            await self.session_store.append_message(
                session_id, "assistant", final_text
            )

            await self._finish_turn(
                session_id, turn_start, iteration + 1, total_usage
            )
            return final_text

        except Exception as e:
            logger.exception("Agent loop error for session %s", session_id)
            self.state_machine.process(Event(type=EventType.STATE_ERROR))
            await self._emit(
                EventType.STATE_ERROR, session_id, {"error": str(e)}
            )
            await self._finish_turn(
                session_id, turn_start, iteration + 1, total_usage
            )
            raise

        finally:
            self._cancel_flags.pop(session_id, None)

    def cancel(self, session_id: str) -> None:
        """Cancel a running agent loop."""
        self._cancel_flags[session_id] = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _finish_turn(
        self,
        session_id: str,
        turn_start: float,
        iterations: int,
        usage: dict[str, int],
    ) -> None:
        """Emit conversation_end + turn.end — called from every exit path."""
        self.state_machine.process(Event(type=EventType.STATE_CONVERSATION_END))
        await self._emit_state(EventType.STATE_CONVERSATION_END, session_id)
        await self._emit(
            EventType.TURN_END, session_id,
            {
                "duration": time.time() - turn_start,
                "iterations": iterations,
                "usage": usage,
            },
        )

    @staticmethod
    def _tool_display_name(tc: Any) -> str:
        """Resolve display name for a tool call.

        For dispatcher tools (action+params pattern), use the action name
        so the frontend shows 'list_models' instead of 'comfyui'.
        """
        if isinstance(tc.input, dict) and "action" in tc.input:
            return str(tc.input["action"])
        return tc.name

    @staticmethod
    def _check_tool_loop(
        recent_tools: list[str], threshold: int = 3
    ) -> str | None:
        """Detect if the agent is stuck calling the same tool repeatedly.

        Returns a warning string to inject into the system prompt, or None.
        """
        if len(recent_tools) < threshold:
            return None
        tail = recent_tools[-threshold:]
        if len(set(tail)) == 1:
            tool_name = tail[0]
            count = len(recent_tools)
            logger.warning(
                "Loop detected: %s called %d times in last %d calls",
                tool_name, threshold, count,
            )
            return (
                f"⚠️ LOOP DETECTED: You have called '{tool_name}' {threshold} "
                f"times in a row. STOP repeating this tool call. "
                f"Either try a completely different approach, or explain "
                f"the problem to the user and ask for guidance."
            )
        return None

    async def _execute_tools(
        self, tool_calls: list[Any], session_id: str
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls in parallel, emitting events for each."""
        # Phase 1: resolve display names, emit executing events
        metas: list[tuple[Any, str]] = []
        self.state_machine.process(Event(type=EventType.STATE_TOOL_PLANNED))
        self.state_machine.process(Event(type=EventType.STATE_TOOL_EXECUTING))

        for tc in tool_calls:
            display = self._tool_display_name(tc)
            metas.append((tc, display))
            await self._emit(
                EventType.STATE_TOOL_EXECUTING, session_id,
                {"tool_name": display, "tool_id": tc.id},
            )

        # Phase 2: execute all tools in parallel
        async def _run_one(tc: Any) -> Any:
            return await self.tool_executor.execute(tc.name, tc.input)

        raw = await asyncio.gather(
            *(_run_one(tc) for tc, _ in metas),
            return_exceptions=True,
        )

        # Phase 3: emit results, build tool_result blocks
        results: list[dict[str, Any]] = []
        any_failed = False

        for (tc, display), outcome in zip(metas, raw):
            if isinstance(outcome, BaseException):
                any_failed = True
                error_text = f"Tool '{display}' failed: {outcome}"
                await self._emit(
                    EventType.STATE_TOOL_FAILED, session_id,
                    {"tool_name": display, "error": error_text},
                )
                results.append(
                    build_tool_result_block(tc.id, error_text, True)
                )
                await self._emit(
                    EventType.MESSAGE_TOOL_RESULT, session_id,
                    {"tool_name": display, "result": error_text[:500]},
                )
                continue

            result = outcome
            if result.is_error:
                any_failed = True
                await self._emit(
                    EventType.STATE_TOOL_FAILED, session_id,
                    {"tool_name": display, "error": result.text},
                )
            else:
                await self._emit(
                    EventType.STATE_TOOL_COMPLETED, session_id,
                    {"tool_name": display},
                )
                # Forward workflow to plugin for canvas loading
                if result.data.get("workflow"):
                    await self._emit(
                        EventType.WORKFLOW_SUBMITTED, session_id,
                        {
                            "workflow": result.data["workflow"],
                            "prompt_id": result.data.get("prompt_id", ""),
                        },
                    )

            results.append(
                build_tool_result_block(tc.id, result.text, result.is_error)
            )
            await self._emit(
                EventType.MESSAGE_TOOL_RESULT, session_id,
                {"tool_name": display, "result": result.text[:500]},
            )

        # State machine: batch done → back to thinking
        if any_failed:
            self.state_machine.process(Event(type=EventType.STATE_TOOL_FAILED))
        else:
            self.state_machine.process(Event(type=EventType.STATE_TOOL_COMPLETED))

        return results

    async def _emit(
        self,
        event_type: EventType,
        session_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event through the event bus."""
        await self.event_bus.emit(
            Event(type=event_type, session_id=session_id, data=data or {})
        )

    async def _emit_state(self, event_type: EventType, session_id: str) -> None:
        """Emit a state-only event (no data)."""
        await self._emit(event_type, session_id)
