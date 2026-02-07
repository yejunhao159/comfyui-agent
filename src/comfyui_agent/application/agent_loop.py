"""Core Agent loop.

The central orchestrator that coordinates LLM, tools, and state management.
Inspired by OpenCode's agent.go:276-311 pattern.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from comfyui_agent.application.state_machine import AgentStateMachine
from comfyui_agent.domain.models.events import AgentState, Event, EventType
from comfyui_agent.domain.tools.base import Tool, ToolResult
from comfyui_agent.infrastructure.event_bus import EventBus
from comfyui_agent.infrastructure.llm_client import LLMClient, LLMResponse, ToolSchema
from comfyui_agent.infrastructure.session_store import SessionStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a ComfyUI assistant. You help users create, manage, and debug ComfyUI workflows through natural language.

You have access to tools that let you:
- Check system status (GPU, VRAM, version)
- List available models (checkpoints, loras, VAE, etc.)
- Search and inspect node types and their parameters
- Submit workflows for execution
- Monitor the execution queue
- View execution history and outputs
- Interrupt running executions

When the user asks you to generate an image, you should:
1. First check what models are available
2. Build a workflow in ComfyUI's API format
3. Submit it for execution
4. Report the results

ComfyUI workflow API format is a dict of node_id -> {class_type, inputs}.
Example minimal txt2img workflow:
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
  "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a photo of a cat", "clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad quality", "clip": ["1", 1]}},
  "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
  "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
  "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "output"}}
}

Always verify node types exist before using them. Use comfyui_get_node_info to check."""


class AgentLoop:
    """Core agent loop: user input → LLM → tool calls → repeat → response.

    This is the heart of the agent. It implements the classic
    ReAct (Reason + Act) pattern:

    1. User sends a message
    2. LLM reasons about what to do
    3. If LLM wants to use a tool → execute it → feed result back → goto 2
    4. If LLM has a final answer → return it to user
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        session_store: SessionStore,
        event_bus: EventBus,
        max_iterations: int = 20,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.tools = {t.info().name: t for t in tools}
        self.tool_schemas = [
            ToolSchema(
                name=t.info().name,
                description=t.info().description,
                input_schema=t.info().parameters,
            )
            for t in tools
        ]
        self.session_store = session_store
        self.event_bus = event_bus
        self.state_machine = AgentStateMachine()
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self._cancel_flags: dict[str, bool] = {}

    async def run(self, session_id: str, user_input: str) -> str:
        """Run the agent loop for a user message.

        Args:
            session_id: Session identifier
            user_input: User's natural language input

        Returns:
            Agent's final text response
        """
        # Emit conversation start
        await self.event_bus.emit(Event(
            type=EventType.STATE_CONVERSATION_START,
            session_id=session_id,
        ))
        self.state_machine.process(Event(type=EventType.STATE_CONVERSATION_START))

        # Load history and append user message
        messages = await self.session_store.load_messages(session_id)
        messages.append({"role": "user", "content": user_input})

        await self.event_bus.emit(Event(
            type=EventType.MESSAGE_USER,
            session_id=session_id,
            data={"content": user_input},
        ))

        # Emit turn start
        turn_start = time.time()
        await self.event_bus.emit(Event(
            type=EventType.TURN_START,
            session_id=session_id,
        ))

        total_usage = {"input_tokens": 0, "output_tokens": 0}
        self._cancel_flags[session_id] = False

        try:
            for iteration in range(self.max_iterations):
                # Check cancellation
                if self._cancel_flags.get(session_id, False):
                    logger.info("Agent cancelled for session %s", session_id)
                    break

                logger.info("Iteration %d/%d for session %s", iteration + 1, self.max_iterations, session_id)

                # Emit thinking state
                self.state_machine.process(Event(type=EventType.STATE_THINKING))

                # Call LLM
                response = await self.llm.chat(
                    messages=messages,
                    tools=self.tool_schemas if self.tool_schemas else None,
                    system=self.system_prompt,
                )

                # Accumulate usage
                for k in total_usage:
                    total_usage[k] += response.usage.get(k, 0)

                # LLM wants to call tools
                if response.has_tool_calls():
                    # Add assistant message with tool calls to history
                    assistant_msg = self._build_assistant_message(response)
                    messages.append(assistant_msg)

                    await self.event_bus.emit(Event(
                        type=EventType.MESSAGE_ASSISTANT,
                        session_id=session_id,
                        data={"content": response.text, "tool_calls": len(response.tool_calls)},
                    ))

                    # Execute each tool
                    tool_results_content = []
                    for tc in response.tool_calls:
                        self.state_machine.process(Event(type=EventType.STATE_TOOL_PLANNED))
                        await self.event_bus.emit(Event(
                            type=EventType.STATE_TOOL_EXECUTING,
                            session_id=session_id,
                            data={"tool_name": tc.name, "tool_id": tc.id},
                        ))
                        self.state_machine.process(Event(type=EventType.STATE_TOOL_EXECUTING))

                        result = await self._execute_tool(tc.name, tc.input)

                        if result.is_error:
                            self.state_machine.process(Event(type=EventType.STATE_TOOL_FAILED))
                            await self.event_bus.emit(Event(
                                type=EventType.STATE_TOOL_FAILED,
                                session_id=session_id,
                                data={"tool_name": tc.name, "error": result.text},
                            ))
                        else:
                            self.state_machine.process(Event(type=EventType.STATE_TOOL_COMPLETED))
                            await self.event_bus.emit(Event(
                                type=EventType.STATE_TOOL_COMPLETED,
                                session_id=session_id,
                                data={"tool_name": tc.name},
                            ))

                        # Anthropic API requires each tool_result as a separate user message
                        tool_results_content.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result.text,
                            "is_error": result.is_error,
                        })

                        await self.event_bus.emit(Event(
                            type=EventType.MESSAGE_TOOL_RESULT,
                            session_id=session_id,
                            data={"tool_name": tc.name, "result": result.text[:500]},
                        ))

                    # Add tool results — Anthropic expects role=user with tool_result content blocks
                    messages.append({"role": "user", "content": tool_results_content})
                    continue

                # LLM has a final answer
                self.state_machine.process(Event(type=EventType.STATE_RESPONDING))
                messages.append({"role": "assistant", "content": response.text})

                await self.event_bus.emit(Event(
                    type=EventType.MESSAGE_ASSISTANT,
                    session_id=session_id,
                    data={"content": response.text, "tool_calls": 0},
                ))

                # Save and return
                await self.session_store.save_messages(session_id, messages)

                self.state_machine.process(Event(type=EventType.STATE_CONVERSATION_END))
                await self.event_bus.emit(Event(
                    type=EventType.STATE_CONVERSATION_END,
                    session_id=session_id,
                ))

                # Emit turn end with stats
                await self.event_bus.emit(Event(
                    type=EventType.TURN_END,
                    session_id=session_id,
                    data={
                        "duration": time.time() - turn_start,
                        "iterations": iteration + 1,
                        "usage": total_usage,
                    },
                ))

                return response.text

            # Max iterations reached
            logger.warning("Max iterations reached for session %s", session_id)
            final_text = "I've reached the maximum number of steps. Here's what I've done so far."
            messages.append({"role": "assistant", "content": final_text})
            await self.session_store.save_messages(session_id, messages)
            self.state_machine.reset()
            return final_text

        except Exception as e:
            logger.exception("Agent loop error for session %s", session_id)
            self.state_machine.process(Event(type=EventType.STATE_ERROR))
            await self.event_bus.emit(Event(
                type=EventType.STATE_ERROR,
                session_id=session_id,
                data={"error": str(e)},
            ))
            self.state_machine.reset()
            raise

        finally:
            self._cancel_flags.pop(session_id, None)

    def cancel(self, session_id: str) -> None:
        """Cancel a running agent loop."""
        self._cancel_flags[session_id] = True

    async def _execute_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool with error isolation and timeout."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {tool_name}")

        try:
            result = await asyncio.wait_for(tool.run(params), timeout=60.0)
            logger.info("Tool %s completed: %s", tool_name, "error" if result.is_error else "ok")
            return result
        except asyncio.TimeoutError:
            logger.warning("Tool %s timed out", tool_name)
            return ToolResult.error(f"Tool '{tool_name}' timed out after 60 seconds")
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return ToolResult.error(f"Tool '{tool_name}' failed: {e}")

    def _build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
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
