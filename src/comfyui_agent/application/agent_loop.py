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

## comfyui Tool — Available Actions

Use the `comfyui` tool with {"action": "<name>", "params": {...}} format.

### Discovery
- search_nodes(query?, category?) — Search nodes by keyword or browse categories. No params = list all categories.
- get_node_detail(node_class) — Get inputs/outputs/description for a specific node type.
- validate_workflow(workflow) — Validate workflow before submitting. workflow = API format dict.

### Execution
- queue_prompt(workflow) — Submit workflow for execution. Always validate first.

### Monitoring
- system_stats() — GPU/VRAM status and version info.
- list_models(folder?) — List available models. folder: checkpoints, loras, vae, controlnet, upscale_models, embeddings, clip. Default: checkpoints.
- get_queue() — Current execution queue status.
- get_history(prompt_id?) — Execution history. With prompt_id: details + output image URLs.
- interrupt() — Stop current execution.

### Management
- upload_image(url?, filepath?, filename?) — Upload image for img2img/ControlNet. Provide url or filepath.
- download_model(url, folder, filename?) — Download model from URL (HuggingFace, Civitai). Use get_folder_paths first.
- install_custom_node(git_url) — Install custom node from git repo. Restart ComfyUI after.
- free_memory(unload_models?, free_memory?) — Free GPU VRAM and RAM. Both default to true.
- get_folder_paths() — Show where models and outputs are stored.

## Workflow Building Process

1. Search for relevant nodes: comfyui(action="search_nodes", params={"query": "..."})
2. Get node details: comfyui(action="get_node_detail", params={"node_class": "..."})
3. Build workflow in API format
4. Validate: comfyui(action="validate_workflow", params={"workflow": {...}})
5. Submit: comfyui(action="queue_prompt", params={"workflow": {...}})
6. Check results: comfyui(action="get_history", params={"prompt_id": "..."})

## Model Management

- Use list_models to check available models first
- If no models are available, use download_model to download from HuggingFace or Civitai
- Use get_folder_paths to see where models should be stored
- Use free_memory before loading large models if VRAM is low

## ComfyUI Workflow API Format

A workflow is a dict of node_id -> {class_type, inputs}.
Node connections use [source_node_id, output_index] format.

Example txt2img:
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
  "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a photo of a cat", "clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad quality", "clip": ["1", 1]}},
  "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
  "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
  "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "output"}}
}

## Important Rules

- Always search_nodes and get_node_detail before using a node type
- Always validate_workflow before queue_prompt
- Use the actual model names from list_models, not guessed names
- Node connections: [node_id_string, output_index_int]"""


MAX_TOOL_RESULT_CHARS = 15000


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
        """Execute a tool with error isolation, timeout, and output truncation."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {tool_name}")

        try:
            result = await asyncio.wait_for(tool.run(params), timeout=60.0)
            # Truncate large outputs (OpenCode pattern: keep first/last halves)
            if len(result.text) > MAX_TOOL_RESULT_CHARS:
                half = MAX_TOOL_RESULT_CHARS // 2
                mid_lines = result.text[half:-half].count("\n")
                result.text = (
                    f"{result.text[:half]}\n\n"
                    f"... [{mid_lines} lines truncated] ...\n\n"
                    f"{result.text[-half:]}"
                )
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
