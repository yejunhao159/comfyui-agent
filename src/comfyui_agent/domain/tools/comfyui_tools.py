"""ComfyUI tools for the agent.

Each tool wraps one or more ComfyUI API calls and exposes them
to the LLM via the Tool interface.
"""

from __future__ import annotations

import json
from typing import Any

from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult
from comfyui_agent.infrastructure.comfyui_client import ComfyUIClient


class SystemStatsTool(Tool):
    """Get ComfyUI system stats (GPU, VRAM, version)."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_system_stats",
            description="Get ComfyUI system statistics including GPU info, VRAM usage, and version.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            stats = await self.client.get_system_stats()
            return ToolResult.success(json.dumps(stats, indent=2))
        except Exception as e:
            return ToolResult.error(f"Failed to get system stats: {e}")


class ListModelsTool(Tool):
    """List available models (checkpoints, loras, vae, etc.)."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_list_models",
            description=(
                "List available models in ComfyUI. "
                "Folder can be: checkpoints, loras, vae, controlnet, upscale_models, embeddings, clip, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Model folder to list (default: checkpoints)",
                        "default": "checkpoints",
                    }
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        folder = params.get("folder", "checkpoints")
        try:
            models = await self.client.list_models(folder)
            if not models:
                return ToolResult.success(f"No models found in '{folder}' folder.")
            text = f"Models in '{folder}' ({len(models)}):\n"
            for m in models:
                text += f"  - {m}\n"
            return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to list models: {e}")


class GetNodeInfoTool(Tool):
    """Get information about ComfyUI nodes."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_node_info",
            description=(
                "Get detailed information about a ComfyUI node type, including its inputs, outputs, and parameters. "
                "If no node_class is specified, returns a summary of all available node types."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_class": {
                        "type": "string",
                        "description": "Specific node class name to get info for (e.g., 'KSampler', 'CheckpointLoaderSimple'). Leave empty to list all.",
                    }
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        node_class = params.get("node_class")
        try:
            if node_class:
                info = await self.client.get_object_info(node_class)
                if node_class in info:
                    return ToolResult.success(json.dumps(info[node_class], indent=2))
                return ToolResult.error(f"Node class '{node_class}' not found.")
            else:
                all_info = await self.client.get_object_info()
                names = sorted(all_info.keys())
                text = f"Available node types ({len(names)}):\n"
                for name in names:
                    node = all_info[name]
                    category = node.get("category", "unknown")
                    display = node.get("display_name", name)
                    text += f"  - {name} [{category}] ({display})\n"
                return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get node info: {e}")


class QueuePromptTool(Tool):
    """Submit a workflow to ComfyUI for execution."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_queue_prompt",
            description=(
                "Submit a ComfyUI workflow (prompt) for execution. "
                "The workflow must be in ComfyUI's API format (dict of node_id -> node_config). "
                "Returns the prompt_id for tracking."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "ComfyUI workflow in API format (node_id -> {class_type, inputs})",
                    }
                },
                "required": ["workflow"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        workflow = params.get("workflow")
        if not workflow:
            return ToolResult.error("workflow parameter is required")
        try:
            result = await self.client.queue_prompt(workflow)
            prompt_id = result.get("prompt_id", "unknown")
            return ToolResult.success(
                f"Workflow submitted successfully. prompt_id: {prompt_id}",
                data={"prompt_id": prompt_id},
            )
        except Exception as e:
            return ToolResult.error(f"Failed to queue prompt: {e}")


class GetQueueTool(Tool):
    """Get the current ComfyUI queue status."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_queue",
            description="Get the current ComfyUI execution queue status (running and pending items).",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            queue = await self.client.get_queue()
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            text = f"Queue status:\n  Running: {len(running)}\n  Pending: {len(pending)}\n"
            if running:
                for item in running:
                    text += f"  [running] prompt_id: {item[1]}\n"
            if pending:
                for item in pending[:10]:
                    text += f"  [pending] prompt_id: {item[1]}\n"
                if len(pending) > 10:
                    text += f"  ... and {len(pending) - 10} more\n"
            return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get queue: {e}")


class GetHistoryTool(Tool):
    """Get execution history and outputs."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_history",
            description=(
                "Get execution history from ComfyUI. "
                "If prompt_id is given, returns details for that specific execution including outputs. "
                "Otherwise returns recent history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt_id": {
                        "type": "string",
                        "description": "Specific prompt_id to get history for",
                    }
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        prompt_id = params.get("prompt_id")
        try:
            history = await self.client.get_history(prompt_id)
            if prompt_id and prompt_id in history:
                entry = history[prompt_id]
                outputs = entry.get("outputs", {})
                status = entry.get("status", {})
                text = f"Execution {prompt_id}:\n"
                text += f"  Status: {status.get('status_str', 'unknown')}\n"
                if outputs:
                    text += "  Outputs:\n"
                    for node_id, output in outputs.items():
                        if "images" in output:
                            for img in output["images"]:
                                url = self.client.get_image_url(
                                    img["filename"], img.get("subfolder", ""), img.get("type", "output")
                                )
                                text += f"    Node {node_id}: {url}\n"
                return ToolResult.success(text)
            else:
                entries = list(history.keys())[-10:]
                text = f"Recent executions ({len(history)} total, showing last {len(entries)}):\n"
                for pid in entries:
                    status = history[pid].get("status", {}).get("status_str", "unknown")
                    text += f"  - {pid} [{status}]\n"
                return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get history: {e}")


class InterruptTool(Tool):
    """Interrupt the currently running execution."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_interrupt",
            description="Interrupt the currently running ComfyUI execution.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            await self.client.interrupt()
            return ToolResult.success("Execution interrupted.")
        except Exception as e:
            return ToolResult.error(f"Failed to interrupt: {e}")


def create_all_tools(client: ComfyUIClient) -> list[Tool]:
    """Create all ComfyUI tools with the given client."""
    return [
        SystemStatsTool(client),
        ListModelsTool(client),
        GetNodeInfoTool(client),
        QueuePromptTool(client),
        GetQueueTool(client),
        GetHistoryTool(client),
        InterruptTool(client),
    ]
