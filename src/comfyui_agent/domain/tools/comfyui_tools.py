"""ComfyUI tools for the agent.

Tools are organized by purpose:
- Discovery: search_nodes, get_node_detail, list_categories
- Execution: queue_prompt, validate_workflow
- Monitoring: system_stats, list_models, get_queue, get_history, interrupt
"""

from __future__ import annotations

import json
from typing import Any

from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult
from comfyui_agent.infrastructure.comfyui_client import ComfyUIClient
from comfyui_agent.knowledge.node_index import NodeIndex

MAX_TOOL_OUTPUT = 15000  # chars — OpenCode uses 30K, we use 15K for LLM efficiency


def truncate_output(text: str, max_len: int = MAX_TOOL_OUTPUT) -> str:
    """Truncate tool output, keeping first and last portions."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    mid_lines = text[half:-half].count("\n")
    return f"{text[:half]}\n\n... [{mid_lines} lines truncated] ...\n\n{text[-half:]}"


# ============================================================
# Discovery Tools
# ============================================================


class SearchNodesTool(Tool):
    """Search ComfyUI nodes by keyword, or browse by category."""

    def __init__(self, node_index: NodeIndex) -> None:
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_search_nodes",
            description=(
                "Search for ComfyUI node types by keyword or browse by category. "
                "Use this to find the right nodes for a task. "
                "Examples: search_nodes(query='upscale'), search_nodes(category='loaders'), "
                "search_nodes() to list all categories."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword (e.g., 'sampler', 'upscale', 'controlnet')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Browse a specific category (e.g., 'loaders', 'sampling')",
                    },
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query")
        category = params.get("category")

        if query:
            return ToolResult.success(self.index.search(query))
        elif category:
            return ToolResult.success(self.index.list_category(category))
        else:
            return ToolResult.success(self.index.list_categories())


class GetNodeDetailTool(Tool):
    """Get condensed detail for a specific node type."""

    def __init__(self, node_index: NodeIndex) -> None:
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_node_detail",
            description=(
                "Get detailed information about a specific ComfyUI node type: "
                "its inputs (required/optional with types), outputs, and description. "
                "Use search_nodes first to find the right node class name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_class": {
                        "type": "string",
                        "description": "Exact node class name (e.g., 'KSampler', 'CheckpointLoaderSimple')",
                    },
                },
                "required": ["node_class"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        node_class = params.get("node_class", "")
        if not node_class:
            return ToolResult.error("node_class is required")
        return ToolResult.success(self.index.get_detail(node_class))


class ValidateWorkflowTool(Tool):
    """Validate a workflow before submitting it."""

    def __init__(self, node_index: NodeIndex) -> None:
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_validate_workflow",
            description=(
                "Validate a ComfyUI workflow before submitting. "
                "Checks that all node types exist, required inputs are provided, "
                "and connections are valid. Always validate before queue_prompt."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "ComfyUI workflow in API format (node_id -> {class_type, inputs})",
                    },
                },
                "required": ["workflow"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        workflow = params.get("workflow")
        if not workflow:
            return ToolResult.error("workflow is required")
        return ToolResult.success(self.index.validate_workflow(workflow))


# ============================================================
# Execution Tools
# ============================================================


class QueuePromptTool(Tool):
    """Submit a workflow to ComfyUI for execution."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_queue_prompt",
            description=(
                "Submit a ComfyUI workflow for execution. "
                "The workflow must be in ComfyUI's API format (dict of node_id -> {class_type, inputs}). "
                "Always use comfyui_validate_workflow first to check for errors."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "ComfyUI workflow in API format",
                    },
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
                f"Workflow submitted. prompt_id: {prompt_id}",
                data={"prompt_id": prompt_id},
            )
        except Exception as e:
            return ToolResult.error(f"Failed to queue prompt: {e}")


# ============================================================
# Monitoring Tools
# ============================================================


class SystemStatsTool(Tool):
    """Get ComfyUI system stats."""

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
    """List available models."""

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
                    },
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        folder = params.get("folder", "checkpoints")
        try:
            models = await self.client.list_models(folder)
            if not models:
                return ToolResult.success(f"No models found in '{folder}'.")
            text = f"Models in '{folder}' ({len(models)}):\n"
            for m in models:
                text += f"  - {m}\n"
            return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to list models: {e}")


class GetQueueTool(Tool):
    """Get queue status."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_queue",
            description="Get the current ComfyUI execution queue status.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            queue = await self.client.get_queue()
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            text = f"Queue: {len(running)} running, {len(pending)} pending\n"
            for item in running:
                text += f"  [running] {item[1]}\n"
            for item in pending[:10]:
                text += f"  [pending] {item[1]}\n"
            if len(pending) > 10:
                text += f"  ... and {len(pending) - 10} more\n"
            return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get queue: {e}")


class GetHistoryTool(Tool):
    """Get execution history."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_history",
            description=(
                "Get execution history. If prompt_id is given, returns details "
                "including output image URLs. Otherwise returns recent history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt_id": {
                        "type": "string",
                        "description": "Specific prompt_id to get details for",
                    },
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
                                    img["filename"],
                                    img.get("subfolder", ""),
                                    img.get("type", "output"),
                                )
                                text += f"    Node {node_id}: {url}\n"
                return ToolResult.success(text)
            else:
                entries = list(history.keys())[-10:]
                text = f"Recent executions ({len(history)} total, showing {len(entries)}):\n"
                for pid in entries:
                    st = history[pid].get("status", {}).get("status_str", "unknown")
                    text += f"  - {pid} [{st}]\n"
                return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get history: {e}")


class InterruptTool(Tool):
    """Interrupt running execution."""

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


# ============================================================
# Factory
# ============================================================


def create_all_tools(client: ComfyUIClient, node_index: NodeIndex) -> list[Tool]:
    """Create all ComfyUI tools."""
    return [
        # Discovery
        SearchNodesTool(node_index),
        GetNodeDetailTool(node_index),
        ValidateWorkflowTool(node_index),
        # Execution
        QueuePromptTool(client),
        # Monitoring
        SystemStatsTool(client),
        ListModelsTool(client),
        GetQueueTool(client),
        GetHistoryTool(client),
        InterruptTool(client),
    ]
