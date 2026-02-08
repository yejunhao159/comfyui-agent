"""Monitoring tools — system stats, models, queue, history, interrupt."""

from __future__ import annotations

import json
from typing import Any

from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult


class SystemStatsTool(Tool):
    """Get ComfyUI system stats."""

    def __init__(self, client: ComfyUIPort) -> None:
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

    def __init__(self, client: ComfyUIPort) -> None:
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

    def __init__(self, client: ComfyUIPort) -> None:
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

    def __init__(self, client: ComfyUIPort) -> None:
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

    def __init__(self, client: ComfyUIPort) -> None:
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
