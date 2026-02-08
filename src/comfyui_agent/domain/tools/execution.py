"""Execution tools — submit workflows to ComfyUI."""

from __future__ import annotations

from typing import Any

from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult


class QueuePromptTool(Tool):
    """Submit a workflow to ComfyUI for execution."""

    def __init__(self, client: ComfyUIPort) -> None:
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
