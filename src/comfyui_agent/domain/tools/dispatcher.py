"""ComfyUI dispatcher — single tool that routes to all operations."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult
from comfyui_agent.domain.tools.factory import create_internal_tools

if TYPE_CHECKING:
    from comfyui_agent.knowledge.node_index import NodeIndex

_ACTION_NAMES = [
    "search_nodes",
    "get_node_detail",
    "validate_workflow",
    "queue_prompt",
    "system_stats",
    "list_models",
    "get_queue",
    "get_history",
    "interrupt",
    "upload_image",
    "download_model",
    "install_custom_node",
    "free_memory",
    "get_folder_paths",
    "refresh_index",
]

_TOOL_DESCRIPTION = """\
Execute ComfyUI operations.

## Discovery
- search_nodes(query?, category?) — Search nodes by keyword or browse categories.
- get_node_detail(node_class) — Get inputs/outputs/description for a specific node type.
- validate_workflow(workflow) — Validate workflow before submitting.

## Execution
- queue_prompt(workflow) — Submit workflow for execution. Always validate first.

## Monitoring
- system_stats() — GPU/VRAM status and version info.
- list_models(folder?) — List available models.
- get_queue() — Current execution queue status.
- get_history(prompt_id?) — Execution history.
- interrupt() — Stop current execution.

## Management
- upload_image(url?, filepath?, filename?) — Upload image for img2img/ControlNet.
- download_model(url, folder, filename?) — Download model from URL.
- install_custom_node(git_url) — Install custom node from git repo.
- free_memory(unload_models?, free_memory?) — Free GPU VRAM and RAM.
- get_folder_paths() — Show where models and outputs are stored.
- refresh_index() — Rebuild node index after installing custom nodes."""


class ComfyUIDispatcher(Tool):
    """Single dispatcher tool that routes to all ComfyUI operations.

    Instead of exposing 15 separate tools, this exposes one tool with
    an action+params pattern, reducing per-request token overhead.
    """

    def __init__(self, client: ComfyUIPort, node_index: NodeIndex) -> None:
        self._tools: dict[str, Tool] = {}
        for t in create_internal_tools(client, node_index):
            name = t.info().name.replace("comfyui_", "")
            self._tools[name] = t

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": _ACTION_NAMES,
                        "description": "The operation to perform",
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters",
                    },
                },
                "required": ["action"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        action_params = params.get("params", {})

        tool = self._tools.get(action)
        if not tool:
            return ToolResult.error(
                f"Unknown action: '{action}'. Available: {list(self._tools.keys())}"
            )

        return await tool.run(action_params)
