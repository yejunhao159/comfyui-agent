"""Discovery tools — search nodes, get details, validate workflows."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult

if TYPE_CHECKING:
    from comfyui_agent.knowledge.node_index import NodeIndex


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
