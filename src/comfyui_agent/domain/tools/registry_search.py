"""Registry search tool — look up custom node packages on Comfy Registry.

Queries api.comfy.org to find custom node packages by ID, returning
metadata like description, downloads, stars, repository URL, and
available versions. Complements the local NodeIndex which only knows
about already-installed nodes.
"""

from __future__ import annotations

from typing import Any

from comfyui_agent.domain.ports import WebPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult


class RegistrySearchTool(Tool):
    """Look up a custom node package on the Comfy Registry (api.comfy.org)."""

    def __init__(self, web: WebPort) -> None:
        self._web = web

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfy_registry",
            description=(
                "Look up a custom node package on the official Comfy Registry "
                "(api.comfy.org) by its package ID.\n\n"
                "Use cases:\n"
                "- Check if a custom node pack exists before recommending install_custom_node\n"
                "- Get the GitHub repository URL for a node pack\n"
                "- See download count and stars to gauge popularity\n"
                "- Find the package description and available versions\n\n"
                "The node_id is the registry package name (e.g. 'comfyui-impact-pack', "
                "'comfyui-rembg', 'rgthree-comfy'). This is different from the node "
                "class_type used in workflows — use web_search to discover package names "
                "if you only know the node class_type.\n\n"
                "Returns: name, description, downloads, github_stars, repository URL, "
                "latest version, license, and tags."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The registry package ID (e.g. 'comfyui-impact-pack')",
                    },
                },
                "required": ["node_id"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        node_id = params.get("node_id", "").strip()
        if not node_id:
            return ToolResult.error("node_id parameter is required")

        try:
            data = await self._web.search_registry(node_id)
        except Exception as e:
            return ToolResult.error(f"Registry lookup failed: {e}")

        if data is None:
            return ToolResult.success(
                f"Package '{node_id}' not found in the Comfy Registry. "
                "Try web_search to find the correct package name."
            )

        # Format a concise summary
        lines: list[str] = [
            f"Registry: {data.get('id', node_id)}",
            f"Name: {data.get('name', 'N/A')}",
            f"Description: {data.get('description', 'N/A')}",
            f"Downloads: {data.get('downloads', 0)}",
            f"GitHub Stars: {data.get('github_stars', 0)}",
            f"Repository: {data.get('repository', 'N/A')}",
            f"License: {data.get('license', 'N/A')}",
            f"Status: {data.get('status', 'N/A')}",
        ]

        latest = data.get("latest_version")
        if isinstance(latest, dict):
            lines.append(f"Latest Version: {latest.get('version', 'N/A')}")
            deps = latest.get("dependencies", [])
            if deps:
                lines.append(f"Dependencies: {', '.join(deps[:10])}")

        tags = data.get("tags", [])
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")

        return ToolResult.success("\n".join(lines))
