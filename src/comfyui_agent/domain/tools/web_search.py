"""Web search tool — search the internet for information.

Allows the agent to search the web when it needs external knowledge,
such as documentation, tutorials, model information, or troubleshooting.
"""

from __future__ import annotations

from typing import Any

from comfyui_agent.domain.ports import WebPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult


class WebSearchTool(Tool):
    """Search the web and return summarized results."""

    def __init__(self, web: WebPort) -> None:
        self._web = web

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="web_search",
            description=(
                "Search the web for ComfyUI ecosystem information and external resources.\n\n"
                "Best for:\n"
                "- Find open-source ComfyUI workflows and templates (search ComfyWorkflows.com, "
                "Civitai, GitHub)\n"
                "- Find model download URLs (HuggingFace, Civitai)\n"
                "- Discover custom node packages and their GitHub repos\n"
                "- Research workflow techniques, LoRA guides, or prompt engineering tips\n"
                "- Troubleshoot ComfyUI errors or compatibility issues\n\n"
                "Tips for effective ComfyUI searches:\n"
                "- Add 'comfyui workflow' to find shareable workflow templates\n"
                "- Add 'site:civitai.com' or 'site:github.com' to target specific platforms\n"
                "- Search for node pack names to find their repos and documentation\n\n"
                "Returns a ranked list of results with title, URL, and snippet. "
                "Use web_fetch to read the full content of any result URL — especially "
                "useful for reading workflow JSON files or node documentation from GitHub."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — include 'comfyui' for ecosystem-specific results",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        if not query:
            return ToolResult.error("query parameter is required")

        max_results = min(params.get("max_results", 5), 10)

        try:
            results = await self._web.search(query, max_results=max_results)
        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")

        if not results:
            return ToolResult.success("No results found.")

        lines: list[str] = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return ToolResult.success("\n".join(lines))
