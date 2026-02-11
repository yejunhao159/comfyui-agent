"""Web fetch tool — retrieve and extract content from URLs.

Allows the agent to fetch web pages, documentation, and API responses.
Inspired by OpenCode's fetch tool, adapted for our async Tool ABC pattern.
"""

from __future__ import annotations

from typing import Any

from comfyui_agent.domain.ports import WebPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult, truncate_output


class WebFetchTool(Tool):
    """Fetch content from a URL and return extracted text."""

    def __init__(self, web: WebPort) -> None:
        self._web = web

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="web_fetch",
            description=(
                "Fetch and extract readable content from a URL.\n\n"
                "Use cases:\n"
                "- Read workflow JSON files from GitHub repos (raw.githubusercontent.com)\n"
                "- Read documentation pages, blog posts, or tutorials\n"
                "- Inspect ComfyUI workflow templates shared on community sites\n"
                "- Check model cards on HuggingFace or Civitai\n"
                "- Download text content the user links to\n\n"
                "Workflow research flow: web_search finds URLs → web_fetch reads the content "
                "→ study the workflow design → adapt it for the user's needs.\n\n"
                "Automatically extracts readable text from HTML (strips nav, ads, scripts). "
                "For non-HTML content (JSON, plain text, XML), returns raw content.\n\n"
                "Limits: max 5MB response, max 120s timeout, HTTP/HTTPS only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 30, max: 120)",
                    },
                },
                "required": ["url"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        url = params.get("url", "")
        if not url:
            return ToolResult.error("url parameter is required")

        if not url.startswith(("http://", "https://")):
            return ToolResult.error("URL must start with http:// or https://")

        timeout = min(params.get("timeout", 30), 120)

        try:
            result = await self._web.fetch_url(url, timeout=timeout)
        except Exception as e:
            return ToolResult.error(f"Failed to fetch URL: {e}")

        status = result.get("status_code", 0)
        if status != 200:
            return ToolResult.error(f"HTTP {status} for {url}")

        content = result.get("content", "")
        content_type = result.get("content_type", "")

        output = f"URL: {url}\nContent-Type: {content_type}\n\n{content}"
        return ToolResult.success(truncate_output(output))
