"""Tests for web_fetch and web_search tools + WebClient helpers."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from comfyui_agent.domain.tools.web_fetch import WebFetchTool
from comfyui_agent.domain.tools.web_search import WebSearchTool
from comfyui_agent.infrastructure.clients.web_client import (
    _extract_text_from_html,
    _parse_ddg_html,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_web() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def fetch_tool(mock_web: AsyncMock) -> WebFetchTool:
    return WebFetchTool(web=mock_web)


@pytest.fixture
def search_tool(mock_web: AsyncMock) -> WebSearchTool:
    return WebSearchTool(web=mock_web)


# ---------------------------------------------------------------------------
# WebFetchTool tests
# ---------------------------------------------------------------------------

class TestWebFetchTool:
    def test_info_has_correct_name(self, fetch_tool: WebFetchTool) -> None:
        info = fetch_tool.info()
        assert info.name == "web_fetch"
        assert "url" in info.parameters["properties"]

    async def test_fetch_success(
        self, fetch_tool: WebFetchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.fetch_url.return_value = {
            "content": "Hello World",
            "content_type": "text/plain",
            "status_code": 200,
            "url": "https://example.com",
        }
        result = await fetch_tool.run({"url": "https://example.com"})
        assert not result.is_error
        assert "Hello World" in result.text
        mock_web.fetch_url.assert_called_once_with("https://example.com", timeout=30)

    async def test_fetch_empty_url(self, fetch_tool: WebFetchTool) -> None:
        result = await fetch_tool.run({"url": ""})
        assert result.is_error
        assert "required" in result.text

    async def test_fetch_invalid_protocol(self, fetch_tool: WebFetchTool) -> None:
        result = await fetch_tool.run({"url": "ftp://example.com"})
        assert result.is_error
        assert "http" in result.text.lower()

    async def test_fetch_non_200(
        self, fetch_tool: WebFetchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.fetch_url.return_value = {
            "content": "",
            "content_type": "text/html",
            "status_code": 404,
            "url": "https://example.com/missing",
        }
        result = await fetch_tool.run({"url": "https://example.com/missing"})
        assert result.is_error
        assert "404" in result.text

    async def test_fetch_timeout_capped(
        self, fetch_tool: WebFetchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.fetch_url.return_value = {
            "content": "ok",
            "content_type": "text/plain",
            "status_code": 200,
            "url": "https://example.com",
        }
        await fetch_tool.run({"url": "https://example.com", "timeout": 999})
        mock_web.fetch_url.assert_called_once_with("https://example.com", timeout=120)

    async def test_fetch_exception(
        self, fetch_tool: WebFetchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.fetch_url.side_effect = RuntimeError("connection refused")
        result = await fetch_tool.run({"url": "https://example.com"})
        assert result.is_error
        assert "connection refused" in result.text


# ---------------------------------------------------------------------------
# WebSearchTool tests
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    def test_info_has_correct_name(self, search_tool: WebSearchTool) -> None:
        info = search_tool.info()
        assert info.name == "web_search"
        assert "query" in info.parameters["properties"]

    async def test_search_success(
        self, search_tool: WebSearchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.search.return_value = [
            {"title": "Result 1", "url": "https://a.com", "snippet": "First"},
            {"title": "Result 2", "url": "https://b.com", "snippet": "Second"},
        ]
        result = await search_tool.run({"query": "comfyui tutorial"})
        assert not result.is_error
        assert "Result 1" in result.text
        assert "Result 2" in result.text
        assert "https://a.com" in result.text
        mock_web.search.assert_called_once_with("comfyui tutorial", max_results=5)

    async def test_search_empty_query(self, search_tool: WebSearchTool) -> None:
        result = await search_tool.run({"query": ""})
        assert result.is_error
        assert "required" in result.text

    async def test_search_no_results(
        self, search_tool: WebSearchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.search.return_value = []
        result = await search_tool.run({"query": "xyznonexistent"})
        assert not result.is_error
        assert "No results" in result.text

    async def test_search_max_results_capped(
        self, search_tool: WebSearchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.search.return_value = []
        await search_tool.run({"query": "test", "max_results": 50})
        mock_web.search.assert_called_once_with("test", max_results=10)

    async def test_search_exception(
        self, search_tool: WebSearchTool, mock_web: AsyncMock
    ) -> None:
        mock_web.search.side_effect = RuntimeError("API error")
        result = await search_tool.run({"query": "test"})
        assert result.is_error
        assert "API error" in result.text


# ---------------------------------------------------------------------------
# HTML extraction tests
# ---------------------------------------------------------------------------

class TestHTMLExtraction:
    def test_extract_text_strips_tags(self) -> None:
        html = "<html><body><p>Hello <b>World</b></p></body></html>"
        text = _extract_text_from_html(html)
        assert "Hello" in text
        assert "World" in text
        assert "<" not in text

    def test_extract_text_removes_scripts(self) -> None:
        html = "<html><script>alert('xss')</script><p>Content</p></html>"
        text = _extract_text_from_html(html)
        assert "Content" in text
        assert "alert" not in text

    def test_extract_text_removes_styles(self) -> None:
        html = "<html><style>.x{color:red}</style><p>Visible</p></html>"
        text = _extract_text_from_html(html)
        assert "Visible" in text
        assert "color" not in text

    def test_extract_text_decodes_entities(self) -> None:
        html = "<p>A &amp; B &lt; C</p>"
        text = _extract_text_from_html(html)
        assert "A & B < C" in text


class TestDDGParsing:
    def test_parse_ddg_empty(self) -> None:
        results = _parse_ddg_html("<html></html>", 5)
        assert results == []

    def test_parse_ddg_respects_max(self) -> None:
        # Build fake DDG-like HTML with multiple results
        items = ""
        for i in range(10):
            items += (
                f'<a class="result__a" href="https://example.com/{i}">Title {i}</a>'
                f'<td class="result__snippet">Snippet {i}</td>'
            )
        results = _parse_ddg_html(items, 3)
        assert len(results) == 3
        assert results[0]["title"] == "Title 0"
        assert results[2]["url"] == "https://example.com/2"


# ---------------------------------------------------------------------------
# Factory integration test
# ---------------------------------------------------------------------------

class TestFactoryIntegration:
    def test_create_all_tools_without_web(self) -> None:
        """Without web port, no web tools are created."""
        from unittest.mock import MagicMock
        from comfyui_agent.domain.tools.factory import create_all_tools

        client = MagicMock()
        node_index = MagicMock()
        tools = create_all_tools(client, node_index)
        names = [t.info().name for t in tools]
        assert "web_fetch" not in names
        assert "web_search" not in names

    def test_create_all_tools_with_web(self) -> None:
        """With web port, web tools are included."""
        from unittest.mock import MagicMock
        from comfyui_agent.domain.tools.factory import create_all_tools

        client = MagicMock()
        node_index = MagicMock()
        web = MagicMock()
        tools = create_all_tools(client, node_index, web=web)
        names = [t.info().name for t in tools]
        assert "web_fetch" in names
        assert "web_search" in names
