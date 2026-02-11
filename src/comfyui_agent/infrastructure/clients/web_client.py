"""Web client — implements WebPort for HTTP fetching and web search.

Provides two capabilities:
1. fetch_url: GET any URL, extract readable text from HTML
2. search: Web search via Tavily API (if configured) or DuckDuckGo HTML fallback

Uses aiohttp for all HTTP operations. HTML text extraction uses a lightweight
approach with regex-based tag stripping (no heavy dependencies like BeautifulSoup).
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any
from urllib.parse import quote_plus

import aiohttp

logger = logging.getLogger(__name__)

_MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5MB
_USER_AGENT = "comfyui-agent/1.0"


class WebClient:
    """WebPort implementation using aiohttp.

    Supports Tavily API for search (recommended, needs API key),
    with DuckDuckGo HTML scraping as zero-config fallback.
    """

    def __init__(
        self,
        tavily_api_key: str = "",
        timeout: int = 30,
    ) -> None:
        self._tavily_api_key = tavily_api_key
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": _USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def fetch_url(self, url: str, timeout: int = 30) -> dict[str, Any]:
        """Fetch content from a URL, extracting text from HTML."""
        session = await self._get_session()
        effective_timeout = aiohttp.ClientTimeout(total=min(timeout, 120))

        async with session.get(url, timeout=effective_timeout) as resp:
            status = resp.status
            content_type = resp.content_type or ""
            raw = await resp.content.read(_MAX_RESPONSE_SIZE)
            text = raw.decode("utf-8", errors="replace")

        # Extract readable text from HTML
        if "html" in content_type:
            text = _extract_text_from_html(text)

        return {
            "content": text,
            "content_type": content_type,
            "status_code": status,
            "url": str(url),
        }

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search the web. Uses Tavily if configured, else DuckDuckGo fallback."""
        if self._tavily_api_key:
            return await self._search_tavily(query, max_results)
        return await self._search_ddg(query, max_results)

    async def _search_tavily(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search via Tavily API (https://tavily.com)."""
        session = await self._get_session()
        payload = {
            "api_key": self._tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
        }
        async with session.post(
            "https://api.tavily.com/search", json=payload
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Tavily API error {resp.status}: {body[:200]}")
            data = await resp.json()

        results: list[dict[str, Any]] = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        return results

    async def _search_ddg(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Fallback search via DuckDuckGo HTML (no API key needed)."""
        session = await self._get_session()
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"DuckDuckGo returned HTTP {resp.status}")
            text = await resp.text()

        return _parse_ddg_html(text, max_results)

    async def search_registry(self, node_id: str) -> dict[str, Any] | None:
        """Look up a custom node package on the Comfy Registry."""
        session = await self._get_session()
        url = f"https://api.comfy.org/nodes/{quote_plus(node_id)}"
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    logger.warning("Registry API returned %d for %s", resp.status, node_id)
                    return None
                return await resp.json()  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("Registry lookup failed for %s: %s", node_id, exc)
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ---------------------------------------------------------------------------
# HTML processing helpers (lightweight, no external deps beyond stdlib)
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\n{3,}")


def _extract_text_from_html(raw_html: str) -> str:
    """Extract readable text from HTML, stripping tags and boilerplate."""
    # Remove script/style/noscript blocks
    text = _SCRIPT_STYLE_RE.sub("", raw_html)
    # Strip all remaining tags
    text = _TAG_RE.sub("\n", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Collapse excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


_DDG_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
    r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
    re.DOTALL | re.IGNORECASE,
)


def _parse_ddg_html(raw_html: str, max_results: int) -> list[dict[str, Any]]:
    """Parse DuckDuckGo HTML search results page."""
    results: list[dict[str, Any]] = []
    for match in _DDG_RESULT_RE.finditer(raw_html):
        if len(results) >= max_results:
            break
        href = html.unescape(match.group(1))
        title = _TAG_RE.sub("", html.unescape(match.group(2))).strip()
        snippet = _TAG_RE.sub("", html.unescape(match.group(3))).strip()
        if href and title:
            results.append({"title": title, "url": href, "snippet": snippet})
    return results
