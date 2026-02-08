"""mitmproxy addon for debugging ComfyUI Agent WebSocket traffic.

Usage:
    mitmproxy -s scripts/mitm_agent.py -p 8080 --mode reverse:http://127.0.0.1:5200

Then point the plugin's WebSocket to ws://127.0.0.1:8080/api/chat/ws

Or use as transparent proxy:
    mitmproxy -s scripts/mitm_agent.py -p 8080

Features:
- Color-coded event logging (tool=yellow, stream=dim, state=green, error=red)
- WebSocket message filtering (hide noisy stream.text_delta by default)
- Request/response timing
- JSON pretty-print for readability
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from mitmproxy import ctx, http, websocket

# ── Configuration ──────────────────────────────────────────────────
# Set to True to also log stream.text_delta events (very noisy)
SHOW_TEXT_DELTA = False
# Set to True to log stream.tool_call_delta events
SHOW_TOOL_DELTA = False
# Set to True to log full response body for HTTP requests
SHOW_HTTP_BODY = False
# Max chars to display for result/content fields
MAX_CONTENT_LEN = 300


# ── ANSI colors ────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


EVENT_COLORS = {
    "state.conversation_start": C.GREEN + C.BOLD,
    "state.conversation_end": C.GREEN + C.BOLD,
    "state.thinking": C.GREEN,
    "state.responding": C.GREEN,
    "state.tool_executing": C.YELLOW + C.BOLD,
    "state.tool_completed": C.YELLOW,
    "state.tool_failed": C.RED + C.BOLD,
    "state.error": C.RED + C.BOLD,
    "message.user": C.CYAN,
    "message.assistant": C.BLUE,
    "message.tool_result": C.YELLOW,
    "turn.start": C.MAGENTA,
    "turn.end": C.MAGENTA + C.BOLD,
    "stream.text_delta": C.DIM,
    "stream.tool_call_start": C.YELLOW,
    "stream.tool_call_delta": C.DIM,
    "stream.message_stop": C.DIM,
    "workflow.submitted": C.CYAN + C.BOLD,
}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _truncate(s: str, max_len: int = MAX_CONTENT_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... ({len(s)} chars)"


def _format_event(data: dict) -> str:
    """Format a server event for display."""
    msg_type = data.get("type", "")
    event_type = data.get("event_type", "")
    et = event_type or msg_type
    color = EVENT_COLORS.get(et, C.WHITE)

    parts = [f"{color}[{et}]{C.RESET}"]

    payload = data.get("data", {})
    if isinstance(payload, dict):
        # Show key fields inline
        for key in ("tool_name", "tool_id", "content", "result", "error",
                     "duration", "iterations", "usage", "action"):
            if key in payload:
                val = payload[key]
                if isinstance(val, str) and len(val) > 80:
                    val = _truncate(val, 80)
                parts.append(f"{C.DIM}{key}={C.RESET}{val}")

    # For response/error top-level messages
    if msg_type == "response":
        content = data.get("content", "")
        parts.append(_truncate(str(content), 100))
    elif msg_type == "error":
        parts.append(f"{C.RED}{data.get('error', '')}{C.RESET}")
    elif msg_type == "session_created":
        parts.append(f"session={data.get('session_id', '')}")

    return " ".join(parts)


def _format_client_msg(data: dict) -> str:
    """Format a client WebSocket message."""
    msg_type = data.get("type", "")
    if msg_type == "chat":
        msg = data.get("message", "")
        return f"{C.CYAN + C.BOLD}[chat]{C.RESET} {_truncate(msg, 120)}"
    elif msg_type == "cancel":
        return f"{C.RED}[cancel]{C.RESET} session={data.get('session_id', '')}"
    elif msg_type == "ping":
        return f"{C.DIM}[ping]{C.RESET}"
    return f"[{msg_type}] {json.dumps(data, ensure_ascii=False)[:100]}"


# ── Addon class ────────────────────────────────────────────────────
class AgentDebugAddon:
    def __init__(self):
        self._http_timers: dict[str, float] = {}

    def websocket_message(self, flow: http.HTTPFlow):
        assert flow.websocket is not None
        msg = flow.websocket.messages[-1]

        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            ctx.log.info(f"{_ts()} [ws] (non-JSON) {msg.content[:100]}")
            return

        # Filter noisy events
        et = data.get("event_type", "")
        if et == "stream.text_delta" and not SHOW_TEXT_DELTA:
            return
        if et == "stream.tool_call_delta" and not SHOW_TOOL_DELTA:
            return

        direction = "◀ SERVER" if msg.from_server else "▶ CLIENT"
        if msg.from_server:
            formatted = _format_event(data)
        else:
            formatted = _format_client_msg(data)

        ctx.log.info(f"{_ts()} {C.DIM}{direction}{C.RESET} {formatted}")

    def request(self, flow: http.HTTPFlow):
        self._http_timers[flow.id] = time.time()
        method = flow.request.method
        path = flow.request.path
        ctx.log.info(
            f"{_ts()} {C.BLUE}→ {method} {path}{C.RESET}"
        )
        if SHOW_HTTP_BODY and flow.request.content:
            try:
                body = json.loads(flow.request.content)
                ctx.log.info(f"  body: {json.dumps(body, ensure_ascii=False)[:200]}")
            except (json.JSONDecodeError, TypeError):
                pass

    def response(self, flow: http.HTTPFlow):
        elapsed = time.time() - self._http_timers.pop(flow.id, time.time())
        status = flow.response.status_code if flow.response else "?"
        path = flow.request.path
        color = C.GREEN if str(status).startswith("2") else C.RED
        ctx.log.info(
            f"{_ts()} {color}← {status} {path}{C.RESET} "
            f"{C.DIM}({elapsed*1000:.0f}ms){C.RESET}"
        )
        if SHOW_HTTP_BODY and flow.response and flow.response.content:
            try:
                body = json.loads(flow.response.content)
                ctx.log.info(
                    f"  body: {json.dumps(body, ensure_ascii=False)[:200]}"
                )
            except (json.JSONDecodeError, TypeError):
                pass


addons = [AgentDebugAddon()]
