"""Quick test: send a message through the proxy and print events.

Usage:
    # Terminal 1: start mitmweb
    mitmweb -s scripts/mitm_agent.py -p 9080 --set web_port=9081 \
            --mode reverse:http://127.0.0.1:5200

    # Terminal 2: run this test
    python scripts/test_via_proxy.py [port] [message]

    # Default: port=9080, message="列出可用的 checkpoint 模型"
"""

from __future__ import annotations

import asyncio
import json
import sys

import aiohttp

PROXY_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9080
MESSAGE = sys.argv[2] if len(sys.argv) > 2 else "列出可用的 checkpoint 模型"

# ANSI colors
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"

COLOR_MAP = {
    "state.conversation_start": GREEN + BOLD,
    "state.conversation_end": GREEN + BOLD,
    "state.thinking": GREEN,
    "state.tool_executing": YELLOW + BOLD,
    "state.tool_completed": YELLOW,
    "state.tool_failed": RED + BOLD,
    "message.tool_result": YELLOW,
    "message.user": CYAN,
    "message.assistant": CYAN + DIM,
    "turn.start": MAGENTA,
    "turn.end": MAGENTA + BOLD,
    "stream.text_delta": DIM,
    "stream.tool_call_start": YELLOW + DIM,
    "stream.tool_call_delta": DIM,
    "stream.message_stop": DIM,
}

SKIP = {"stream.text_delta", "stream.tool_call_delta"}


async def main():
    url = f"ws://127.0.0.1:{PROXY_PORT}/api/chat/ws"
    print(f"{BOLD}Connecting to {url}{RESET}")
    print(f"{BOLD}Message: {MESSAGE}{RESET}\n")

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            # Handshake
            await ws.send_json({"type": "ping"})
            await ws.receive_json()
            print(f"{GREEN}✓ Connected{RESET}\n")

            # Send message
            await ws.send_json({"type": "chat", "message": MESSAGE})

            # Collect events
            text_chars = 0
            tool_delta_chars = 0

            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=120)
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break

                    data = json.loads(msg.data)
                    et = data.get("event_type", data.get("type", ""))

                    # Count but skip noisy events
                    if et == "stream.text_delta":
                        text_chars += len(data.get("data", {}).get("text", ""))
                        continue
                    if et == "stream.tool_call_delta":
                        tool_delta_chars += len(
                            data.get("data", {}).get("partial_json", "")
                        )
                        continue

                    color = COLOR_MAP.get(et, "")
                    payload = data.get("data", {})

                    # Build display line
                    line = f"{color}[{et}]{RESET}"

                    if isinstance(payload, dict):
                        parts = []
                        for k in (
                            "tool_name", "tool_id", "content", "result",
                            "error", "duration", "iterations", "usage",
                        ):
                            if k in payload:
                                v = payload[k]
                                if isinstance(v, str) and len(v) > 100:
                                    v = v[:100] + "..."
                                parts.append(f"{DIM}{k}={RESET}{v}")
                        if parts:
                            line += " " + " ".join(parts)

                    if data.get("type") == "response":
                        content = str(data.get("content", ""))
                        line += f" {content[:120]}..."
                    elif data.get("type") == "session_created":
                        line += f" session={data.get('session_id', '')}"

                    print(line)

                    if data.get("type") in ("response", "error"):
                        break

                except asyncio.TimeoutError:
                    print(f"{RED}TIMEOUT{RESET}")
                    break

            # Summary
            print(f"\n{BOLD}── Summary ──{RESET}")
            print(f"  stream.text_delta: {text_chars} chars total")
            print(f"  stream.tool_call_delta: {tool_delta_chars} chars total")


if __name__ == "__main__":
    asyncio.run(main())
