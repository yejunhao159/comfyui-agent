"""Web API server for the ComfyUI Agent.

Provides:
- POST /api/chat — Send a message, get streamed response via SSE
- GET  /api/chat/ws — WebSocket for bidirectional streaming
- GET  /api/sessions — List sessions
- POST /api/sessions — Create session
- DELETE /api/sessions/{id} — Delete session
- GET  /api/health — Health check (agent + ComfyUI status)
- GET  / — Serve the web chat UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web
import aiohttp_cors

from comfyui_agent.application.agent_loop import AgentLoop
from comfyui_agent.application.context_manager import ContextManager
from comfyui_agent.application.message_converter import api_messages_to_chat_items
from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.domain.tools.factory import create_all_tools
from comfyui_agent.infrastructure.clients.comfyui_client import ComfyUIClient
from comfyui_agent.infrastructure.config import AppConfig
from comfyui_agent.infrastructure.event_bus import EventBus
from comfyui_agent.infrastructure.clients.llm_client import LLMClient
from comfyui_agent.infrastructure.logging_setup import setup_logging
from comfyui_agent.infrastructure.persistence.session_store import SessionStore
from comfyui_agent.knowledge.node_index import NodeIndex

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class WebServer:
    """aiohttp-based web server for the agent."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.event_bus = EventBus()
        self.comfyui = ComfyUIClient(
            base_url=config.comfyui.base_url,
            ws_url=config.comfyui.ws_url,
            timeout=config.comfyui.timeout,
            event_bus=self.event_bus,
        )
        api_key = config.llm.resolve_api_key()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Set it via environment variable or in config.yaml"
            )
        self.llm = LLMClient(
            api_key=api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            base_url=config.llm.base_url,
            event_bus=self.event_bus,
        )
        self.session_store = SessionStore(db_path=config.agent.session_db)
        self.node_index = NodeIndex()
        tools = create_all_tools(self.comfyui, self.node_index)
        context_manager = ContextManager(
            model=config.llm.model,
            max_output_tokens=config.llm.max_tokens,
            context_budget=config.agent.context_budget,
        )
        self.agent = AgentLoop(
            llm=self.llm,
            tools=tools,
            session_store=self.session_store,
            event_bus=self.event_bus,
            max_iterations=config.agent.max_iterations,
            context_manager=context_manager,
        )
        self._ws_clients: dict[str, list[web.WebSocketResponse]] = {}

    def create_app(self) -> web.Application:
        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)

        # API routes
        health = app.router.add_get("/api/health", self.handle_health)
        sessions_list = app.router.add_get("/api/sessions", self.handle_list_sessions)
        sessions_create = app.router.add_post("/api/sessions", self.handle_create_session)
        sessions_delete = app.router.add_delete(
            "/api/sessions/{session_id}", self.handle_delete_session
        )
        sessions_messages = app.router.add_get(
            "/api/sessions/{session_id}/messages", self.handle_session_messages
        )
        chat = app.router.add_post("/api/chat", self.handle_chat)
        chat_ws = app.router.add_get("/api/chat/ws", self.handle_chat_ws)

        # CORS — allow ComfyUI frontend (and other origins) to access the API
        cors = aiohttp_cors.setup(
            app,
            defaults={
                origin: aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    allow_headers="*",
                    allow_methods="*",
                )
                for origin in self.config.server.cors_origins
            },
        )
        for route in [health, sessions_list, sessions_create, sessions_delete, sessions_messages, chat, chat_ws]:
            cors.add(route)

        # Static files (web UI)
        if STATIC_DIR.exists():
            app.router.add_get("/", self.handle_index)
            app.router.add_static("/static", STATIC_DIR, name="static")

        return app

    async def _on_startup(self, app: web.Application) -> None:
        logger.info("Starting ComfyUI Agent server...")
        comfyui_ok = await self.comfyui.health_check()
        if comfyui_ok:
            logger.info("ComfyUI connected at %s", self.config.comfyui.base_url)
            await self.comfyui.connect_ws()
            # Build node index for on-demand discovery
            await self.node_index.build(self.comfyui)
            logger.info(
                "Node index: %d nodes in %d categories",
                self.node_index.node_count,
                len(self.node_index.categories),
            )
        else:
            logger.warning(
                "ComfyUI not reachable at %s", self.config.comfyui.base_url
            )

    async def _on_shutdown(self, app: web.Application) -> None:
        logger.info("Shutting down...")
        await self.comfyui.close()
        await self.llm.close()
        await self.session_store.close()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def handle_index(self, request: web.Request) -> web.Response:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="Web UI not found", status=404)

    async def handle_health(self, request: web.Request) -> web.Response:
        comfyui_ok = await self.comfyui.health_check()
        stats = None
        if comfyui_ok:
            try:
                stats = await self.comfyui.get_system_stats()
            except Exception:
                pass
        return web.json_response({
            "status": "ok",
            "comfyui": {
                "connected": comfyui_ok,
                "url": self.config.comfyui.base_url,
                "stats": stats,
            },
            "llm": {
                "model": self.config.llm.model,
            },
            "node_index": {
                "built": self.node_index.is_built,
                "node_count": self.node_index.node_count,
                "categories": len(self.node_index.categories),
            },
        })

    async def handle_list_sessions(self, request: web.Request) -> web.Response:
        sessions = await self.session_store.list_sessions()
        return web.json_response({"sessions": sessions})

    async def handle_create_session(self, request: web.Request) -> web.Response:
        body = await request.json() if request.content_length else {}
        title = body.get("title", "New Session")
        session_id = await self.session_store.create_session(title)
        return web.json_response({"session_id": session_id, "title": title})

    async def handle_delete_session(self, request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        await self.session_store.delete_session(session_id)
        return web.json_response({"deleted": session_id})

    async def handle_session_messages(self, request: web.Request) -> web.Response:
        """Load session messages in frontend ChatItem format."""
        session_id = request.match_info["session_id"]
        messages = await self.session_store.load_messages(session_id)
        items = api_messages_to_chat_items(messages)
        return web.json_response({
            "session_id": session_id,
            "items": items,
        })

    async def handle_chat(self, request: web.Request) -> web.Response:
        """HTTP POST chat — returns full response (non-streaming)."""
        body = await request.json()
        session_id = body.get("session_id")
        message = body.get("message", "")

        if not message:
            return web.json_response({"error": "message is required"}, status=400)

        if not session_id:
            session_id = await self.session_store.create_session("API Session")

        try:
            response = await self.agent.run(session_id, message)
            return web.json_response({
                "session_id": session_id,
                "response": response,
            })
        except Exception as e:
            logger.exception("Chat error")
            return web.json_response(
                {"error": str(e), "session_id": session_id}, status=500
            )

    async def handle_chat_ws(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket chat — bidirectional streaming."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("WebSocket client connected")

        # Track this connection
        conn_id = str(uuid.uuid4())
        unsubscribers: list[Any] = []

        try:
            # Subscribe to events and forward to this WebSocket
            async def forward_event(event: Event) -> None:
                if ws.closed:
                    return
                try:
                    await ws.send_json({
                        "type": "event",
                        "event_type": event.type.value,
                        "data": event.data,
                        "session_id": event.session_id,
                        "timestamp": event.timestamp,
                    })
                except Exception:
                    pass

            # Subscribe to all events
            unsub = self.event_bus.on_all(forward_event)
            unsubscribers.append(unsub)

            # Process incoming messages
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "error": "Invalid JSON"})
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", ws.exception())

        finally:
            for unsub in unsubscribers:
                unsub()
            logger.info("WebSocket client disconnected: %s", conn_id)

        return ws

    async def _handle_ws_message(
        self, ws: web.WebSocketResponse, data: dict[str, Any]
    ) -> None:
        """Process a message from a WebSocket client."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
            session_id = data.get("session_id")
            message = data.get("message", "")

            if not message:
                await ws.send_json({"type": "error", "error": "message is required"})
                return

            if not session_id:
                session_id = await self.session_store.create_session("WS Session")
                await ws.send_json({
                    "type": "session_created",
                    "session_id": session_id,
                })

            # Run agent in background so we can keep receiving messages
            asyncio.create_task(self._run_agent_for_ws(ws, session_id, message))

        elif msg_type == "cancel":
            session_id = data.get("session_id", "")
            if session_id:
                self.agent.cancel(session_id)
                await ws.send_json({"type": "cancelled", "session_id": session_id})

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

    async def _run_agent_for_ws(
        self, ws: web.WebSocketResponse, session_id: str, message: str
    ) -> None:
        """Run the agent loop and send the final response via WebSocket."""
        try:
            response = await self.agent.run(session_id, message)
            if not ws.closed:
                await ws.send_json({
                    "type": "response",
                    "session_id": session_id,
                    "content": response,
                })
        except Exception as e:
            logger.exception("Agent error for WS session %s", session_id)
            if not ws.closed:
                await ws.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "error": str(e),
                })


def create_server(config: AppConfig | None = None) -> WebServer:
    if config is None:
        config = AppConfig.from_yaml()
    return WebServer(config)


def run_server() -> None:
    """Entry point for running the web server."""
    config = AppConfig.from_yaml()
    setup_logging(
        level=config.logging.level,
        log_dir=config.logging.log_dir,
    )
    server = create_server(config)
    app = server.create_app()
    logger.info(
        "Starting server on %s:%d", config.server.host, config.server.port
    )
    web.run_app(app, host=config.server.host, port=config.server.port)
