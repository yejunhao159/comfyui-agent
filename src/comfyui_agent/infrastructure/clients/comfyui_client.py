"""ComfyUI HTTP + WebSocket client.

Independent process communication with ComfyUI via its native API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

import aiohttp

from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for communicating with ComfyUI's HTTP and WebSocket API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6006",
        ws_url: str = "ws://127.0.0.1:6006/ws",
        timeout: int = 30,
        event_bus: EventBus | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.client_id = str(uuid.uuid4())
        self.event_bus = event_bus
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task[None] | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    # ============================================================
    # HTTP API Methods
    # ============================================================

    async def _get(self, path: str, **kwargs: Any) -> Any:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.get(url, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, data: Any = None, **kwargs: Any) -> Any:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.post(url, json=data, **kwargs) as resp:
            resp.raise_for_status()
            content_type = resp.content_type or ""
            if "json" in content_type:
                return await resp.json()
            text = await resp.text()
            if text.strip():
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"status": "ok", "raw": text}
            return {"status": "ok"}

    async def get_system_stats(self) -> dict[str, Any]:
        """Get system statistics (VRAM, version, etc.)."""
        return await self._get("/api/system_stats")

    async def get_object_info(self, node_class: str | None = None) -> dict[str, Any]:
        """Get node definitions. If node_class is given, get info for that node only."""
        if node_class:
            return await self._get(f"/api/object_info/{node_class}")
        return await self._get("/api/object_info")

    async def get_queue(self) -> dict[str, Any]:
        """Get current queue status (running and pending)."""
        return await self._get("/api/queue")

    async def get_history(self, prompt_id: str | None = None, max_items: int = 200) -> dict[str, Any]:
        """Get execution history."""
        if prompt_id:
            return await self._get(f"/api/history/{prompt_id}")
        return await self._get(f"/api/history?max_items={max_items}")

    async def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Submit a workflow for execution.

        Args:
            workflow: ComfyUI workflow JSON (prompt format)

        Returns:
            Dict with prompt_id and other info
        """
        data = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        result = await self._post("/api/prompt", data=data)
        logger.info("Queued prompt: %s", result.get("prompt_id", "unknown"))
        return result

    async def interrupt(self) -> None:
        """Interrupt the currently running prompt."""
        await self._post("/api/interrupt")
        logger.info("Interrupted current execution")

    async def clear_queue(self) -> None:
        """Clear all pending items from the queue."""
        await self._post("/api/queue", data={"clear": True})

    async def delete_queue_item(self, delete_ids: list[str]) -> None:
        """Delete specific items from the queue."""
        await self._post("/api/queue", data={"delete": delete_ids})

    async def list_models(self, folder: str = "checkpoints") -> list[str]:
        """List available models in a folder (checkpoints, loras, vae, etc.)."""
        result = await self._get(f"/api/models/{folder}")
        return result if isinstance(result, list) else []

    async def get_embeddings(self) -> list[str]:
        """List available embeddings."""
        return await self._get("/api/embeddings")

    async def upload_image(
        self, image_data: bytes, filename: str, subfolder: str = "", overwrite: bool = False
    ) -> dict[str, Any]:
        """Upload an image to ComfyUI."""
        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("image", image_data, filename=filename, content_type="image/png")
        if subfolder:
            form.add_field("subfolder", subfolder)
        form.add_field("overwrite", str(overwrite).lower())

        async with session.post(f"{self.base_url}/api/upload/image", data=form) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image from ComfyUI."""
        session = await self._get_session()
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with session.get(f"{self.base_url}/api/view", params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    def get_image_url(self, filename: str, subfolder: str = "", folder_type: str = "output") -> str:
        """Get the URL for an image."""
        return f"{self.base_url}/api/view?filename={filename}&subfolder={subfolder}&type={folder_type}"

    async def get_folder_paths(self) -> dict[str, Any]:
        """Get all folder paths configuration (where models are stored)."""
        return await self._get("/internal/folder_paths")

    async def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Free VRAM/RAM by unloading models and clearing caches."""
        await self._post("/api/free", data={
            "unload_models": unload_models,
            "free_memory": free_memory,
        })
        logger.info("Memory freed (unload=%s, free=%s)", unload_models, free_memory)

    # ============================================================
    # WebSocket Methods
    # ============================================================

    async def connect_ws(self) -> None:
        """Connect to ComfyUI WebSocket for real-time events."""
        session = await self._get_session()
        url = f"{self.ws_url}?clientId={self.client_id}"
        self._ws = await session.ws_connect(url)
        self._ws_task = asyncio.create_task(self._ws_listener())
        logger.info("WebSocket connected to %s", url)

    async def disconnect_ws(self) -> None:
        """Disconnect WebSocket."""
        if self._ws_task:
            self._ws_task.cancel()
            self._ws_task = None
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket disconnected")

    async def _ws_listener(self) -> None:
        """Background task that listens to WebSocket messages and emits events."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_ws_message(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary messages are preview images
                    if self.event_bus:
                        await self.event_bus.emit(Event(
                            type=EventType.COMFYUI_PREVIEW,
                            data={"image_data": msg.data},
                        ))
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("WebSocket listener error")

    async def _handle_ws_message(self, message: dict[str, Any]) -> None:
        """Process a WebSocket message and emit corresponding event."""
        if not self.event_bus:
            return

        msg_type = message.get("type", "")
        data = message.get("data", {})

        event_map: dict[str, EventType] = {
            "progress": EventType.COMFYUI_PROGRESS,
            "executing": EventType.COMFYUI_EXECUTING,
            "executed": EventType.COMFYUI_EXECUTED,
            "execution_error": EventType.COMFYUI_ERROR,
            "status": EventType.COMFYUI_QUEUE_UPDATE,
        }

        event_type = event_map.get(msg_type)
        if event_type:
            await self.event_bus.emit(Event(type=event_type, data=data))

    async def wait_for_prompt(self, prompt_id: str, timeout: float = 300.0) -> dict[str, Any]:
        """Wait for a prompt to complete execution.

        Polls history until the prompt appears in completed state.
        If WebSocket is connected, also listens for real-time events.
        """
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            history = await self.get_history(prompt_id)
            if prompt_id in history:
                prompt_history = history[prompt_id]
                status = prompt_history.get("status", {})
                if status.get("completed", False) or "outputs" in prompt_history:
                    return prompt_history
                if status.get("status_str") == "error":
                    raise RuntimeError(
                        f"Prompt {prompt_id} failed: {status.get('messages', [])}"
                    )
            await asyncio.sleep(1.0)

        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

    # ============================================================
    # Lifecycle
    # ============================================================

    async def health_check(self) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            await self.get_system_stats()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close all connections."""
        await self.disconnect_ws()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
