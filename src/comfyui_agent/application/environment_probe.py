"""Environment probe — active ComfyUI environment sensing.

Collects system stats, GPU info, models, queue status, and node index
summary into an EnvironmentSnapshot. Each sub-collection is independent:
a single API failure never crashes the whole probe.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from comfyui_agent.domain.models.context import EnvironmentSnapshot
from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.knowledge.node_index import NodeIndex

logger = logging.getLogger(__name__)


class EnvironmentProbe:
    """Actively probes ComfyUI for runtime environment information."""

    def __init__(
        self,
        client: ComfyUIPort,
        node_index: NodeIndex,
        refresh_interval: float = 300.0,
    ) -> None:
        self._client = client
        self._node_index = node_index
        self._refresh_interval = refresh_interval
        self._cached: EnvironmentSnapshot | None = None

    async def collect(self) -> EnvironmentSnapshot:
        """Collect a full environment snapshot. Never raises."""
        snap = EnvironmentSnapshot(collected_at=time.time())

        # 1. Connection & system stats
        try:
            ok = await self._client.health_check()
            snap.connection_ok = ok
        except Exception as exc:
            snap.errors.append(f"health_check: {exc}")

        if snap.connection_ok:
            await self._collect_system_stats(snap)
            await self._collect_models(snap)
            await self._collect_queue(snap)

        # Node index (local, no API call)
        self._collect_node_index(snap)

        self._cached = snap
        return snap

    async def get_snapshot(self) -> EnvironmentSnapshot:
        """Return cached snapshot if fresh, otherwise re-collect."""
        if self._cached is not None:
            age = time.time() - self._cached.collected_at
            if age < self._refresh_interval:
                return self._cached
        return await self.collect()

    async def refresh(self) -> None:
        """Force re-collect and update cache."""
        await self.collect()

    # ------------------------------------------------------------------
    # Private sub-collectors
    # ------------------------------------------------------------------

    async def _collect_system_stats(self, snap: EnvironmentSnapshot) -> None:
        try:
            stats: dict[str, Any] = await self._client.get_system_stats()
            system = stats.get("system", {})
            snap.comfyui_version = system.get("comfyui_version", "")
            devices = stats.get("devices", [])
            if devices:
                dev = devices[0]
                snap.gpu_name = dev.get("name", "")
                snap.vram_total_mb = dev.get("vram_total", 0) / (1024 * 1024)
                snap.vram_free_mb = dev.get("vram_free", 0) / (1024 * 1024)
        except Exception as exc:
            snap.errors.append(f"system_stats: {exc}")

    async def _collect_models(self, snap: EnvironmentSnapshot) -> None:
        try:
            snap.checkpoint_models = await self._client.list_models("checkpoints")
        except Exception as exc:
            snap.errors.append(f"list_models: {exc}")

    async def _collect_queue(self, snap: EnvironmentSnapshot) -> None:
        try:
            q: dict[str, Any] = await self._client.get_queue()
            snap.queue_running = len(q.get("queue_running", []))
            snap.queue_pending = len(q.get("queue_pending", []))
        except Exception as exc:
            snap.errors.append(f"get_queue: {exc}")

    def _collect_node_index(self, snap: EnvironmentSnapshot) -> None:
        if self._node_index.is_built:
            snap.node_count = self._node_index.node_count
            snap.node_categories = self._node_index.categories
