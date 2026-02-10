"""Canvas state tracker — monitors ComfyUI workflow submissions.

Listens to WORKFLOW_SUBMITTED events and maintains a text summary
of the most recently submitted workflow for prompt injection.
"""

from __future__ import annotations

import logging
from typing import Any

from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.domain.ports import EventBusPort

logger = logging.getLogger(__name__)

_EMPTY_CANVAS = "Canvas is empty — no workflow has been submitted yet."


class CanvasState:
    """Tracks the latest workflow on the ComfyUI canvas."""

    def __init__(self, event_bus: EventBusPort) -> None:
        self._summary: str = ""
        self._prompt_id: str = ""
        event_bus.on(EventType.WORKFLOW_SUBMITTED, self._on_workflow_submitted)

    def get_summary(self) -> str:
        """Return a text summary of the current canvas state."""
        return self._summary or _EMPTY_CANVAS

    async def _on_workflow_submitted(self, event: Event) -> None:
        """Handle WORKFLOW_SUBMITTED event."""
        workflow = event.data.get("workflow")
        if not isinstance(workflow, dict):
            logger.warning("WORKFLOW_SUBMITTED missing valid workflow data")
            return

        self._prompt_id = event.data.get("prompt_id", "")
        try:
            self._summary = self._build_summary(workflow)
        except Exception as exc:
            logger.warning("Failed to build canvas summary: %s", exc)

    @staticmethod
    def _build_summary(workflow: dict[str, Any]) -> str:
        """Build a concise text summary from a workflow dict."""
        if not workflow:
            return ""

        class_types: list[str] = []
        checkpoint = ""
        prompt_text = ""
        width = 0
        height = 0

        for _node_id, node in workflow.items():
            ct = node.get("class_type", "")
            if ct:
                class_types.append(ct)
            inputs = node.get("inputs", {})
            if ct == "CheckpointLoaderSimple":
                checkpoint = inputs.get("ckpt_name", "")
            if ct == "CLIPTextEncode" and not prompt_text:
                prompt_text = inputs.get("text", "")
            if ct == "EmptyLatentImage":
                width = inputs.get("width", 0)
                height = inputs.get("height", 0)

        lines = [f"## Canvas ({len(workflow)} nodes)"]
        lines.append(f"- Node types: {', '.join(class_types)}")
        if checkpoint:
            lines.append(f"- Checkpoint: {checkpoint}")
        if prompt_text:
            preview = prompt_text[:80] + ("..." if len(prompt_text) > 80 else "")
            lines.append(f"- Prompt: {preview}")
        if width and height:
            lines.append(f"- Size: {width}×{height}")
        return "\n".join(lines)
