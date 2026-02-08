"""Tool factory — creates and assembles all ComfyUI tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.domain.tools.base import Tool
from comfyui_agent.domain.tools.discovery import (
    GetConnectableTool,
    GetNodeDetailTool,
    SearchNodesTool,
    ValidateWorkflowTool,
)
from comfyui_agent.domain.tools.execution import QueuePromptTool
from comfyui_agent.domain.tools.management import (
    DownloadModelTool,
    FreeMemoryTool,
    GetFolderPathsTool,
    InstallCustomNodeTool,
    RefreshNodeIndexTool,
    UploadImageTool,
)
from comfyui_agent.domain.tools.monitoring import (
    GetHistoryTool,
    GetQueueTool,
    InterruptTool,
    ListModelsTool,
    SystemStatsTool,
)

if TYPE_CHECKING:
    from comfyui_agent.knowledge.node_index import NodeIndex


def create_internal_tools(client: ComfyUIPort, node_index: NodeIndex) -> list[Tool]:
    """Create all internal ComfyUI tools (used by the dispatcher)."""
    return [
        # Discovery
        SearchNodesTool(node_index),
        GetNodeDetailTool(node_index),
        GetConnectableTool(node_index),
        ValidateWorkflowTool(node_index),
        # Execution
        QueuePromptTool(client),
        # Monitoring
        SystemStatsTool(client),
        ListModelsTool(client),
        GetQueueTool(client),
        GetHistoryTool(client),
        InterruptTool(client),
        # Management
        UploadImageTool(client),
        DownloadModelTool(client),
        InstallCustomNodeTool(client),
        FreeMemoryTool(client),
        GetFolderPathsTool(client),
        RefreshNodeIndexTool(client, node_index),
    ]


def create_all_tools(client: ComfyUIPort, node_index: NodeIndex) -> list[Tool]:
    """Create all ComfyUI tools as a single dispatcher.

    Returns a list with one ComfyUIDispatcher that routes to all 15 operations.
    """
    from comfyui_agent.domain.tools.dispatcher import ComfyUIDispatcher

    return [ComfyUIDispatcher(client, node_index)]
