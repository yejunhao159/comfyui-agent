"""Group dispatchers — split the monolithic ComfyUIDispatcher into 4 focused tools.

Each dispatcher exposes a subset of ComfyUI operations as a single tool
with action+params routing. This gives the LLM shorter, more focused
tool descriptions for better decision-making.

Groups:
  comfyui_discover — node search, detail, connectable, validate
  comfyui_execute  — queue_prompt, interrupt
  comfyui_monitor  — system_stats, list_models, get_queue, get_history
  comfyui_manage   — upload, download, install, free_memory, folder_paths, refresh
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult

if TYPE_CHECKING:
    from comfyui_agent.knowledge.node_index import NodeIndex


class _GroupDispatcher(Tool):
    """Base class for group dispatchers — routes action to internal tools."""

    def __init__(self, tools: dict[str, Tool]) -> None:
        self._tools = tools

    async def run(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        action_params = params.get("params", {})
        tool = self._tools.get(action)
        if not tool:
            return ToolResult.error(
                f"Unknown action: '{action}'. Available: {list(self._tools.keys())}"
            )
        return await tool.run(action_params)


class DiscoveryDispatcher(_GroupDispatcher):
    """Node discovery and workflow validation tools."""

    def __init__(self, client: ComfyUIPort, node_index: NodeIndex) -> None:
        from comfyui_agent.domain.tools.discovery import (
            GetConnectableTool,
            GetNodeDetailTool,
            SearchNodesTool,
            ValidateWorkflowTool,
        )

        tools: dict[str, Tool] = {}
        for t in [
            SearchNodesTool(node_index),
            GetNodeDetailTool(node_index),
            GetConnectableTool(node_index),
            ValidateWorkflowTool(node_index),
        ]:
            name = t.info().name.replace("comfyui_", "")
            tools[name] = t
        super().__init__(tools)

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_discover",
            description=(
                "Discover ComfyUI nodes and validate workflows. This is your primary "
                "research tool — always start here when building or modifying workflows.\n\n"
                "Actions:\n"
                "- search_nodes(query?, category?) — Search nodes by keyword (e.g. 'upscale', "
                "'controlnet') or browse a category. Returns top 10 matches with class_name, "
                "display_name, category, and description. Call with no args to list all categories.\n"
                "- get_node_detail(node_class) — Get full specification of a node type: "
                "required/optional inputs with types and allowed values, output types and names. "
                "Only call for complex nodes (KSampler, ControlNetApply, etc.) — skip simple "
                "nodes like CLIPTextEncode, EmptyLatentImage, VAEDecode, SaveImage whose "
                "inputs are obvious.\n"
                "- get_connectable(output_type?) — Given a data type (MODEL, CLIP, LATENT, "
                "CONDITIONING, IMAGE, VAE, etc.), list which nodes produce it and which consume it. "
                "Critical for finding compatible nodes when building pipelines. "
                "Call with no args for a summary of all connection types.\n"
                "- validate_workflow(workflow) — Check a workflow dict for errors: missing nodes, "
                "invalid connections, type mismatches, missing required inputs. "
                "Always call this before submitting a workflow with comfyui_execute. "
                "If validation fails, fix the specific error and re-validate ONCE."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search_nodes", "get_node_detail", "get_connectable", "validate_workflow"],
                        "description": "The discovery operation to perform",
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters: search_nodes({query?, category?}), get_node_detail({node_class}), get_connectable({output_type?}), validate_workflow({workflow})",
                    },
                },
                "required": ["action"],
            },
        )


class ExecuteDispatcher(_GroupDispatcher):
    """Workflow execution and control tools."""

    def __init__(self, client: ComfyUIPort, node_index: NodeIndex) -> None:
        from comfyui_agent.domain.tools.execution import QueuePromptTool
        from comfyui_agent.domain.tools.monitoring import InterruptTool

        tools: dict[str, Tool] = {}
        for t in [QueuePromptTool(client), InterruptTool(client)]:
            name = t.info().name.replace("comfyui_", "")
            tools[name] = t
        super().__init__(tools)

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_execute",
            description=(
                "Submit workflows to ComfyUI for execution and control running jobs.\n\n"
                "Actions:\n"
                "- queue_prompt(workflow) — Submit a workflow dict for execution. The workflow "
                "must be in ComfyUI API format: {node_id: {class_type, inputs}}. Node connections "
                "use [source_node_id, output_index] references. Always validate_workflow first. "
                "Returns a prompt_id for tracking. IMPORTANT: After queue_prompt succeeds, "
                "IMMEDIATELY give a final text response to the user — tell them the workflow "
                "was submitted with the prompt_id and describe what it will produce. "
                "Do NOT call any more tools after a successful queue_prompt.\n"
                "- interrupt() — Cancel the currently running execution immediately. "
                "Use when the user wants to stop a long-running generation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["queue_prompt", "interrupt"],
                        "description": "The execution operation to perform",
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters: queue_prompt({workflow}), interrupt(no params)",
                    },
                },
                "required": ["action"],
            },
        )


class MonitorDispatcher(_GroupDispatcher):
    """System monitoring and status tools."""

    def __init__(self, client: ComfyUIPort, node_index: NodeIndex) -> None:
        from comfyui_agent.domain.tools.monitoring import (
            GetHistoryTool,
            GetQueueTool,
            ListModelsTool,
            SystemStatsTool,
        )

        tools: dict[str, Tool] = {}
        for t in [
            SystemStatsTool(client),
            ListModelsTool(client),
            GetQueueTool(client),
            GetHistoryTool(client),
        ]:
            name = t.info().name.replace("comfyui_", "")
            tools[name] = t
        super().__init__(tools)

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_monitor",
            description=(
                "Monitor ComfyUI system status, available resources, and execution history.\n\n"
                "Actions:\n"
                "- system_stats() — Get GPU device info, VRAM usage (total/free), "
                "ComfyUI version, and Python version. Useful for checking if the system "
                "can handle a workload.\n"
                "- list_models(folder?) — List model files in a folder. Folder can be: "
                "checkpoints, loras, vae, controlnet, upscale_models, embeddings, clip, "
                "clip_vision, etc. Defaults to 'checkpoints'. IMPORTANT: Always use the "
                "exact filenames returned by this tool in workflow inputs — never guess "
                "or fabricate model names.\n"
                "- get_queue() — Show how many jobs are running and pending in the queue.\n"
                "- get_history(prompt_id?) — Get execution results. With a prompt_id, "
                "returns output details including image filenames and node outputs. "
                "Without, lists recent executions. Use this to check results after "
                "queue_prompt or to diagnose execution errors."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["system_stats", "list_models", "get_queue", "get_history"],
                        "description": "The monitoring operation to perform",
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters: system_stats(no params), list_models({folder?}), get_queue(no params), get_history({prompt_id?})",
                    },
                },
                "required": ["action"],
            },
        )


class ManageDispatcher(_GroupDispatcher):
    """Resource management tools."""

    def __init__(self, client: ComfyUIPort, node_index: NodeIndex) -> None:
        from comfyui_agent.domain.tools.management import (
            DownloadModelTool,
            FreeMemoryTool,
            GetFolderPathsTool,
            InstallCustomNodeTool,
            RefreshNodeIndexTool,
            UploadImageTool,
        )

        tools: dict[str, Tool] = {}
        for t in [
            UploadImageTool(client),
            DownloadModelTool(client),
            InstallCustomNodeTool(client),
            FreeMemoryTool(client),
            GetFolderPathsTool(client),
            RefreshNodeIndexTool(client, node_index),
        ]:
            name = t.info().name.replace("comfyui_", "")
            tools[name] = t
        super().__init__(tools)

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_manage",
            description=(
                "Manage ComfyUI resources: upload images, download models, install custom "
                "nodes, and manage GPU memory.\n\n"
                "Actions:\n"
                "- upload_image(url?, filepath?, filename?) — Upload an image to ComfyUI's "
                "input directory for use in img2img, ControlNet, or other image-input workflows. "
                "Provide either a URL (downloaded automatically) or a local filepath. "
                "Returns the filename to reference in workflow inputs.\n"
                "- download_model(url, folder, filename?) — Download a model file from URL "
                "(HuggingFace, Civitai, or direct link) into a model folder. Use "
                "get_folder_paths() first to see available folders and their disk paths.\n"
                "- install_custom_node(git_url) — Clone a custom node repository into "
                "ComfyUI's custom_nodes/ directory and install its dependencies. "
                "Requires ComfyUI restart to take effect. After restart, call "
                "refresh_index to update the node search index.\n"
                "- free_memory(unload_models?, free_memory?) — Release GPU VRAM by "
                "unloading models and clearing caches. Useful before loading large models "
                "or when VRAM is running low.\n"
                "- get_folder_paths() — List all ComfyUI storage directories: where models, "
                "outputs, inputs, and custom nodes are stored on disk.\n"
                "- refresh_index() — Rebuild the node search index from ComfyUI's current "
                "node registry. Required after installing new custom nodes and restarting ComfyUI."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "upload_image", "download_model", "install_custom_node",
                            "free_memory", "get_folder_paths", "refresh_index",
                        ],
                        "description": "The management operation to perform",
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters: upload_image({url?, filepath?, filename?}), download_model({url, folder, filename?}), install_custom_node({git_url}), free_memory({unload_models?, free_memory?}), get_folder_paths(no params), refresh_index(no params)",
                    },
                },
                "required": ["action"],
            },
        )
