"""ComfyUI tools for the agent.

Tools are organized by purpose:
- Discovery: search_nodes, get_node_detail, validate_workflow
- Execution: queue_prompt
- Monitoring: system_stats, list_models, get_queue, get_history, interrupt
- Management: upload_image, download_model, install_custom_node, free_memory, get_folder_paths
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult
from comfyui_agent.infrastructure.comfyui_client import ComfyUIClient
from comfyui_agent.knowledge.node_index import NodeIndex

MAX_TOOL_OUTPUT = 15000  # chars — OpenCode uses 30K, we use 15K for LLM efficiency


def truncate_output(text: str, max_len: int = MAX_TOOL_OUTPUT) -> str:
    """Truncate tool output, keeping first and last portions."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    mid_lines = text[half:-half].count("\n")
    return f"{text[:half]}\n\n... [{mid_lines} lines truncated] ...\n\n{text[-half:]}"


# ============================================================
# Discovery Tools
# ============================================================


class SearchNodesTool(Tool):
    """Search ComfyUI nodes by keyword, or browse by category."""

    def __init__(self, node_index: NodeIndex) -> None:
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_search_nodes",
            description=(
                "Search for ComfyUI node types by keyword or browse by category. "
                "Use this to find the right nodes for a task. "
                "Examples: search_nodes(query='upscale'), search_nodes(category='loaders'), "
                "search_nodes() to list all categories."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword (e.g., 'sampler', 'upscale', 'controlnet')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Browse a specific category (e.g., 'loaders', 'sampling')",
                    },
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query")
        category = params.get("category")

        if query:
            return ToolResult.success(self.index.search(query))
        elif category:
            return ToolResult.success(self.index.list_category(category))
        else:
            return ToolResult.success(self.index.list_categories())


class GetNodeDetailTool(Tool):
    """Get condensed detail for a specific node type."""

    def __init__(self, node_index: NodeIndex) -> None:
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_node_detail",
            description=(
                "Get detailed information about a specific ComfyUI node type: "
                "its inputs (required/optional with types), outputs, and description. "
                "Use search_nodes first to find the right node class name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_class": {
                        "type": "string",
                        "description": "Exact node class name (e.g., 'KSampler', 'CheckpointLoaderSimple')",
                    },
                },
                "required": ["node_class"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        node_class = params.get("node_class", "")
        if not node_class:
            return ToolResult.error("node_class is required")
        return ToolResult.success(self.index.get_detail(node_class))


class ValidateWorkflowTool(Tool):
    """Validate a workflow before submitting it."""

    def __init__(self, node_index: NodeIndex) -> None:
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_validate_workflow",
            description=(
                "Validate a ComfyUI workflow before submitting. "
                "Checks that all node types exist, required inputs are provided, "
                "and connections are valid. Always validate before queue_prompt."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "ComfyUI workflow in API format (node_id -> {class_type, inputs})",
                    },
                },
                "required": ["workflow"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        workflow = params.get("workflow")
        if not workflow:
            return ToolResult.error("workflow is required")
        return ToolResult.success(self.index.validate_workflow(workflow))


# ============================================================
# Execution Tools
# ============================================================


class QueuePromptTool(Tool):
    """Submit a workflow to ComfyUI for execution."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_queue_prompt",
            description=(
                "Submit a ComfyUI workflow for execution. "
                "The workflow must be in ComfyUI's API format (dict of node_id -> {class_type, inputs}). "
                "Always use comfyui_validate_workflow first to check for errors."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "ComfyUI workflow in API format",
                    },
                },
                "required": ["workflow"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        workflow = params.get("workflow")
        if not workflow:
            return ToolResult.error("workflow parameter is required")
        try:
            result = await self.client.queue_prompt(workflow)
            prompt_id = result.get("prompt_id", "unknown")
            return ToolResult.success(
                f"Workflow submitted. prompt_id: {prompt_id}",
                data={"prompt_id": prompt_id},
            )
        except Exception as e:
            return ToolResult.error(f"Failed to queue prompt: {e}")


# ============================================================
# Monitoring Tools
# ============================================================


class SystemStatsTool(Tool):
    """Get ComfyUI system stats."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_system_stats",
            description="Get ComfyUI system statistics including GPU info, VRAM usage, and version.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            stats = await self.client.get_system_stats()
            return ToolResult.success(json.dumps(stats, indent=2))
        except Exception as e:
            return ToolResult.error(f"Failed to get system stats: {e}")


class ListModelsTool(Tool):
    """List available models."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_list_models",
            description=(
                "List available models in ComfyUI. "
                "Folder can be: checkpoints, loras, vae, controlnet, upscale_models, embeddings, clip, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Model folder to list (default: checkpoints)",
                        "default": "checkpoints",
                    },
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        folder = params.get("folder", "checkpoints")
        try:
            models = await self.client.list_models(folder)
            if not models:
                return ToolResult.success(f"No models found in '{folder}'.")
            text = f"Models in '{folder}' ({len(models)}):\n"
            for m in models:
                text += f"  - {m}\n"
            return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to list models: {e}")


class GetQueueTool(Tool):
    """Get queue status."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_queue",
            description="Get the current ComfyUI execution queue status.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            queue = await self.client.get_queue()
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            text = f"Queue: {len(running)} running, {len(pending)} pending\n"
            for item in running:
                text += f"  [running] {item[1]}\n"
            for item in pending[:10]:
                text += f"  [pending] {item[1]}\n"
            if len(pending) > 10:
                text += f"  ... and {len(pending) - 10} more\n"
            return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get queue: {e}")


class GetHistoryTool(Tool):
    """Get execution history."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_history",
            description=(
                "Get execution history. If prompt_id is given, returns details "
                "including output image URLs. Otherwise returns recent history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt_id": {
                        "type": "string",
                        "description": "Specific prompt_id to get details for",
                    },
                },
                "required": [],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        prompt_id = params.get("prompt_id")
        try:
            history = await self.client.get_history(prompt_id)
            if prompt_id and prompt_id in history:
                entry = history[prompt_id]
                outputs = entry.get("outputs", {})
                status = entry.get("status", {})
                text = f"Execution {prompt_id}:\n"
                text += f"  Status: {status.get('status_str', 'unknown')}\n"
                if outputs:
                    text += "  Outputs:\n"
                    for node_id, output in outputs.items():
                        if "images" in output:
                            for img in output["images"]:
                                url = self.client.get_image_url(
                                    img["filename"],
                                    img.get("subfolder", ""),
                                    img.get("type", "output"),
                                )
                                text += f"    Node {node_id}: {url}\n"
                return ToolResult.success(text)
            else:
                entries = list(history.keys())[-10:]
                text = f"Recent executions ({len(history)} total, showing {len(entries)}):\n"
                for pid in entries:
                    st = history[pid].get("status", {}).get("status_str", "unknown")
                    text += f"  - {pid} [{st}]\n"
                return ToolResult.success(text)
        except Exception as e:
            return ToolResult.error(f"Failed to get history: {e}")


class InterruptTool(Tool):
    """Interrupt running execution."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_interrupt",
            description="Interrupt the currently running ComfyUI execution.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            await self.client.interrupt()
            return ToolResult.success("Execution interrupted.")
        except Exception as e:
            return ToolResult.error(f"Failed to interrupt: {e}")


logger = logging.getLogger(__name__)

# ============================================================
# Management Tools
# ============================================================


class UploadImageTool(Tool):
    """Upload an image to ComfyUI for use in workflows (img2img, ControlNet, etc.)."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_upload_image",
            description=(
                "Upload an image to ComfyUI from a URL or local file path. "
                "The uploaded image can then be used in workflows (img2img, ControlNet, etc.). "
                "Returns the filename to use in workflow inputs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the image to download and upload",
                    },
                    "filepath": {
                        "type": "string",
                        "description": "Local file path of the image to upload",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name to save the image as (optional, auto-generated if not provided)",
                    },
                },
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        url = params.get("url")
        filepath = params.get("filepath")
        filename = params.get("filename", "")

        if not url and not filepath:
            return ToolResult.error("Either 'url' or 'filepath' is required")

        try:
            if url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        image_data = await resp.read()
                if not filename:
                    # Extract filename from URL
                    filename = url.split("/")[-1].split("?")[0]
                    if not filename or "." not in filename:
                        filename = "uploaded_image.png"
            else:
                path = Path(filepath)
                if not path.exists():
                    return ToolResult.error(f"File not found: {filepath}")
                image_data = path.read_bytes()
                if not filename:
                    filename = path.name

            result = await self.client.upload_image(image_data, filename, overwrite=True)
            saved_name = result.get("name", filename)
            subfolder = result.get("subfolder", "")
            return ToolResult.success(
                f"Image uploaded: {saved_name}"
                + (f" (subfolder: {subfolder})" if subfolder else "")
                + f"\nUse this in workflow inputs as: \"{saved_name}\""
            )
        except Exception as e:
            return ToolResult.error(f"Failed to upload image: {e}")


class DownloadModelTool(Tool):
    """Download a model file from URL to ComfyUI's model directory."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_download_model",
            description=(
                "Download a model file from a URL to ComfyUI's model directory. "
                "Supports HuggingFace, Civitai, and direct download URLs. "
                "Use comfyui_get_folder_paths first to see available model folders. "
                "Common folders: checkpoints, loras, vae, controlnet, upscale_models, embeddings, clip."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Download URL for the model file",
                    },
                    "folder": {
                        "type": "string",
                        "description": "Target model folder (e.g., 'checkpoints', 'loras', 'vae')",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename to save as (optional, auto-detected from URL)",
                    },
                },
                "required": ["url", "folder"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        url = params.get("url", "")
        folder = params.get("folder", "")
        filename = params.get("filename", "")

        if not url or not folder:
            return ToolResult.error("'url' and 'folder' are required")

        try:
            # Get the actual filesystem path for this folder
            folder_paths = await self.client.get_folder_paths()
            paths = folder_paths.get(folder)
            if not paths or not isinstance(paths, list) or len(paths) == 0:
                available = [k for k in folder_paths if isinstance(folder_paths[k], list)]
                return ToolResult.error(
                    f"Unknown folder '{folder}'. Available: {', '.join(sorted(available))}"
                )

            # Use the first path for this folder type
            target_dir = Path(paths[0][0]) if isinstance(paths[0], list) else Path(paths[0])
            target_dir.mkdir(parents=True, exist_ok=True)

            # Determine filename
            if not filename:
                filename = _extract_filename_from_url(url)
            if not filename:
                return ToolResult.error(
                    "Could not determine filename from URL. Please provide 'filename' parameter."
                )

            target_path = target_dir / filename
            if target_path.exists():
                size_mb = target_path.stat().st_size / (1024 * 1024)
                return ToolResult.success(
                    f"Model already exists: {target_path} ({size_mb:.1f} MB)"
                )

            # Download with progress
            downloaded = 0
            async with aiohttp.ClientSession() as session:
                async with session.get(url, allow_redirects=True) as resp:
                    resp.raise_for_status()
                    total = resp.content_length
                    with open(target_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)

            size_mb = downloaded / (1024 * 1024)
            return ToolResult.success(
                f"Model downloaded: {filename} ({size_mb:.1f} MB)\n"
                f"Saved to: {target_path}\n"
                f"Folder: {folder}"
            )
        except Exception as e:
            return ToolResult.error(f"Failed to download model: {e}")


class InstallCustomNodeTool(Tool):
    """Install a ComfyUI custom node from a git repository."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_install_custom_node",
            description=(
                "Install a ComfyUI custom node from a git repository URL. "
                "Clones the repo into ComfyUI's custom_nodes/ directory and installs dependencies. "
                "ComfyUI needs to be restarted after installation for the new nodes to be available."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "git_url": {
                        "type": "string",
                        "description": "Git repository URL (e.g., 'https://github.com/user/ComfyUI-NodePack')",
                    },
                },
                "required": ["git_url"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        git_url = params.get("git_url", "")
        if not git_url:
            return ToolResult.error("'git_url' is required")

        # Validate URL format
        if not re.match(r"https?://", git_url):
            return ToolResult.error("git_url must start with http:// or https://")

        try:
            # Find ComfyUI's custom_nodes directory
            folder_paths = await self.client.get_folder_paths()
            custom_nodes_paths = folder_paths.get("custom_nodes")
            if not custom_nodes_paths:
                # Fallback: derive from base_url or use common path
                return ToolResult.error(
                    "Could not determine custom_nodes directory. "
                    "Please check ComfyUI installation."
                )
            custom_nodes_dir = Path(
                custom_nodes_paths[0][0]
                if isinstance(custom_nodes_paths[0], list)
                else custom_nodes_paths[0]
            )

            # Extract repo name
            repo_name = git_url.rstrip("/").split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]

            target_dir = custom_nodes_dir / repo_name
            if target_dir.exists():
                return ToolResult.success(
                    f"Custom node '{repo_name}' already installed at {target_dir}"
                )

            # Git clone
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", git_url, str(target_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode != 0:
                return ToolResult.error(
                    f"git clone failed: {stderr.decode().strip()}"
                )

            # Install requirements if present
            req_file = target_dir / "requirements.txt"
            pip_msg = ""
            if req_file.exists():
                # Use the same Python that ComfyUI runs with
                python_path = _find_comfyui_python(custom_nodes_dir)
                proc = await asyncio.create_subprocess_exec(
                    python_path, "-m", "pip", "install", "-r", str(req_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
                if proc.returncode == 0:
                    pip_msg = "\nDependencies installed from requirements.txt"
                else:
                    pip_msg = f"\nWarning: pip install failed: {stderr.decode()[:200]}"

            return ToolResult.success(
                f"Custom node '{repo_name}' installed at {target_dir}{pip_msg}\n"
                f"Note: Restart ComfyUI for the new nodes to be available."
            )
        except asyncio.TimeoutError:
            return ToolResult.error("Installation timed out")
        except Exception as e:
            return ToolResult.error(f"Failed to install custom node: {e}")


class FreeMemoryTool(Tool):
    """Free VRAM/RAM by unloading models."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_free_memory",
            description=(
                "Free GPU VRAM and system RAM by unloading models and clearing caches. "
                "Useful when running low on memory before loading a new model."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "unload_models": {
                        "type": "boolean",
                        "description": "Unload all loaded models (default: true)",
                    },
                    "free_memory": {
                        "type": "boolean",
                        "description": "Free cached memory (default: true)",
                    },
                },
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        unload = params.get("unload_models", True)
        free = params.get("free_memory", True)
        try:
            # Get stats before
            stats_before = await self.client.get_system_stats()
            dev = stats_before.get("devices", [{}])[0]
            vram_before = dev.get("vram_free", 0)

            await self.client.free_memory(unload_models=unload, free_memory=free)

            # Get stats after
            stats_after = await self.client.get_system_stats()
            dev = stats_after.get("devices", [{}])[0]
            vram_after = dev.get("vram_free", 0)

            freed = (vram_after - vram_before) / (1024 * 1024)
            return ToolResult.success(
                f"Memory freed. VRAM: {vram_after / (1024**3):.1f} GB available"
                + (f" (+{freed:.0f} MB)" if freed > 0 else "")
            )
        except Exception as e:
            return ToolResult.error(f"Failed to free memory: {e}")


class GetFolderPathsTool(Tool):
    """Get ComfyUI's model folder paths."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_folder_paths",
            description=(
                "Get the filesystem paths where ComfyUI stores models, outputs, and other files. "
                "Use this to know where to download models or find generated images."
            ),
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            paths = await self.client.get_folder_paths()
            # Format nicely — only show folders that have paths
            lines = ["ComfyUI folder paths:\n"]
            for folder, folder_paths in sorted(paths.items()):
                if isinstance(folder_paths, list) and folder_paths:
                    dirs = []
                    for p in folder_paths:
                        if isinstance(p, list):
                            dirs.append(p[0])
                        else:
                            dirs.append(str(p))
                    lines.append(f"  {folder}: {', '.join(dirs)}")
            return ToolResult.success(truncate_output("\n".join(lines)))
        except Exception as e:
            return ToolResult.error(f"Failed to get folder paths: {e}")


# ============================================================
# Helpers
# ============================================================


def _extract_filename_from_url(url: str) -> str:
    """Extract a reasonable filename from a download URL."""
    # Handle HuggingFace URLs
    if "huggingface.co" in url:
        # https://huggingface.co/user/repo/resolve/main/model.safetensors
        parts = url.split("/")
        for i, p in enumerate(parts):
            if p in ("resolve", "blob") and i + 2 < len(parts):
                return parts[-1].split("?")[0]

    # Handle Civitai URLs
    if "civitai.com" in url:
        # Try to get filename from content-disposition later; use URL for now
        parts = url.split("/")
        for p in reversed(parts):
            if "." in p:
                return p.split("?")[0]

    # Generic: last path segment
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    if "." in name:
        return name
    return ""


def _find_comfyui_python(custom_nodes_dir: Path) -> str:
    """Find the Python executable that ComfyUI uses."""
    comfyui_root = custom_nodes_dir.parent
    # Check common venv locations
    for venv_dir in [comfyui_root / ".venv", comfyui_root / "venv"]:
        python = venv_dir / "bin" / "python"
        if python.exists():
            return str(python)
    # Fallback to system python
    return "python3"


# ============================================================
# Dispatcher — single tool that routes to all ComfyUI operations
# ============================================================

# Action names (strip "comfyui_" prefix from internal tool names)
_ACTION_NAMES = [
    "search_nodes",
    "get_node_detail",
    "validate_workflow",
    "queue_prompt",
    "system_stats",
    "list_models",
    "get_queue",
    "get_history",
    "interrupt",
    "upload_image",
    "download_model",
    "install_custom_node",
    "free_memory",
    "get_folder_paths",
]


class ComfyUIDispatcher(Tool):
    """Single dispatcher tool that routes to all ComfyUI operations.

    Instead of exposing 14 separate tools (~1366 tokens of schema per request),
    this exposes one tool with an action+params pattern (~200 tokens).
    The LLM learns available actions from the system prompt.
    """

    def __init__(self, client: ComfyUIClient, node_index: NodeIndex) -> None:
        self._tools: dict[str, Tool] = {}
        for t in _create_internal_tools(client, node_index):
            # Strip "comfyui_" prefix: "comfyui_search_nodes" -> "search_nodes"
            name = t.info().name.replace("comfyui_", "")
            self._tools[name] = t

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui",
            description=(
                "Execute ComfyUI operations. "
                "See system prompt for available actions and their parameters."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": _ACTION_NAMES,
                        "description": "The operation to perform",
                    },
                    "params": {
                        "type": "object",
                        "description": "Action-specific parameters (see system prompt for each action)",
                    },
                },
                "required": ["action"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        action_params = params.get("params", {})

        tool = self._tools.get(action)
        if not tool:
            return ToolResult.error(
                f"Unknown action: '{action}'. Available: {list(self._tools.keys())}"
            )

        return await tool.run(action_params)


# ============================================================
# Factory
# ============================================================


def _create_internal_tools(client: ComfyUIClient, node_index: NodeIndex) -> list[Tool]:
    """Create all internal ComfyUI tools (used by the dispatcher)."""
    return [
        # Discovery
        SearchNodesTool(node_index),
        GetNodeDetailTool(node_index),
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
    ]


def create_all_tools(client: ComfyUIClient, node_index: NodeIndex) -> list[Tool]:
    """Create all ComfyUI tools as a single dispatcher.

    Returns a list with one ComfyUIDispatcher that routes to all 14 operations.
    This reduces per-request token overhead from ~1366 to ~200 tokens.
    """
    return [ComfyUIDispatcher(client, node_index)]
