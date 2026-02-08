"""Management tools — upload, download, install, free memory, folder paths, refresh index."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

import aiohttp

from comfyui_agent.domain.ports import ComfyUIPort
from comfyui_agent.domain.tools.base import Tool, ToolInfo, ToolResult, truncate_output

if TYPE_CHECKING:
    from comfyui_agent.knowledge.node_index import NodeIndex


def _extract_filename_from_url(url: str) -> str:
    """Extract a reasonable filename from a download URL."""
    if "huggingface.co" in url:
        parts = url.split("/")
        for i, p in enumerate(parts):
            if p in ("resolve", "blob") and i + 2 < len(parts):
                return parts[-1].split("?")[0]
    if "civitai.com" in url:
        parts = url.split("/")
        for p in reversed(parts):
            if "." in p:
                return p.split("?")[0]
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    if "." in name:
        return name
    return ""


def _find_comfyui_python(custom_nodes_dir: Path) -> str:
    """Find the Python executable that ComfyUI uses."""
    comfyui_root = custom_nodes_dir.parent
    for venv_dir in [comfyui_root / ".venv", comfyui_root / "venv"]:
        python = venv_dir / "bin" / "python"
        if python.exists():
            return str(python)
    return "python3"


class UploadImageTool(Tool):
    """Upload an image to ComfyUI for use in workflows."""

    def __init__(self, client: ComfyUIPort) -> None:
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
                    "url": {"type": "string", "description": "URL of the image to download and upload"},
                    "filepath": {"type": "string", "description": "Local file path of the image to upload"},
                    "filename": {"type": "string", "description": "Name to save the image as (optional)"},
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
                + f'\nUse this in workflow inputs as: "{saved_name}"'
            )
        except Exception as e:
            return ToolResult.error(f"Failed to upload image: {e}")


class DownloadModelTool(Tool):
    """Download a model file from URL to ComfyUI's model directory."""

    def __init__(self, client: ComfyUIPort) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_download_model",
            description=(
                "Download a model file from a URL to ComfyUI's model directory. "
                "Supports HuggingFace, Civitai, and direct download URLs. "
                "Use comfyui_get_folder_paths first to see available model folders."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Download URL for the model file"},
                    "folder": {"type": "string", "description": "Target model folder (e.g., 'checkpoints', 'loras')"},
                    "filename": {"type": "string", "description": "Filename to save as (optional)"},
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
            folder_paths = await self.client.get_folder_paths()
            paths = folder_paths.get(folder)
            if not paths or not isinstance(paths, list) or len(paths) == 0:
                available = [k for k in folder_paths if isinstance(folder_paths[k], list)]
                return ToolResult.error(f"Unknown folder '{folder}'. Available: {', '.join(sorted(available))}")

            target_dir = Path(paths[0][0]) if isinstance(paths[0], list) else Path(paths[0])
            target_dir.mkdir(parents=True, exist_ok=True)

            if not filename:
                filename = _extract_filename_from_url(url)
            if not filename:
                return ToolResult.error("Could not determine filename from URL. Please provide 'filename' parameter.")

            target_path = target_dir / filename
            if target_path.exists():
                size_mb = target_path.stat().st_size / (1024 * 1024)
                return ToolResult.success(f"Model already exists: {target_path} ({size_mb:.1f} MB)")

            downloaded = 0
            async with aiohttp.ClientSession() as session:
                async with session.get(url, allow_redirects=True) as resp:
                    resp.raise_for_status()
                    with open(target_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)

            size_mb = downloaded / (1024 * 1024)
            return ToolResult.success(f"Model downloaded: {filename} ({size_mb:.1f} MB)\nSaved to: {target_path}\nFolder: {folder}")
        except Exception as e:
            return ToolResult.error(f"Failed to download model: {e}")


class InstallCustomNodeTool(Tool):
    """Install a ComfyUI custom node from a git repository."""

    def __init__(self, client: ComfyUIPort) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_install_custom_node",
            description=(
                "Install a ComfyUI custom node from a git repository URL. "
                "Clones the repo into ComfyUI's custom_nodes/ directory and installs dependencies."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "git_url": {"type": "string", "description": "Git repository URL"},
                },
                "required": ["git_url"],
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        git_url = params.get("git_url", "")
        if not git_url:
            return ToolResult.error("'git_url' is required")
        if not re.match(r"https?://", git_url):
            return ToolResult.error("git_url must start with http:// or https://")

        try:
            folder_paths = await self.client.get_folder_paths()
            custom_nodes_paths = folder_paths.get("custom_nodes")
            if not custom_nodes_paths:
                return ToolResult.error("Could not determine custom_nodes directory.")
            custom_nodes_dir = Path(
                custom_nodes_paths[0][0] if isinstance(custom_nodes_paths[0], list) else custom_nodes_paths[0]
            )

            repo_name = git_url.rstrip("/").split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]

            target_dir = custom_nodes_dir / repo_name
            if target_dir.exists():
                return ToolResult.success(f"Custom node '{repo_name}' already installed at {target_dir}")

            proc = await asyncio.create_subprocess_exec(
                "git", "clone", git_url, str(target_dir),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode != 0:
                return ToolResult.error(f"git clone failed: {stderr.decode().strip()}")

            req_file = target_dir / "requirements.txt"
            pip_msg = ""
            if req_file.exists():
                python_path = _find_comfyui_python(custom_nodes_dir)
                proc = await asyncio.create_subprocess_exec(
                    python_path, "-m", "pip", "install", "-r", str(req_file),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
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

    def __init__(self, client: ComfyUIPort) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_free_memory",
            description="Free GPU VRAM and system RAM by unloading models and clearing caches.",
            parameters={
                "type": "object",
                "properties": {
                    "unload_models": {"type": "boolean", "description": "Unload all loaded models (default: true)"},
                    "free_memory": {"type": "boolean", "description": "Free cached memory (default: true)"},
                },
            },
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        unload = params.get("unload_models", True)
        free = params.get("free_memory", True)
        try:
            stats_before = await self.client.get_system_stats()
            dev = stats_before.get("devices", [{}])[0]
            vram_before = dev.get("vram_free", 0)

            await self.client.free_memory(unload_models=unload, free_memory=free)

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

    def __init__(self, client: ComfyUIPort) -> None:
        self.client = client

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_get_folder_paths",
            description="Get the filesystem paths where ComfyUI stores models, outputs, and other files.",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            paths = await self.client.get_folder_paths()
            lines = ["ComfyUI folder paths:\n"]
            for folder, folder_paths in sorted(paths.items()):
                if isinstance(folder_paths, list) and folder_paths:
                    dirs = []
                    for p in folder_paths:
                        dirs.append(p[0] if isinstance(p, list) else str(p))
                    lines.append(f"  {folder}: {', '.join(dirs)}")
            return ToolResult.success(truncate_output("\n".join(lines)))
        except Exception as e:
            return ToolResult.error(f"Failed to get folder paths: {e}")


class RefreshNodeIndexTool(Tool):
    """Refresh the node index by re-fetching from ComfyUI."""

    def __init__(self, client: ComfyUIPort, node_index: NodeIndex) -> None:
        self.client = client
        self.index = node_index

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="comfyui_refresh_index",
            description="Rebuild the node index after installing custom nodes.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            old_count = self.index.node_count
            await self.index.build(self.client)
            new_count = self.index.node_count
            diff = new_count - old_count
            msg = f"Node index refreshed: {new_count} nodes in {len(self.index.categories)} categories"
            if diff > 0:
                msg += f" (+{diff} new nodes)"
            elif diff < 0:
                msg += f" ({diff} nodes removed)"
            return ToolResult.success(msg)
        except Exception as e:
            return ToolResult.error(f"Failed to refresh node index: {e}")
