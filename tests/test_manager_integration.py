"""Tests for ComfyUI Manager integration in management tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfyui_agent.domain.tools.management import (
    DownloadModelTool,
    InstallCustomNodeTool,
    _extract_filename_from_url,
)


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock ComfyUIPort with Manager methods."""
    client = AsyncMock()
    client.manager_available = AsyncMock(return_value=True)
    client.manager_install_model = AsyncMock(return_value={"status": "ok"})
    client.manager_install_node = AsyncMock(
        return_value={"status": "ok", "message": "Installation success."}
    )
    client.get_folder_paths = AsyncMock(
        return_value={
            "checkpoints": [["/models/checkpoints"]],
            "loras": [["/models/loras"]],
            "controlnet": [["/models/controlnet"]],
            "custom_nodes": [["/custom_nodes"]],
        }
    )
    return client


@pytest.fixture
def mock_client_no_manager() -> AsyncMock:
    """Create a mock ComfyUIPort without Manager."""
    client = AsyncMock()
    client.manager_available = AsyncMock(return_value=False)
    client.get_folder_paths = AsyncMock(
        return_value={
            "checkpoints": [["/models/checkpoints"]],
            "loras": [["/models/loras"]],
            "custom_nodes": [["/custom_nodes"]],
        }
    )
    return client


# ============================================================
# DownloadModelTool tests
# ============================================================


class TestDownloadModelTool:
    """Tests for DownloadModelTool with Manager integration."""

    def test_info_describes_manager_delegation(self, mock_client: AsyncMock) -> None:
        tool = DownloadModelTool(mock_client)
        info = tool.info()
        assert info.name == "comfyui_download_model"
        assert "Manager" in info.description
        assert "url" in info.parameters["properties"]
        assert "folder" in info.parameters["properties"]

    async def test_requires_url_and_folder(self, mock_client: AsyncMock) -> None:
        tool = DownloadModelTool(mock_client)
        result = await tool.run({"url": "", "folder": ""})
        assert result.is_error
        assert "'url' and 'folder' are required" in result.text

    async def test_requires_filename_detectable(self, mock_client: AsyncMock) -> None:
        tool = DownloadModelTool(mock_client)
        result = await tool.run({"url": "https://example.com/noext", "folder": "checkpoints"})
        assert result.is_error
        assert "filename" in result.text.lower()

    async def test_delegates_to_manager_when_available(self, mock_client: AsyncMock) -> None:
        tool = DownloadModelTool(mock_client)
        result = await tool.run({
            "url": "https://huggingface.co/repo/resolve/main/model.safetensors",
            "folder": "checkpoints",
        })
        assert not result.is_error
        assert "Manager" in result.text
        assert "background" in result.text.lower()

    async def test_manager_download_fires_background_task(self, mock_client: AsyncMock) -> None:
        """Verify that Manager download is non-blocking (returns immediately)."""
        tool = DownloadModelTool(mock_client)
        result = await tool.run({
            "url": "https://huggingface.co/repo/resolve/main/big-model.safetensors",
            "folder": "checkpoints",
            "filename": "big-model.safetensors",
        })
        assert not result.is_error
        assert "list_models" in result.text  # Tells user how to check progress

    async def test_explicit_filename_used(self, mock_client: AsyncMock) -> None:
        tool = DownloadModelTool(mock_client)
        result = await tool.run({
            "url": "https://example.com/download",
            "folder": "checkpoints",
            "filename": "my-model.safetensors",
        })
        assert not result.is_error
        assert "my-model.safetensors" in result.text

    async def test_fallback_to_direct_when_no_manager(
        self, mock_client_no_manager: AsyncMock,
    ) -> None:
        """When Manager is not available, should attempt direct download."""
        tool = DownloadModelTool(mock_client_no_manager)
        # This will fail because we can't actually download, but it should
        # attempt the direct path (not the Manager path)
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content.iter_chunked = MagicMock(return_value=iter([]))
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            result = await tool.run({
                "url": "https://huggingface.co/repo/resolve/main/model.safetensors",
                "folder": "checkpoints",
            })
            # Should have tried direct download (not Manager)
            mock_client_no_manager.manager_install_model.assert_not_called()


# ============================================================
# InstallCustomNodeTool tests
# ============================================================


class TestInstallCustomNodeTool:
    """Tests for InstallCustomNodeTool with Manager integration."""

    def test_info_describes_manager_and_cnr(self, mock_client: AsyncMock) -> None:
        tool = InstallCustomNodeTool(mock_client)
        info = tool.info()
        assert info.name == "comfyui_install_custom_node"
        assert "CNR" in info.description or "Manager" in info.description
        assert "node_id" in info.parameters["properties"]

    async def test_requires_node_id(self, mock_client: AsyncMock) -> None:
        tool = InstallCustomNodeTool(mock_client)
        result = await tool.run({})
        assert result.is_error
        assert "node_id" in result.text

    async def test_installs_via_manager_with_cnr_id(self, mock_client: AsyncMock) -> None:
        tool = InstallCustomNodeTool(mock_client)
        result = await tool.run({"node_id": "comfyui-impact-pack"})
        assert not result.is_error
        assert "Manager" in result.text
        mock_client.manager_install_node.assert_called_once_with(
            node_id="comfyui-impact-pack",
            version="latest",
        )

    async def test_installs_via_manager_with_version(self, mock_client: AsyncMock) -> None:
        tool = InstallCustomNodeTool(mock_client)
        result = await tool.run({"node_id": "comfyui-impact-pack", "version": "1.2.3"})
        assert not result.is_error
        mock_client.manager_install_node.assert_called_once_with(
            node_id="comfyui-impact-pack",
            version="1.2.3",
        )

    async def test_backward_compat_git_url_param(self, mock_client: AsyncMock) -> None:
        """Old 'git_url' parameter should still work."""
        tool = InstallCustomNodeTool(mock_client)
        result = await tool.run({"git_url": "https://github.com/user/repo"})
        assert not result.is_error
        # Should use Manager with the git URL as node_id
        mock_client.manager_install_node.assert_called_once()

    async def test_cnr_id_without_manager_fails_gracefully(
        self, mock_client_no_manager: AsyncMock,
    ) -> None:
        """CNR package IDs require Manager — should fail with helpful message."""
        tool = InstallCustomNodeTool(mock_client_no_manager)
        result = await tool.run({"node_id": "comfyui-impact-pack"})
        assert result.is_error
        assert "Manager" in result.text
        assert "git URL" in result.text.lower() or "git URL" in result.text

    async def test_git_url_without_manager_uses_git_clone(
        self, mock_client_no_manager: AsyncMock,
    ) -> None:
        """Git URLs should work even without Manager (via git clone fallback)."""
        tool = InstallCustomNodeTool(mock_client_no_manager)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            with patch("pathlib.Path.exists", return_value=False):
                result = await tool.run({
                    "node_id": "https://github.com/user/ComfyUI-CustomNode"
                })
                # Should attempt git clone
                mock_exec.assert_called_once()
                args = mock_exec.call_args[0]
                assert args[0] == "git"
                assert args[1] == "clone"

    async def test_manager_install_failure_returns_error(self, mock_client: AsyncMock) -> None:
        mock_client.manager_install_node = AsyncMock(
            side_effect=RuntimeError("Manager install failed: conflict")
        )
        tool = InstallCustomNodeTool(mock_client)
        result = await tool.run({"node_id": "broken-node"})
        assert result.is_error
        assert "failed" in result.text.lower() or "Failed" in result.text

    async def test_restart_note_in_success(self, mock_client: AsyncMock) -> None:
        tool = InstallCustomNodeTool(mock_client)
        result = await tool.run({"node_id": "some-node"})
        assert not result.is_error
        assert "restart" in result.text.lower() or "Restart" in result.text


# ============================================================
# Helper function tests
# ============================================================


class TestExtractFilename:
    def test_huggingface_resolve_url(self) -> None:
        url = "https://huggingface.co/user/repo/resolve/main/model.safetensors"
        assert _extract_filename_from_url(url) == "model.safetensors"

    def test_huggingface_blob_url(self) -> None:
        url = "https://huggingface.co/user/repo/blob/main/model.safetensors"
        assert _extract_filename_from_url(url) == "model.safetensors"

    def test_civitai_url(self) -> None:
        url = "https://civitai.com/api/download/models/12345/model.safetensors"
        assert _extract_filename_from_url(url) == "model.safetensors"

    def test_direct_url(self) -> None:
        url = "https://example.com/files/model.safetensors"
        assert _extract_filename_from_url(url) == "model.safetensors"

    def test_url_with_query_params(self) -> None:
        url = "https://example.com/model.safetensors?token=abc"
        assert _extract_filename_from_url(url) == "model.safetensors"

    def test_no_extension_returns_empty(self) -> None:
        url = "https://example.com/noext"
        assert _extract_filename_from_url(url) == ""
