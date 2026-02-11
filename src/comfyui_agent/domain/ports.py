"""Port interfaces for dependency inversion.

Domain layer defines these interfaces (ports).
Infrastructure layer implements them (adapters).
This ensures the domain layer has zero external dependencies.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from comfyui_agent.domain.models.context import IdentityFeature
from comfyui_agent.domain.models.events import Event, EventType


class ComfyUIPort(Protocol):
    """Interface for ComfyUI communication."""

    async def get_system_stats(self) -> dict[str, Any]: ...
    async def get_object_info(self, node_class: str | None = None) -> dict[str, Any]: ...
    async def get_queue(self) -> dict[str, Any]: ...
    async def get_history(self, prompt_id: str | None = None) -> dict[str, Any]: ...
    async def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]: ...
    async def interrupt(self) -> None: ...
    async def list_models(self, folder: str) -> list[str]: ...
    async def health_check(self) -> bool: ...
    def get_image_url(self, filename: str, subfolder: str, folder_type: str) -> str: ...
    async def upload_image(self, image_data: bytes, filename: str, overwrite: bool = False) -> dict[str, Any]: ...
    async def get_folder_paths(self) -> dict[str, Any]: ...
    async def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> None: ...
    async def close(self) -> None: ...
    async def connect_ws(self) -> None: ...

    # --- ComfyUI Manager integration ---
    async def manager_available(self) -> bool:
        """Check if ComfyUI Manager extension is installed and reachable."""
        ...

    async def manager_install_model(
        self, name: str, url: str, filename: str, save_path: str, model_type: str = "checkpoint",
    ) -> dict[str, Any]:
        """Install a model via ComfyUI Manager's /model/install endpoint.

        Manager handles the download internally (supports aria2 for large files).
        This is a blocking call — returns only after download completes.
        """
        ...

    async def manager_install_node(
        self, node_id: str, version: str = "latest", channel: str = "default", mode: str = "default",
    ) -> dict[str, Any]:
        """Install a custom node via ComfyUI Manager's /customnode/install endpoint.

        Uses Manager's unified install system (CNR packages, nightly, git URLs).
        """
        ...

    async def manager_get_node_list(self, mode: str = "default") -> dict[str, Any]:
        """Get the list of available custom nodes from Manager."""
        ...

    async def manager_reboot(self) -> None:
        """Request ComfyUI restart via Manager's /manager/reboot endpoint."""
        ...


class LLMPort(Protocol):
    """Interface for LLM communication."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Any: ...

    async def close(self) -> None: ...


class SessionPort(Protocol):
    """Interface for session persistence."""

    async def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None: ...
    async def load_messages(self, session_id: str) -> list[dict[str, Any]]: ...
    async def list_sessions(self) -> list[dict[str, Any]]: ...
    async def create_session(self, title: str) -> str: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def close(self) -> None: ...
    async def append_message(self, session_id: str, role: str, content: Any) -> int: ...
    async def load_messages_from(self, session_id: str, from_id: int = 0) -> list[dict[str, Any]]: ...
    async def get_session_meta(self, session_id: str) -> dict[str, Any]: ...
    async def update_session_meta(self, session_id: str, **kwargs: Any) -> None: ...
    async def create_child_session(self, parent_id: str, title: str) -> str: ...


EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBusPort(Protocol):
    """Interface for event pub/sub communication."""

    def on(self, event_type: EventType, handler: EventHandler) -> Callable[[], None]: ...
    def on_prefix(self, prefix: str, handler: EventHandler) -> Callable[[], None]: ...
    def on_all(self, handler: EventHandler) -> Callable[[], None]: ...
    async def emit(self, event: Event) -> None: ...
    def emit_sync(self, event: Event) -> None: ...


class IdentityPort(Protocol):
    """Interface for role identity loading and experience persistence.

    Abstracts the RoleX file system storage, allowing the application
    layer to load identity features and save experiences without
    knowing the storage details.
    """

    def load_identity(self, role_name: str) -> list[IdentityFeature]: ...
    def save_experience(self, role_name: str, exp_name: str, gherkin_source: str) -> None: ...


class WebPort(Protocol):
    """Interface for web fetching and searching.

    Abstracts HTTP fetching and web search API calls,
    allowing tools to access the web without knowing
    the underlying HTTP client or search provider.
    """

    async def fetch_url(self, url: str, timeout: int = 30) -> dict[str, Any]:
        """Fetch content from a URL.

        Returns dict with keys: content, content_type, status_code, url.
        """
        ...

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search the web for a query.

        Returns list of dicts with keys: title, url, snippet.
        """
        ...

    async def search_registry(self, node_id: str) -> dict[str, Any] | None:
        """Look up a custom node package on the Comfy Registry (api.comfy.org).

        Returns node metadata dict or None if not found.
        """
        ...

    async def close(self) -> None: ...


