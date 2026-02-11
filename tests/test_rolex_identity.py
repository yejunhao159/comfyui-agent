"""Tests for RoleX identity loading and tool group dispatchers.

Covers:
- RolexIdentityLoader: file scanning, type detection, Gherkin parsing
- features_to_sections: IdentityFeature → ContextSection conversion
- Group dispatchers: routing, unknown action handling
- IdentityConfig: config loading with identity section
- ExperienceSynthesizer: event-driven experience detection
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from comfyui_agent.domain.models.context import (
    ContextSection,
    IdentityFeature,
    IdentityType,
    IntentResult,
    SectionCategory,
)
from comfyui_agent.infrastructure.identity.rolex_loader import (
    RolexIdentityLoader,
    _detect_identity_type,
    _extract_feature_name,
    features_to_sections,
)


# ---------------------------------------------------------------
# _detect_identity_type
# ---------------------------------------------------------------


def test_detect_persona() -> None:
    assert _detect_identity_type("persona.identity.feature") == IdentityType.PERSONA


def test_detect_knowledge() -> None:
    assert _detect_identity_type("python-async.knowledge.identity.feature") == IdentityType.KNOWLEDGE


def test_detect_experience() -> None:
    assert _detect_identity_type("auth-lessons.experience.identity.feature") == IdentityType.EXPERIENCE


def test_detect_voice() -> None:
    assert _detect_identity_type("casual.voice.identity.feature") == IdentityType.VOICE


def test_detect_unknown_defaults_to_knowledge() -> None:
    assert _detect_identity_type("something.identity.feature") == IdentityType.KNOWLEDGE


# ---------------------------------------------------------------
# _extract_feature_name
# ---------------------------------------------------------------


def test_extract_feature_name() -> None:
    content = "Feature: My Cool Feature\n  Scenario: test"
    assert _extract_feature_name(content) == "My Cool Feature"


def test_extract_feature_name_missing() -> None:
    assert _extract_feature_name("no feature here") == "unnamed"


# ---------------------------------------------------------------
# RolexIdentityLoader
# ---------------------------------------------------------------


@pytest.fixture
def rolex_dir() -> Path:
    """Create a temporary .rolex directory with test identity files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        identity_dir = Path(tmpdir) / "roles" / "test-role" / "identity"
        identity_dir.mkdir(parents=True)

        # Persona
        (identity_dir / "persona.identity.feature").write_text(
            "Feature: Test Persona\n"
            "  Scenario: I am a test\n"
            "    Given I exist\n"
            "    Then I test things\n"
        )

        # Knowledge
        (identity_dir / "python.knowledge.identity.feature").write_text(
            "Feature: Python Knowledge\n"
            "  Scenario: I know Python\n"
            "    Given Python is a language\n"
            "    Then I can write it\n"
        )

        # Experience
        (identity_dir / "debug-lesson.experience.identity.feature").write_text(
            "Feature: Debug Lesson\n"
            "  Scenario: Learned debugging\n"
            "    Given a bug occurred\n"
            "    Then I learned to check logs\n"
        )

        yield Path(tmpdir)


def test_load_identity(rolex_dir: Path) -> None:
    loader = RolexIdentityLoader(rolex_dir=str(rolex_dir))
    features = loader.load_identity("test-role")

    assert len(features) == 3
    types = {f.type for f in features}
    assert types == {IdentityType.PERSONA, IdentityType.KNOWLEDGE, IdentityType.EXPERIENCE}


def test_load_identity_missing_role(rolex_dir: Path) -> None:
    loader = RolexIdentityLoader(rolex_dir=str(rolex_dir))
    features = loader.load_identity("nonexistent")
    assert features == []


def test_save_experience(rolex_dir: Path) -> None:
    loader = RolexIdentityLoader(rolex_dir=str(rolex_dir))
    loader.save_experience("test-role", "new-exp", "Feature: New Experience\n  Scenario: test\n")

    saved = rolex_dir / "roles" / "test-role" / "identity" / "new-exp.experience.identity.feature"
    assert saved.exists()
    assert "New Experience" in saved.read_text()


def test_save_experience_creates_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = RolexIdentityLoader(rolex_dir=tmpdir)
        loader.save_experience("new-role", "exp1", "Feature: Exp\n")

        saved = Path(tmpdir) / "roles" / "new-role" / "identity" / "exp1.experience.identity.feature"
        assert saved.exists()


# ---------------------------------------------------------------
# features_to_sections
# ---------------------------------------------------------------


def test_features_to_sections_mapping() -> None:
    features = [
        IdentityFeature(type=IdentityType.PERSONA, name="Persona", content="persona content"),
        IdentityFeature(type=IdentityType.VOICE, name="Voice", content="voice content"),
        IdentityFeature(type=IdentityType.KNOWLEDGE, name="Python", content="python content"),
        IdentityFeature(type=IdentityType.EXPERIENCE, name="Debug", content="debug content"),
    ]
    sections = features_to_sections(features)

    assert len(sections) == 4

    # Check categories
    cats = {s.name: s.category for s in sections}
    assert cats["identity_persona_Persona"] == SectionCategory.IDENTITY
    assert cats["identity_voice_Voice"] == SectionCategory.IDENTITY
    assert cats["knowledge_Python"] == SectionCategory.KNOWLEDGE
    assert cats["experience_Debug"] == SectionCategory.EXPERIENCE

    # Check priorities
    priorities = {s.name: s.priority for s in sections}
    assert priorities["identity_persona_Persona"] == 0
    assert priorities["identity_voice_Voice"] == 1


def test_features_to_sections_empty() -> None:
    assert features_to_sections([]) == []


# ---------------------------------------------------------------
# Group Dispatchers
# ---------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.get_system_stats = AsyncMock(return_value={"devices": []})
    client.list_models = AsyncMock(return_value=["model1.safetensors"])
    client.get_queue = AsyncMock(return_value={"queue_running": [], "queue_pending": []})
    client.get_history = AsyncMock(return_value={})
    client.queue_prompt = AsyncMock(return_value={"prompt_id": "test-123"})
    client.interrupt = AsyncMock()
    client.upload_image = AsyncMock(return_value={"name": "test.png"})
    client.get_folder_paths = AsyncMock(return_value={})
    client.free_memory = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.get_object_info = AsyncMock(return_value={})
    return client


@pytest.fixture
def mock_node_index() -> MagicMock:
    idx = MagicMock()
    idx.search = MagicMock(return_value="search results")
    idx.get_detail = MagicMock(return_value="detail results")
    idx.get_connectable = MagicMock(return_value="connectable results")
    idx.validate_workflow = MagicMock(return_value="valid")
    idx.list_categories = MagicMock(return_value="categories")
    idx.list_category = MagicMock(return_value="category nodes")
    idx.get_type_summary = MagicMock(return_value="type summary")
    idx.node_count = 100
    idx.categories = ["loaders", "sampling"]
    return idx


@pytest.mark.asyncio
async def test_discovery_dispatcher(mock_client: MagicMock, mock_node_index: MagicMock) -> None:
    from comfyui_agent.domain.tools.dispatchers import DiscoveryDispatcher

    d = DiscoveryDispatcher(mock_client, mock_node_index)
    info = d.info()
    assert info.name == "comfyui_discover"
    assert "search_nodes" in info.parameters["properties"]["action"]["enum"]

    result = await d.run({"action": "search_nodes", "params": {"query": "sampler"}})
    assert not result.is_error


@pytest.mark.asyncio
async def test_execute_dispatcher(mock_client: MagicMock, mock_node_index: MagicMock) -> None:
    from comfyui_agent.domain.tools.dispatchers import ExecuteDispatcher

    d = ExecuteDispatcher(mock_client, mock_node_index)
    info = d.info()
    assert info.name == "comfyui_execute"
    assert "queue_prompt" in info.parameters["properties"]["action"]["enum"]
    assert "interrupt" in info.parameters["properties"]["action"]["enum"]


@pytest.mark.asyncio
async def test_monitor_dispatcher(mock_client: MagicMock, mock_node_index: MagicMock) -> None:
    from comfyui_agent.domain.tools.dispatchers import MonitorDispatcher

    d = MonitorDispatcher(mock_client, mock_node_index)
    info = d.info()
    assert info.name == "comfyui_monitor"
    assert "system_stats" in info.parameters["properties"]["action"]["enum"]

    result = await d.run({"action": "system_stats", "params": {}})
    assert not result.is_error


@pytest.mark.asyncio
async def test_manage_dispatcher(mock_client: MagicMock, mock_node_index: MagicMock) -> None:
    from comfyui_agent.domain.tools.dispatchers import ManageDispatcher

    d = ManageDispatcher(mock_client, mock_node_index)
    info = d.info()
    assert info.name == "comfyui_manage"
    assert "upload_image" in info.parameters["properties"]["action"]["enum"]
    assert "free_memory" in info.parameters["properties"]["action"]["enum"]


@pytest.mark.asyncio
async def test_dispatcher_unknown_action(mock_client: MagicMock, mock_node_index: MagicMock) -> None:
    from comfyui_agent.domain.tools.dispatchers import DiscoveryDispatcher

    d = DiscoveryDispatcher(mock_client, mock_node_index)
    result = await d.run({"action": "nonexistent", "params": {}})
    assert result.is_error
    assert "Unknown action" in result.text


# ---------------------------------------------------------------
# IdentityConfig
# ---------------------------------------------------------------


def test_identity_config_defaults() -> None:
    from comfyui_agent.infrastructure.config import IdentityConfig

    cfg = IdentityConfig()
    assert cfg.rolex_dir == "~/.rolex"
    assert cfg.role_name == ""


def test_app_config_has_identity() -> None:
    from comfyui_agent.infrastructure.config import AppConfig

    cfg = AppConfig()
    assert cfg.identity.role_name == ""
    assert cfg.identity.rolex_dir == "~/.rolex"


# ---------------------------------------------------------------
# IntentResult knowledge_tags
# ---------------------------------------------------------------


def test_intent_result_knowledge_tags() -> None:
    result = IntentResult(
        topics=["workflow"],
        knowledge_tags=["comfyui", "python"],
    )
    assert result.knowledge_tags == ["comfyui", "python"]


def test_intent_result_default_knowledge_tags() -> None:
    result = IntentResult()
    assert result.knowledge_tags == []


# ---------------------------------------------------------------
# PromptBuilder KNOWLEDGE filtering with knowledge_tags
# ---------------------------------------------------------------


def test_prompt_builder_knowledge_tag_filtering() -> None:
    from comfyui_agent.application.prompt_builder import PromptBuilder

    builder = PromptBuilder(token_budget=100000)

    # Register identity (always included)
    builder.register_section(ContextSection(
        name="identity",
        category=SectionCategory.IDENTITY,
        content="identity",
        priority=0,
        token_estimate=10,
    ))

    # Register two knowledge sections
    builder.register_section(ContextSection(
        name="knowledge_python",
        category=SectionCategory.KNOWLEDGE,
        content="python knowledge",
        priority=0,
        token_estimate=10,
    ))
    builder.register_section(ContextSection(
        name="knowledge_comfyui",
        category=SectionCategory.KNOWLEDGE,
        content="comfyui knowledge",
        priority=1,
        token_estimate=10,
    ))

    # With knowledge_tags filtering
    intent = IntentResult(
        topics=["test"],
        environment_needed=False,
        suggested_sections=["workflow_strategy", "rules"],
        knowledge_tags=["python"],
    )
    result = builder.build(intent_result=intent)

    assert "python knowledge" in result
    assert "comfyui knowledge" not in result


def test_prompt_builder_no_knowledge_tags_includes_all() -> None:
    from comfyui_agent.application.prompt_builder import PromptBuilder

    builder = PromptBuilder(token_budget=100000)

    builder.register_section(ContextSection(
        name="identity",
        category=SectionCategory.IDENTITY,
        content="identity",
        priority=0,
        token_estimate=10,
    ))
    builder.register_section(ContextSection(
        name="knowledge_python",
        category=SectionCategory.KNOWLEDGE,
        content="python knowledge",
        priority=0,
        token_estimate=10,
    ))
    builder.register_section(ContextSection(
        name="knowledge_comfyui",
        category=SectionCategory.KNOWLEDGE,
        content="comfyui knowledge",
        priority=1,
        token_estimate=10,
    ))

    # No knowledge_tags → all knowledge included
    intent = IntentResult(
        topics=["test"],
        environment_needed=False,
        suggested_sections=["workflow_strategy", "rules"],
        knowledge_tags=[],
    )
    result = builder.build(intent_result=intent)

    assert "python knowledge" in result
    assert "comfyui knowledge" in result


# ---------------------------------------------------------------------------
# RegistrySearchTool tests
# ---------------------------------------------------------------------------


class TestRegistrySearchTool:
    """Tests for RegistrySearchTool — Comfy Registry package lookup."""

    def _make_tool(self, web: MagicMock | None = None) -> "RegistrySearchTool":
        from comfyui_agent.domain.tools.registry_search import RegistrySearchTool

        if web is None:
            web = AsyncMock()
        return RegistrySearchTool(web)

    def test_info_returns_correct_name(self) -> None:
        tool = self._make_tool()
        info = tool.info()
        assert info.name == "comfy_registry"
        assert "node_id" in info.parameters["properties"]
        assert "node_id" in info.parameters["required"]

    @pytest.mark.asyncio
    async def test_run_success(self) -> None:
        web = AsyncMock()
        web.search_registry.return_value = {
            "id": "comfyui-impact-pack",
            "name": "ComfyUI Impact Pack",
            "description": "A powerful node pack",
            "downloads": 50000,
            "github_stars": 1200,
            "repository": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
            "license": "MIT",
            "status": "active",
            "latest_version": {"version": "4.5.0", "dependencies": ["dep-a"]},
            "tags": ["segmentation", "detection"],
        }
        tool = self._make_tool(web)
        result = await tool.run({"node_id": "comfyui-impact-pack"})

        assert not result.is_error
        assert "comfyui-impact-pack" in result.text
        assert "50000" in result.text
        assert "1200" in result.text
        assert "4.5.0" in result.text
        assert "segmentation" in result.text

    @pytest.mark.asyncio
    async def test_run_not_found(self) -> None:
        web = AsyncMock()
        web.search_registry.return_value = None
        tool = self._make_tool(web)
        result = await tool.run({"node_id": "nonexistent-pack"})

        assert not result.is_error
        assert "not found" in result.text.lower()

    @pytest.mark.asyncio
    async def test_run_empty_node_id(self) -> None:
        tool = self._make_tool()
        result = await tool.run({"node_id": ""})
        assert result.is_error
        assert "required" in result.text.lower()

    @pytest.mark.asyncio
    async def test_run_missing_node_id(self) -> None:
        tool = self._make_tool()
        result = await tool.run({})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_run_exception_handling(self) -> None:
        web = AsyncMock()
        web.search_registry.side_effect = RuntimeError("Connection refused")
        tool = self._make_tool(web)
        result = await tool.run({"node_id": "some-pack"})

        assert result.is_error
        assert "failed" in result.text.lower()


# ---------------------------------------------------------------------------
# create_all_tools includes registry tool
# ---------------------------------------------------------------------------


def test_create_all_tools_includes_registry() -> None:
    """Verify create_all_tools adds RegistrySearchTool when web is provided."""
    from comfyui_agent.domain.tools.factory import create_all_tools

    client = MagicMock()
    node_index = MagicMock()
    node_index.search.return_value = []
    node_index.get_detail.return_value = None
    node_index.get_connectable.return_value = {}
    node_index.validate_workflow.return_value = {"valid": True}
    web = AsyncMock()

    tools = create_all_tools(client, node_index, web=web)
    tool_names = [t.info().name for t in tools]

    assert "comfy_registry" in tool_names
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
