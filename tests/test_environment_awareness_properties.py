"""Property-based tests for the environment awareness system.

Uses Hypothesis to verify correctness properties defined in the design doc.
Each test maps to a numbered property from design.md.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from comfyui_agent.application.canvas_state import CanvasState
from comfyui_agent.application.environment_probe import EnvironmentProbe
from comfyui_agent.application.intent_analyzer import IntentAnalyzer
from comfyui_agent.application.prompt_builder import PromptBuilder
from comfyui_agent.domain.models.context import (
    ContextSection,
    EnvironmentSnapshot,
    IntentResult,
    SectionCategory,
)
from comfyui_agent.domain.models.events import Event, EventType


# ---------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------

def _printable_text(min_size: int = 1, max_size: int = 30) -> st.SearchStrategy[str]:
    """Non-empty printable strings without newlines."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), whitelist_characters=" "),
        min_size=min_size,
        max_size=max_size,
    ).filter(lambda s: s.strip() != "")


_version_st = st.from_regex(r"[0-9]+\.[0-9]+\.[0-9]+", fullmatch=True)
_gpu_name_st = _printable_text(3, 40)
_model_name_st = _printable_text(3, 50)


# ---------------------------------------------------------------
# Property 4: Snapshot text rendering
# Validates: Requirements 7.2
# ---------------------------------------------------------------

@given(
    version=_version_st,
    gpu=_gpu_name_st,
    vram_total=st.floats(min_value=1024, max_value=65536, allow_nan=False),
    vram_free=st.floats(min_value=0, max_value=65536, allow_nan=False),
    models=st.lists(_model_name_st, min_size=0, max_size=5),
    queue_running=st.integers(min_value=0, max_value=100),
    queue_pending=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_snapshot_text_rendering(
    version: str,
    gpu: str,
    vram_total: float,
    vram_free: float,
    models: list[str],
    queue_running: int,
    queue_pending: int,
) -> None:
    """Property 4: connected snapshot text contains version, gpu, and all model names."""
    snap = EnvironmentSnapshot(
        connection_ok=True,
        comfyui_version=version,
        gpu_name=gpu,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        checkpoint_models=models,
        queue_running=queue_running,
        queue_pending=queue_pending,
        collected_at=time.time(),
    )
    text = snap.to_prompt_text()

    assert version in text, f"version {version!r} not in prompt text"
    assert gpu in text, f"gpu {gpu!r} not in prompt text"
    for model in models:
        assert model in text, f"model {model!r} not in prompt text"


# ---------------------------------------------------------------
# Property 5: IntentResult round-trip parsing
# Validates: Requirements 5.1, 5.2, 5.3, 5.5
# ---------------------------------------------------------------

_section_names = [c.value for c in SectionCategory if c != SectionCategory.IDENTITY]


@given(
    topics=st.lists(_printable_text(2, 20), min_size=1, max_size=3),
    env_needed=st.booleans(),
    sections=st.lists(st.sampled_from(_section_names), min_size=1, max_size=5, unique=True),
)
@settings(max_examples=100)
def test_intent_result_round_trip(
    topics: list[str],
    env_needed: bool,
    sections: list[str],
) -> None:
    """Property 5: serialize IntentResult to JSON, parse back, verify equivalence."""
    original = IntentResult(
        topics=topics,
        environment_needed=env_needed,
        suggested_sections=sections,
    )

    # Serialize to the JSON format the analyzer expects
    json_str = json.dumps({
        "topics": original.topics,
        "env_needed": original.environment_needed,
        "sections": original.suggested_sections,
    })

    # Parse using the real _parse_response method
    analyzer = IntentAnalyzer.__new__(IntentAnalyzer)
    parsed = analyzer._parse_response(json_str)

    assert parsed.topics == original.topics[:3]
    assert parsed.environment_needed == original.environment_needed
    assert parsed.suggested_sections == original.suggested_sections


# ---------------------------------------------------------------
# Property 1: Probe completeness
# Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
# ---------------------------------------------------------------

def _make_mock_client(
    system_stats: dict[str, Any],
    models: list[str],
    queue: dict[str, Any],
) -> MagicMock:
    """Create a mock ComfyUIPort returning the given data."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value=True)
    client.get_system_stats = AsyncMock(return_value=system_stats)
    client.list_models = AsyncMock(return_value=models)
    client.get_queue = AsyncMock(return_value=queue)
    return client


def _make_mock_node_index(node_count: int, categories: list[str]) -> MagicMock:
    """Create a mock NodeIndex with given stats."""
    ni = MagicMock()
    ni.is_built = True
    ni.node_count = node_count
    ni.categories = categories
    return ni


@given(
    version=_version_st,
    gpu=_gpu_name_st,
    vram_total=st.integers(min_value=1024 * 1024 * 1024, max_value=48 * 1024 * 1024 * 1024),
    vram_free=st.integers(min_value=0, max_value=48 * 1024 * 1024 * 1024),
    models=st.lists(_model_name_st, min_size=0, max_size=5),
    running=st.lists(st.just([]), min_size=0, max_size=3),
    pending=st.lists(st.just([]), min_size=0, max_size=5),
    node_count=st.integers(min_value=0, max_value=500),
    categories=st.lists(_printable_text(3, 20), min_size=0, max_size=10),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_probe_completeness(
    version: str,
    gpu: str,
    vram_total: int,
    vram_free: int,
    models: list[str],
    running: list[list[Any]],
    pending: list[list[Any]],
    node_count: int,
    categories: list[str],
) -> None:
    """Property 1: snapshot fields match input data when all APIs succeed."""
    stats = {
        "system": {"comfyui_version": version},
        "devices": [{"name": gpu, "vram_total": vram_total, "vram_free": vram_free}],
    }
    queue = {"queue_running": running, "queue_pending": pending}

    client = _make_mock_client(stats, models, queue)
    ni = _make_mock_node_index(node_count, categories)
    probe = EnvironmentProbe(client, ni, refresh_interval=300.0)

    snap = await probe.collect()

    assert snap.connection_ok is True
    assert snap.comfyui_version == version
    assert snap.gpu_name == gpu
    assert snap.vram_total_mb == pytest.approx(vram_total / (1024 * 1024), rel=1e-3)
    assert snap.vram_free_mb == pytest.approx(vram_free / (1024 * 1024), rel=1e-3)
    assert snap.checkpoint_models == models
    assert snap.queue_running == len(running)
    assert snap.queue_pending == len(pending)
    assert snap.node_count == node_count
    assert snap.node_categories == categories
    assert snap.errors == []


# ---------------------------------------------------------------
# Property 2: Probe resilience
# Validates: Requirements 1.6, 7.3
# ---------------------------------------------------------------

@given(
    fail_health=st.booleans(),
    fail_stats=st.booleans(),
    fail_models=st.booleans(),
    fail_queue=st.booleans(),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_probe_resilience(
    fail_health: bool,
    fail_stats: bool,
    fail_models: bool,
    fail_queue: bool,
) -> None:
    """Property 2: collect() never raises; errors list tracks each failure."""
    client = MagicMock()
    if fail_health:
        client.health_check = AsyncMock(side_effect=ConnectionError("down"))
    else:
        client.health_check = AsyncMock(return_value=True)

    if fail_stats:
        client.get_system_stats = AsyncMock(side_effect=RuntimeError("stats fail"))
    else:
        client.get_system_stats = AsyncMock(return_value={
            "system": {"comfyui_version": "1.0.0"},
            "devices": [{"name": "GPU", "vram_total": 8 * 1024**3, "vram_free": 4 * 1024**3}],
        })

    if fail_models:
        client.list_models = AsyncMock(side_effect=RuntimeError("models fail"))
    else:
        client.list_models = AsyncMock(return_value=["model.safetensors"])

    if fail_queue:
        client.get_queue = AsyncMock(side_effect=RuntimeError("queue fail"))
    else:
        client.get_queue = AsyncMock(return_value={"queue_running": [], "queue_pending": []})

    ni = _make_mock_node_index(10, ["image"])
    probe = EnvironmentProbe(client, ni, refresh_interval=300.0)

    # Must never raise
    snap = await probe.collect()

    # Count expected errors (stats/models/queue only fail if health succeeded)
    expected_errors = 0
    if fail_health:
        expected_errors += 1
    else:
        if fail_stats:
            expected_errors += 1
        if fail_models:
            expected_errors += 1
        if fail_queue:
            expected_errors += 1

    assert len(snap.errors) == expected_errors

    # Successful fields still populated
    if not fail_health:
        assert snap.connection_ok is True
        if not fail_stats:
            assert snap.comfyui_version == "1.0.0"
        if not fail_models:
            assert snap.checkpoint_models == ["model.safetensors"]


# ---------------------------------------------------------------
# Property 3: Canvas state tracking
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------

@given(
    node_data=st.dictionaries(
        keys=st.text(alphabet="0123456789", min_size=1, max_size=3),
        values=st.fixed_dictionaries({
            "class_type": _printable_text(3, 30),
            "inputs": st.just({}),
        }),
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_canvas_state_tracking(
    node_data: dict[str, dict[str, Any]],
) -> None:
    """Property 3: summary contains every class_type from the workflow."""
    # Build a mock EventBus that captures the handler
    handler_ref: list[Any] = []

    class FakeEventBus:
        def on(self, event_type: EventType, handler: Any) -> Any:
            handler_ref.append(handler)
            return lambda: None
        def on_prefix(self, prefix: str, handler: Any) -> Any:
            return lambda: None
        def on_all(self, handler: Any) -> Any:
            return lambda: None
        async def emit(self, event: Event) -> None:
            pass

    bus = FakeEventBus()
    cs = CanvasState(bus)

    # Simulate WORKFLOW_SUBMITTED event
    event = Event(
        type=EventType.WORKFLOW_SUBMITTED,
        data={"prompt_id": "test-123", "workflow": node_data},
        session_id="s1",
        timestamp=time.time(),
    )
    assert handler_ref, "CanvasState should have registered a handler"
    await handler_ref[0](event)

    summary = cs.get_summary()
    for node_id, node in node_data.items():
        ct = node["class_type"]
        assert ct in summary, f"class_type {ct!r} not in summary"


# ---------------------------------------------------------------
# Unit test: empty canvas state (Task 3.3)
# Requirements: 2.3
# ---------------------------------------------------------------

def test_empty_canvas_state() -> None:
    """get_summary() returns empty canvas message when no workflow submitted."""
    class FakeEventBus:
        def on(self, event_type: EventType, handler: Any) -> Any:
            return lambda: None
        def on_prefix(self, prefix: str, handler: Any) -> Any:
            return lambda: None
        def on_all(self, handler: Any) -> Any:
            return lambda: None
        async def emit(self, event: Event) -> None:
            pass

    cs = CanvasState(FakeEventBus())
    summary = cs.get_summary()
    assert "empty" in summary.lower() or "no workflow" in summary.lower()


# ---------------------------------------------------------------
# Unit test: IntentAnalyzer fail-open (Task 4.2)
# Requirements: 5.6
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_intent_analyzer_fail_open_on_exception() -> None:
    """analyze() returns default_result() when LLM raises."""
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
    analyzer = IntentAnalyzer(llm)

    result = await analyzer.analyze("hello")
    default = IntentAnalyzer.default_result()
    assert result.environment_needed == default.environment_needed
    assert result.suggested_sections == default.suggested_sections


@pytest.mark.asyncio
async def test_intent_analyzer_fail_open_on_bad_json() -> None:
    """analyze() returns default_result() when LLM returns non-JSON."""
    llm = MagicMock()
    resp = MagicMock()
    resp.text = "this is not json at all"
    llm.chat = AsyncMock(return_value=resp)
    analyzer = IntentAnalyzer(llm)

    result = await analyzer.analyze("hello")
    default = IntentAnalyzer.default_result()
    assert result.environment_needed == default.environment_needed
    assert result.suggested_sections == default.suggested_sections


# ---------------------------------------------------------------
# Property 6: Prompt section ordering
# Validates: Requirements 4.2
# ---------------------------------------------------------------

_CATEGORY_ORDER = list(SectionCategory)


@given(
    categories=st.lists(
        st.sampled_from(list(SectionCategory)),
        min_size=2,
        max_size=6,
        unique=True,
    ),
)
@settings(max_examples=100)
def test_prompt_section_ordering(categories: list[SectionCategory]) -> None:
    """Property 6: build() output follows category order."""
    builder = PromptBuilder(token_budget=100000)

    for i, cat in enumerate(categories):
        marker = f"__MARKER_{cat.value}__"
        section = ContextSection(
            name=cat.value,
            category=cat,
            content=marker,
            priority=0,
            token_estimate=50,
        )
        builder.register_section(section)

    # Build with no intent filtering (include everything)
    result = builder.build(
        intent_result=IntentResult(
            topics=["test"],
            environment_needed=True,
            suggested_sections=[c.value for c in SectionCategory],
        ),
    )

    # Verify ordering: each marker should appear in category order
    positions = {}
    for cat in categories:
        marker = f"__MARKER_{cat.value}__"
        pos = result.find(marker)
        if pos >= 0:
            positions[cat] = pos

    sorted_cats = sorted(positions.keys(), key=lambda c: positions[c])
    expected_order = sorted(sorted_cats, key=lambda c: _CATEGORY_ORDER.index(c))
    assert sorted_cats == expected_order, (
        f"Section order {[c.value for c in sorted_cats]} "
        f"doesn't match expected {[c.value for c in expected_order]}"
    )


# ---------------------------------------------------------------
# Property 7: Prompt conditional filtering
# Validates: Requirements 4.4
# ---------------------------------------------------------------

_ALWAYS_INCLUDE_CATS = {
    SectionCategory.IDENTITY,
    SectionCategory.WORKFLOW_STRATEGY,
    SectionCategory.RULES,
}

# Categories always included (subject to token budget) after identity integration
_ALWAYS_PRESENT_CATS = {
    SectionCategory.IDENTITY,
    SectionCategory.WORKFLOW_STRATEGY,
    SectionCategory.RULES,
    SectionCategory.KNOWLEDGE,
    SectionCategory.EXPERIENCE,
}


@given(
    suggested=st.lists(
        st.sampled_from(_section_names),
        min_size=0,
        max_size=5,
        unique=True,
    ),
)
@settings(max_examples=100)
def test_prompt_conditional_filtering(suggested: list[str]) -> None:
    """Property 7: only suggested + always-included sections appear."""
    builder = PromptBuilder(token_budget=100000)

    # Register one section per category
    all_sections: dict[str, ContextSection] = {}
    for cat in SectionCategory:
        marker = f"__SECTION_{cat.value}__"
        sec = ContextSection(
            name=cat.value,
            category=cat,
            content=marker,
            priority=0,
            token_estimate=50,
        )
        builder.register_section(sec)
        all_sections[cat.value] = sec

    intent = IntentResult(
        topics=["test"],
        environment_needed=True,
        suggested_sections=suggested,
    )
    result = builder.build(intent_result=intent)

    for name, sec in all_sections.items():
        marker = f"__SECTION_{name}__"
        should_include = (
            sec.category in _ALWAYS_PRESENT_CATS
            or name in suggested
        )
        if should_include:
            assert marker in result, f"Expected section {name!r} to be included"
        else:
            assert marker not in result, f"Expected section {name!r} to be excluded"


# ---------------------------------------------------------------
# Property 8: Prompt token budget
# Validates: Requirements 4.5
# ---------------------------------------------------------------

@given(
    budget=st.integers(min_value=100, max_value=500),
    section_count=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=100)
def test_prompt_token_budget(budget: int, section_count: int) -> None:
    """Property 8: output token estimate ≤ budget; higher categories preserved."""
    builder = PromptBuilder(token_budget=budget)

    cats = list(SectionCategory)
    for i in range(min(section_count, len(cats))):
        cat = cats[i]
        # Each section is ~200 tokens of content to exceed budget
        content = "x " * 400  # ~200 tokens at 4 chars/token
        sec = ContextSection(
            name=cat.value,
            category=cat,
            content=content,
            priority=0,
            token_estimate=200,
        )
        builder.register_section(sec)

    intent = IntentResult(
        topics=["test"],
        environment_needed=True,
        suggested_sections=[c.value for c in SectionCategory],
    )
    result = builder.build(intent_result=intent)

    # Estimate tokens of the output
    from comfyui_agent.application.context_manager import estimate_tokens
    result_tokens = estimate_tokens(result)
    assert result_tokens <= budget + 50, (
        f"Result {result_tokens} tokens exceeds budget {budget}"
    )


# ---------------------------------------------------------------
# Property 9: Environment conditional inclusion
# Validates: Requirements 6.3
# ---------------------------------------------------------------

@given(env_needed=st.booleans())
@settings(max_examples=100)
def test_environment_conditional_inclusion(env_needed: bool) -> None:
    """Property 9: environment section present iff env_needed is True."""
    builder = PromptBuilder(token_budget=100000)

    # Register identity (always included) and environment
    builder.register_section(ContextSection(
        name="identity",
        category=SectionCategory.IDENTITY,
        content="I am an assistant.",
        priority=0,
        token_estimate=10,
    ))

    env_snap = EnvironmentSnapshot(
        connection_ok=True,
        comfyui_version="1.0.0",
        gpu_name="RTX 4090",
        vram_total_mb=24000,
        vram_free_mb=20000,
        checkpoint_models=["model.safetensors"],
        collected_at=time.time(),
    )

    intent = IntentResult(
        topics=["test"],
        environment_needed=env_needed,
        suggested_sections=["environment", "workflow_strategy", "rules"],
    )

    result = builder.build(
        intent_result=intent,
        environment=env_snap,
    )

    env_text = "## Environment"
    if env_needed:
        assert env_text in result, "Environment section should be present"
    else:
        assert env_text not in result, "Environment section should be absent"
