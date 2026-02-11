"""Experience synthesizer — learns from practice (Kantian Synthesis).

Monitors agent events for learning opportunities and persists
experiences as Gherkin .feature files via IdentityPort.

Three learning layers:
  Layer 1 (Passive): Tool failure patterns, validation recovery, workflow tracking
  Layer 2 (Active): Post-conversation self-bootstrapping via LLM reflection
  Layer 3 (Hot-load): New experiences immediately registered to PromptBuilder

Every conversation triggers self-bootstrapping — the agent reflects on what
happened, extracts experience in Gherkin format (RoleX synthesize), and
injects it into the prompt for the next conversation.

This is a posteriori learning — from concrete encounters to structured experience.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.domain.ports import EventBusPort, IdentityPort

if TYPE_CHECKING:
    from comfyui_agent.application.prompt_builder import PromptBuilder
    from comfyui_agent.domain.ports import LLMPort

logger = logging.getLogger(__name__)

# Minimum interval between experience saves (seconds)
_SAVE_COOLDOWN = 120  # 2 minutes — tighter than before for faster learning

# Max tokens for reflection LLM call — generous to allow rich Gherkin output
_REFLECTION_MAX_TOKENS = 2000

# RoleX synthesize methodology — injected into the reflection prompt
# so the LLM knows HOW to write experiences in Gherkin format
_ROLEX_SYNTHESIZE_GUIDE = """\
You are performing Kantian Synthesis (综合) — transforming a raw encounter into \
structured experience. Write your reflection as a Gherkin Feature file following \
the RoleX experience format:

```gherkin
Feature: <Experience Title — what was learned>
  <Optional: one-line context about why this matters>

  Scenario: <Specific lesson or pattern discovered>
    Given <the situation or context>
    When <what happened or what action was taken>
    Then <what was learned or what the outcome was>
    And <additional insight or implication>
```

Rules:
- Feature name should be a clear, reusable lesson title
- Each Scenario captures ONE concrete learning
- Given/When/Then should be specific, not generic
- Include node names, connection types, or parameter values when relevant
- Multiple Scenarios are OK if the conversation had multiple learnings
- Focus on ComfyUI workflow patterns, node combinations, user preferences, \
or error recovery strategies
"""

_REFLECTION_PROMPT = """\
Review this completed ComfyUI agent conversation and extract learnings.

{synthesize_guide}

Conversation context:
- Tool calls: {tool_count}
- Tools used: {tools_used}
- Duration: {duration:.1f}s
- Errors: {error_count}
{workflow_info}{correction_info}

Based on this conversation, write a Gherkin experience Feature.
If the conversation was trivial (simple greeting, no real work), respond with \
exactly "NONE"."""


class ExperienceSynthesizer:
    """Detects learning opportunities and persists experiences via RoleX.

    Self-bootstrapping: every conversation triggers reflection → experience
    extraction → hot-load into PromptBuilder. The agent gets smarter with
    each conversation.
    """

    def __init__(
        self,
        identity_port: IdentityPort,
        event_bus: EventBusPort,
        role_name: str,
        llm: LLMPort | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._identity = identity_port
        self._role_name = role_name
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._last_save_time = 0.0

        # Per-session tracking
        self._tool_failures: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._validation_failures: dict[str, str] = {}
        self._session_stats: dict[str, _SessionStats] = {}

        # Subscribe to events
        event_bus.on(EventType.STATE_TOOL_FAILED, self._on_tool_failed)
        event_bus.on(EventType.STATE_TOOL_COMPLETED, self._on_tool_completed)
        event_bus.on(EventType.WORKFLOW_SUBMITTED, self._on_workflow_submitted)
        event_bus.on(EventType.TURN_END, self._on_turn_end)
        event_bus.on(EventType.MESSAGE_USER, self._on_user_message)

    # ------------------------------------------------------------------
    # Layer 1: Passive learning — event tracking
    # ------------------------------------------------------------------

    async def _on_tool_failed(self, event: Event) -> None:
        """Track tool failures for pattern detection."""
        sid = event.session_id
        tool_name = event.data.get("tool_name", "")
        error = event.data.get("error", "")

        self._tool_failures[sid].append({"tool": tool_name, "error": error[:200]})
        stats = self._ensure_stats(sid)
        stats.error_count += 1
        stats.tools_used.add(tool_name)
        stats.add_event(f"✗ {tool_name}: {error[:200]}")

        if "validate" in tool_name:
            self._validation_failures[sid] = error[:300]

    async def _on_tool_completed(self, event: Event) -> None:
        """Track tool completions and detect recovery patterns."""
        sid = event.session_id
        tool_name = event.data.get("tool_name", "")
        result_text = str(event.data.get("result", ""))[:200]

        stats = self._ensure_stats(sid)
        stats.tool_count += 1
        stats.tools_used.add(tool_name)
        stats.add_event(f"✓ {tool_name}: {result_text}")

        # Validation recovery: fail → success
        if "validate" in tool_name and sid in self._validation_failures:
            prev_error = self._validation_failures.pop(sid)
            self._save_and_hotload(
                f"validation-recovery-{int(time.time())}",
                self._format_validation_experience(prev_error),
            )

        if sid in self._tool_failures:
            self._tool_failures[sid].clear()

    async def _on_workflow_submitted(self, event: Event) -> None:
        """Record workflow node composition for reflection."""
        sid = event.session_id
        workflow = event.data.get("workflow", {})
        stats = self._ensure_stats(sid)

        if isinstance(workflow, dict):
            for node in workflow.values():
                if isinstance(node, dict) and "class_type" in node:
                    stats.workflow_nodes.append(node["class_type"])
        stats.workflow_submitted = True

    async def _on_user_message(self, event: Event) -> None:
        """Detect user correction signals for reflection context."""
        sid = event.session_id
        content = event.data.get("content", "")
        correction_signals = [
            "不要", "不对", "错了", "应该", "别这样", "换一个",
            "wrong", "don't", "should", "instead", "not what",
        ]
        if any(s in content.lower() for s in correction_signals):
            self._ensure_stats(sid).user_corrections += 1

    # ------------------------------------------------------------------
    # Layer 2: Active self-bootstrapping — post-conversation reflection
    # ------------------------------------------------------------------

    async def _on_turn_end(self, event: Event) -> None:
        """Self-bootstrap: reflect only when the conversation is worth learning from.

        Trigger conditions (any one is enough):
        - A workflow was submitted (node combination worth remembering)
        - User corrected the agent (preference/mistake worth recording)
        - Errors occurred but were recovered (error pattern worth noting)
        - 5+ tool calls (complex interaction, likely has learnings)

        Simple queries and greetings are skipped — no wasted LLM calls.
        """
        sid = event.session_id
        stats = self._session_stats.get(sid)
        if not stats:
            self._cleanup_session(sid)
            return

        duration = event.data.get("duration", 0)

        # Decide whether this conversation is worth reflecting on
        worth_reflecting = (
            stats.workflow_submitted
            or stats.user_corrections > 0
            or (stats.error_count > 0 and stats.tool_count > stats.error_count)
            or stats.tool_count >= 5
        )

        if not worth_reflecting:
            logger.debug("Skipping reflection: conversation not notable enough")
            self._cleanup_session(sid)
            return

        # Active reflection via LLM
        if self._llm:
            try:
                await self._reflect_with_llm(stats, duration)
            except Exception as exc:
                logger.warning("Self-bootstrap reflection failed: %s", exc)

        self._cleanup_session(sid)

    async def _reflect_with_llm(
        self, stats: _SessionStats, duration: float
    ) -> None:
        """Use LLM to generate structured experience from conversation."""
        workflow_info = ""
        if stats.workflow_nodes:
            unique_nodes = sorted(set(stats.workflow_nodes))
            workflow_info = f"- Workflow nodes used: {', '.join(unique_nodes)}\n"

        correction_info = ""
        if stats.user_corrections > 0:
            correction_info = f"- User corrections detected: {stats.user_corrections}\n"

        prompt = _REFLECTION_PROMPT.format(
            synthesize_guide=_ROLEX_SYNTHESIZE_GUIDE,
            tool_count=stats.tool_count,
            tools_used=", ".join(sorted(stats.tools_used)) or "none",
            duration=duration,
            error_count=stats.error_count,
            workflow_info=workflow_info,
            correction_info=correction_info,
        )

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            system="You are a concise experience recorder for a ComfyUI workflow agent. "
                   "Output only valid Gherkin Feature text, or exactly NONE.",
            max_tokens=_REFLECTION_MAX_TOKENS,
        )

        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        if text.upper() == "NONE" or not text.startswith("Feature:"):
            logger.debug("Reflection: no notable experience extracted")
            return

        name = f"reflection-{int(time.time())}"
        self._save_and_hotload(name, text)
        logger.info("Self-bootstrap: experience extracted and hot-loaded: %s", name)

    # ------------------------------------------------------------------
    # Layer 3: Save + hot-load into PromptBuilder
    # ------------------------------------------------------------------

    def _save_and_hotload(self, name: str, gherkin: str) -> None:
        """Persist experience to RoleX and inject into current PromptBuilder."""
        now = time.time()
        if now - self._last_save_time < _SAVE_COOLDOWN:
            logger.debug("Experience save skipped (cooldown: %ds remaining)",
                         int(_SAVE_COOLDOWN - (now - self._last_save_time)))
            return

        # Save to filesystem via IdentityPort
        try:
            self._identity.save_experience(self._role_name, name, gherkin)
            self._last_save_time = now
            logger.info("Experience persisted: %s", name)
        except Exception as exc:
            logger.warning("Failed to persist experience: %s", exc)
            return

        # Hot-load: register to PromptBuilder for immediate effect
        if self._prompt_builder:
            from comfyui_agent.domain.models.context import (
                ContextSection,
                SectionCategory,
            )
            section = ContextSection(
                name=f"experience_{name}",
                category=SectionCategory.EXPERIENCE,
                content=gherkin,
                priority=99,  # Low priority — trimmed first under token budget
            )
            self._prompt_builder.register_section(section)
            logger.info("Hot-loaded experience into prompt: %s", name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_stats(self, session_id: str) -> _SessionStats:
        if session_id not in self._session_stats:
            self._session_stats[session_id] = _SessionStats()
        return self._session_stats[session_id]

    def _cleanup_session(self, session_id: str) -> None:
        self._session_stats.pop(session_id, None)
        self._tool_failures.pop(session_id, None)
        self._validation_failures.pop(session_id, None)

    @staticmethod
    def _format_validation_experience(error: str) -> str:
        return (
            "Feature: Workflow Validation Recovery\n"
            "  Scenario: Validation error corrected\n"
            f"    Given a workflow validation failed with: {error}\n"
            "    When the workflow was corrected and re-validated\n"
            "    Then the validation succeeded\n"
            "    And this error pattern should be avoided in future workflows\n"
        )


class _SessionStats:
    """Per-session statistics for reflection."""

    __slots__ = (
        "tool_count", "error_count", "tools_used",
        "workflow_nodes", "workflow_submitted", "user_corrections",
        "key_events",
    )

    # Cap key_events to avoid unbounded growth
    _MAX_KEY_EVENTS = 15

    def __init__(self) -> None:
        self.tool_count: int = 0
        self.error_count: int = 0
        self.tools_used: set[str] = set()
        self.workflow_nodes: list[str] = []
        self.workflow_submitted: bool = False
        self.user_corrections: int = 0
        self.key_events: list[str] = []

    def add_event(self, summary: str) -> None:
        """Append a key event summary, respecting the cap."""
        if len(self.key_events) < self._MAX_KEY_EVENTS:
            self.key_events.append(summary)
