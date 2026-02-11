"""Environment awareness data models.

Domain-layer data structures for environment sensing, intent analysis,
and modular prompt construction. Pure data — no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SectionCategory(str, Enum):
    """Ordered categories for system prompt sections."""

    IDENTITY = "identity"
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"
    ENVIRONMENT = "environment"
    WORKFLOW_STRATEGY = "workflow_strategy"
    TOOL_REFERENCE = "tool_reference"
    RULES = "rules"
    ERROR_HANDLING = "error_handling"


class IdentityType(str, Enum):
    """Types of RoleX identity features."""

    PERSONA = "persona"
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"
    VOICE = "voice"


@dataclass
class EnvironmentSnapshot:
    """Point-in-time snapshot of the ComfyUI runtime environment."""

    connection_ok: bool = False
    comfyui_version: str = ""
    gpu_name: str = ""
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    checkpoint_models: list[str] = field(default_factory=list)
    queue_running: int = 0
    queue_pending: int = 0
    node_count: int = 0
    node_categories: list[str] = field(default_factory=list)
    collected_at: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render as human-readable text for system prompt injection."""
        lines = ["## Environment"]
        if not self.connection_ok:
            lines.append("⚠ ComfyUI is NOT connected")
            if self.errors:
                lines.append(f"Errors: {', '.join(self.errors)}")
            return "\n".join(lines)

        lines.append(f"- ComfyUI: v{self.comfyui_version}")
        lines.append(f"- GPU: {self.gpu_name}")
        lines.append(
            f"- VRAM: {self.vram_free_mb:.0f}MB free / {self.vram_total_mb:.0f}MB total"
        )
        lines.append(
            f"- Checkpoints: {', '.join(self.checkpoint_models) or 'none'}"
        )
        lines.append(
            f"- Queue: {self.queue_running} running, {self.queue_pending} pending"
        )
        lines.append(
            f"- Nodes: {self.node_count} types in "
            f"{len(self.node_categories)} categories"
        )
        if self.errors:
            lines.append(f"- Probe errors: {', '.join(self.errors)}")
        return "\n".join(lines)


@dataclass
class IntentResult:
    """Result of lightweight intent pre-analysis."""

    topics: list[str] = field(default_factory=list)
    environment_needed: bool = True
    suggested_sections: list[str] = field(default_factory=list)
    knowledge_tags: list[str] = field(default_factory=list)


@dataclass
class ContextSection:
    """An independent block within the system prompt."""

    name: str
    category: SectionCategory
    content: str
    priority: int = 0
    token_estimate: int = 0


@dataclass
class IdentityFeature:
    """A parsed RoleX identity feature file.

    Represents one .identity.feature file from the RoleX storage.
    The content field holds the original Gherkin text.
    """

    type: IdentityType
    name: str
    content: str
    source_file: str = ""
