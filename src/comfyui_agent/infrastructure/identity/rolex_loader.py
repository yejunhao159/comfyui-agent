"""RoleX identity loader — reads .rolex identity features from filesystem.

Implements IdentityPort by scanning the RoleX directory structure:
  {rolex_dir}/roles/{role_name}/identity/*.identity.feature

Uses lightweight regex parsing for Gherkin — no external parser needed.
Inspired by RoleX LocalPlatform's detectIdentityType() and identity() methods.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from comfyui_agent.domain.models.context import (
    ContextSection,
    IdentityFeature,
    IdentityType,
    SectionCategory,
)

logger = logging.getLogger(__name__)

# Regex for extracting Feature name from Gherkin
_FEATURE_RE = re.compile(r"^\s*Feature:\s*(.+)$", re.MULTILINE)


def _detect_identity_type(filename: str) -> IdentityType:
    """Detect identity type from filename suffix.

    Follows RoleX convention:
      persona.identity.feature → PERSONA
      *.knowledge.identity.feature → KNOWLEDGE
      *.experience.identity.feature → EXPERIENCE
      *.voice.identity.feature → VOICE
    """
    if filename == "persona.identity.feature":
        return IdentityType.PERSONA
    if filename.endswith(".knowledge.identity.feature"):
        return IdentityType.KNOWLEDGE
    if filename.endswith(".experience.identity.feature"):
        return IdentityType.EXPERIENCE
    if filename.endswith(".voice.identity.feature"):
        return IdentityType.VOICE
    # Default to knowledge for unrecognized patterns
    return IdentityType.KNOWLEDGE


def _extract_feature_name(content: str) -> str:
    """Extract the Feature name from Gherkin text."""
    match = _FEATURE_RE.search(content)
    return match.group(1).strip() if match else "unnamed"


class RolexIdentityLoader:
    """Loads RoleX identity features from the filesystem.

    Implements IdentityPort protocol. Reads .identity.feature files
    from the RoleX directory and converts them to IdentityFeature objects.
    """

    def __init__(self, rolex_dir: str = "~/.rolex") -> None:
        self._rolex_dir = Path(rolex_dir).expanduser()

    def load_identity(self, role_name: str) -> list[IdentityFeature]:
        """Load all identity features for a role.

        Scans {rolex_dir}/roles/{role_name}/identity/*.identity.feature
        and parses each file into an IdentityFeature.
        """
        identity_dir = self._rolex_dir / "roles" / role_name / "identity"
        if not identity_dir.exists():
            logger.warning("Identity dir not found: %s", identity_dir)
            return []

        features: list[IdentityFeature] = []
        for path in sorted(identity_dir.iterdir()):
            if not path.name.endswith(".identity.feature"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
                id_type = _detect_identity_type(path.name)
                name = _extract_feature_name(content)
                features.append(
                    IdentityFeature(
                        type=id_type,
                        name=name,
                        content=content,
                        source_file=str(path),
                    )
                )
                logger.debug("Loaded identity: %s (%s)", name, id_type.value)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        logger.info(
            "Loaded %d identity features for role '%s'",
            len(features), role_name,
        )
        return features

    def save_experience(
        self, role_name: str, exp_name: str, gherkin_source: str
    ) -> None:
        """Save an experience feature to the identity directory.

        Writes to {rolex_dir}/roles/{role_name}/identity/{exp_name}.experience.identity.feature
        """
        identity_dir = self._rolex_dir / "roles" / role_name / "identity"
        identity_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{exp_name}.experience.identity.feature"
        path = identity_dir / filename
        path.write_text(gherkin_source, encoding="utf-8")
        logger.info("Saved experience: %s → %s", exp_name, path)


def features_to_sections(
    features: list[IdentityFeature],
    role_name: str = "",
) -> list[ContextSection]:
    """Convert IdentityFeature list to ContextSection list for PromptBuilder.

    Mapping:
      persona → IDENTITY category, priority=0 (highest)
      voice   → IDENTITY category, priority=1
      knowledge → KNOWLEDGE category, priority by sort order
      experience → EXPERIENCE category, priority by sort order

    When role_name is provided and a persona feature exists, an additional
    identity directive section is injected to instruct the LLM to prefix
    responses with [RoleName] and embody the persona.
    """
    sections: list[ContextSection] = []
    knowledge_idx = 0
    experience_idx = 0
    has_persona = False

    for feature in features:
        if feature.type == IdentityType.PERSONA:
            has_persona = True
            sections.append(ContextSection(
                name=f"identity_persona_{feature.name}",
                category=SectionCategory.IDENTITY,
                content=feature.content,
                priority=0,
            ))
        elif feature.type == IdentityType.VOICE:
            sections.append(ContextSection(
                name=f"identity_voice_{feature.name}",
                category=SectionCategory.IDENTITY,
                content=feature.content,
                priority=1,
            ))
        elif feature.type == IdentityType.KNOWLEDGE:
            sections.append(ContextSection(
                name=f"knowledge_{feature.name}",
                category=SectionCategory.KNOWLEDGE,
                content=feature.content,
                priority=knowledge_idx,
            ))
            knowledge_idx += 1
        elif feature.type == IdentityType.EXPERIENCE:
            sections.append(ContextSection(
                name=f"experience_{feature.name}",
                category=SectionCategory.EXPERIENCE,
                content=feature.content,
                priority=experience_idx,
            ))
            experience_idx += 1

    # Inject identity directive when persona is loaded
    if has_persona and role_name:
        # Capitalize first letter for display
        display_name = role_name[0].upper() + role_name[1:] if role_name else ""
        directive = (
            f"You have been given a persona identity above. "
            f"You MUST prefix every response with [{display_name}] "
            f"to indicate your active identity. "
            f"Embody this persona in your communication style, "
            f"thinking approach, and problem-solving methodology. "
            f"Your experiences and knowledge shape how you respond."
        )
        sections.append(ContextSection(
            name="identity_directive",
            category=SectionCategory.IDENTITY,
            content=directive,
            priority=2,  # After persona (0) and voice (1)
        ))

    return sections
