"""Modular prompt builder — assembles system prompt from ContextSections.

Replaces the static prompt string with a structured, priority-ordered,
token-budget-aware prompt assembly pipeline.
"""

from __future__ import annotations

import logging

from comfyui_agent.application.context_manager import estimate_tokens
from comfyui_agent.domain.models.context import (
    ContextSection,
    EnvironmentSnapshot,
    IntentResult,
    SectionCategory,
)

logger = logging.getLogger(__name__)

# Categories that are always included regardless of intent filtering
_ALWAYS_INCLUDE = {
    SectionCategory.IDENTITY,
    SectionCategory.WORKFLOW_STRATEGY,
    SectionCategory.RULES,
}

# Knowledge is always included but subject to token budget trimming
_KNOWLEDGE_INCLUDE = {
    SectionCategory.KNOWLEDGE,
}

# Ordered list for rendering
_CATEGORY_ORDER = list(SectionCategory)


class PromptBuilder:
    """Assembles a system prompt from registered ContextSection instances."""

    def __init__(self, token_budget: int = 12000) -> None:
        self._token_budget = token_budget if token_budget > 0 else 12000
        self._sections: dict[str, ContextSection] = {}

    def register_section(self, section: ContextSection) -> None:
        """Register (or replace) a context section."""
        if not section.token_estimate:
            section.token_estimate = estimate_tokens(section.content)
        self._sections[section.name] = section

    def build(
        self,
        intent_result: IntentResult | None = None,
        environment: EnvironmentSnapshot | None = None,
        canvas_summary: str = "",
    ) -> str:
        """Assemble the final system prompt string.

        1. Collect all registered sections
        2. Inject dynamic content (environment, canvas)
        3. Filter by intent
        4. Sort by category order, then priority
        5. Enforce token budget
        6. Render with headers
        """
        sections = list(self._sections.values())

        # Inject dynamic environment section
        if environment:
            env_text = environment.to_prompt_text()
            env_section = ContextSection(
                name="environment",
                category=SectionCategory.ENVIRONMENT,
                content=env_text,
                priority=0,
                token_estimate=estimate_tokens(env_text),
            )
            # Replace or add
            sections = [s for s in sections if s.name != "environment"]
            sections.append(env_section)

        # Inject dynamic canvas section (append to environment)
        if canvas_summary and canvas_summary.strip():
            canvas_section = ContextSection(
                name="canvas",
                category=SectionCategory.ENVIRONMENT,
                content=canvas_summary,
                priority=1,
                token_estimate=estimate_tokens(canvas_summary),
            )
            sections = [s for s in sections if s.name != "canvas"]
            sections.append(canvas_section)

        # Filter by intent
        if intent_result:
            suggested = set(intent_result.suggested_sections)
            knowledge_tags = [t.lower() for t in intent_result.knowledge_tags]
            sections = [
                s for s in sections
                if s.category in _ALWAYS_INCLUDE
                or s.category in _KNOWLEDGE_INCLUDE
                or s.name in suggested
                or s.category.value in suggested
            ]
            # Filter KNOWLEDGE sections by knowledge_tags when tags are present
            if knowledge_tags:
                sections = [
                    s for s in sections
                    if s.category != SectionCategory.KNOWLEDGE
                    or any(tag in s.name.lower() for tag in knowledge_tags)
                ]
            # EXPERIENCE is always included (subject to token budget)
            # Re-add any experience sections that were filtered out
            all_experience = [
                s for s in list(self._sections.values())
                if s.category == SectionCategory.EXPERIENCE
            ]
            existing_names = {s.name for s in sections}
            for exp in all_experience:
                if exp.name not in existing_names:
                    sections.append(exp)
            # If environment not needed, drop environment sections
            if not intent_result.environment_needed:
                sections = [
                    s for s in sections
                    if s.category != SectionCategory.ENVIRONMENT
                ]

        # Sort: category order first, then priority within category
        cat_index = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
        sections.sort(key=lambda s: (cat_index.get(s.category, 99), s.priority))

        # Enforce token budget
        sections = self._apply_budget(sections)

        # Render
        if not sections:
            return "You are a ComfyUI assistant."

        parts: list[str] = []
        for section in sections:
            parts.append(section.content)

        return "\n\n".join(parts)

    def _apply_budget(
        self, sections: list[ContextSection]
    ) -> list[ContextSection]:
        """Keep sections within token budget, dropping lowest-priority last."""
        total = sum(s.token_estimate for s in sections)
        if total <= self._token_budget:
            return sections

        # Drop from the end (lowest category priority) until within budget
        kept: list[ContextSection] = []
        running = 0
        for section in sections:
            if running + section.token_estimate > self._token_budget:
                logger.info(
                    "Token budget: dropping section '%s' (%d tokens)",
                    section.name, section.token_estimate,
                )
                continue
            kept.append(section)
            running += section.token_estimate

        return kept



def create_default_sections() -> list[ContextSection]:
    """Create the default ContextSection set for the system prompt.

    Focuses on WHO the agent is and HOW it should behave.
    Tool-specific WHAT/WHEN details live in each tool's info().description.
    """
    return [
        ContextSection(
            name="identity",
            category=SectionCategory.IDENTITY,
            content=(
                "You are deepractice 生图助手, a ComfyUI workflow assistant. "
                "You help users create, manage, and debug ComfyUI image generation "
                "workflows through natural language conversation."
            ),
            priority=0,
        ),
        ContextSection(
            name="workflow_strategy",
            category=SectionCategory.WORKFLOW_STRATEGY,
            content=(
                "## Workflow Building — MANDATORY Steps\n\n"
                "When building or modifying a workflow, you MUST follow these steps IN ORDER. "
                "Do NOT skip steps or jump straight to writing JSON.\n\n"
                "### Step 1: Research (for unfamiliar workflow types)\n"
                "If the request involves advanced techniques (ControlNet, LoRA, Inpainting, "
                "IP-Adapter, AnimateDiff, etc.), use web_search to find reference workflows first:\n"
                "  web_search('comfyui workflow <technique>') → web_fetch(url) → study the design\n\n"
                "### Step 2: Discover nodes\n"
                "Call comfyui_discover → search_nodes to find the node types you need.\n"
                "Never assume node class_type names — always verify they exist.\n\n"
                "### Step 3: Get exact model filenames\n"
                "Call comfyui_monitor → list_models(folder='checkpoints') to get real filenames.\n"
                "NEVER guess model names. Use the exact string returned by list_models.\n"
                "Also list_models for loras, vae, controlnet etc. if needed.\n\n"
                "### Step 4: Inspect key nodes\n"
                "Call comfyui_discover → get_node_detail for complex nodes (KSampler, "
                "ControlNetApply, etc.) to learn their exact input names, types, and allowed values.\n\n"
                "### Step 5: Plan with Link Notation\n"
                "Write out the node chain as typed links BEFORE writing any JSON:\n"
                "  CheckpointLoaderSimple_0 --MODEL--> KSampler_0.model\n"
                "  CheckpointLoaderSimple_0 --CLIP--> CLIPTextEncode_0.clip\n"
                "  CLIPTextEncode_0 --CONDITIONING--> KSampler_0.positive\n"
                "  EmptyLatentImage_0 --LATENT--> KSampler_0.latent_image\n"
                "  KSampler_0 --LATENT--> VAEDecode_0.samples\n"
                "  CheckpointLoaderSimple_0 --VAE--> VAEDecode_0.vae\n"
                "  VAEDecode_0 --IMAGE--> SaveImage_0.images\n"
                "Each link: source_node --TYPE--> target_node.input_name\n"
                "This catches type mismatches before you write JSON.\n\n"
                "### Step 6: Build workflow JSON\n"
                "Convert the link plan to ComfyUI API format: {node_id: {class_type, inputs}}.\n"
                "Node connections use [source_node_id_string, output_index_int] format.\n\n"
                "### Step 7: Validate\n"
                "Call comfyui_discover → validate_workflow. Fix errors and re-validate ONCE.\n\n"
                "### Step 8: Submit\n"
                "Call comfyui_execute → queue_prompt. After success, IMMEDIATELY respond to the "
                "user with the prompt_id and what the workflow will produce. Do NOT call more tools.\n\n"
                "## ComfyUI Workflow API Format\n\n"
                "Example txt2img:\n"
                "{\n"
                '  "1": {"class_type": "CheckpointLoaderSimple", '
                '"inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},\n'
                '  "2": {"class_type": "CLIPTextEncode", '
                '"inputs": {"text": "a photo of a cat", "clip": ["1", 1]}},\n'
                '  "3": {"class_type": "CLIPTextEncode", '
                '"inputs": {"text": "bad quality", "clip": ["1", 1]}},\n'
                '  "4": {"class_type": "EmptyLatentImage", '
                '"inputs": {"width": 512, "height": 512, "batch_size": 1}},\n'
                '  "5": {"class_type": "KSampler", "inputs": '
                '{"model": ["1", 0], "positive": ["2", 0], '
                '"negative": ["3", 0], "latent_image": ["4", 0], '
                '"seed": 42, "steps": 20, "cfg": 7.0, '
                '"sampler_name": "euler", "scheduler": "normal", '
                '"denoise": 1.0}},\n'
                '  "6": {"class_type": "VAEDecode", '
                '"inputs": {"samples": ["5", 0], "vae": ["1", 2]}},\n'
                '  "7": {"class_type": "SaveImage", '
                '"inputs": {"images": ["6", 0], '
                '"filename_prefix": "output"}}\n'
                "}"
            ),
            priority=0,
        ),
        ContextSection(
            name="rules",
            category=SectionCategory.RULES,
            content=(
                "## Rules\n\n"
                "### Workflow Building Checklist\n"
                "Before calling queue_prompt, verify:\n"
                "- [ ] All model filenames come from list_models (never guessed)\n"
                "- [ ] Key nodes inspected via get_node_detail (KSampler, ControlNet, etc.)\n"
                "- [ ] Link Notation plan written out showing all connections\n"
                "- [ ] All connections use [node_id_string, output_index_int] format\n"
                "- [ ] validate_workflow passed\n\n"
                "### General Rules\n"
                "- Be efficient: combine what you know, don't over-call tools\n"
                "- After 5+ tool calls without resolution, summarize progress "
                "and ask the user for guidance\n"
                "- If a tool call fails, try a DIFFERENT approach — "
                "do NOT repeat the same call\n"
                "- Never call the same tool more than 3 times in a row\n"
                "- When stuck, explain the situation to the user\n"
                "- After queue_prompt succeeds, respond immediately — no more tool calls"
            ),
            priority=0,
        ),
    ]


