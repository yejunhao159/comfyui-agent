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
            sections = [
                s for s in sections
                if s.category in _ALWAYS_INCLUDE
                or s.name in suggested
                or s.category.value in suggested
            ]
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
    """Create the default ContextSection set from the original static prompt.

    Splits the monolithic prompt into structured sections matching
    SectionCategory ordering.
    """
    return [
        ContextSection(
            name="identity",
            category=SectionCategory.IDENTITY,
            content=(
                "You are a ComfyUI assistant. You help users create, manage, "
                "and debug ComfyUI workflows through natural language.\n\n"
                'Use the `comfyui` tool with {"action": "<name>", "params": {...}} '
                "format. See the tool description for available actions."
            ),
            priority=0,
        ),
        ContextSection(
            name="workflow_strategy",
            category=SectionCategory.WORKFLOW_STRATEGY,
            content=(
                "## Workflow Building Strategy\n\n"
                "Think in LINKS first, then convert to JSON.\n\n"
                "Step 1: Plan the node chain using link notation:\n"
                "  CheckpointLoaderSimple_0 --MODEL--> KSampler_0.model\n"
                "  CheckpointLoaderSimple_0 --CLIP--> CLIPTextEncode_0.clip\n"
                "  CheckpointLoaderSimple_0 --CLIP--> CLIPTextEncode_1.clip\n"
                "  CLIPTextEncode_0 --CONDITIONING--> KSampler_0.positive\n"
                "  CLIPTextEncode_1 --CONDITIONING--> KSampler_0.negative\n"
                "  EmptyLatentImage_0 --LATENT--> KSampler_0.latent_image\n"
                "  KSampler_0 --LATENT--> VAEDecode_0.samples\n"
                "  CheckpointLoaderSimple_0 --VAE--> VAEDecode_0.vae\n"
                "  VAEDecode_0 --IMAGE--> SaveImage_0.images\n\n"
                "Step 2: Convert to API JSON format:\n"
                "  Each unique NodeType_N becomes a node entry with a string ID.\n"
                "  Each link becomes an input reference: [source_node_id, output_index].\n\n"
                "Use get_connectable(output_type) to check which nodes can produce "
                "or consume a given type.\n\n"
                "## Workflow Building Process\n\n"
                '1. Search for relevant nodes: comfyui(action="search_nodes", '
                'params={"query": "..."})\n'
                "2. Check type compatibility: comfyui(action=\"get_connectable\", "
                'params={"output_type": "MODEL"})\n'
                "3. Get node details for KEY nodes only (checkpoint loader, sampler) "
                "— skip simple nodes like CLIPTextEncode, EmptyLatentImage, "
                "VAEDecode, SaveImage\n"
                "4. Plan the link chain, then build workflow in API format\n"
                '5. Validate: comfyui(action="validate_workflow", '
                'params={"workflow": {...}})\n'
                '6. Submit: comfyui(action="queue_prompt", '
                'params={"workflow": {...}})\n'
                "7. IMMEDIATELY give a final text response to the user "
                "— do NOT call more tools after queue_prompt"
            ),
            priority=0,
        ),
        ContextSection(
            name="tool_reference",
            category=SectionCategory.TOOL_REFERENCE,
            content=(
                "## ComfyUI Workflow API Format\n\n"
                "A workflow is a dict of node_id -> {class_type, inputs}.\n"
                "Node connections use [source_node_id, output_index] format.\n\n"
                "Example txt2img:\n"
                "{\n"
                '  "1": {"class_type": "CheckpointLoaderSimple", '
                '"inputs": {"ckpt_name": "model.safetensors"}},\n'
                '  "2": {"class_type": "CLIPTextEncode", '
                '"inputs": {"text": "a photo of a cat", "clip": ["1", 1]}},\n'
                '  "3": {"class_type": "CLIPTextEncode", '
                '"inputs": {"text": "bad quality", "clip": ["1", 1]}},\n'
                '  "4": {"class_type": "EmptyLatentImage", '
                '"inputs": {"width": 1024, "height": 1024, "batch_size": 1}},\n'
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
                "}\n\n"
                "## CRITICAL: When to Stop Calling Tools\n\n"
                "After queue_prompt succeeds, you MUST immediately give a "
                "final text response:\n"
                "- Tell the user the workflow was submitted\n"
                "- Mention the prompt_id so they can track it\n"
                "- Describe what the workflow will produce\n"
                "- Do NOT call any more tools after queue_prompt succeeds\n\n"
                "Other stopping conditions:\n"
                "- After answering a question with text, just respond\n"
                "- If you're unsure what to do next, ask the user\n"
                "- After 5 tool calls, summarize what you've done and respond\n\n"
                "NEVER call tools endlessly. Your goal is to help the user, "
                "not to keep calling tools."
            ),
            priority=0,
        ),
        ContextSection(
            name="rules",
            category=SectionCategory.RULES,
            content=(
                "## Rules\n\n"
                "- Always search_nodes and get_node_detail before using a "
                "node type you're unsure about\n"
                "- Always validate_workflow before queue_prompt\n"
                "- Use the actual model names from list_models, not guessed names\n"
                "- Node connections: [node_id_string, output_index_int]\n"
                "- After install_custom_node, use refresh_index to update "
                "the node index\n"
                "- Be efficient: combine what you know, don't call "
                "get_node_detail for every single node"
            ),
            priority=0,
        ),
        ContextSection(
            name="error_handling",
            category=SectionCategory.ERROR_HANDLING,
            content=(
                "## Error Handling\n\n"
                "- If a tool call fails, analyze the error and try a "
                "DIFFERENT approach — do NOT repeat the same call\n"
                "- If validate_workflow fails, fix the specific error "
                "mentioned, then re-validate ONCE\n"
                "- If queue_prompt fails, explain the error to the user "
                "and ask if they want to retry\n"
                "- Never call the same tool more than 3 times in a row "
                "— if stuck, explain the situation to the user\n"
                "- When an execution error occurs, check get_history for "
                "details before attempting fixes"
            ),
            priority=0,
        ),
    ]

