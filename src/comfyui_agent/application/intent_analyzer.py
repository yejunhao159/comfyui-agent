"""Intent pre-analyzer — lightweight LLM classification of user input.

Uses a single compact LLM call to determine:
- Topic tags (2-3 keywords)
- Whether environment info is needed
- Which context sections to inject

Fail-open: any error returns a default result that includes everything.
"""

from __future__ import annotations

import json
import logging

from comfyui_agent.domain.models.context import IntentResult, SectionCategory
from comfyui_agent.domain.ports import LLMPort

logger = logging.getLogger(__name__)

_ALL_SECTION_NAMES = [c.value for c in SectionCategory if c != SectionCategory.IDENTITY]

_ANALYSIS_PROMPT = """\
Classify this ComfyUI user message. Respond in JSON only.
{{"topics": ["tag1", "tag2"], "env_needed": true/false, "sections": ["section_name", ...]}}

Rules:
- topics: 2-3 keyword tags describing the intent
- env_needed: true if message asks about GPU, models, system status, or needs model names for workflow building
- sections: which context sections to include. Options: environment, workflow_strategy, tool_reference, rules, error_handling

Message: {user_input}"""


class IntentAnalyzer:
    """Pre-analyzes user intent before the main ReAct loop."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def analyze(self, user_input: str) -> IntentResult:
        """Analyze user input. Returns default_result() on any failure."""
        try:
            prompt = self._build_analysis_prompt(user_input)
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You are a classifier. Output JSON only, no explanation.",
            )
            return self._parse_response(response.text)
        except Exception as exc:
            logger.warning("Intent analysis failed, using defaults: %s", exc)
            return self.default_result()

    def _build_analysis_prompt(self, user_input: str) -> str:
        return _ANALYSIS_PROMPT.format(user_input=user_input)

    def _parse_response(self, response_text: str) -> IntentResult:
        """Parse LLM JSON response into IntentResult."""
        try:
            # Strip markdown code fences if present
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            data = json.loads(text)
            return IntentResult(
                topics=data.get("topics", [])[:3],
                environment_needed=bool(data.get("env_needed", True)),
                suggested_sections=data.get("sections", _ALL_SECTION_NAMES),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse intent response: %s", exc)
            return self.default_result()

    @staticmethod
    def default_result() -> IntentResult:
        """Fail-open default: include everything."""
        return IntentResult(
            topics=["general"],
            environment_needed=True,
            suggested_sections=list(_ALL_SECTION_NAMES),
        )
