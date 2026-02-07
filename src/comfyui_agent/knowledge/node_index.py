"""Node index for on-demand discovery.

Builds a local index of ComfyUI nodes at startup, organized by category.
Supports keyword search, category browsing, and condensed detail views.
Avoids dumping all 100+ nodes into the LLM context at once.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from comfyui_agent.infrastructure.comfyui_client import ComfyUIClient

logger = logging.getLogger(__name__)


class NodeIndex:
    """In-memory index of ComfyUI nodes for fast search and browsing.

    Built once at startup from /api/object_info, then queried by tools.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}  # class_name → raw info
        self._by_category: dict[str, list[str]] = {}  # category → [class_names]
        self._search_corpus: dict[str, str] = {}  # class_name → searchable text
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    async def build(self, client: ComfyUIClient) -> None:
        """Fetch all node info from ComfyUI and build the index."""
        logger.info("Building node index...")
        try:
            all_info = await client.get_object_info()
        except Exception:
            logger.exception("Failed to fetch object_info")
            return

        self._nodes = all_info
        self._by_category.clear()
        self._search_corpus.clear()

        for class_name, info in all_info.items():
            # Index by category
            category = info.get("category", "uncategorized")
            self._by_category.setdefault(category, []).append(class_name)

            # Build search corpus: name + display_name + category + description
            display = info.get("display_name", class_name)
            desc = info.get("description", "")
            corpus = f"{class_name} {display} {category} {desc}".lower()
            self._search_corpus[class_name] = corpus

        self._built = True
        logger.info(
            "Node index built: %d nodes in %d categories",
            len(self._nodes),
            len(self._by_category),
        )

    def list_categories(self) -> str:
        """Return a summary of all categories with node counts."""
        if not self._built:
            return "Node index not built yet. ComfyUI may not be connected."
        lines = [f"Node categories ({len(self._by_category)}):"]
        for cat in sorted(self._by_category.keys()):
            nodes = self._by_category[cat]
            lines.append(f"  [{cat}] ({len(nodes)} nodes)")
        return "\n".join(lines)

    def list_category(self, category: str) -> str:
        """List all nodes in a specific category."""
        if not self._built:
            return "Node index not built yet."
        # Fuzzy match category name
        matched = None
        for cat in self._by_category:
            if cat.lower() == category.lower():
                matched = cat
                break
        if not matched:
            # Try partial match
            for cat in self._by_category:
                if category.lower() in cat.lower():
                    matched = cat
                    break
        if not matched:
            return f"Category '{category}' not found. Use search_nodes to find nodes."

        nodes = self._by_category[matched]
        lines = [f"Nodes in [{matched}] ({len(nodes)}):"]
        for name in sorted(nodes):
            info = self._nodes[name]
            display = info.get("display_name", name)
            lines.append(f"  - {name} ({display})")
        return "\n".join(lines)

    def search(self, query: str, limit: int = 20) -> str:
        """Search nodes by keyword. Matches against name, category, description."""
        if not self._built:
            return "Node index not built yet."
        query_lower = query.lower()
        terms = query_lower.split()

        scored: list[tuple[int, str]] = []
        for class_name, corpus in self._search_corpus.items():
            score = 0
            for term in terms:
                if term in corpus:
                    score += 1
                # Bonus for exact class name match
                if term in class_name.lower():
                    score += 2
            if score > 0:
                scored.append((score, class_name))

        scored.sort(key=lambda x: (-x[0], x[1]))
        results = scored[:limit]

        if not results:
            return f"No nodes found matching '{query}'."

        truncated = len(scored) > limit
        lines = [f"Search results for '{query}' ({len(scored)} matches, showing {len(results)}):"]
        for score, name in results:
            info = self._nodes[name]
            display = info.get("display_name", name)
            category = info.get("category", "")
            lines.append(f"  - {name} [{category}] ({display})")
        if truncated:
            lines.append(f"  ... {len(scored) - limit} more results. Refine your search.")
        return "\n".join(lines)

    def get_detail(self, class_name: str) -> str:
        """Get condensed detail for a specific node (not raw JSON)."""
        if not self._built:
            return "Node index not built yet."
        info = self._nodes.get(class_name)
        if not info:
            # Try case-insensitive match
            for name in self._nodes:
                if name.lower() == class_name.lower():
                    info = self._nodes[name]
                    class_name = name
                    break
        if not info:
            return f"Node '{class_name}' not found."

        lines = [f"Node: {class_name}"]
        lines.append(f"  Display: {info.get('display_name', class_name)}")
        lines.append(f"  Category: {info.get('category', 'unknown')}")
        if info.get("description"):
            lines.append(f"  Description: {info['description']}")

        # Inputs — condensed format
        input_info = info.get("input", {})
        required = input_info.get("required", {})
        optional = input_info.get("optional", {})

        if required:
            lines.append("  Required inputs:")
            for param_name, param_spec in required.items():
                lines.append(f"    {param_name}: {self._format_param(param_spec)}")

        if optional:
            lines.append("  Optional inputs:")
            for param_name, param_spec in optional.items():
                lines.append(f"    {param_name}: {self._format_param(param_spec)}")

        # Outputs
        output_names = info.get("output_name", [])
        output_types = info.get("output", [])
        if output_types:
            lines.append("  Outputs:")
            for i, otype in enumerate(output_types):
                oname = output_names[i] if i < len(output_names) else f"output_{i}"
                lines.append(f"    [{i}] {oname}: {otype}")

        return "\n".join(lines)

    def validate_workflow(self, workflow: dict[str, Any]) -> str:
        """Validate a workflow before submission."""
        if not self._built:
            return "Node index not built yet. Cannot validate."

        errors: list[str] = []
        warnings: list[str] = []

        for node_id, node_config in workflow.items():
            class_type = node_config.get("class_type", "")
            if not class_type:
                errors.append(f"Node {node_id}: missing class_type")
                continue

            if class_type not in self._nodes:
                errors.append(f"Node {node_id}: unknown class_type '{class_type}'")
                continue

            node_info = self._nodes[class_type]
            required_inputs = node_info.get("input", {}).get("required", {})
            provided_inputs = node_config.get("inputs", {})

            # Check required inputs
            for param_name in required_inputs:
                if param_name not in provided_inputs:
                    errors.append(
                        f"Node {node_id} ({class_type}): missing required input '{param_name}'"
                    )

            # Check for unknown inputs
            all_inputs = set(required_inputs.keys())
            optional_inputs = node_info.get("input", {}).get("optional", {})
            all_inputs.update(optional_inputs.keys())
            for param_name in provided_inputs:
                if param_name not in all_inputs:
                    warnings.append(
                        f"Node {node_id} ({class_type}): unknown input '{param_name}'"
                    )

        if not errors and not warnings:
            return f"Workflow valid: {len(workflow)} nodes, all checks passed."

        lines = []
        if errors:
            lines.append(f"Errors ({len(errors)}):")
            for e in errors:
                lines.append(f"  ✗ {e}")
        if warnings:
            lines.append(f"Warnings ({len(warnings)}):")
            for w in warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)

    def _format_param(self, param_spec: Any) -> str:
        """Format a parameter spec into a concise string."""
        if not isinstance(param_spec, (list, tuple)) or len(param_spec) == 0:
            return str(param_spec)

        type_info = param_spec[0]

        # It's a type reference like "MODEL", "CLIP", "LATENT"
        if isinstance(type_info, str):
            return type_info

        # It's a list of options (enum)
        if isinstance(type_info, list):
            if len(type_info) <= 5:
                return f"enum[{', '.join(str(x) for x in type_info)}]"
            return f"enum[{', '.join(str(x) for x in type_info[:3])}, ... ({len(type_info)} options)]"

        # It has constraints (min, max, default, etc.)
        if isinstance(type_info, str) and len(param_spec) > 1:
            constraints = param_spec[1] if len(param_spec) > 1 else {}
            parts = [type_info]
            if isinstance(constraints, dict):
                if "default" in constraints:
                    parts.append(f"default={constraints['default']}")
                if "min" in constraints:
                    parts.append(f"min={constraints['min']}")
                if "max" in constraints:
                    parts.append(f"max={constraints['max']}")
            return " ".join(parts)

        return str(param_spec)
