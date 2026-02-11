"""Node index for on-demand discovery.

Builds a local index of ComfyUI nodes at startup, organized by category.
Supports keyword search, category browsing, and condensed detail views.
Avoids dumping all 100+ nodes into the LLM context at once.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from comfyui_agent.domain.ports import ComfyUIPort

logger = logging.getLogger(__name__)


class NodeIndex:
    """In-memory index of ComfyUI nodes for fast search and browsing.

    Built once at startup from /api/object_info, then queried by tools.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}  # class_name → raw info
        self._by_category: dict[str, list[str]] = {}  # category → [class_names]
        self._search_corpus: dict[str, str] = {}  # class_name → searchable text (legacy)
        # Structured search fields for weighted scoring
        self._search_fields: dict[str, _SearchFields] = {}
        # Inverted index: token → set of class_names (for O(k) lookup)
        self._inverted_index: dict[str, set[str]] = {}
        # Type compatibility indexes
        # type → [(class_name, output_index, output_name), ...]
        self._type_producers: dict[str, list[tuple[str, int, str]]] = {}
        # type → [(class_name, input_name), ...]
        self._type_consumers: dict[str, list[tuple[str, str]]] = {}
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

    async def build(self, client: ComfyUIPort) -> None:
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
        self._search_fields.clear()
        self._inverted_index.clear()
        self._type_producers.clear()
        self._type_consumers.clear()

        for class_name, info in all_info.items():
            # Index by category
            category = info.get("category") or "uncategorized"
            self._by_category.setdefault(category, []).append(class_name)

            # Build structured search fields
            # Use `or` instead of default arg — key may exist with None value
            display = info.get("display_name") or class_name
            desc = info.get("description") or ""
            fields = _SearchFields(
                class_name=class_name.lower(),
                display_name=display.lower(),
                category=category.lower(),
                description=desc.lower(),
            )
            self._search_fields[class_name] = fields

            # Legacy corpus (kept for backward compat with test fixtures)
            corpus = f"{class_name} {display} {category} {desc}".lower()
            self._search_corpus[class_name] = corpus

            # Build inverted index from all fields
            tokens = set(_tokenize(class_name) + _tokenize(display)
                         + _tokenize(category) + _tokenize(desc))
            for token in tokens:
                self._inverted_index.setdefault(token, set()).add(class_name)

            # Build type producer index from outputs
            self._index_outputs(class_name, info)
            # Build type consumer index from inputs
            self._index_inputs(class_name, info)

        self._built = True
        logger.info(
            "Node index built: %d nodes in %d categories, %d connection types",
            len(self._nodes),
            len(self._by_category),
            len(self._type_producers | self._type_consumers),
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
            display = info.get("display_name") or name
            lines.append(f"  - {name} ({display})")
        return "\n".join(lines)

    def search(self, query: str, limit: int = 20) -> str:
        """Search nodes by keyword with weighted scoring.

        Scoring weights (per term):
          - class_name exact match: 10
          - class_name contains term: 5
          - display_name contains term: 4
          - category contains term: 2
          - description contains term: 1

        Uses an inverted index for candidate selection (O(k) instead of O(n)),
        then scores only the candidates.
        """
        if not self._built:
            return "Node index not built yet."
        query_lower = query.lower()
        terms = query_lower.split()

        # Gather candidates from inverted index
        candidates: set[str] = set()
        for term in terms:
            # Exact token match
            if term in self._inverted_index:
                candidates.update(self._inverted_index[term])
            # Prefix/substring match on tokens (for partial queries)
            for token, names in self._inverted_index.items():
                if term in token or token in term:
                    candidates.update(names)

        if not candidates:
            return f"No nodes found matching '{query}'."

        # Score candidates with weighted fields
        scored: list[tuple[float, str]] = []
        for class_name in candidates:
            fields = self._search_fields.get(class_name)
            if not fields:
                continue
            score = 0.0
            for term in terms:
                # class_name: highest weight
                if term == fields.class_name:
                    score += 10
                elif term in fields.class_name:
                    score += 5
                # display_name
                if term in fields.display_name:
                    score += 4
                # category
                if term in fields.category:
                    score += 2
                # description
                if term in fields.description:
                    score += 1
            if score > 0:
                scored.append((score, class_name))

        scored.sort(key=lambda x: (-x[0], x[1]))
        results = scored[:limit]

        if not results:
            return f"No nodes found matching '{query}'."

        truncated = len(scored) > limit
        lines = [f"Search results for '{query}' ({len(scored)} matches, showing {len(results)}):"]
        for _score, name in results:
            info = self._nodes[name]
            display = info.get("display_name") or name
            category = info.get("category") or ""
            io = self._io_summary(info)
            lines.append(f"  - {name} [{category}] ({display}) — {io}")
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
        lines.append(f"  Display: {info.get('display_name') or class_name}")
        lines.append(f"  Category: {info.get('category') or 'unknown'}")
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

    def _index_outputs(self, class_name: str, info: dict[str, Any]) -> None:
        """Index a node's output types as producers."""
        output_types = info.get("output", [])
        output_names = info.get("output_name", [])
        for i, otype in enumerate(output_types):
            if not isinstance(otype, str):
                continue
            oname = output_names[i] if i < len(output_names) else f"output_{i}"
            self._type_producers.setdefault(otype, []).append(
                (class_name, i, oname)
            )

    def _index_inputs(self, class_name: str, info: dict[str, Any]) -> None:
        """Index a node's input types as consumers."""
        input_info = info.get("input", {})
        for section in ("required", "optional"):
            for param_name, param_spec in input_info.get(section, {}).items():
                if (
                    isinstance(param_spec, (list, tuple))
                    and len(param_spec) > 0
                    and isinstance(param_spec[0], str)
                    and param_spec[0].isupper()
                ):
                    self._type_consumers.setdefault(param_spec[0], []).append(
                        (class_name, param_name)
                    )

    def get_connectable(self, output_type: str, limit: int = 20) -> str:
        """Given an output type, list nodes that can receive it."""
        if not self._built:
            return "Node index not built yet."
        output_type = output_type.upper()

        consumers = self._type_consumers.get(output_type, [])
        producers = self._type_producers.get(output_type, [])

        if not consumers and not producers:
            return f"No nodes found for type '{output_type}'."

        lines = [f"Type: {output_type}"]

        if producers:
            lines.append(f"\n  Produced by ({len(producers)} nodes):")
            for class_name, idx, oname in producers[:limit]:
                display = self._nodes[class_name].get("display_name") or class_name
                lines.append(f"    {class_name} [{display}] → output[{idx}] {oname}")

        if consumers:
            lines.append(f"\n  Consumed by ({len(consumers)} nodes):")
            for class_name, input_name in consumers[:limit]:
                display = self._nodes[class_name].get("display_name") or class_name
                lines.append(f"    {class_name} [{display}] ← input.{input_name}")

        return "\n".join(lines)

    def get_type_summary(self) -> str:
        """Return a summary of all connection types and their node counts."""
        if not self._built:
            return "Node index not built yet."
        all_types = sorted(set(self._type_producers) | set(self._type_consumers))
        if not all_types:
            return "No connection types found."

        lines = [f"Connection types ({len(all_types)}):"]
        for t in all_types:
            p_count = len(self._type_producers.get(t, []))
            c_count = len(self._type_consumers.get(t, []))
            lines.append(f"  {t}: {p_count} producers, {c_count} consumers")
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

    @staticmethod
    def _io_summary(info: dict[str, Any]) -> str:
        """Build a compact I/O type summary for search results.

        Example: "IN: model(MODEL), latent_image(LATENT) → OUT: LATENT"
        Only shows typed connections (MODEL, CLIP, etc.), not scalar params.
        """
        # Collect typed inputs (uppercase type references)
        typed_inputs: list[str] = []
        input_info = info.get("input", {})
        for section in ("required", "optional"):
            for param_name, param_spec in input_info.get(section, {}).items():
                if (
                    isinstance(param_spec, (list, tuple))
                    and len(param_spec) > 0
                    and isinstance(param_spec[0], str)
                    and param_spec[0].isupper()
                ):
                    typed_inputs.append(f"{param_name}({param_spec[0]})")

        # Collect outputs
        output_types = info.get("output", [])
        out_str = ", ".join(str(t) for t in output_types) if output_types else "none"

        if typed_inputs:
            in_str = ", ".join(typed_inputs)
            return f"IN: {in_str} → OUT: {out_str}"
        return f"OUT: {out_str}"


# ============================================================
# Module-level helpers (not part of the class)
# ============================================================

# Regex for splitting identifiers: CamelCase, snake_case, slashes
_SPLIT_RE = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)")


class _SearchFields:
    """Structured search fields for a single node, all lowercase."""

    __slots__ = ("class_name", "display_name", "category", "description")

    def __init__(
        self,
        class_name: str,
        display_name: str,
        category: str,
        description: str,
    ) -> None:
        self.class_name = class_name
        self.display_name = display_name
        self.category = category
        self.description = description


def _tokenize(text: str) -> list[str]:
    """Split text into searchable tokens.

    Handles CamelCase (KSampler → k, sampler), snake_case,
    slashes (image/upscaling → image, upscaling), and spaces.
    Returns lowercase tokens with length >= 2.
    """
    # Split on non-alphanumeric boundaries
    parts = re.split(r"[^a-zA-Z0-9]+", text)
    tokens: list[str] = []
    for part in parts:
        lower = part.lower()
        if len(lower) >= 2:
            tokens.append(lower)
        # Also split CamelCase sub-tokens
        sub = _SPLIT_RE.findall(part)
        for s in sub:
            sl = s.lower()
            if len(sl) >= 2 and sl != lower:
                tokens.append(sl)
    return tokens
