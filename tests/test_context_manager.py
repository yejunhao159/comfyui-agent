"""Tests for application.context_manager."""

import pytest

from comfyui_agent.application.context_manager import (
    ContextManager,
    _resolve_context_size,
    estimate_messages_tokens,
    estimate_tokens,
)


# ------------------------------------------------------------------
# Token estimation
# ------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world") >= 1

    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_proportional(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("a" * 400)
        assert long > short


class TestEstimateMessagesTokens:
    def test_single_text_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0

    def test_tool_result_message(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "data"}
            ]},
        ]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0


# ------------------------------------------------------------------
# Model resolution
# ------------------------------------------------------------------

class TestResolveContextSize:
    def test_known_model(self):
        assert _resolve_context_size("claude-sonnet-4-5-20250929") == 200_000

    def test_unknown_model_returns_default(self):
        assert _resolve_context_size("gpt-4o") == 200_000

    def test_prefix_match(self):
        assert _resolve_context_size("claude-opus-4-6-some-variant") == 200_000


# ------------------------------------------------------------------
# ContextManager
# ------------------------------------------------------------------

def _make_messages(n_turns: int, tool_result_size: int = 100) -> list[dict]:
    """Build a realistic conversation with n_turns of user->assistant->tool_result."""
    msgs: list[dict] = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"Question {i}"})
        msgs.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check..."},
                {"type": "tool_use", "id": f"tool_{i}", "name": "test", "input": {}},
            ],
        })
        msgs.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"tool_{i}",
                    "content": "x" * tool_result_size,
                    "is_error": False,
                },
            ],
        })
    # Final assistant response
    msgs.append({"role": "assistant", "content": "Here is the answer."})
    return msgs


class TestContextManager:
    def test_no_compaction_when_under_budget(self):
        cm = ContextManager(model="claude-sonnet-4-5-20250929")
        msgs = _make_messages(2, tool_result_size=50)
        result = cm.prepare_messages(msgs)
        assert result == msgs

    def test_tool_result_truncation(self):
        cm = ContextManager(context_budget=20000, max_output_tokens=1000)
        # 20 turns × 5000 chars ≈ 25000 tokens, budget ≈ 9000
        msgs = _make_messages(20, tool_result_size=5000)
        original_tokens = estimate_messages_tokens(msgs)
        result = cm.prepare_messages(msgs)
        assert estimate_messages_tokens(result) < original_tokens
        assert result[-1] == msgs[-1]

    def test_truncated_tool_results_have_marker(self):
        cm = ContextManager(context_budget=20000, max_output_tokens=1000)
        msgs = _make_messages(20, tool_result_size=5000)
        result = cm.prepare_messages(msgs)
        found_truncated = False
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "[truncated" in str(block.get("content", "")):
                        found_truncated = True
        assert found_truncated

    def test_emergency_trim(self):
        cm = ContextManager(context_budget=500)
        msgs = _make_messages(10, tool_result_size=2000)
        result = cm.prepare_messages(msgs)
        assert len(result) < len(msgs)

    def test_auto_budget_from_model(self):
        cm = ContextManager(model="claude-sonnet-4-5-20250929", max_output_tokens=8192)
        assert cm.history_budget == 181_808

    def test_explicit_budget_overrides_model(self):
        cm = ContextManager(model="claude-sonnet-4-5-20250929", context_budget=50000)
        assert cm.history_budget == 50000 - 2000 - 3000 - 8192 - 5000
