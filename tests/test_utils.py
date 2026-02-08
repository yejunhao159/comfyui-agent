"""Tests for tool output truncation and config loading."""

from __future__ import annotations

import os
import tempfile

import pytest

from comfyui_agent.domain.tools.base import truncate_output
from comfyui_agent.infrastructure.config import AppConfig


class TestTruncateOutput:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert truncate_output(text, max_len=100) == text

    def test_exact_limit_unchanged(self):
        text = "x" * 100
        assert truncate_output(text, max_len=100) == text

    def test_long_text_truncated(self):
        text = "line\n" * 10000  # 50K chars
        result = truncate_output(text, max_len=1000)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_keeps_first_and_last(self):
        lines = [f"line_{i:04d}" for i in range(1000)]
        text = "\n".join(lines)
        result = truncate_output(text, max_len=200)
        assert "line_0000" in result  # first line preserved
        assert "line_0999" in result  # last line preserved

    def test_shows_truncated_count(self):
        text = "a\n" * 10000
        result = truncate_output(text, max_len=500)
        assert "lines truncated" in result


class TestAppConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.comfyui.base_url == "http://127.0.0.1:6006"
        assert config.llm.model == "claude-sonnet-4-5-20250929"
        assert config.agent.max_iterations == 20
        assert config.server.port == 5200
        assert config.logging.level == "INFO"

    def test_from_yaml(self):
        yaml_content = """
comfyui:
  base_url: "http://localhost:8188"
  timeout: 60
llm:
  model: "claude-opus-4-6"
  max_tokens: 4096
agent:
  max_iterations: 10
server:
  port: 3000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = AppConfig.from_yaml(f.name)

        os.unlink(f.name)

        assert config.comfyui.base_url == "http://localhost:8188"
        assert config.comfyui.timeout == 60
        assert config.llm.model == "claude-opus-4-6"
        assert config.llm.max_tokens == 4096
        assert config.agent.max_iterations == 10
        assert config.server.port == 3000

    def test_from_missing_yaml(self):
        config = AppConfig.from_yaml("/nonexistent/path.yaml")
        assert config.comfyui.base_url == "http://127.0.0.1:6006"

    def test_resolve_api_key_from_env(self):
        config = AppConfig()
        os.environ["ANTHROPIC_API_KEY"] = "test-key-123"
        try:
            assert config.llm.resolve_api_key() == "test-key-123"
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_resolve_api_key_from_config(self):
        config = AppConfig(llm={"api_key": "config-key"})
        assert config.llm.resolve_api_key() == "config-key"

    def test_resolve_api_key_empty(self):
        config = AppConfig()
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            assert config.llm.resolve_api_key() == ""
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old
