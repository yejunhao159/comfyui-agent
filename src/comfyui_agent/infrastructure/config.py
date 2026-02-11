"""Configuration management for comfyui-agent."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ComfyUIConfig(BaseModel):
    base_url: str = "http://127.0.0.1:6006"
    ws_url: str = "ws://127.0.0.1:6006/ws"
    timeout: int = 30


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 8192
    temperature: float = 0.7
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    max_retries: int = 5
    retry_base_delay_ms: int = 2000
    retry_max_delay_ms: int = 60000

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return os.environ.get("ANTHROPIC_API_KEY", "")


class AgentConfig(BaseModel):
    max_iterations: int = 20
    session_db: str = "data/sessions.db"
    context_budget: int = 0  # 0 = auto-resolve from model name


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5200
    cors_origins: list[str] = ["*"]


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"
    log_dir: str = "data/logs"


class IdentityConfig(BaseModel):
    """RoleX identity configuration."""

    rolex_dir: str = "~/.rolex"
    role_name: str = ""  # Empty = skip identity loading


class WebConfig(BaseModel):
    """Web search and fetch configuration."""

    tavily_api_key: str = ""  # Empty = use DuckDuckGo fallback
    timeout: int = 30

    def resolve_tavily_key(self) -> str:
        if self.tavily_api_key:
            return self.tavily_api_key
        return os.environ.get("TAVILY_API_KEY", "")




class AppConfig(BaseSettings):
    comfyui: ComfyUIConfig = ComfyUIConfig()
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()
    web: WebConfig = WebConfig()
    identity: IdentityConfig = IdentityConfig()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> AppConfig:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.from_yaml()
    return _config


def set_config(config: AppConfig) -> None:
    global _config
    _config = config
