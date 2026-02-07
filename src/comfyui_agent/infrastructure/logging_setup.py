"""Structured logging setup.

Provides:
- Console output: colored, human-readable (for development)
- File output: JSON lines (for production/debugging)
- Request tracing with correlation IDs
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", log_dir: str = "data/logs") -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for structlog
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # File handler — JSON lines for machine parsing
    file_handler = logging.FileHandler(
        log_path / "agent.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(log_level)

    # Console handler — colored for humans
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[file_handler, console_handler],
        force=True,
    )

    # Suppress noisy libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set formatters on handlers
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(json_formatter)
