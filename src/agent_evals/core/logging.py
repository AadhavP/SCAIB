"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure structlog for application logging.

    Args:
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: Whether to output logs in JSON format instead of console formatting.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger instance by name.

    Args:
        name: Name of module/component logger.

    Returns:
        BoundLogger instance.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
