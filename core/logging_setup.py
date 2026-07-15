"""Central logging configuration.

Uses structlog configured to wrap Python's standard `logging` module
(the recommended structlog production pattern): every `logging.getLogger(...)`
call already used throughout the codebase (agents, core, api) is rendered
through structlog's processor pipeline, so we get structured, leveled,
timestamped log lines everywhere without having to change every call
site to `structlog.get_logger()`.

Call `configure_logging(settings)` once, as early as possible (API
lifespan startup, or the top of each standalone script).
"""
from __future__ import annotations

import logging
import sys

import structlog

from config.settings import Settings


def configure_logging(settings: Settings, json_logs: bool = False) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
