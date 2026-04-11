"""
utils/logger.py — Structured logging with structlog.

- JSON renderer for file output (.agent/logs/agent.jsonl)
- Colored console renderer for terminal when verbose
- Standard log levels (DEBUG, INFO, WARNING, ERROR)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import structlog


def setup_logging(verbose: bool = False, log_dir: str | Path | None = None) -> None:
    """
    Configure structlog for the CLI agent.

    Args:
        verbose: If True, show DEBUG-level colored output in console.
        log_dir: Directory for JSONL log file. Defaults to .agent/logs/.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Shared processors for all outputs
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure standard logging (for third-party libs)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    # Set up JSONL file logging if a log directory is provided
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "agent.jsonl", mode="a")
        file_handler.setLevel(logging.DEBUG)  # Always capture everything in file
        logging.getLogger().addHandler(file_handler)

    # Choose renderer based on verbosity
    if verbose:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            level_styles=structlog.dev.ConsoleRenderer.get_default_level_styles(),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib handler
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Apply formatter to all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a bound structlog logger with the given name.

    Usage:
        log = get_logger("agent.core")
        log.info("Starting agent loop", iteration=1)
    """
    return structlog.get_logger(name)


class SessionLogger:
    """Append structured per-event records to `.agent/logs/session.jsonl`."""

    def __init__(
        self,
        path: str | Path = ".agent/logs/session.jsonl",
        session_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid4().hex

    def log_event(self, event_type: str, **fields: object) -> None:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
        }
        payload.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            # Logging should never crash the session.
            pass
