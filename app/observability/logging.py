"""Centralised logging and observability configuration."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone


_STANDARD_LOG_ATTRS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
)


class JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines for easy ingestion by log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        for key, val in record.__dict__.items():
            if key in _STANDARD_LOG_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(val, default=str)
            except (TypeError, ValueError):
                val = str(val)
            log_entry[key] = val

        return json.dumps(log_entry, default=str)


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with structured JSON output.

    Parameters
    ──────────
    level : str, optional
        Logging level name (DEBUG, INFO, WARNING, …).
        Defaults to the ``LOG_LEVEL`` env var or ``INFO``.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers to avoid duplicates on re‑init
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Silence noisy third‑party loggers
    for noisy in ("httpcore", "httpx", "openai", "deepseek", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Logging initialised at level %s", log_level)
