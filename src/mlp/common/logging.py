"""Small structured logger with bounded metadata and correlation IDs."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import UTC, datetime

request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Serialize approved record attributes only, never arbitrary object extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        correlation_id = request_id.get()
        if correlation_id is not None:
            payload["request_id"] = correlation_id
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO", as_json: bool = False) -> logging.Logger:
    """Configure only our logger, leaving unrelated libraries untouched."""
    logger = logging.getLogger("mlp")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        JSONFormatter() if as_json else logging.Formatter("%(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger
