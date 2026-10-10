from __future__ import annotations

import io
import json
import logging

from mlp.common.logging import JSONFormatter, configure_logging, request_id


def test_json_log_with_correlation_id() -> None:
    stream = io.StringIO()
    logger = configure_logging(as_json=True)
    logger.handlers[0].setStream(stream)
    token = request_id.set("request-test-001")
    try:
        logger.info("model_loaded")
    finally:
        request_id.reset(token)
    payload = json.loads(stream.getvalue())
    assert payload["event"] == "model_loaded"
    assert payload["request_id"] == "request-test-001"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "mlp"
    assert "timestamp" in payload


def test_unapproved_record_extras_are_not_logged() -> None:
    record = logging.LogRecord(
        "mlp", logging.INFO, __file__, 1, "request_accepted", (), None
    )
    record.private_image = "sensitive-content"
    message = JSONFormatter().format(record)
    assert "sensitive-content" not in message
    assert "private_image" not in message


def test_logger_is_idempotent() -> None:
    first = configure_logging()
    second = configure_logging()
    assert first is second
    assert len(second.handlers) == 1
    assert second.propagate is False
