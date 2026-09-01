from __future__ import annotations

import logging

from app.logging_setup import log_event, redact_log_text


def test_log_text_redacts_url_credentials() -> None:
    value = redact_log_text("open rtsp://admin:secret@camera.local/live")
    assert "secret" not in value
    assert "admin:***@camera.local" in value


def test_log_text_redacts_standalone_secret_fields() -> None:
    value = redact_log_text("password=plain-secret token:token-secret")
    assert "plain-secret" not in value
    assert "token-secret" not in value
    assert "password=***" in value
    assert "token:***" in value


def test_log_event_redacts_url_fields(caplog) -> None:
    logger = logging.getLogger("hardening-test")
    with caplog.at_level(logging.INFO):
        log_event(logger, logging.INFO, "connect", url="rtsp://user:password@camera/live")

    assert "password" not in caplog.text
    assert "user:***@camera" in caplog.text

def test_log_event_does_not_redact_or_format_when_level_is_disabled(monkeypatch) -> None:
    import app.logging_setup as logging_setup

    logger = logging.getLogger("disabled-hot-path")
    logger.setLevel(logging.INFO)
    calls = 0

    def unexpected_redaction(value: object) -> str:
        nonlocal calls
        calls += 1
        return str(value)

    monkeypatch.setattr(logging_setup, "redact_log_text", unexpected_redaction)
    logging_setup.log_event(logger, logging.DEBUG, "frame", sequence=123, detail="ignored")

    assert calls == 0
