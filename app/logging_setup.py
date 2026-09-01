"""Small, dependency-free logging helpers used by the CLI tools."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.video.base import redact_url


_URL_PATTERN = re.compile(r"(?P<url>(?:rtsp|rtsps|https?|ftp)://[^\s]+)", re.IGNORECASE)
_SECRET_PATTERN = re.compile(
    r"(?P<prefix>\b(?:password|passwd|pwd|secret|token|api[_-]?key)\b\s*[:=]\s*)"
    r"(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)


def redact_log_text(value: object) -> str:
    """Redact credentials from URL-like values before logging them."""

    text = str(value).replace("\n", " ")
    text = _URL_PATTERN.sub(lambda match: redact_url(match.group("url")), text)
    return _SECRET_PATTERN.sub(r"\g<prefix>***", text)


def configure_logging(level: str = "INFO") -> None:
    """Configure one readable process-wide logging handler."""

    normalized = str(level).upper()
    numeric_level = getattr(logging, normalized, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a compact key=value event suitable for console diagnostics."""

    # Redaction is intentionally skipped when the event would be discarded.
    # This function is used in per-frame hot paths, so eagerly formatting DEBUG
    # events would otherwise pay regex/string costs even at INFO level.
    if not logger.isEnabledFor(level):
        return

    parts = [f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={redact_log_text(value)}")
    logger.log(level, " ".join(parts))
