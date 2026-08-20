"""Validation helpers for the local YOLOE text categories."""

from __future__ import annotations

from collections.abc import Sequence


MAX_PROMPT_COUNT = 20
MAX_PROMPT_LENGTH = 64


def normalize_prompts(value: str | Sequence[str], *, max_count: int = MAX_PROMPT_COUNT) -> tuple[str, ...]:
    """Normalize comma-separated or sequence prompt values deterministically."""

    raw_values = value.split(",") if isinstance(value, str) else value
    if not isinstance(raw_values, Sequence):
        raise ValueError("prompts must be a comma-separated string or sequence")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        prompt = str(raw).strip()
        if not prompt:
            continue
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"each prompt must be at most {MAX_PROMPT_LENGTH} characters"
            )
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(prompt)

    if not normalized:
        raise ValueError("at least one prompt category is required")
    if len(normalized) > max_count:
        raise ValueError(f"at most {max_count} prompt categories are supported")
    return tuple(normalized)
