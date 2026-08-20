"""Structured results for local hardening and environment checks.

The checks in this module are deliberately small data contracts.  Operational
probes remain in the command-line checker so they can be run against the
selected configuration without adding lifecycle side effects to the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HardeningStatus = Literal["PASS", "INFO", "DEFERRED", "FAIL"]


@dataclass(frozen=True)
class HardeningCheck:
    """One hardening result with a stable, JSON-serializable shape."""

    name: str
    status: HardeningStatus
    detail: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("hardening check name cannot be empty")
        if self.status not in {"PASS", "INFO", "DEFERRED", "FAIL"}:
            raise ValueError("invalid hardening check status")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class HardeningReport:
    """Collection of hardening checks and their aggregate outcome."""

    checks: tuple[HardeningCheck, ...]

    @property
    def failed(self) -> bool:
        return any(check.status == "FAIL" for check in self.checks)

    @property
    def status(self) -> HardeningStatus:
        return "FAIL" if self.failed else "PASS"

    def to_dict(self) -> dict[str, object]:
        counts = {status: 0 for status in ("PASS", "INFO", "DEFERRED", "FAIL")}
        for check in self.checks:
            counts[check.status] += 1
        return {
            "status": self.status,
            "failed": self.failed,
            "counts": counts,
            "checks": [check.to_dict() for check in self.checks],
        }


def normalize_status(state: str) -> HardeningStatus:
    """Map legacy environment labels to the hardening status vocabulary."""

    normalized = str(state).upper()
    if normalized in {"OK", "PASS"}:
        return "PASS"
    if normalized in {"DEFERRED"}:
        return "DEFERRED"
    if normalized in {"FAIL", "ERROR"}:
        return "FAIL"
    return "INFO"
