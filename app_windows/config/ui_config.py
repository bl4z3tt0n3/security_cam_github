"""Configuration helpers that keep the monitor on the central YAML contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig


DEFAULT_SLOT_COUNT = 6


@dataclass(frozen=True)
class UiSettings:
    display_fps: float = 15.0
    start_maximized: bool = True
    remember_window_geometry: bool = True
    show_person_boxes: bool = True

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "UiSettings":
        settings = config.windows_ui
        return cls(
            display_fps=settings.display_fps,
            start_maximized=settings.start_maximized,
            remember_window_geometry=settings.remember_window_geometry,
            show_person_boxes=settings.show_person_boxes,
        )


def choose_config_path(repo_root: Path, requested: Path | None = None) -> Path:
    """Prefer the local ignored configuration, then the safe example file."""

    if requested is not None:
        return requested
    local = repo_root / "config" / "config.local.yaml"
    if local.is_file():
        return local
    return repo_root / "config" / "config.example.yaml"
