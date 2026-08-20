"""Shared visual tokens and styles for the surveillance operator UI.

The theme deliberately keeps the video canvas dark and uses the Mistral-inspired
warm palette for actions, surfaces, text, and status accents.  It contains no
behaviour or widget wiring so the existing UI contracts remain independent from
the visual system.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from PySide6.QtGui import QFontDatabase


COLORS: Final[dict[str, str]] = {
    "shell": "#1f1f1f",
    "surface": "#2c2c2c",
    "surface_subtle": "#3a302b",
    "surface_warm": "#fff8e0",
    "surface_light": "#fffaeb",
    "ink": "#1f1f1f",
    "cream": "#fffaeb",
    "beige": "#e6d5a8",
    "slate": "#6a6a6a",
    "hairline": "#4a4a4a",
    "muted": "#a8a8a8",
    "action": "#fa520f",
    "action_deep": "#cc3a05",
    "success": "#4fbf83",
    "warning": "#ffb83e",
    "error": "#df5d54",
    "video": "#000000",
}

_FONT_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "ui": ("Inter", "Segoe UI", "Arial"),
    "display": ("PP Editorial Old", "Georgia", "Times New Roman"),
    "mono": ("JetBrains Mono", "Cascadia Mono", "Consolas"),
}
_FONT_FALLBACKS: Final[dict[str, str]] = {
    "ui": "Segoe UI",
    "display": "Georgia",
    "mono": "Consolas",
}


def resolve_font_family(role: str, available_families: Iterable[str] | None = None) -> str:
    """Return the first available family, with a deterministic native fallback.

    ``available_families`` is injectable so the fallback contract can be tested
    without relying on the fonts installed on the test machine.
    """

    if role not in _FONT_CANDIDATES:
        raise ValueError(f"unknown font role: {role}")
    available = (
        {value.casefold(): value for value in available_families}
        if available_families is not None
        else {value.casefold(): value for value in QFontDatabase.families()}
    )
    for candidate in _FONT_CANDIDATES[role]:
        resolved = available.get(candidate.casefold())
        if resolved is not None:
            return resolved
    return _FONT_FALLBACKS[role]


def _font(role: str) -> str:
    return f'"{resolve_font_family(role)}"'


def main_window_stylesheet() -> str:
    return f"""
    QMainWindow {{
        background: {COLORS['shell']};
        color: {COLORS['cream']};
    }}
    QStatusBar {{
        background: {COLORS['surface']};
        color: {COLORS['beige']};
        border-top: 1px solid {COLORS['hairline']};
        padding: 4px 10px;
        font-family: {_font('ui')};
    }}
    QPushButton {{
        min-height: 40px;
        padding: 8px 14px;
        color: {COLORS['cream']};
        background: {COLORS['surface_subtle']};
        border: 1px solid {COLORS['slate']};
        border-radius: 8px;
        font-family: {_font('ui')};
    }}
    QPushButton:hover {{
        background: #4a382f;
        border-color: {COLORS['action']};
    }}
    QPushButton:focus {{
        border: 2px solid {COLORS['action']};
    }}
    QPushButton:disabled {{
        color: {COLORS['muted']};
        background: #252525;
        border-color: {COLORS['hairline']};
    }}
    QLineEdit, QComboBox, QSpinBox {{
        min-height: 40px;
        padding: 6px 10px;
        color: {COLORS['cream']};
        background: {COLORS['surface']};
        border: 1px solid {COLORS['slate']};
        border-radius: 8px;
        selection-background-color: {COLORS['action']};
        font-family: {_font('ui')};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border: 2px solid {COLORS['action']};
    }}
    QCheckBox {{
        min-height: 40px;
        color: {COLORS['cream']};
        spacing: 8px;
        font-family: {_font('ui')};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {COLORS['slate']};
        border-radius: 4px;
        background: {COLORS['surface']};
    }}
    QCheckBox::indicator:checked {{
        background: {COLORS['action']};
        border-color: {COLORS['action']};
    }}
    """


def camera_tile_stylesheet() -> str:
    return f"""
    QFrame#CameraTile {{
        background: {COLORS['shell']};
        border: 1px solid {COLORS['hairline']};
        border-radius: 12px;
    }}
    QFrame#CameraTile:hover {{
        border: 1px solid {COLORS['action']};
    }}
    QFrame#CameraTile:focus {{
        border: 2px solid {COLORS['action']};
    }}
    QLabel#CameraVideo {{
        background: {COLORS['video']};
        color: {COLORS['muted']};
        border-radius: 8px;
    }}
    QLabel#CameraName, QLabel#CameraStatus {{
        background: rgba(31, 31, 31, 225);
        padding: 6px 10px;
        border-radius: 8px;
        font-family: {_font('ui')};
    }}
    QLabel#CameraName {{
        color: {COLORS['cream']};
        font-weight: 600;
    }}
    QLabel#CameraMessage {{
        color: {COLORS['beige']};
        font-family: {_font('ui')};
        font-size: 13px;
    }}
    """


def focus_view_stylesheet() -> str:
    return f"""
    QWidget#CameraFocusView {{ background: {COLORS['shell']}; }}
    QWidget#FocusVideoSurface {{ background: {COLORS['video']}; }}
    QLabel#FocusVideo {{
        background: {COLORS['video']};
        color: {COLORS['muted']};
    }}
    QLabel#FocusTitle {{
        color: {COLORS['cream']};
        font-family: {_font('display')};
        font-size: 20px;
        font-weight: 600;
    }}
    QLabel#FocusMeta {{
        color: {COLORS['beige']};
        padding: 6px 0;
        font-family: {_font('ui')};
    }}
    QPushButton#BackButton {{
        color: {COLORS['cream']};
        background: {COLORS['surface_subtle']};
        border: 1px solid {COLORS['slate']};
        border-radius: 8px;
        padding: 8px 12px;
    }}
    QPushButton#BackButton:hover {{
        background: #4a382f;
        border-color: {COLORS['action']};
    }}
    QPushButton#RotateButton, QPushButton#MirrorButton {{
        color: {COLORS['cream']};
        background: rgba(44, 44, 44, 235);
        border: 1px solid {COLORS['slate']};
        border-radius: 8px;
        padding: 8px 12px;
    }}
    QPushButton#RotateButton:hover, QPushButton#MirrorButton:hover {{
        background: #4a382f;
        border-color: {COLORS['action']};
    }}
    QPushButton#MirrorButton:checked {{
        background: {COLORS['action']};
        border-color: {COLORS['action']};
        color: #ffffff;
    }}
    QPushButton#RotateButton:disabled, QPushButton#MirrorButton:disabled {{
        color: {COLORS['muted']};
        border-color: {COLORS['hairline']};
    }}
    QSplitter::handle {{ background: {COLORS['hairline']}; }}
    """


def configuration_panel_stylesheet() -> str:
    return f"""
    QWidget#CameraConfigurationPanel {{
        background: {COLORS['surface']};
        color: {COLORS['cream']};
    }}
    QLabel#ConfigurationPanelTitle {{
        color: {COLORS['cream']};
        font-family: {_font('display')};
        font-size: 18px;
        font-weight: 600;
    }}
    QLabel#ConfigurationPanelHint {{
        color: {COLORS['beige']};
        font-family: {_font('ui')};
    }}
    QLabel#ConfigurationPanelSummary {{
        color: {COLORS['beige']};
        font-family: {_font('ui')};
    }}
    QScrollArea#CameraConfigurationScroll {{
        background: {COLORS['surface']};
        border: 0;
    }}
    """


def configuration_editor_stylesheet() -> str:
    return f"""
    QGroupBox {{
        color: {COLORS['cream']};
        background: {COLORS['surface_subtle']};
        border: 1px solid {COLORS['hairline']};
        border-radius: 12px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        font-family: {_font('ui')};
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {COLORS['cream']};
        background: {COLORS['surface_subtle']};
    }}
    QLabel#cameraIdLabel, QLabel#urlPreview {{ color: {COLORS['beige']}; }}
    QLineEdit#urlPreview {{ font-size: 12px; }}
    QLabel#cameraConfigStatus {{ color: {COLORS['beige']}; }}
    QDialog {{ background: {COLORS['shell']}; color: {COLORS['cream']}; }}
    QScrollArea#cameraConfigurationScrollArea {{
        background: {COLORS['shell']};
        border: 0;
    }}
    QLabel#cameraConfigurationSummary {{ color: {COLORS['beige']}; }}
    """


def status_color(status_name: str) -> str:
    return {
        "LIVE": COLORS["success"],
        "CONNECTING": COLORS["warning"],
        "RECONNECTING": COLORS["warning"],
        "OFFLINE": COLORS["error"],
        "ERROR": COLORS["error"],
        "DISABLED": COLORS["muted"],
        "NOT_CONFIGURED": COLORS["muted"],
    }.get(status_name, COLORS["muted"])


def status_badge_stylesheet(status_name: str) -> str:
    return f"""
    color: {status_color(status_name)};
    background: rgba(31, 31, 31, 235);
    padding: 6px 10px;
    border: 1px solid {status_color(status_name)};
    border-radius: 8px;
    font-family: {_font('ui')};
    font-weight: 600;
    """


def status_text_stylesheet(*, error: bool = False) -> str:
    color = COLORS["error"] if error else COLORS["beige"]
    return f"color: {color}; font-family: {_font('ui')};"
