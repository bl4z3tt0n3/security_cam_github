from __future__ import annotations

from app_windows.ui.theme import (
    COLORS,
    camera_tile_stylesheet,
    main_window_stylesheet,
    resolve_font_family,
    status_color,
)


def test_font_resolver_prefers_explicit_mistral_families() -> None:
    assert resolve_font_family("ui", ["Inter", "Segoe UI"]) == "Inter"
    assert resolve_font_family("display", ["PP Editorial Old", "Georgia"]) == "PP Editorial Old"
    assert resolve_font_family("mono", ["JetBrains Mono", "Consolas"]) == "JetBrains Mono"


def test_font_resolver_has_native_fallbacks_when_assets_are_unavailable() -> None:
    assert resolve_font_family("ui", []) == "Segoe UI"
    assert resolve_font_family("display", []) == "Georgia"
    assert resolve_font_family("mono", []) == "Consolas"


def test_theme_reserves_orange_for_action_and_keeps_semantic_status_colors(qapp) -> None:
    stylesheet = main_window_stylesheet()
    tile_stylesheet = camera_tile_stylesheet()

    assert COLORS["action"] in stylesheet
    assert "min-height: 40px" in stylesheet
    assert COLORS["action"] in tile_stylesheet
    assert status_color("LIVE") == COLORS["success"]
    assert status_color("RECONNECTING") == COLORS["warning"]
    assert status_color("ERROR") == COLORS["error"]
    assert status_color("NOT_CONFIGURED") == COLORS["muted"]
