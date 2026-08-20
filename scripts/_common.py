"""Shared argument and source construction helpers for CLI scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    AppConfig,
    CameraConfig,
    ConfigurationError,
    get_camera,
    load_config,
    validate_stream_url,
)
from app.video.factory import create_opencv_source


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.local.yaml"
EXAMPLE_CONFIG = REPO_ROOT / "config" / "config.example.yaml"


@dataclass(frozen=True)
class Target:
    url: str
    config: AppConfig | None
    camera: CameraConfig | None


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        help="Concrete RTSP/HTTP stream URL; credentials are never printed in reports.",
    )
    group.add_argument(
        "--config",
        type=Path,
        help="YAML configuration containing the target camera.",
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help=(
            "Camera id used with --config. If omitted, single-camera scripts "
            "use huawei_p30 and the multi-camera runner uses all enabled cameras."
        ),
    )


def resolve_target(args: argparse.Namespace) -> Target:
    if args.url:
        return Target(url=validate_stream_url(args.url), config=None, camera=None)

    config_path = Path(args.config)
    config = load_config(config_path)
    camera = get_camera(config, args.camera_id or "huawei_p30")
    if not camera.enabled:
        raise ConfigurationError(f"camera '{camera.id}' is disabled in {config_path}")
    return Target(url=validate_stream_url(camera.stream_url), config=config, camera=camera)


def resolve_targets(args: argparse.Namespace) -> tuple[Target, ...]:
    """Resolve one or all enabled camera targets without sharing source state."""

    if args.url:
        return (Target(url=validate_stream_url(args.url), config=None, camera=None),)

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.camera_id:
        cameras = (get_camera(config, args.camera_id),)
    else:
        cameras = tuple(camera for camera in config.cameras if camera.enabled)
        if not cameras:
            raise ConfigurationError(f"no enabled cameras found in {config_path}")

    targets: list[Target] = []
    for camera in cameras:
        if not camera.enabled:
            raise ConfigurationError(f"camera '{camera.id}' is disabled in {config_path}")
        try:
            url = validate_stream_url(camera.stream_url)
        except ConfigurationError as exc:
            raise ConfigurationError(f"camera '{camera.id}': {exc}") from exc
        targets.append(Target(url=url, config=config, camera=camera))
    return tuple(targets)


def build_source(target: Target, args: argparse.Namespace):
    config = target.config
    video = config.video if config is not None else None
    return create_opencv_source(
        target.url,
        video=video,
        backend=getattr(args, "backend", None),
        open_timeout_s=getattr(args, "open_timeout", None),
        read_timeout_s=getattr(args, "read_timeout", None),
    )


def add_runtime_arguments(parser: argparse.ArgumentParser, *, default_duration: float) -> None:
    parser.add_argument(
        "--duration",
        type=float,
        default=default_duration,
        help="Test duration in seconds.",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=None,
        help="Maximum wait for a frame.",
    )
    parser.add_argument(
        "--open-timeout",
        type=float,
        default=None,
        help="Open timeout in seconds.",
    )
    parser.add_argument("--reconnect-attempts", type=int, default=3)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--backend", choices=("auto", "opencv", "ffmpeg"), default=None)


def print_stream_info(info: object) -> None:
    print(f"backend: {getattr(info, 'backend', 'n/d')}")
    print(
        f"resolution: {getattr(info, 'width', None) or 'n/d'}x"
        f"{getattr(info, 'height', None) or 'n/d'}"
    )
    fps = getattr(info, "declared_fps", None)
    print(f"declared FPS: {fps:.2f}" if fps else "declared FPS: n/d")
    print(f"codec: {getattr(info, 'codec', None) or 'n/d'}")
