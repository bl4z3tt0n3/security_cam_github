"""Entry point for the local PySide6 six-camera monitor."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from app.config import ConfigurationError, load_config, validate_stream_url
from app.inference import InferenceGate
from app.logging_setup import configure_logging, log_event, redact_log_text
from app.video.factory import create_opencv_source

from app_windows.config.camera_config import runtime_stream_url
from app_windows.config.persistence import CameraConfigRepository
from app_windows.config.credentials import (
    CredentialStore,
    CredentialStoreError,
    DpapiCredentialStore,
    InMemoryCredentialStore,
)
from app_windows.config.ui_config import UiSettings, choose_config_path
from app_windows.models.camera_view_state import CameraSlot, camera_slots_from_config
from app_windows.models.person_detection_state import PersonDetectionSettings
from app_windows.video.fake_provider import fake_camera_factory, fake_connection_source_factory
from app_windows.video.frame_provider import BackendFrameProvider


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Windows monitor for six independent camera streams."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration; defaults to config.local.yaml or the example.",
    )
    parser.add_argument(
        "--fake-cameras",
        action="store_true",
        help="Run six clearly labelled synthetic cameras without network or hardware.",
    )
    parser.add_argument(
        "--fake-offline-camera",
        default=None,
        help="Camera id to keep offline in fake mode.",
    )
    parser.add_argument(
        "--fake-reconnect-camera",
        default=None,
        help="Camera id to repeatedly disconnect in fake mode.",
    )
    parser.add_argument("--log-level", default=None, help="Logging level override.")
    return parser


def _load_config(config_path: Path):
    try:
        return load_config(config_path)
    except ConfigurationError as exc:
        raise ConfigurationError(f"{config_path}: {exc}") from exc


def _fake_slots(slots: tuple[CameraSlot, ...]) -> tuple[CameraSlot, ...]:
    return tuple(
        slot.with_runtime_source(f"fake://{slot.camera_id}/live") for slot in slots
    )


def _build_provider_factory(
    *,
    config,
    fake_mode: bool,
    fake_offline_camera: str | None,
    fake_reconnect_camera: str | None,
    logger: logging.Logger,
    credentials: CredentialStore | None = None,
):
    if fake_mode:
        return fake_camera_factory(
            offline_camera_id=fake_offline_camera,
            reconnect_camera_id=fake_reconnect_camera,
            logger=logger,
        )

    def create(slot: CameraSlot) -> BackendFrameProvider:
        password = credentials.get(slot.camera_id) if credentials is not None else None
        url = validate_stream_url(runtime_stream_url(slot.stream_url, password))
        source = create_opencv_source(
            url,
            video=config.video,
            rtsp_transport=slot.rtsp_transport,
            logger=logger,
        )
        video = config.video
        return BackendFrameProvider(
            slot.camera_id,
            source,
            read_timeout_s=video.read_timeout_seconds,
            reconnect_delay_s=video.reconnect_delay_seconds,
            max_reconnect_attempts=video.max_reconnect_attempts,
            max_buffer_frames=video.max_buffer_frames,
            stop_timeout_s=max(
                1.0,
                video.open_timeout_seconds + 1.0,
                video.read_timeout_seconds + 1.0,
            ),
            logger=logger,
        )

    return create


def _run(args: argparse.Namespace) -> int:
    configure_logging(args.log_level or "INFO")
    logger = logging.getLogger("app_windows")
    config_path = choose_config_path(REPO_ROOT, args.config)
    try:
        config = _load_config(config_path)
        migrated_path = CameraConfigRepository(REPO_ROOT).migrate_person_detection(
            current_path=config_path
        )
        if migrated_path != config_path:
            config_path = migrated_path
            config = _load_config(config_path)
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    log_event(
        logger,
        logging.INFO,
        "windows_video_runtime_config",
        config_path=config_path,
        max_reconnect_attempts=config.video.max_reconnect_attempts,
        reconnect_delay=f"{config.video.reconnect_delay_seconds:.2f}s",
        read_timeout=f"{config.video.read_timeout_seconds:.2f}s",
        rtsp_transport=config.video.rtsp_transport,
    )

    slots = camera_slots_from_config(config)
    if args.fake_cameras:
        slots = _fake_slots(slots)
        logger.warning("event=fake_mode_enabled camera_count=%s", len(slots))

    if args.fake_cameras:
        credentials: CredentialStore = InMemoryCredentialStore()
    else:
        try:
            credentials = DpapiCredentialStore(
                REPO_ROOT / "config" / "config.local.secrets.json"
            )
        except CredentialStoreError:
            # The GUI remains useful for password-free streams on non-Windows
            # development hosts; saving a password will report the store error.
            credentials = InMemoryCredentialStore()

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from app_windows.inference.person_detection_controller import PersonDetectionController
        from app_windows.inference.face_recognition_controller import FaceRecognitionController
        from app_windows.monitor_controller import CameraMonitorController
        from app_windows.ui.main_window import MainWindow
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            print(
                "PySide6 non è installato. Eseguire: "
                "python -m pip install -e \".[dev,windows]\"",
                file=sys.stderr,
            )
            return 2
        raise

    provider_factory = _build_provider_factory(
        config=config,
        fake_mode=args.fake_cameras,
        fake_offline_camera=args.fake_offline_camera,
        fake_reconnect_camera=args.fake_reconnect_camera,
        logger=logger,
        credentials=credentials,
    )
    if args.fake_cameras:
        connection_source_factory = fake_connection_source_factory
    else:
        def connection_source_factory(url: str, transport: str):
            return create_opencv_source(
                url,
                video=config.video,
                rtsp_transport=transport,
                logger=logger,
            )

    ui_settings = UiSettings.from_app_config(config)
    controller = CameraMonitorController(
        slots,
        provider_factory,
        display_fps=ui_settings.display_fps,
        read_timeout_s=config.video.read_timeout_seconds,
        logger=logger,
    )
    inference_gate = InferenceGate()
    person_detection_controller = PersonDetectionController(
        repo_root=REPO_ROOT,
        settings=PersonDetectionSettings.from_app_config(config, repo_root=REPO_ROOT),
        inference_gate=inference_gate,
        logger=logger,
    )
    face_recognition_controller = FaceRecognitionController(
        repo_root=REPO_ROOT,
        config=config,
        inference_gate=inference_gate,
        logger=logger,
    )

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Local Security Monitor")
    window = MainWindow(
        slots,
        controller,
        ui_settings=ui_settings,
        simulation=args.fake_cameras,
        config_path=config_path,
        credentials=credentials,
        repo_root=REPO_ROOT,
        connection_source_factory=connection_source_factory,
        read_timeout_s=config.video.read_timeout_seconds,
        logger=logger,
        person_detection_controller=person_detection_controller,
        face_recognition_controller=face_recognition_controller,
    )
    window.show()
    # Let Qt paint the shell before any provider starts its acquisition thread.
    QTimer.singleShot(0, window.start_monitoring)
    return qt_app.exec()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fake_offline_camera and not args.fake_cameras:
        print("--fake-offline-camera richiede --fake-cameras", file=sys.stderr)
        return 2
    if args.fake_reconnect_camera and not args.fake_cameras:
        print("--fake-reconnect-camera richiede --fake-cameras", file=sys.stderr)
        return 2
    try:
        return _run(args)
    except Exception as exc:  # keep startup errors readable without a GUI traceback
        print(f"MONITOR ERROR: {redact_log_text(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
