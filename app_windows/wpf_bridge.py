"""Headless bridge used by the native WPF Windows frontend.

The bridge deliberately composes the existing Windows monitor services instead
of reimplementing them.  It owns no camera or inference logic: it starts the
same ``CameraMonitorController`` and ``PersonDetectionController`` used by the
PySide6 frontend, then exposes a small newline-delimited JSON protocol to the
WPF process over redirected standard streams.
"""

from __future__ import annotations

import argparse
import mmap
import os
import struct
from dataclasses import replace
import json
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Slot

from app.config import ConfigurationError, load_config
from app.inference import InferenceGate
from app.face import face_capability_matrix
from app.face.registry import LANDMARKER_SPEC, model_path
from app.face.storage import PersonStore
from app.logging_setup import configure_logging, log_event, redact_log_text
from app.video.base import redact_url
from app.video.factory import create_opencv_source

from app_windows.config.camera_config import (
    CameraDraft,
    draft_from_slot,
    runtime_stream_url,
    validate_camera_draft,
)
from app_windows.config.connection_test import (
    AsyncConnectionTester,
    ConnectionTestResult,
)
from app_windows.config.credentials import (
    CredentialStore,
    CredentialStoreError,
    DpapiCredentialStore,
    InMemoryCredentialStore,
)
from app_windows.config.persistence import CameraConfigRepository
from app_windows.config.ui_config import UiSettings, choose_config_path
from app_windows.inference.person_detection_controller import PersonDetectionController
from app_windows.inference.face_recognition_controller import FaceRecognitionController
from app_windows.main import _build_provider_factory, _fake_slots
from app_windows.models.camera_view_state import CameraSlot, CameraViewSnapshot
from app_windows.models.person_detection_state import (
    PersonDetectionSettings,
    PersonDetectionSnapshot,
)
from app_windows.models.face_recognition_state import (
    FaceGalleryState,
    FaceRecognitionSettings,
    FaceRecognitionSnapshot,
)
from app_windows.monitor_controller import CameraMonitorController
from app_windows.video.fake_provider import fake_connection_source_factory


REPO_ROOT = Path(__file__).resolve().parents[1]

_FRAME_MAGIC = b"LSCF"
_FRAME_VERSION = 1
_FRAME_HEADER = struct.Struct("<4sIQQIIII")


class SharedFramePublisher:
    """Publish latest BGR frames through a Windows named memory mapping.

    JSON carries only the mapping name and geometry.  A monotonically
    increasing write epoch in the fixed header lets the WPF reader detect and
    retry a frame that was overwritten while it was copying the pixel bytes.
    """

    def __init__(self, camera_id: str) -> None:
        if os.name != "nt":
            raise RuntimeError("WPF shared preview transport is available on Windows only")
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in camera_id) or "camera"
        self._prefix = f"LocalSecurityCamPreview_{os.getpid()}_{safe_id}"
        self._generation = 0
        self._mapping: mmap.mmap | None = None
        self._mapping_name: str | None = None
        self._capacity = 0
        self._epoch = 0

    def close(self) -> None:
        mapping = self._mapping
        self._mapping = None
        self._mapping_name = None
        self._capacity = 0
        if mapping is not None:
            mapping.close()

    def _ensure_mapping(self, required_bytes: int) -> None:
        total = _FRAME_HEADER.size + required_bytes
        if self._mapping is not None and self._capacity >= total:
            return
        old = self._mapping
        self._generation += 1
        self._mapping_name = f"{self._prefix}_{self._generation}"
        self._mapping = mmap.mmap(
            -1,
            total,
            tagname=self._mapping_name,
            access=mmap.ACCESS_WRITE,
        )
        self._capacity = total
        if old is not None:
            old.close()

    def publish(self, sequence: int, frame: Any) -> dict[str, Any]:
        image = np.asarray(frame)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("preview frame must be a non-empty HxWx3 BGR image")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)

        height, width = int(image.shape[0]), int(image.shape[1])
        stride = int(image.strides[0])
        byte_count = stride * height
        self._ensure_mapping(byte_count)
        mapping = self._mapping
        name = self._mapping_name
        assert mapping is not None and name is not None

        self._epoch += 1
        if self._epoch % 2 == 0:
            self._epoch += 1
        writing_epoch = self._epoch
        _FRAME_HEADER.pack_into(
            mapping,
            0,
            _FRAME_MAGIC,
            _FRAME_VERSION,
            writing_epoch,
            int(sequence),
            width,
            height,
            stride,
            byte_count,
        )
        mapping.seek(_FRAME_HEADER.size)
        mapping.write(memoryview(image).cast("B"))
        self._epoch += 1
        _FRAME_HEADER.pack_into(
            mapping,
            0,
            _FRAME_MAGIC,
            _FRAME_VERSION,
            self._epoch,
            int(sequence),
            width,
            height,
            stride,
            byte_count,
        )
        return {
            "frame_shm_name": name,
            "frame_byte_count": byte_count,
            "frame_stride": stride,
            "frame_width": width,
            "frame_height": height,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Security Monitor backend bridge for the native WPF frontend."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fake-cameras", action="store_true")
    parser.add_argument("--fake-offline-camera", default=None)
    parser.add_argument("--fake-reconnect-camera", default=None)
    parser.add_argument("--log-level", default=None)
    return parser


class BridgeRuntime(QObject):
    """Run the existing Qt-backed services without constructing a Qt widget."""

    def __init__(self, qt_app: QCoreApplication, args: argparse.Namespace) -> None:
        super().__init__()
        self._qt_app = qt_app
        self._args = args
        self._logger = logging.getLogger("app_windows.wpf_bridge")
        self._repo_root = REPO_ROOT
        self._repository: CameraConfigRepository
        self._credentials: CredentialStore
        self._config_path: Path
        self._config: Any
        self._slots: tuple[CameraSlot, ...]
        self._controller: CameraMonitorController
        self._person_controller: PersonDetectionController
        self._face_controller: FaceRecognitionController
        self._connection_tester: AsyncConnectionTester
        self._source_factory: Any
        self._fake_mode = bool(args.fake_cameras)
        self._command_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._command_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._stopping = False
        self._frame_publishers: dict[str, SharedFramePublisher] = {}
        self._active_camera_id: str | None = None
        self._last_preview_publish: dict[str, float] = {}
        self._last_snapshot_key: dict[str, tuple[Any, ...]] = {}

        self._prepare_backend()
        # Populate the source gallery before the first hello payload even when
        # the optional face runtime is disabled or not loadable.  The WPF
        # shell can therefore render enrollment folders immediately instead
        # of requiring a manual refresh command.
        self._face_controller.refresh_gallery()
        self._command_timer = QTimer(self)
        self._command_timer.setInterval(20)
        self._command_timer.timeout.connect(self._drain_commands)

        self._controller.snapshot_changed.connect(self._on_snapshot)
        self._controller.camera_reconfigured.connect(self._on_camera_reconfigured)
        self._person_controller.snapshot_changed.connect(self._on_person_snapshot)
        self._face_controller.snapshot_changed.connect(self._on_face_snapshot)
        self._face_controller.gallery_changed.connect(self._on_face_gallery)
        self._face_controller.capabilities_changed.connect(self._on_face_capabilities)
        self._connection_tester.started.connect(self._on_connection_test_started)
        self._connection_tester.finished.connect(self._on_connection_test_finished)

    def _prepare_backend(self) -> None:
        requested_config = self._args.config
        self._config_path = choose_config_path(self._repo_root, requested_config)
        try:
            self._config = load_config(self._config_path)
        except ConfigurationError as exc:
            raise ConfigurationError(f"{self._config_path}: {exc}") from exc

        migrated_path = CameraConfigRepository(self._repo_root).migrate_person_detection(
            current_path=self._config_path
        )
        if migrated_path != self._config_path:
            self._config_path = migrated_path
            self._config = load_config(self._config_path)

        self._repository = CameraConfigRepository(self._repo_root)
        if self._fake_mode:
            self._credentials = InMemoryCredentialStore()
        else:
            try:
                self._credentials = DpapiCredentialStore(
                    self._repo_root / "config" / "config.local.secrets.json"
                )
            except CredentialStoreError:
                self._credentials = InMemoryCredentialStore()

        from app_windows.models.camera_view_state import camera_slots_from_config

        slots = camera_slots_from_config(self._config)
        self._slots = _fake_slots(slots) if self._fake_mode else slots
        ui_settings = UiSettings.from_app_config(self._config)
        provider_factory = _build_provider_factory(
            config=self._config,
            fake_mode=self._fake_mode,
            fake_offline_camera=self._args.fake_offline_camera,
            fake_reconnect_camera=self._args.fake_reconnect_camera,
            logger=self._logger,
            credentials=self._credentials,
        )
        if self._fake_mode:
            self._source_factory = fake_connection_source_factory
        else:

            def source_factory(url: str, transport: str):
                return create_opencv_source(
                    url,
                    video=self._config.video,
                    rtsp_transport=transport,
                    logger=self._logger,
                )

            self._source_factory = source_factory

        self._controller = CameraMonitorController(
            self._slots,
            provider_factory,
            display_fps=ui_settings.display_fps,
            read_timeout_s=self._config.video.read_timeout_seconds,
            logger=self._logger,
        )
        inference_gate = InferenceGate()
        self._person_controller = PersonDetectionController(
            repo_root=self._repo_root,
            settings=PersonDetectionSettings.from_app_config(
                self._config,
                repo_root=self._repo_root,
            ),
            inference_gate=inference_gate,
            logger=self._logger,
        )
        self._face_controller = FaceRecognitionController(
            repo_root=self._repo_root,
            config=self._config,
            inference_gate=inference_gate,
            logger=self._logger,
        )
        self._connection_tester = AsyncConnectionTester(
            self._source_factory,
            read_timeout_s=self._config.video.read_timeout_seconds,
            existing_probe=self._controller.probe_existing_camera,
            logger=self._logger,
            parent=self,
        )

    def start(self) -> None:
        ui_settings = UiSettings.from_app_config(self._config)
        log_event(
            self._logger,
            logging.INFO,
            "wpf_bridge_started",
            config_path=self._config_path,
            simulation=self._fake_mode,
        )
        self._emit(
            "hello",
            {
                "simulation": self._fake_mode,
                "config_path": str(self._config_path),
                "ui": {
                    "start_maximized": ui_settings.start_maximized,
                    "remember_window_geometry": ui_settings.remember_window_geometry,
                    "display_fps": ui_settings.display_fps,
                },
                "cameras": [self._camera_data(slot) for slot in self._slots],
                "person_detection": self._person_settings_data(
                    self._person_controller.settings
                ),
                "face_detection": self._face_settings_data(self._face_controller.settings),
                "face_recognition": self._face_settings_data(self._face_controller.settings),
                "face_gallery": self._face_controller.gallery.to_dict(),
                "face_capabilities": [
                    row.to_dict()
                    for row in face_capability_matrix(
                        self._repo_root,
                        configured_recognition={
                            "recognizer_id": self._face_controller.settings.recognizer_id,
                            "backend": self._face_controller.settings.recognizer_backend,
                            "model": self._face_controller.settings.recognizer_model,
                            "device": self._face_controller.settings.recognizer_device,
                        },
                    )
                ],
            },
        )
        self._command_thread = threading.Thread(
            target=self._read_commands,
            name="wpf-bridge-stdin",
            daemon=True,
        )
        self._command_thread.start()
        self._command_timer.start()
        self._controller.start()
        self._person_controller.start()
        self._face_controller.start()
        self._emit_person_snapshot(self._person_controller.snapshot)
        self._emit_face_snapshot(self._face_controller.snapshot)
        self._emit_face_gallery(self._face_controller.gallery)

    def _read_commands(self) -> None:
        try:
            for line in sys.stdin:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    command = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    self._command_queue.put(
                        {"command": "invalid", "error": f"JSON non valido: {exc.msg}"}
                    )
                    continue
                if isinstance(command, dict):
                    self._command_queue.put(command)
        except (OSError, ValueError):
            pass
        finally:
            self._command_queue.put({"command": "shutdown"})

    @Slot()
    def _drain_commands(self) -> None:
        for _ in range(12):
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command)
            if self._stopping:
                return

    def _handle_command(self, command: dict[str, Any]) -> None:
        name = str(command.get("command", "")).strip().lower()
        data = command.get("data")
        if not isinstance(data, dict):
            data = {}
        try:
            if name == "shutdown":
                self._shutdown()
            elif name == "set_active_camera":
                self._set_active_camera(data.get("camera_id"))
            elif name == "save_camera":
                self._save_camera(data)
            elif name == "test_connection":
                self._test_connection(data)
            elif name == "set_person_detection":
                self._set_person_detection(data)
            elif name == "set_face_detection":
                self._set_face_settings(data, detection_only=True)
            elif name == "set_face_recognition":
                self._set_face_settings(data, detection_only=False)
            elif name == "set_face_gallery_root":
                self._set_face_gallery_root(data)
            elif name == "enroll_person":
                self._enroll_person(data)
            elif name == "import_enrollment":
                self._import_enrollment(data)
            elif name == "remove_person":
                self._remove_person(data)
            elif name in {"refresh_face_gallery", "refresh_gallery"}:
                self._emit_face_gallery(self._face_controller.refresh_gallery())
            elif name in {"refresh_face_capabilities", "refresh_capabilities"}:
                self._emit_face_capabilities(self._face_controller.refresh_capabilities())
            elif name == "ping":
                self._emit("pong", {})
            elif name == "invalid":
                self._emit("error", {"message": str(command.get("error", "Comando non valido"))})
            else:
                self._emit("error", {"message": f"Comando non supportato: {name}"})
        except Exception as exc:  # the frontend must receive a readable result
            message = redact_log_text(exc)
            self._logger.exception("WPF bridge command failed: %s", message)
            self._emit("error", {"message": message, "command": name})

    def _set_active_camera(self, camera_id: object) -> None:
        normalized = str(camera_id).strip() if camera_id else None
        if normalized is not None and not any(
            slot.camera_id == normalized for slot in self._controller.slots
        ):
            raise ValueError(f"camera '{normalized}' non disponibile")
        provider = self._controller.provider_for(normalized) if normalized else None
        self._active_camera_id = normalized
        if normalized is not None:
            self._last_preview_publish.pop(normalized, None)
        self._person_controller.set_active_camera(normalized, provider)
        self._face_controller.set_active_camera(normalized, provider)
        self._emit("active_camera", {"camera_id": normalized})

    def _save_camera(self, data: dict[str, Any]) -> None:
        draft = self._draft_from_data(data)
        result = self._repository.save(
            (draft,),
            current_path=self._config_path,
            credentials=self._credentials,
        )
        self._config_path = result.path
        value = result.values[0]
        self._emit(
            "camera_save_result",
            {
                "camera_id": value.draft.camera_id,
                "ok": True,
                "path": str(self._config_path),
                "url": redact_url(value.stream_url) if value.stream_url else None,
            },
        )
        try:
            self._controller.apply_camera_slot(value.to_slot())
        except Exception:
            # The YAML write succeeded, but the existing controller may be
            # applying a previous edit. The frontend can surface the error and
            # the next edit will retry through the same controller contract.
            raise

    def _test_connection(self, data: dict[str, Any]) -> None:
        draft = self._draft_from_data(data)
        validated = validate_camera_draft(draft)
        if validated.stream_url is None:
            raise ValueError("configurare un URL prima del test")
        test_url = runtime_stream_url(validated.stream_url, validated.credential_value)
        if test_url is None:
            raise ValueError("URL stream non configurato")
        self._connection_tester.start(
            draft.camera_id,
            test_url,
            validated.draft.transport,
        )

    def _set_person_detection(self, data: dict[str, Any]) -> None:
        prompts_value = data.get("prompts", ["person"])
        if isinstance(prompts_value, str):
            prompts = tuple(value.strip() for value in prompts_value.split(",") if value.strip())
        elif isinstance(prompts_value, (list, tuple)):
            prompts = tuple(str(value).strip() for value in prompts_value if str(value).strip())
        else:
            prompts = ("person",)
        prompts = prompts or ("person",)
        backend = str(data.get("backend") or "yoloe").strip().lower()
        model_value = data.get("model")
        model = str(model_value).strip() if model_value else None
        if backend == "fake":
            model = None
        settings = PersonDetectionSettings(
            enabled=bool(data.get("enabled", False)),
            backend=backend,
            model=model,
            confidence_threshold=float(data.get("confidence_threshold", 0.5)),
            inference_fps=float(data.get("inference_fps", 2.0)),
            device=str(data.get("device") or "auto").strip().lower(),
            precision=str(data.get("precision") or "fp16").strip().lower(),
            fallback_device=str(data.get("fallback_device") or "none").strip().lower(),
            image_size=int(data.get("image_size", 640)),
            classes=prompts,
            prompts=prompts,
            show_boxes=bool(data.get("show_boxes", True)),
            show_masks=bool(data.get("show_masks", False)),
        )
        self._config_path = self._repository.save_person_detection(
            settings,
            current_path=self._config_path,
        )
        self._person_controller.update_settings(settings)
        self._emit(
            "person_settings_saved",
            {"ok": True, "path": str(self._config_path), "settings": self._person_settings_data(settings)},
        )

    def _set_face_settings(self, data: dict[str, Any], *, detection_only: bool) -> None:
        current = self._face_controller.settings
        updates: dict[str, Any] = {}
        if detection_only:
            landmarker_id = str(
                data.get("landmarker_id", current.landmarker_id)
            ).strip().lower()
            landmarker_model = data.get("landmarker_model", current.landmarker_model)
            if "landmarker_id" in data:
                landmarker_model = resolve_landmarker_model(landmarker_id, self._repo_root)
            updates = {
                "face_detection_enabled": bool(data.get("enabled", data.get("face_detection_enabled", current.face_detection_enabled))),
                "detector_id": data.get("detector_id", current.detector_id),
                "detector_backend": str(data.get("backend", data.get("detector_backend", current.detector_backend))).strip().lower(),
                "detector_model": data.get("model", data.get("detector_model", current.detector_model)),
                "detector_device": str(data.get("device", data.get("detector_device", current.detector_device))).strip().lower(),
                "detector_confidence_threshold": float(data.get("confidence_threshold", current.detector_confidence_threshold)),
                "detector_inference_fps": float(data.get("inference_fps", current.detector_inference_fps)),
                "landmarks_enabled": bool(data.get("landmarks_enabled", current.landmarks_enabled)),
                "landmarker_id": landmarker_id,
                "landmarker_model": landmarker_model,
                "landmarker_device": str(data.get("landmarker_device", current.landmarker_device)).strip().lower(),
            }
        else:
            recognition_threshold = recognition_threshold_from_payload(
                current.recognition_threshold,
                data,
            )
            updates = {
                "recognition_enabled": bool(data.get("enabled", data.get("recognition_enabled", current.recognition_enabled))),
                "recognizer_id": data.get("recognizer_id", current.recognizer_id),
                "recognizer_backend": str(data.get("backend", data.get("recognizer_backend", current.recognizer_backend))).strip().lower(),
                "recognizer_model": data.get("model", data.get("recognizer_model", current.recognizer_model)),
                "recognizer_device": str(data.get("device", data.get("recognizer_device", current.recognizer_device))).strip().lower(),
                "recognition_threshold": recognition_threshold,
                "recognition_inference_fps": float(data.get("inference_fps", current.recognition_inference_fps)),
                "min_confirmations": int(data.get("min_confirmations", current.min_confirmations)),
                "confirmation_window_seconds": float(data.get("confirmation_window_seconds", current.confirmation_window_seconds)),
            }
        settings = replace(current, **updates)
        self._config_path = self._repository.save_face_analysis(
            settings,
            current_path=self._config_path,
        )
        self._face_controller.update_settings(settings)
        self._emit(
            "face_settings_saved",
            {
                "ok": True,
                "path": str(self._config_path),
                "section": "detection" if detection_only else "recognition",
                "settings": self._face_settings_data(settings),
            },
        )

    def _set_face_gallery_root(self, data: dict[str, Any]) -> None:
        raw_root = data.get("root") or data.get("directory")
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ValueError("set_face_gallery_root richiede una directory")
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"cartella Face gallery non trovata: {root}")

        self._config_path = self._repository.save_face_gallery_root(
            root,
            current_path=self._config_path,
        )
        self._config = load_config(self._config_path)
        self._face_controller.set_enrollment_root(root)
        self._face_controller.refresh_gallery()
        self._emit(
            "face_gallery_root_saved",
            {
                "ok": True,
                "path": str(self._config_path),
                "root": str(root),
            },
        )

    def _enroll_person(self, data: dict[str, Any]) -> None:
        person_id = str(data.get("person_id") or "").strip()
        images = data.get("images", data.get("images_dir"))
        if person_id and not images:
            PersonStore.validate_person_id(person_id)
            images = self._face_controller.enrollment_root / person_id
        name = str(data.get("name") or person_id).strip()
        if not name or not images:
            raise ValueError("enroll_person richiede una persona selezionata valida")
        report = self._face_controller.enroll_person(
            name,
            str(images),
            person_id=person_id or None,
            overwrite=bool(data.get("overwrite", False)),
        )
        self._emit(
            "face_enrollment_result",
            {
                "ok": report.record is not None,
                "name": report.name,
                "processed_count": report.processed_count,
                "accepted_count": report.accepted_count,
                "rejected_count": report.rejected_count,
                "images": [
                    {
                        "image": result.image_name,
                        "accepted": result.accepted,
                        "reasons": list(result.reasons),
                        "embedding_dimension": result.embedding_dimension,
                    }
                    for result in report.images
                ],
                "person_id": report.record.person_id if report.record is not None else None,
            },
        )

    def _remove_person(self, data: dict[str, Any]) -> None:
        person_id = str(data.get("person_id") or "").strip()
        if not person_id:
            raise ValueError("remove_person richiede person_id")
        self._face_controller.remove_person(person_id)

    def _import_enrollment(self, data: dict[str, Any]) -> None:
        root = data.get("root") or data.get("directory")
        report = self._face_controller.import_enrollment(str(root) if root else None)
        self._emit(
            "face_enrollment_batch_result",
            {"ok": not report.errors, **report.to_dict()},
        )

    def _face_settings_data(self, settings: FaceRecognitionSettings) -> dict[str, Any]:
        return settings.as_dict()

    def _draft_from_data(self, data: dict[str, Any]) -> CameraDraft:
        camera_id = str(data.get("camera_id") or "").strip()
        slot = next((value for value in self._controller.slots if value.camera_id == camera_id), None)
        if slot is None:
            raise ValueError(f"camera '{camera_id}' non disponibile")
        raw_port = data.get("port")
        if raw_port in (None, "", 0, "0"):
            port = None
        else:
            port = int(raw_port)
        return CameraDraft(
            camera_id=camera_id,
            slot_index=int(data.get("slot_index", slot.slot_index)),
            name=str(data.get("name") or ""),
            enabled=bool(data.get("enabled", False)),
            scheme=str(data.get("scheme") or "rtsp"),
            host=str(data.get("host") or ""),
            port=port,
            path=str(data.get("path") or ""),
            username=str(data.get("username") or ""),
            transport=str(data.get("transport") or "tcp"),
            query=str(data.get("query") or ""),
            fragment=str(data.get("fragment") or ""),
            password=str(data.get("password") or ""),
            clear_password=bool(data.get("clear_password", False)),
            existing_password=self._credentials.get(camera_id),
        )

    def _camera_data(self, slot: CameraSlot) -> dict[str, Any]:
        try:
            draft = draft_from_slot(slot, self._credentials)
        except ConfigurationError:
            if not (slot.stream_url or "").lower().startswith("fake://"):
                raise
            draft = CameraDraft(
                camera_id=slot.camera_id,
                slot_index=slot.slot_index,
                name=slot.name,
                enabled=slot.enabled,
                transport=slot.rtsp_transport,
            )
        return {
            "slot_index": slot.slot_index,
            "camera_id": slot.camera_id,
            "name": slot.name,
            "enabled": slot.enabled,
            "configured": slot.configured,
            "stream_url": redact_url(slot.stream_url) if slot.stream_url else None,
            "rtsp_transport": slot.rtsp_transport,
            "editor": {
                "scheme": draft.scheme,
                "host": draft.host,
                "port": draft.port,
                "path": draft.path,
                "username": draft.username,
                "transport": draft.transport,
                "query": draft.query,
                "fragment": draft.fragment,
                "password_stored": draft.password_is_stored,
            },
        }

    def _person_settings_data(self, settings: PersonDetectionSettings) -> dict[str, Any]:
        return {
            "enabled": settings.enabled,
            "backend": settings.backend,
            "model": settings.model,
            "confidence_threshold": settings.confidence_threshold,
            "inference_fps": settings.inference_fps,
            "device": settings.device,
            "precision": settings.precision,
            "fallback_device": settings.fallback_device,
            "image_size": settings.image_size,
            "classes": list(settings.classes),
            "prompts": list(settings.prompts),
            "show_boxes": settings.show_boxes,
            "show_masks": settings.show_masks,
        }

    @Slot(str, bool, str)
    def _on_camera_reconfigured(self, camera_id: str, success: bool, message: str) -> None:
        self._emit(
            "camera_reconfigured",
            {"camera_id": camera_id, "ok": success, "message": redact_log_text(message)},
        )

    @Slot(str)
    def _on_connection_test_started(self, camera_id: str) -> None:
        self._emit("connection_test_started", {"camera_id": camera_id})

    @Slot(object)
    def _on_connection_test_finished(self, result: object) -> None:
        if not isinstance(result, ConnectionTestResult):
            return
        self._emit(
            "connection_test_result",
            {
                "camera_id": result.camera_id,
                "ok": result.success,
                "message": redact_log_text(result.message),
                "url": result.url,
            },
        )

    @Slot(str, object)
    def _on_snapshot(self, camera_id: str, value: object) -> None:
        if not isinstance(value, CameraViewSnapshot):
            return
        snapshot = value
        packet = snapshot.frame
        sequence = packet.sequence if packet is not None else None
        key = (snapshot.status.value, snapshot.message, sequence)
        if self._last_snapshot_key.get(camera_id) == key:
            return
        self._last_snapshot_key[camera_id] = key
        payload: dict[str, Any] = {
            "camera_id": camera_id,
            "slot_index": snapshot.slot.slot_index,
            "name": snapshot.slot.name,
            "enabled": snapshot.slot.enabled,
            "configured": snapshot.slot.configured,
            "status": snapshot.status.value,
            "message": snapshot.message,
            "last_frame_age_s": snapshot.last_frame_age_s,
            "display_fps": snapshot.display_fps,
            "frame_sequence": sequence,
            "stream_width": snapshot.stream_info.width if snapshot.stream_info else None,
            "stream_height": snapshot.stream_info.height if snapshot.stream_info else None,
            "codec": snapshot.stream_info.codec if snapshot.stream_info else None,
            "dropped_frames": (
                snapshot.worker_snapshot.dropped_frames
                if snapshot.worker_snapshot is not None
                else 0
            ),
        }
        if packet is not None and self._preview_due(camera_id):
            try:
                publisher = self._frame_publishers.get(camera_id)
                if publisher is None:
                    publisher = SharedFramePublisher(camera_id)
                    self._frame_publishers[camera_id] = publisher
                frame = (
                    packet.frame
                    if camera_id == self._active_camera_id
                    else self._thumbnail_frame(packet.frame)
                )
                payload.update(publisher.publish(packet.sequence, frame))
            except Exception as exc:
                self._logger.warning(
                    "camera=%s shared preview publish failed: %s",
                    camera_id,
                    redact_log_text(exc),
                )
        self._emit("snapshot", payload)

    def _preview_due(self, camera_id: str) -> bool:
        """Keep focus preview at UI FPS while throttling background tiles."""

        now = time.monotonic()
        if camera_id == self._active_camera_id:
            self._last_preview_publish[camera_id] = now
            return True
        previous = self._last_preview_publish.get(camera_id, 0.0)
        if now - previous < 0.2:  # 5 FPS is enough for small grid tiles.
            return False
        self._last_preview_publish[camera_id] = now
        return True

    @staticmethod
    def _thumbnail_frame(frame: Any, max_width: int = 480) -> Any:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[1] <= max_width:
            return image
        try:
            import cv2

            width = max_width
            height = max(1, int(round(image.shape[0] * width / image.shape[1])))
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        except Exception:
            # Preview optimization must never break camera status delivery.
            return image

    @Slot(object)
    def _on_person_snapshot(self, value: object) -> None:
        if isinstance(value, PersonDetectionSnapshot):
            self._face_controller.set_person_snapshot(value)
            self._emit_person_snapshot(value)

    def _emit_person_snapshot(self, snapshot: PersonDetectionSnapshot) -> None:
        detections = []
        for item in snapshot.detections:
            detections.append(
                {
                    "bbox": list(item.bbox),
                    "confidence": item.confidence,
                    "label": item.label,
                    "mask_polygon": [list(point) for point in item.mask_polygon]
                    if item.mask_polygon
                    else None,
                }
            )
        self._emit(
            "person_detection",
            {
                "camera_id": snapshot.camera_id,
                "status": snapshot.status.value,
                "status_label": snapshot.status.label,
                "message": snapshot.message,
                "model_name": snapshot.model_name,
                "model_path": snapshot.model_path,
                "requested_device": snapshot.requested_device,
                "actual_device": snapshot.actual_device,
                "device_verified": snapshot.device_verified,
                "provider": snapshot.provider,
                "backend": snapshot.backend,
                "precision": snapshot.precision,
                "inference_fps": snapshot.inference_fps,
                "latency_ms": snapshot.latency_ms,
                "person_count": snapshot.person_count,
                "detection_count": snapshot.detection_count,
                "source_width": snapshot.source_width,
                "source_height": snapshot.source_height,
                "result_monotonic": snapshot.result_monotonic,
                "detections": detections,
            },
        )

    @Slot(object)
    def _on_face_snapshot(self, value: object) -> None:
        if isinstance(value, FaceRecognitionSnapshot):
            self._emit_face_snapshot(value)

    def _emit_face_snapshot(self, snapshot: FaceRecognitionSnapshot) -> None:
        settings = self._face_controller.settings
        recognition_payload = snapshot.to_dict()
        recognition_payload["status"] = (
            snapshot.recognition_status or snapshot.status
        ).value
        recognition_payload["message"] = snapshot.recognition_message or snapshot.message
        recognition_payload["error"] = snapshot.recognition_error or snapshot.error
        recognition_payload.update(
            {
                "detector_backend": snapshot.detector_backend or settings.detector_backend,
                "detector_model": snapshot.detector_model or settings.detector_model,
                "recognizer_backend": snapshot.recognizer_backend or settings.recognizer_backend,
                "recognizer_model": snapshot.recognizer_model or settings.recognizer_model,
            }
        )
        self._emit("face_recognition_state", recognition_payload)
        # Keep a distinct detection message so clients can show the face stage
        # independently from recognition while sharing the same overlays.
        self._emit(
            "face_detection_state",
            {
                "camera_id": snapshot.camera_id,
                "status": (snapshot.detection_status or snapshot.status).value,
                "status_label": (snapshot.detection_status or snapshot.status).label,
                "message": snapshot.detection_message or snapshot.message,
                "error": snapshot.detection_error or snapshot.error,
                "detector_id": snapshot.detector_id,
                "detector_backend": snapshot.detector_backend or settings.detector_backend,
                "detector_model": snapshot.detector_model or settings.detector_model,
                "requested_device": snapshot.requested_detector_device,
                "actual_device": snapshot.actual_detector_device,
                "device_verified": snapshot.actual_detector_device is not None,
                "landmarker_id": settings.landmarker_id,
                "landmarker_model": settings.landmarker_model,
                "landmarker_device": settings.landmarker_device,
                "face_count": snapshot.face_count,
                "frame_sequence": snapshot.frame_sequence,
                "overlays": [overlay.to_dict() for overlay in snapshot.overlays],
            },
        )

    @Slot(object)
    def _on_face_gallery(self, value: object) -> None:
        if isinstance(value, FaceGalleryState):
            self._emit_face_gallery(value)

    def _emit_face_gallery(self, gallery: FaceGalleryState) -> None:
        self._emit("face_gallery_state", gallery.to_dict())

    @Slot(object)
    def _on_face_capabilities(self, value: object) -> None:
        if isinstance(value, (list, tuple)):
            self._emit_face_capabilities(value)

    def _emit_face_capabilities(self, value: object) -> None:
        self._emit("face_capabilities", {"items": face_capability_rows(value)})

    def _emit(self, message_type: str, data: dict[str, Any]) -> None:
        line = json.dumps(
            {"type": message_type, "data": data},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._write_lock:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                self._shutdown()

    def _shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._command_timer.stop()
        try:
            self._person_controller.stop(timeout_s=1.5)
            self._face_controller.stop(timeout_s=1.5)
        finally:
            self._controller.stop(timeout_s=1.0)
            for publisher in self._frame_publishers.values():
                publisher.close()
            self._frame_publishers.clear()
        self._emit("stopped", {})
        QTimer.singleShot(0, self._qt_app.quit)


def face_capability_rows(value: object) -> list[dict[str, Any]]:
    """Normalize capability dataclasses and mappings for the WPF DTO contract."""

    rows: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return rows
    for row in value:
        if isinstance(row, dict):
            rows.append(dict(row))
        elif hasattr(row, "to_dict"):
            rows.append(dict(row.to_dict()))
        else:
            raise TypeError(f"unsupported face capability row: {type(row).__name__}")
    return rows


def resolve_landmarker_model(model_id: str, repo_root: Path) -> str:
    """Resolve a registry landmarker ID to its portable configured path.

    The WPF client persists the registry-relative path, while the capability
    check below proves that the paired local artifacts exist before a new
    selection is accepted.  Runtime factories resolve the same path against
    the repository root.
    """

    normalized = str(model_id).strip().lower()
    if normalized != LANDMARKER_SPEC.model_id:
        raise ValueError(f"unsupported face landmarker model: {model_id}")
    resolved = model_path(LANDMARKER_SPEC, Path(repo_root).resolve())
    required = (resolved, resolved.with_suffix(".bin"))
    missing = tuple(str(path) for path in required if not path.is_file())
    if missing:
        raise ValueError("face landmarker artifact missing: " + ", ".join(missing))
    return LANDMARKER_SPEC.relative_path


def recognition_threshold_from_payload(
    current: float | None,
    data: dict[str, Any],
) -> float | None:
    """Apply explicit recognition threshold updates, including ``null``."""

    if "threshold" not in data:
        return current
    raw = data["threshold"]
    return None if raw is None else float(raw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fake_offline_camera and not args.fake_cameras:
        print("--fake-offline-camera richiede --fake-cameras", file=sys.stderr)
        return 2
    if args.fake_reconnect_camera and not args.fake_cameras:
        print("--fake-reconnect-camera richiede --fake-cameras", file=sys.stderr)
        return 2
    configure_logging(args.log_level or "INFO")
    try:
        qt_app = QCoreApplication(sys.argv)
        qt_app.setApplicationName("Local Security Monitor WPF bridge")
        runtime = BridgeRuntime(qt_app, args)
        runtime.start()
        return qt_app.exec()
    except (ConfigurationError, CredentialStoreError) as exc:
        print(f"BRIDGE CONFIGURATION ERROR: {redact_log_text(exc)}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"BRIDGE ERROR: {redact_log_text(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
