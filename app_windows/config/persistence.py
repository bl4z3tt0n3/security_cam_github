"""Atomic, secret-free persistence for camera editor changes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml

from app.config import ConfigurationError
from app_windows.models.camera_view_state import CameraSlot
from app_windows.models.person_detection_state import (
    DEFAULT_YOLOE_MODEL,
    PersonDetectionSettings,
)

from .camera_config import CameraDraft, ValidatedCameraDraft, validate_camera_draft
from .credentials import CredentialStore


@dataclass(frozen=True)
class CameraSaveResult:
    path: Path
    values: tuple[ValidatedCameraDraft, ...]

    @property
    def slots(self) -> tuple[CameraSlot, ...]:
        return tuple(value.to_slot() for value in self.values)


class CameraConfigRepository:
    """Update only selected camera mappings while retaining the YAML document."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root)

    @property
    def local_path(self) -> Path:
        return self.repo_root / "config" / "config.local.yaml"

    def writable_path(self, current_path: Path | str) -> Path:
        current = Path(current_path)
        example = self.repo_root / "config" / "config.example.yaml"
        try:
            is_example = current.resolve() == example.resolve()
        except OSError:
            is_example = current.name == example.name
        return self.local_path if is_example else current

    def save(
        self,
        drafts: Iterable[CameraDraft],
        *,
        current_path: Path | str,
        credentials: CredentialStore,
    ) -> CameraSaveResult:
        drafts_tuple = tuple(drafts)
        if not drafts_tuple:
            return CameraSaveResult(path=self.writable_path(current_path), values=())

        validated = tuple(validate_camera_draft(draft) for draft in drafts_tuple)
        ids = [value.draft.camera_id for value in validated]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("non è possibile salvare due volte la stessa camera")

        target = self.writable_path(current_path)
        current = Path(current_path)
        if target.is_file():
            source = target
        elif current.is_file():
            source = current
        elif _same_path(target, self.local_path):
            # A caller may already point at config.local.yaml even when this
            # is the first save. Seed it from the tracked example just as the
            # GUI does when it starts with config.example.yaml.
            source = self.repo_root / "config" / "config.example.yaml"
        else:
            source = current
        document = _read_yaml_mapping(source)
        raw_cameras = document.get("cameras")
        if raw_cameras is None:
            raw_cameras = []
            document["cameras"] = raw_cameras
        if not isinstance(raw_cameras, list):
            raise ConfigurationError("la sezione cameras deve essere una lista YAML")

        global_transport = _global_transport(document)
        entries_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_cameras:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                entries_by_id[item["id"]] = item

        for value in validated:
            draft = value.draft
            entry = entries_by_id.get(draft.camera_id)
            if entry is None:
                entry = {"id": draft.camera_id, "source_type": "opencv"}
                raw_cameras.append(entry)
                entries_by_id[draft.camera_id] = entry

            entry["enabled"] = draft.enabled
            normalized_name = draft.name.strip()
            if normalized_name:
                entry["name"] = normalized_name
            else:
                entry.pop("name", None)

            # This is the only URL written by the editor, and it never contains
            # the password.  The runtime adds the DPAPI value in memory.
            if value.stream_url is None:
                entry.pop("stream_url", None)
            else:
                entry["stream_url"] = value.stream_url

            if draft.transport == global_transport:
                entry.pop("rtsp_transport", None)
            else:
                entry["rtsp_transport"] = draft.transport

        old_credentials: dict[str, str | None] = {}
        credential_updates: dict[str, str | None] = {}
        for value in validated:
            if not value.credential_update_required:
                continue
            camera_id = value.draft.camera_id
            old_credentials[camera_id] = credentials.get(camera_id)
            credential_updates[camera_id] = value.credential_value

        if credential_updates:
            credentials.apply(credential_updates)
        try:
            _atomic_write_yaml(target, document)
        except Exception:
            if credential_updates:
                try:
                    credentials.apply(old_credentials)
                except Exception:
                    # Do not mask the original persistence error and never log
                    # the values involved in the rollback.
                    pass
            raise

        return CameraSaveResult(path=target, values=validated)

    def save_person_detection(
        self,
        settings: PersonDetectionSettings,
        *,
        current_path: Path | str,
    ) -> Path:
        """Persist inference settings without touching cameras or credentials."""

        target = self.writable_path(current_path)
        current = Path(current_path)
        if target.is_file():
            source = target
        elif current.is_file():
            source = current
        elif _same_path(target, self.local_path):
            source = self.repo_root / "config" / "config.example.yaml"
        else:
            source = current

        document = _read_yaml_mapping(source)
        person_detection = document.setdefault("person_detection", {})
        if not isinstance(person_detection, dict):
            raise ConfigurationError("la sezione person_detection deve essere una mappa YAML")
        person_detection["enabled"] = settings.enabled
        person_detection["backend"] = settings.backend
        person_detection["model"] = settings.model
        person_detection["confidence_threshold"] = settings.confidence_threshold
        person_detection["precision"] = settings.precision
        person_detection["device"] = settings.device
        person_detection["fallback_device"] = settings.fallback_device
        person_detection["image_size"] = settings.image_size
        person_detection["classes"] = list(settings.classes)
        person_detection["prompts"] = list(settings.prompts)
        person_detection["show_masks"] = settings.show_masks

        inference = document.setdefault("inference", {})
        if not isinstance(inference, dict):
            raise ConfigurationError("la sezione inference deve essere una mappa YAML")
        inference["person_detection_fps"] = settings.inference_fps

        windows_ui = document.setdefault("windows_ui", {})
        if not isinstance(windows_ui, dict):
            raise ConfigurationError("la sezione windows_ui deve essere una mappa YAML")
        windows_ui["show_person_boxes"] = settings.show_boxes

        _atomic_write_yaml(target, document)
        return target

    def save_face_analysis(
        self,
        settings: Any,
        *,
        current_path: Path | str,
    ) -> Path:
        """Persist canonical face-stage settings without secrets or live data."""

        target = self.writable_path(current_path)
        current = Path(current_path)
        if target.is_file():
            source = target
        elif current.is_file():
            source = current
        elif _same_path(target, self.local_path):
            source = self.repo_root / "config" / "config.example.yaml"
        else:
            source = current

        document = _read_yaml_mapping(source)
        face = document.setdefault("face_detection", {})
        landmarks = document.setdefault("face_landmarks", {})
        recognition = document.setdefault("recognition", {})
        if not all(isinstance(value, dict) for value in (face, landmarks, recognition)):
            raise ConfigurationError("le sezioni face_detection/face_landmarks/recognition devono essere mappe YAML")

        face.update(
            {
                "enabled": bool(settings.face_detection_enabled),
                "detector_id": settings.detector_id,
                "backend": settings.detector_backend,
                "model": settings.detector_model,
                "device": settings.detector_device,
                "confidence_threshold": float(settings.detector_confidence_threshold),
                "inference_fps": float(settings.detector_inference_fps),
            }
        )
        for legacy in (
            "backend_id",
            "model_id",
            "roi_mode",
            "show_boxes",
            "show_confidence",
            "show_detector",
            "show_inference_time",
            "show_landmarks",
        ):
            face.pop(legacy, None)
        landmarks.update(
            {
                "enabled": bool(settings.landmarks_enabled),
                "landmarker_id": settings.landmarker_id,
                "backend": settings.landmarker_backend,
                "model": settings.landmarker_model,
                "device": settings.landmarker_device,
            }
        )
        recognition.update(
            {
                "enabled": bool(settings.recognition_enabled),
                "recognizer_id": settings.recognizer_id,
                "backend": settings.recognizer_backend,
                "model": settings.recognizer_model,
                "device": settings.recognizer_device,
                "threshold": settings.recognition_threshold,
                "inference_fps": float(settings.recognition_inference_fps),
                "min_confirmations": int(settings.min_confirmations),
                "confirmation_window_seconds": float(settings.confirmation_window_seconds),
            }
        )
        for legacy in ("backend_id", "model_id"):
            recognition.pop(legacy, None)
        _atomic_write_yaml(target, document)
        return target

    def save_face_gallery_root(
        self,
        root: Path | str,
        *,
        current_path: Path | str,
    ) -> Path:
        """Persist the configured face-enrollment source without touching other settings."""

        target = self.writable_path(current_path)
        current = Path(current_path)
        if target.is_file():
            source = target
        elif current.is_file():
            source = current
        elif _same_path(target, self.local_path):
            source = self.repo_root / "config" / "config.example.yaml"
        else:
            source = current

        document = _read_yaml_mapping(source)
        storage = document.setdefault("storage", {})
        if not isinstance(storage, dict):
            raise ConfigurationError("la sezione storage deve essere una mappa YAML")
        storage["enrollment_dir"] = str(Path(root).expanduser())
        _atomic_write_yaml(target, document)
        return target

    def migrate_person_detection(self, *, current_path: Path | str) -> Path:
        """Migrate the old absent ONNX setting into the local YOLOE config.

        Only the ignored local file is ever written when the caller points at
        ``config.example.yaml``. The source document is copied as a mapping,
        so cameras, custom sections and all credential-related content remain
        untouched.
        """

        target = self.writable_path(current_path)
        current = Path(current_path)
        if target.is_file():
            source = target
        elif current.is_file():
            source = current
        elif _same_path(target, self.local_path):
            source = self.repo_root / "config" / "config.example.yaml"
        else:
            return current

        document = _read_yaml_mapping(source)
        person_detection = document.get("person_detection")
        if not isinstance(person_detection, dict):
            return current

        changed = False
        configured_model = person_detection.get("model")
        if configured_model is None or str(configured_model).lower().endswith(".onnx"):
            person_detection["backend"] = "yoloe"
            person_detection["model"] = DEFAULT_YOLOE_MODEL
            changed = True
        if "prompts" not in person_detection:
            person_detection["prompts"] = ["person"]
            changed = True
        if "show_masks" not in person_detection:
            person_detection["show_masks"] = False
            changed = True

        if not changed:
            return current
        _atomic_write_yaml(target, document)
        return target


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"impossibile leggere la configurazione locale: {path}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError("la radice della configurazione deve essere una mappa YAML")
    return raw


def _global_transport(document: dict[str, Any]) -> str:
    video = document.get("video")
    if isinstance(video, dict):
        value = video.get("rtsp_transport")
        if isinstance(value, str) and value.lower() in {"auto", "tcp", "udp"}:
            return value.lower()
    return "tcp"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _atomic_write_yaml(path: Path, document: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(
        document,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    _atomic_write_text(path, rendered)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise ConfigurationError(f"impossibile scrivere la configurazione locale: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
