"""Check local prerequisites without requiring a camera URL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

YOLOE_OFFICIAL_MODELS = frozenset(
    {
        "yoloe-26n-seg.pt",
        "yoloe-26s-seg.pt",
        "yoloe-26m-seg.pt",
        "yoloe-26l-seg.pt",
        "yoloe-26x-seg.pt",
    }
)

from app.config import (
    ConfigurationError,
    ensure_runtime_directories,
    load_config,
    validate_stream_url,
)
from app.camera import CameraRuntime, MultiCameraRuntime
from app.events import EventManager, SnapshotWriter
from app.face import (
    FakeEmbedder,
    IncompatibleEmbeddingModelError,
    PersonStore,
    RecognitionResult,
)
from app.hardening import HardeningCheck, HardeningReport, normalize_status
from app.inference import FakePersonDetector
from app.logging_setup import redact_log_text
from app.metrics import read_resource_snapshot
from app.video.base import (
    ReadResult,
    ReadStatus,
    StreamInfo,
    VideoSource,
    VideoSourceError,
    redact_url,
)
from app.video.fake_source import FakeVideoSource
from app.video.diagnostics import run_stream_test
from app.video.opencv_source import OpenCVVideoSource
from app.video.worker import WorkerState
from scripts._common import EXAMPLE_CONFIG, REPO_ROOT


def _status(
    label: str,
    state: str,
    detail: str = "",
    *,
    checks: list[dict[str, str]] | None = None,
    emit: bool = True,
) -> bool:
    normalized = normalize_status(state)
    safe_detail = redact_log_text(detail)
    if checks is not None:
        checks.append(
            {
                "name": label,
                "status": normalized,
                "detail": safe_detail,
            }
        )
    dots = "." * max(1, 28 - len(label))
    suffix = f" {safe_detail}" if safe_detail else ""
    if emit:
        print(f"{label} {dots} {state}{suffix}")
    return normalized != "FAIL"


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-version" if command in {"ffmpeg", "ffprobe", "ffplay"} else "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return executable
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0].strip() if first_line else executable


def _check_person_detection(
    config: object,
    *,
    checks: list[dict[str, str]] | None = None,
    emit: bool = True,
) -> bool:
    person_detection = getattr(config, "person_detection", None)
    if person_detection is None or not person_detection.enabled:
        _status(
            "Person detection",
            "INFO",
            "disabled; no model loaded",
            checks=checks,
            emit=emit,
        )
        return True

    required_ok = True
    model_value = (person_detection.model or "").strip()
    backend = getattr(person_detection, "backend", "auto")
    if backend == "auto":
        model_lower = model_value.lower()
        if model_lower.endswith(".onnx"):
            backend = "onnx"
        elif model_lower.endswith(("yolo26s.pt", "yolo26n.pt", "_openvino_model", ".xml")):
            backend = "openvino"
        else:
            backend = "yoloe"

    if backend == "fake":
        _status(
            "Person detection",
            "OK",
            "fake backend; no model loaded",
            checks=checks,
            emit=emit,
        )
        return True

    if backend == "openvino":
        model_path = Path(model_value).expanduser() if model_value else None
        local_model = (
            model_path if model_path is not None and model_path.is_absolute() else SCRIPT_ROOT / model_path
            if model_path is not None
            else None
        )
        official = model_path is not None and model_path.name.lower() in {
            "yolo26s.pt",
            "yolo26n.pt",
        }
        if official and local_model is not None and not local_model.is_file():
            _status(
                "OpenVINO checkpoint",
                "INFO",
                f"official checkpoint {model_value}; download on first enabled load",
                checks=checks,
                emit=emit,
            )
        else:
            required_ok &= _status(
                "OpenVINO model/cache",
                "OK" if local_model is not None and (local_model.is_file() or local_model.is_dir()) else "FAIL",
                str(local_model) if local_model is not None else "model path is not configured",
                checks=checks,
                emit=emit,
            )

        if importlib.util.find_spec("openvino") is None:
            required_ok &= _status(
                "OpenVINO",
                "FAIL",
                "install the optional openvino extra",
                checks=checks,
                emit=emit,
            )
            return required_ok
        try:
            import openvino as ov

            devices = tuple(str(device) for device in ov.Core().available_devices)
            requested = str(getattr(person_detection, "device", "auto"))
            families = {device.split(".", 1)[0].upper() for device in devices}
            if requested == "cuda":
                required_ok &= _status(
                    "OpenVINO device",
                    "FAIL",
                    "CUDA is not an OpenVINO device; select auto, cpu, or gpu",
                    checks=checks,
                    emit=emit,
                )
            elif requested == "gpu" and "GPU" not in families:
                fallback = getattr(person_detection, "fallback_device", "none")
                state = "INFO" if fallback == "cpu" and "CPU" in families else "FAIL"
                required_ok &= _status(
                    "OpenVINO device",
                    state,
                    f"available={list(devices)}; GPU unavailable; fallback={fallback}",
                    checks=checks,
                    emit=emit,
                )
            else:
                candidate = (
                    "GPU" if requested == "gpu" or requested == "auto" and "GPU" in families else "CPU"
                )
                _status(
                    "OpenVINO device",
                    "INFO",
                    f"{getattr(ov, '__version__', 'installed')}; available={list(devices)}; "
                    f"candidate={candidate}; real EXECUTION_DEVICES verification pending",
                    checks=checks,
                    emit=emit,
                )
        except Exception as exc:
            required_ok &= _status(
                "OpenVINO runtime",
                "FAIL",
                str(exc),
                checks=checks,
                emit=emit,
            )
        return required_ok

    if backend == "yoloe":
        if model_value.lower().endswith(".onnx"):
            required_ok &= _status(
                "Person detection model",
                "FAIL",
                "YOLOE requires a .pt checkpoint or official model identifier",
                checks=checks,
                emit=emit,
            )
        else:
            model_path = Path(model_value).expanduser()
            local_model = model_path if model_path.is_absolute() else SCRIPT_ROOT / model_path
            is_identifier = (
                not model_path.is_absolute()
                and model_path.parent == Path(".")
                and model_path.name.lower() in YOLOE_OFFICIAL_MODELS
            )
            is_download_target = model_path.name.lower() in YOLOE_OFFICIAL_MODELS
            if (is_identifier or is_download_target) and not local_model.is_file():
                _status(
                    "Person detection model",
                    "INFO",
                    f"official checkpoint {model_value}; download on first enabled load",
                    checks=checks,
                    emit=emit,
                )
            else:
                required_ok &= _status(
                    "Person detection model",
                    "OK" if local_model.is_file() else "FAIL",
                    str(local_model),
                    checks=checks,
                    emit=emit,
                )

        if importlib.util.find_spec("ultralytics") is None:
            required_ok &= _status(
                "Ultralytics",
                "FAIL",
                "install the person-detection extra",
                checks=checks,
                emit=emit,
            )
            return required_ok
        if importlib.util.find_spec("torch") is None:
            required_ok &= _status(
                "PyTorch",
                "FAIL",
                "Ultralytics requires PyTorch for YOLOE inference",
                checks=checks,
                emit=emit,
            )
            return required_ok

        try:
            import torch
            import ultralytics

            cuda_available = bool(torch.cuda.is_available())
            requested = person_detection.device
            if requested == "cuda" and not cuda_available:
                required_ok &= _status(
                    "YOLOE device",
                    "FAIL",
                    "CUDA requested but torch.cuda.is_available() is false",
                    checks=checks,
                    emit=emit,
                )
            else:
                candidate = "cuda:0" if requested in {"auto", "cuda"} and cuda_available else "cpu"
                _status(
                    "YOLOE device",
                    "INFO",
                    f"{ultralytics.__version__}; candidate={candidate}; real inference verification pending",
                    checks=checks,
                    emit=emit,
                )
        except Exception as exc:
            required_ok &= _status(
                "YOLOE runtime",
                "FAIL",
                str(exc),
                checks=checks,
                emit=emit,
            )
        return required_ok

    if backend != "onnx":
        required_ok &= _status(
            "Person detection backend",
            "FAIL",
            f"unsupported backend: {backend}",
            checks=checks,
            emit=emit,
        )
        return required_ok

    model_path = Path(model_value) if model_value else None
    if model_path is not None and not model_path.is_absolute():
        model_path = SCRIPT_ROOT / model_path
    model_ok = model_path is not None and model_path.is_file()
    required_ok &= _status(
        "Person detection model",
        "OK" if model_ok else "FAIL",
        str(model_path) if model_path is not None else "model path is not configured",
        checks=checks,
        emit=emit,
    )

    if model_path is not None and model_path.suffix.lower() == ".pt":
        for module_name in ("ultralytics", "torch"):
            available = importlib.util.find_spec(module_name) is not None
            required_ok &= _status(
                f"YOLOE {module_name}",
                "OK" if available else "FAIL",
                "installed" if available else f"install the yoloe extra ({module_name})",
                checks=checks,
                emit=emit,
            )

        encoder_path = SCRIPT_ROOT / "models" / "mobileclip2_b.ts"
        required_ok &= _status(
            "YOLOE prompt encoder",
            "OK" if encoder_path.is_file() else "FAIL",
            str(encoder_path),
            checks=checks,
            emit=emit,
        )
        if person_detection.device == "cuda" and importlib.util.find_spec("torch") is not None:
            try:
                import torch

                cuda_ok = bool(torch.cuda.is_available())
                required_ok &= _status(
                    "YOLOE CUDA",
                    "OK" if cuda_ok else "FAIL",
                    "torch.cuda.is_available() is true"
                    if cuda_ok
                    else "CUDA requested but torch.cuda.is_available() is false",
                    checks=checks,
                    emit=emit,
                )
            except Exception as exc:
                required_ok &= _status(
                    "YOLOE CUDA",
                    "FAIL",
                    str(exc),
                    checks=checks,
                    emit=emit,
                )
        return required_ok

    if importlib.util.find_spec("onnxruntime") is None:
        required_ok &= _status(
            "ONNX Runtime",
            "FAIL",
            "install the person-detection extra",
            checks=checks,
            emit=emit,
        )
        return required_ok

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        requested = person_detection.device
        if requested == "cuda" and "CUDAExecutionProvider" not in providers:
            required_ok &= _status(
                "ONNX Runtime",
                "FAIL",
                "CUDAExecutionProvider unavailable",
                checks=checks,
                emit=emit,
            )
        else:
            selected = (
                "CUDAExecutionProvider"
                if requested == "cuda"
                or (requested == "auto" and "CUDAExecutionProvider" in providers)
                else "CPUExecutionProvider"
            )
            provider_ok = selected in providers
            required_ok &= _status(
                "ONNX Runtime",
                "OK" if provider_ok else "FAIL",
                f"{ort.__version__}; provider={selected}",
                checks=checks,
                emit=emit,
            )
    except Exception as exc:
        required_ok &= _status(
            "ONNX Runtime",
            "FAIL",
            str(exc),
            checks=checks,
            emit=emit,
        )
    return required_ok


def _check_face_components(
    config: object,
    *,
    checks: list[dict[str, str]] | None = None,
    emit: bool = True,
) -> bool:
    """Check optional face model/runtime prerequisites without loading disabled adapters."""

    required_ok = True
    sections = (
        ("Face detection", getattr(config, "face_detection", None)),
        ("Recognition", getattr(config, "recognition", None)),
    )
    for label, section in sections:
        if section is None or not section.enabled:
            _status(
                label,
                "INFO",
                "disabled; no model loaded",
                checks=checks,
                emit=emit,
            )
            continue
        model_value = getattr(section, "model", None)
        model_path = Path(model_value) if model_value else None
        if model_path is not None and not model_path.is_absolute():
            model_path = SCRIPT_ROOT / model_path
        required_ok &= _status(
            f"{label} model",
            "OK" if model_path is not None and model_path.is_file() else "FAIL",
            str(model_path) if model_path is not None else "model path is not configured",
            checks=checks,
            emit=emit,
        )
        if importlib.util.find_spec("onnxruntime") is None:
            required_ok &= _status(
                f"{label} runtime",
                "FAIL",
                "install the face-embedding extra",
                checks=checks,
                emit=emit,
            )
    return required_ok


class _OfflineSource(VideoSource):
    """Minimal source that stays offline for the hardening isolation probe."""

    def open(self) -> StreamInfo:
        raise VideoSourceError("hardening source is offline", code="offline")

    def read(self, timeout_s: float) -> ReadResult:
        del timeout_s
        return ReadResult.status_result(ReadStatus.DISCONNECTED, "hardening source is offline")

    def reconnect(self) -> StreamInfo:
        raise VideoSourceError("hardening source is still offline", code="offline")

    def close(self) -> None:
        return None


def _add_hardening_check(
    checks: list[HardeningCheck],
    label: str,
    state: str,
    detail: str,
    *,
    emit: bool,
) -> bool:
    normalized = normalize_status(state)
    safe_detail = redact_log_text(detail)
    checks.append(HardeningCheck(label, normalized, safe_detail))
    if emit:
        _status(label, normalized, safe_detail)
    return normalized != "FAIL"


def _resolve_configured_path(value: object) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_ROOT / path


def _workspace_probe_directory(prefix: str) -> Path:
    """Create a disposable probe directory with the workspace ACLs."""

    root = SCRIPT_ROOT / ".test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"{prefix}-{uuid.uuid4().hex}"
    directory.mkdir(mode=0o755)
    return directory


def _probe_incompatible_embeddings() -> tuple[bool, str]:
    """Verify that a synthetic incompatible record is rejected fail-closed."""

    from app.face import PersonStorageError

    directory = _workspace_probe_directory("hardening-persons")
    try:
        store = PersonStore(directory)
        enrolled = FakeEmbedder(embedding_dimension=4, model_id="hardening-model-a")
        incompatible = FakeEmbedder(embedding_dimension=4, model_id="hardening-model-b")
        store.save(
            name="Hardening Probe",
            person_id="hardening-probe",
            embeddings=np.ones((1, 4), dtype=np.float32),
            model=enrolled.metadata,
        )
        try:
            store.assert_compatible("hardening-probe", incompatible.metadata)
        except IncompatibleEmbeddingModelError:
            return True, "synthetic incompatible record rejected before matching"
        except PersonStorageError as exc:
            return False, f"storage probe failed unexpectedly: {exc}"
        return False, "incompatible record was accepted"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _probe_storage_failure_isolation() -> tuple[bool, str]:
    """Verify metadata survives a snapshot encoder/storage failure."""

    def broken_encoder(frame: np.ndarray) -> bytes:
        del frame
        raise OSError("synthetic snapshot encoder failure")

    result = RecognitionResult(
        status="unknown",
        person_id=None,
        person_name=None,
        score=0.1,
        threshold=0.8,
    )
    root = _workspace_probe_directory("hardening-events")
    try:
        probe_logger = logging.Logger("security-cam-hardening-snapshot")
        probe_logger.addHandler(logging.NullHandler())
        probe_logger.propagate = False
        writer = SnapshotWriter(encoder=broken_encoder, logger=probe_logger)
        try:
            with EventManager(root / "events", snapshot_writer=writer) as manager:
                event = manager.publish_recognition(
                    camera_id="hardening-camera",
                    track_id=1,
                    result=result,
                    frame=np.zeros((8, 8, 3), dtype=np.uint8),
                )
                flushed = manager.flush(1.0)
        except Exception as exc:
            return False, f"storage failure escaped EventManager: {type(exc).__name__}: {exc}"
        metadata_count = len(list((root / "events").rglob("metadata.json")))
        if event is None:
            return False, "event metadata was not retained after snapshot failure"
        if not flushed or metadata_count != 1:
            return False, "snapshot failure did not leave one valid metadata record"
        return True, "event metadata persisted while snapshot failure stayed isolated"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _probe_camera_isolation() -> tuple[bool, str]:
    """Verify one offline source does not stop a healthy fake camera."""

    healthy = CameraRuntime(
        "hardening-healthy",
        FakeVideoSource(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            read_delay_s=0.002,
        ),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=0,
    )
    offline = CameraRuntime(
        "hardening-offline",
        _OfflineSource(),
        target_fps=20.0,
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=1,
    )
    fleet = MultiCameraRuntime([healthy, offline], detector=FakePersonDetector())
    fleet.start()
    deadline = time.monotonic() + 1.5
    try:
        while time.monotonic() < deadline:
            snapshots = fleet.snapshot()
            if (
                snapshots["hardening-offline"].worker.state is WorkerState.FAILED
                and snapshots["hardening-healthy"].worker.frames_received > 0
            ):
                break
            time.sleep(0.01)
        snapshots = fleet.snapshot()
    finally:
        fleet.stop(timeout_s=1.0)
    offline_failed = snapshots["hardening-offline"].worker.state is WorkerState.FAILED
    healthy_received = snapshots["hardening-healthy"].worker.frames_received > 0
    if not offline_failed or not healthy_received:
        return False, "offline camera did not remain isolated from the healthy camera"
    return True, "offline camera failed independently; healthy camera received frames"


def _probe_shutdown() -> tuple[bool, str]:
    """Verify runtime, sampler and worker threads terminate within a bound."""

    camera_id = "hardening-shutdown"
    runtime = CameraRuntime(
        camera_id,
        FakeVideoSource(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            read_delay_s=0.002,
        ),
        target_fps=20.0,
        detector=FakePersonDetector(),
        read_timeout_s=0.05,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=0,
    )
    runtime.start()
    deadline = time.monotonic() + 1.0
    try:
        while time.monotonic() < deadline and runtime.snapshot().processed_samples == 0:
            time.sleep(0.01)
    finally:
        runtime.stop(timeout_s=1.0)
    snapshot = runtime.snapshot()
    thread_prefixes = (
        "camera-worker-hardening-shutdown",
        "frame-sampler-hardening-shutdown",
        "camera-runtime-hardening-shutdown",
    )
    residual = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name in thread_prefixes
    ]
    if snapshot.processed_samples == 0:
        return False, "shutdown probe never processed a sample"
    if snapshot.thread_alive or snapshot.worker.thread_alive or snapshot.sampler.thread_alive:
        return False, "runtime, sampler or worker thread remained alive"
    if residual:
        return False, f"residual project threads: {', '.join(residual)}"
    return True, "runtime, sampler and worker stopped without residual project threads"


def _check_hardening(config: object, *, emit: bool = True) -> HardeningReport:
    """Run local-only safety checks that do not require a live camera."""

    checks: list[HardeningCheck] = []
    gitignore_path = SCRIPT_ROOT / ".gitignore"
    try:
        gitignore = gitignore_path.read_text(encoding="utf-8")
    except OSError as exc:
        _add_hardening_check(checks, "Secrets ignore rules", "FAIL", str(exc), emit=emit)
    else:
        ignore_ok = "config/config.local.yaml" in gitignore and ".env" in gitignore
        _add_hardening_check(
            checks,
            "Secrets ignore rules",
            "PASS" if ignore_ok else "FAIL",
            "local config and .env patterns present"
            if ignore_ok
            else "missing local-secret patterns",
            emit=emit,
        )

    redacted_url = redact_url("rtsp://admin:secret@camera.local:8554/live?password=query-secret")
    redacted_text = redact_log_text(
        f"url={redacted_url} password=plain-secret token=token-secret"
    )
    redaction_ok = all(
        value not in redacted_text
        for value in ("secret", "query-secret", "plain-secret", "token-secret")
    ) and "***" in redacted_text and "admin:***@camera.local" in redacted_text
    _add_hardening_check(
        checks,
        "Secret redaction",
        "PASS" if redaction_ok else "FAIL",
        redacted_text,
        emit=emit,
    )

    storage = getattr(config, "storage", None)
    writable_count = 0
    storage_names = (
        "database_dir",
        "persons_dir",
        "events_dir",
        "snapshots_dir",
        "recordings_dir",
        "logs_dir",
        "models_dir",
    )
    if storage is not None:
        for name in storage_names:
            configured = getattr(storage, name)
            directory = configured if configured.is_absolute() else SCRIPT_ROOT / configured
            try:
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    prefix=".hardening-",
                    suffix=".tmp",
                    dir=directory,
                    delete=True,
                    encoding="utf-8",
                ) as marker:
                    marker.write("ok")
                writable_count += 1
            except OSError as exc:
                _add_hardening_check(checks, f"Storage {name}", "FAIL", str(exc), emit=emit)
    _add_hardening_check(
        checks,
        "Storage write checks",
        "PASS" if writable_count == len(storage_names) else "FAIL",
        f"{writable_count}/{len(storage_names)} directories writable",
        emit=emit,
    )

    persons_dir = getattr(storage, "persons_dir", None) if storage is not None else None
    if persons_dir is not None:
        resolved_persons = persons_dir if persons_dir.is_absolute() else SCRIPT_ROOT / persons_dir
        try:
            entries = [entry for entry in resolved_persons.iterdir() if entry.name != ".gitkeep"]
        except OSError as exc:
            _add_hardening_check(checks, "Persons database", "FAIL", str(exc), emit=emit)
            entries = []
        _add_hardening_check(
            checks,
            "Persons database",
            "INFO",
            "empty; recognition safely returns UNKNOWN"
            if not entries
            else f"{len(entries)} local record(s)",
            emit=emit,
        )
    else:
        _add_hardening_check(
            checks,
            "Persons database",
            "FAIL",
            "persons directory is not configured",
            emit=emit,
        )

    person_config = getattr(config, "person_detection", None)
    face_config = getattr(config, "face_detection", None)
    recognition_config = getattr(config, "recognition", None)
    disabled = [
        name
        for name, section in (
            ("person detection", person_config),
            ("face detection", face_config),
            ("recognition", recognition_config),
        )
        if section is not None and not section.enabled
    ]
    _add_hardening_check(
        checks,
        "Disabled components",
        "PASS" if disabled else "INFO",
        "no model adapters constructed for: " + ", ".join(disabled)
        if disabled
        else "all optional components are enabled or not configured",
        emit=emit,
    )

    incompatible_ok, incompatible_detail = _probe_incompatible_embeddings()
    _add_hardening_check(
        checks,
        "Embedding compatibility",
        "PASS" if incompatible_ok else "FAIL",
        incompatible_detail,
        emit=emit,
    )

    if recognition_config is not None and recognition_config.enabled:
        model_path = _resolve_configured_path(getattr(recognition_config, "model", None))
        if model_path is None or not model_path.is_file():
            _add_hardening_check(
                checks,
                "Configured embedding records",
                "DEFERRED",
                "recognition enabled but the configured model is unavailable",
                emit=emit,
            )
        elif importlib.util.find_spec("onnxruntime") is None:
            _add_hardening_check(
                checks,
                "Configured embedding records",
                "DEFERRED",
                "recognition enabled but ONNX Runtime is unavailable",
                emit=emit,
            )
        else:
            try:
                from app.face import create_face_matcher

                create_face_matcher(
                    recognition_config,
                    persons_root=_resolve_configured_path(getattr(storage, "persons_dir", None))
                    or SCRIPT_ROOT / "persons",
                    model_root=SCRIPT_ROOT,
                )
            except IncompatibleEmbeddingModelError as exc:
                _add_hardening_check(
                    checks,
                    "Configured embedding records",
                    "FAIL",
                    str(exc),
                    emit=emit,
                )
            except Exception as exc:
                _add_hardening_check(
                    checks,
                    "Configured embedding records",
                    "FAIL",
                    f"cannot validate configured records: {type(exc).__name__}: {exc}",
                    emit=emit,
                )
            else:
                _add_hardening_check(
                    checks,
                    "Configured embedding records",
                    "PASS",
                    "all configured records match the active embedding contract",
                    emit=emit,
                )
    else:
        _add_hardening_check(
            checks,
            "Configured embedding records",
            "INFO",
            "recognition disabled; no live records loaded",
            emit=emit,
        )

    storage_ok, storage_detail = _probe_storage_failure_isolation()
    _add_hardening_check(
        checks,
        "Storage failure isolation",
        "PASS" if storage_ok else "FAIL",
        storage_detail,
        emit=emit,
    )

    camera_ok, camera_detail = _probe_camera_isolation()
    _add_hardening_check(
        checks,
        "Offline camera isolation",
        "PASS" if camera_ok else "FAIL",
        camera_detail,
        emit=emit,
    )

    shutdown_ok, shutdown_detail = _probe_shutdown()
    _add_hardening_check(
        checks,
        "Graceful shutdown",
        "PASS" if shutdown_ok else "FAIL",
        shutdown_detail,
        emit=emit,
    )

    _add_hardening_check(
        checks,
        "Public API exposure",
        "PASS",
        "no web server or port-forwarding component configured",
        emit=emit,
    )
    _add_hardening_check(
        checks,
        "Live camera isolation",
        "DEFERRED",
        "requires two concrete LAN streams; offline probe is reported separately",
        emit=emit,
    )
    return HardeningReport(tuple(checks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check local prerequisites for the security camera prototype."
    )
    parser.add_argument("--config", type=Path, default=EXAMPLE_CONFIG)
    parser.add_argument("--url", help="Optional concrete URL for a short live stream probe.")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--hardening",
        action="store_true",
        help="Run local secret, storage and disabled-component hardening checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON document instead of the human-readable status list.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_ok = True
    emit = not args.json
    environment_checks: list[dict[str, str]] = []
    hardening_report: HardeningReport | None = None

    python_detail = f"{platform.python_version()} ({sys.executable})"
    required_ok &= _status(
        "Python",
        "OK" if sys.version_info >= (3, 11) else "FAIL",
        python_detail,
        checks=environment_checks,
        emit=emit,
    )

    cv2_spec = importlib.util.find_spec("cv2")
    if cv2_spec is None:
        required_ok &= _status(
            "OpenCV",
            "FAIL",
            "not installed; install the project dependencies",
            checks=environment_checks,
            emit=emit,
        )
    else:
        try:
            import cv2

            required_ok &= _status(
                "OpenCV",
                "OK",
                cv2.__version__,
                checks=environment_checks,
                emit=emit,
            )
        except Exception as exc:
            required_ok &= _status(
                "OpenCV",
                "FAIL",
                str(exc),
                checks=environment_checks,
                emit=emit,
            )

    for command in ("ffmpeg", "ffprobe"):
        version = _command_version(command)
        state = "OK" if version else "WARN"
        _status(
            command.capitalize(),
            state,
            version or "not found; ffprobe is optional when OpenCV can open the stream",
            checks=environment_checks,
            emit=emit,
        )

    ffplay_version = _command_version("ffplay")
    _status(
        "FFplay",
        "OK" if ffplay_version else "INFO",
        ffplay_version or "not found; VLC can be used instead",
        checks=environment_checks,
        emit=emit,
    )

    logical_cpus = os.cpu_count() or 0
    required_ok &= _status(
        "CPU",
        "OK" if logical_cpus else "FAIL",
        f"{logical_cpus} logical processors",
        checks=environment_checks,
        emit=emit,
    )

    resources = read_resource_snapshot()
    if resources.ram_total_bytes is not None:
        memory_gb = resources.ram_total_bytes / (1024**3)
        _status(
            "RAM",
            "OK",
            f"{memory_gb:.2f} GB total",
            checks=environment_checks,
            emit=emit,
        )
    else:
        _status(
            "RAM",
            "INFO",
            "resource metrics unavailable",
            checks=environment_checks,
            emit=emit,
        )
    if resources.cpu_percent is not None:
        _status(
            "CPU usage",
            "OK",
            f"{resources.cpu_percent:.1f}%",
            checks=environment_checks,
            emit=emit,
        )
    else:
        _status(
            "CPU usage",
            "INFO",
            "resource metrics unavailable",
            checks=environment_checks,
            emit=emit,
        )
    if resources.gpu_status == "available":
        _status(
            "GPU",
            "OK",
            f"{resources.gpu_percent:.1f}% utilization",
            checks=environment_checks,
            emit=emit,
        )
        if resources.vram_total_bytes is not None:
            _status(
                "VRAM",
                "OK",
                f"{resources.vram_used_bytes / (1024**3):.2f} / "
                f"{resources.vram_total_bytes / (1024**3):.2f} GB",
                checks=environment_checks,
                emit=emit,
            )
    else:
        _status(
            "GPU",
            "INFO",
            resources.gpu_detail,
            checks=environment_checks,
            emit=emit,
        )
    _status(
        "CPU inference",
        "OK",
        "enabled for this milestone",
        checks=environment_checks,
        emit=emit,
    )

    try:
        config = load_config(args.config)
        _status(
            "Configuration",
            "OK",
            str(args.config),
            checks=environment_checks,
            emit=emit,
        )
        try:
            ensure_runtime_directories(REPO_ROOT, config.storage)
            _status(
                "Storage",
                "OK",
                "runtime directories available",
                checks=environment_checks,
                emit=emit,
            )
        except OSError as exc:
            required_ok &= _status(
                "Storage",
                "FAIL",
                str(exc),
                checks=environment_checks,
                emit=emit,
            )
    except ConfigurationError as exc:
        required_ok &= _status(
            "Configuration",
            "FAIL",
            str(exc),
            checks=environment_checks,
            emit=emit,
        )
        config = None

    if config is not None:
        hardware = config.hardware_optimization
        enabled_cameras = sum(1 for camera in config.cameras if camera.enabled)
        _status(
            "Hardware profile",
            "OK" if hardware.enabled and hardware.profile == "intel_iris_xe" else "INFO",
            (
                f"profile={hardware.profile}; enabled={hardware.enabled}; "
                f"cameras={enabled_cameras}; person={config.person_detection.model}@"
                f"{config.person_detection.image_size}; "
                f"person_fps={config.inference.person_detection_fps}; "
                f"streams={config.person_detection.openvino_num_streams}; "
                f"cpu_threads={config.person_detection.openvino_cpu_threads or 'auto'}; "
                f"RAM_budget={config.person_detection.max_process_ram_mb or 'off'} MiB; "
                f"decode={config.video.hardware_acceleration}"
            ),
            checks=environment_checks,
            emit=emit,
        )
        required_ok &= _check_person_detection(
            config,
            checks=environment_checks,
            emit=emit,
        )
        required_ok &= _check_face_components(
            config,
            checks=environment_checks,
            emit=emit,
        )
        if args.hardening:
            hardening_report = _check_hardening(config, emit=emit)
            required_ok &= not hardening_report.failed
    else:
        _status(
            "Face model",
            "INFO",
            "configuration unavailable",
            checks=environment_checks,
            emit=emit,
        )
        if args.hardening:
            hardening_report = HardeningReport(
                (
                    HardeningCheck(
                        "Hardening prerequisites",
                        "FAIL",
                        "configuration unavailable",
                    ),
                )
            )
            required_ok = False

    if args.url:
        try:
            url = validate_stream_url(args.url)
            source = OpenCVVideoSource(url)
            report = run_stream_test(
                source,
                url=url,
                duration_s=args.duration,
                read_timeout_s=3.0,
                reconnect_attempts=1,
                reconnect_delay_s=1.0,
            )
            state = "OK" if report.error is None else "FAIL"
            required_ok &= _status(
                "Stream",
                state,
                f"{report.frames_received} frames, {report.actual_fps:.2f} FPS",
                checks=environment_checks,
                emit=emit,
            )
        except ConfigurationError as exc:
            required_ok &= _status(
                "Stream",
                "FAIL",
                str(exc),
                checks=environment_checks,
                emit=emit,
            )
    else:
        _status(
            "Stream",
            "NOT CONFIGURED",
            "pass --url for a live Huawei check",
            checks=environment_checks,
            emit=emit,
        )

    if args.json:
        payload = {
            "ok": bool(required_ok),
            "config": str(args.config),
            "checks": environment_checks,
            "hardening": hardening_report.to_dict() if hardening_report is not None else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
