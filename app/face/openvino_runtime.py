"""Shared, lazy OpenVINO runtime services.

The core object is process-wide, while compiled-model cache directories are
model/fingerprint/device specific.  Runtime code never downloads or exports
models; it only reads an already-present IR and compiles it on the requested
device.
"""

from __future__ import annotations

from pathlib import Path
import re
import threading
from typing import Any


class OpenVINOUnavailableError(RuntimeError):
    """Raised when OpenVINO is not installed or exposes no usable Core."""


class OpenVINOCoreManager:
    _lock = threading.RLock()
    _core: Any | None = None

    @classmethod
    def core(cls) -> Any:
        with cls._lock:
            if cls._core is None:
                try:
                    from openvino import Core

                    cls._core = Core()
                except Exception as exc:  # pragma: no cover - depends on host installation
                    raise OpenVINOUnavailableError(
                        f"OpenVINO Core is unavailable: {exc}"
                    ) from exc
            return cls._core

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._lock:
            cls._core = None

    @classmethod
    def available_devices(cls, core: Any | None = None) -> tuple[str, ...]:
        instance = core or cls.core()
        return tuple(str(device).upper() for device in instance.available_devices)

    @staticmethod
    def normalize_device_name(value: Any) -> str:
        """Normalize OpenVINO property values such as ``(GPU.0)``."""

        normalized = str(value).strip().strip("()").lower()
        if normalized.startswith("cpu"):
            return "cpu"
        if normalized.startswith("gpu"):
            return "gpu"
        if normalized.startswith("npu"):
            return "npu"
        return normalized

    @classmethod
    def execution_device(cls, compiled: Any, fallback: str | None = None) -> str | None:
        try:
            devices = compiled.get_property("EXECUTION_DEVICES")
            if devices:
                return cls.normalize_device_name(devices[0])
        except Exception:
            pass
        return cls.normalize_device_name(fallback) if fallback else None

    @classmethod
    def compile_model(
        cls,
        core: Any,
        model: Any,
        *,
        device: str,
        model_id: str,
        model_sha256: str | None,
        cache_root: Path | None = None,
    ) -> Any:
        """Compile with an isolated cache when a cache root was configured."""

        requested = str(device).upper()
        cache_dir: Path | None = None
        if cache_root is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_id))
            fingerprint = (model_sha256 or "no-sha256")[:32]
            cache_dir = Path(cache_root) / f"{safe_id}-{fingerprint}-{requested.lower()}"
            cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_dir is not None:
            try:
                return core.compile_model(
                    model,
                    requested,
                    {"CACHE_DIR": str(cache_dir)},
                )
            except (TypeError, ValueError):
                # Older OpenVINO Python bindings accept cache configuration on
                # Core instead of the compile call.  Keep the same isolated
                # directory and retry only the API shape, never the device.
                try:
                    core.set_property(requested, {"CACHE_DIR": str(cache_dir)})
                except Exception:
                    core.set_property({"CACHE_DIR": str(cache_dir)})
        return core.compile_model(model, requested)


__all__ = ["OpenVINOCoreManager", "OpenVINOUnavailableError"]
