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

from app.hardware import ensure_process_memory_budget, resolve_cpu_thread_budget


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
        performance_mode: str = "latency",
        num_streams: int = 0,
        num_requests: int = 0,
        cpu_threads: int = 0,
        max_process_ram_mb: int = 0,
    ) -> Any:
        """Compile with cache plus bounded hardware-aware runtime properties."""

        requested = str(device).upper()
        ensure_process_memory_budget(
            max_process_ram_mb,
            stage=f"OpenVINO compile {model_id}",
        )
        cache_dir: Path | None = None
        if cache_root is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_id))
            fingerprint = (model_sha256 or "no-sha256")[:32]
            cache_dir = Path(cache_root) / f"{safe_id}-{fingerprint}-{requested.lower()}"
            cache_dir.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {}
        if cache_dir is not None:
            config["CACHE_DIR"] = str(cache_dir)

        mode = str(performance_mode).strip().upper()
        streams = max(0, int(num_streams))
        requests = max(0, int(num_requests))
        if streams > 0:
            # Explicit streams are an alternative to PERFORMANCE_HINT. This
            # keeps tuning deterministic on the Iris Xe profile.
            config["NUM_STREAMS"] = streams
        elif mode in {"LATENCY", "THROUGHPUT"}:
            config["PERFORMANCE_HINT"] = mode
            if requests > 0:
                config["PERFORMANCE_HINT_NUM_REQUESTS"] = requests

        if requested.startswith("CPU"):
            config["INFERENCE_NUM_THREADS"] = resolve_cpu_thread_budget(cpu_threads)
            # Complex apps with decode/UI/other models benefit from allowing
            # Windows to schedule the bounded thread set instead of hard pinning.
            config["ENABLE_CPU_PINNING"] = False

        if config:
            try:
                return core.compile_model(model, requested, config)
            except TypeError:
                # Older bindings may only reject CACHE_DIR in compile config.
                # Preserve all performance properties and move only cache setup.
                if cache_dir is None:
                    raise
                try:
                    core.set_property(requested, {"CACHE_DIR": str(cache_dir)})
                except Exception:
                    core.set_property({"CACHE_DIR": str(cache_dir)})
                fallback = dict(config)
                fallback.pop("CACHE_DIR", None)
                return core.compile_model(model, requested, fallback)
        return core.compile_model(model, requested)


__all__ = ["OpenVINOCoreManager", "OpenVINOUnavailableError"]
