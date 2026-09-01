"""Hardware-aware runtime policy for the local surveillance pipeline.

The policy is deliberately conservative for a 16 GB Windows PC with an Intel
Core i7 and an integrated Intel Iris Xe GPU: keep the continuous person model
on the iGPU, reserve CPU capacity for decode/tracking/UI/face stages, and keep
all queues bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import psutil


@dataclass(frozen=True)
class AdaptivePersonProfile:
    model: str
    image_size: int
    inference_fps: float
    performance_mode: str
    num_streams: int
    num_requests: int


def adaptive_person_profile(camera_count: int) -> AdaptivePersonProfile:
    """Return the Intel Iris Xe profile for the current enabled camera count."""

    count = max(1, int(camera_count))
    if count == 1:
        return AdaptivePersonProfile(
            model="models/yolo26s.pt",
            image_size=640,
            inference_fps=3.0,
            performance_mode="latency",
            num_streams=1,
            num_requests=1,
        )
    if count == 2:
        return AdaptivePersonProfile(
            model="models/yolo26s.pt",
            image_size=640,
            inference_fps=2.5,
            performance_mode="throughput",
            num_streams=2,
            num_requests=2,
        )
    if count <= 4:
        return AdaptivePersonProfile(
            model="models/yolo26s.pt",
            image_size=640,
            inference_fps=2.0,
            performance_mode="throughput",
            num_streams=2,
            num_requests=2,
        )
    return AdaptivePersonProfile(
        model="models/yolo26n.pt",
        image_size=512,
        inference_fps=2.0,
        performance_mode="throughput",
        num_streams=2,
        num_requests=2,
    )


def resolve_cpu_thread_budget(configured: int = 0) -> int:
    """Resolve a CPU inference budget while leaving capacity for the rest of the app."""

    if configured > 0:
        return int(configured)
    logical = os.cpu_count() or 8
    # Half of logical CPUs, capped to avoid letting small face models occupy an
    # entire modern i7. The minimum of two keeps older 4-thread CPUs usable.
    return max(2, min(8, logical // 2))


def process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def ensure_process_memory_budget(limit_mb: int, *, stage: str) -> None:
    """Protect both process RSS and system RAM shared with an integrated GPU."""

    if limit_mb <= 0:
        return
    rss = process_rss_mb()
    if rss >= float(limit_mb):
        raise MemoryError(
            f"{stage} refused: process RSS {rss:.0f} MiB reached the "
            f"configured {limit_mb} MiB budget"
        )

    # Iris Xe uses shared system memory. RSS alone does not capture GPU pressure,
    # so keep a reserve for Windows, WPF, decoder surfaces and the iGPU driver.
    minimum_available_mb = max(1024, min(4096, int(limit_mb) // 2))
    available_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
    if available_mb < minimum_available_mb:
        raise MemoryError(
            f"{stage} refused: only {available_mb:.0f} MiB system RAM is available; "
            f"the integrated-GPU profile reserves {minimum_available_mb} MiB"
        )


def apply_hardware_profile(config: Any) -> Any:
    """Mutate one validated AppConfig with the requested hardware policy."""

    hardware = getattr(config, "hardware_optimization", None)
    if hardware is None or not hardware.enabled or hardware.profile == "none":
        return config
    if hardware.profile != "intel_iris_xe":
        return config

    enabled_cameras = sum(1 for camera in config.cameras if camera.enabled)
    enabled_cameras = max(1, enabled_cameras)

    # Bound decoded-frame memory and preview work. The Iris Xe shares system
    # memory, so keeping only the newest frame matters more than deep buffering.
    config.video.max_buffer_frames = 1
    # "auto" tries Intel MFX/Quick Sync first, then D3D11, then software.
    # A measured autotune result can pin one concrete decoder here.
    config.video.hardware_acceleration = hardware.decode_acceleration
    config.windows_ui.background_preview_fps = hardware.background_preview_fps
    config.windows_ui.background_preview_max_width = hardware.background_preview_max_width

    person = config.person_detection
    person.backend = "openvino"
    person.device = "gpu"
    person.fallback_device = "cpu"
    person.precision = "fp16"
    person.classes = ["person"]
    person.prompts = ["person"]
    person.show_masks = False
    person.openvino_cpu_threads = hardware.cpu_threads
    person.max_process_ram_mb = hardware.max_process_ram_mb

    if hardware.adaptive_person_detection:
        adaptive = adaptive_person_profile(enabled_cameras)
        person.model = adaptive.model
        person.image_size = adaptive.image_size
        person.openvino_performance_mode = adaptive.performance_mode
        person.openvino_num_streams = (
            min(hardware.gpu_streams, adaptive.num_streams)
            if hardware.gpu_streams > 0
            else adaptive.num_streams
        )
        person.openvino_num_requests = (
            min(hardware.gpu_num_requests, adaptive.num_requests)
            if hardware.gpu_num_requests > 0
            else adaptive.num_requests
        )
        config.inference.person_detection_fps = adaptive.inference_fps
        person.inference_fps = adaptive.inference_fps
    else:
        person.openvino_performance_mode = hardware.gpu_performance_mode
        person.openvino_num_streams = hardware.gpu_streams
        person.openvino_num_requests = hardware.gpu_num_requests

    if hardware.force_face_cpu:
        config.face_detection.device = "cpu"
        config.face_landmarks.device = "cpu"
        config.recognition.device = "cpu"

    # Small, intermittent face models favor latency and receive only a bounded
    # fraction of the CPU. A value of zero is resolved at runtime from os.cpu_count.
    config.face_detection.openvino_performance_mode = "latency"
    config.face_detection.openvino_cpu_threads = hardware.cpu_threads
    config.face_detection.max_process_ram_mb = hardware.max_process_ram_mb
    config.face_landmarks.openvino_performance_mode = "latency"
    config.face_landmarks.openvino_cpu_threads = hardware.cpu_threads
    config.face_landmarks.max_process_ram_mb = hardware.max_process_ram_mb
    config.recognition.openvino_performance_mode = "latency"
    config.recognition.openvino_cpu_threads = hardware.cpu_threads
    config.recognition.max_process_ram_mb = hardware.max_process_ram_mb
    return config


__all__ = [
    "AdaptivePersonProfile",
    "adaptive_person_profile",
    "apply_hardware_profile",
    "ensure_process_memory_budget",
    "process_rss_mb",
    "resolve_cpu_thread_budget",
]
