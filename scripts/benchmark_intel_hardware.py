"""Benchmark and recommend the Intel i7 + Iris Xe runtime profile.

The default --plan mode is side-effect free. --run explicitly loads/exports
models and, when requested, opens the configured RTSP stream.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from typing import Any

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import PersonDetectionConfig, get_camera, load_config, validate_stream_url
from app.hardware import adaptive_person_profile, process_rss_mb, resolve_cpu_thread_budget
from app.inference import create_person_detector
from app.video.factory import create_opencv_source


CANDIDATES: tuple[dict[str, Any], ...] = tuple(
    {
        "name": f"{model_name}-{image_size}-{streams}stream",
        "model": model,
        "image_size": image_size,
        "performance_mode": "latency" if streams == 1 else "throughput",
        "num_streams": streams,
        "num_requests": streams,
    }
    for model_name, model, image_size in (
        ("yolo26s", "models/yolo26s.pt", 640),
        ("yolo26n", "models/yolo26n.pt", 512),
    )
    for streams in (1, 2, 3, 4)
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Plan or benchmark the Intel Iris Xe hardware profile."
    )
    value.add_argument("--config", type=Path, default=Path("config/config.local.yaml"))
    value.add_argument("--camera-count", type=int, default=None)
    value.add_argument("--iterations", type=int, default=40)
    value.add_argument("--warmup", type=int, default=5)
    value.add_argument("--camera-id", default=None)
    value.add_argument("--decode-seconds", type=float, default=4.0)
    value.add_argument("--run", action="store_true", help="load models and run real benchmarks")
    value.add_argument(
        "--apply",
        action="store_true",
        help="persist the measured recommendation to config.local.yaml; requires --run",
    )
    value.add_argument(
        "--benchmark-decode",
        action="store_true",
        help="with --run, compare software/MFX/D3D11 decoding against a real stream",
    )
    value.add_argument("--json", action="store_true")
    value.add_argument("--output", type=Path, default=None)
    return value


def _load(path: Path):
    if not path.is_file() and path.name == "config.local.yaml":
        path = path.with_name("config.example.yaml")
    return path, load_config(path)


def _camera_count(config: Any, override: int | None) -> int:
    if override is not None:
        if not 1 <= override <= 6:
            raise ValueError("--camera-count must be between 1 and 6")
        return override
    return max(1, sum(1 for camera in config.cameras if camera.enabled))


def _plan(config: Any, count: int) -> dict[str, Any]:
    adaptive = adaptive_person_profile(count)
    return {
        "host": {
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_total_gib": round(psutil.virtual_memory().total / (1024**3), 2),
            "process_rss_mb": round(process_rss_mb(), 1),
        },
        "profile": "intel_iris_xe",
        "camera_count": count,
        "cpu_thread_budget": resolve_cpu_thread_budget(
            config.hardware_optimization.cpu_threads
        ),
        "process_ram_budget_mb": config.hardware_optimization.max_process_ram_mb,
        "adaptive": asdict(adaptive),
        "target_aggregate_person_fps": round(count * adaptive.inference_fps, 2),
        "candidates": list(CANDIDATES),
        "decode_order": ["mfx", "d3d11", "none"],
    }


def _prepare_candidate(
    candidate: dict[str, Any],
    *,
    confidence: float,
    ram_budget_mb: int,
    cpu_threads: int,
):
    config = PersonDetectionConfig(
        enabled=True,
        backend="openvino",
        model=str(candidate["model"]),
        confidence_threshold=confidence,
        precision="fp16",
        device="gpu",
        fallback_device="cpu",
        image_size=int(candidate["image_size"]),
        classes=["person"],
        prompts=["person"],
        show_masks=False,
        openvino_performance_mode=str(candidate["performance_mode"]),
        openvino_num_streams=int(candidate["num_streams"]),
        openvino_num_requests=int(candidate["num_requests"]),
        openvino_cpu_threads=cpu_threads,
        max_process_ram_mb=ram_budget_mb,
    )
    return create_person_detector(config, model_root=ROOT)


def _percentile_ms(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _end_to_end_benchmark(detector: Any, iterations: int, warmup: int) -> dict[str, Any]:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(warmup):
        detector.detect(frame)
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        detector.detect(frame)
        samples.append((time.perf_counter() - started) * 1000.0)
    mean = statistics.fmean(samples)
    return {
        "mean_ms": round(mean, 3),
        "p50_ms": round(_percentile_ms(samples, 0.50), 3),
        "p95_ms": round(_percentile_ms(samples, 0.95), 3),
        "sequential_fps": round(1000.0 / mean, 2) if mean > 0 else None,
        "device": getattr(detector, "device_used", None),
        "provider": getattr(detector, "provider_used", None),
        "runtime_tuned": bool(getattr(detector, "runtime_tuned", False)),
        "runtime_tuning": getattr(detector, "runtime_tuning", {}),
    }


def _raw_openvino_throughput(
    detector: Any,
    candidate: dict[str, Any],
    iterations: int,
    warmup: int,
) -> dict[str, Any] | None:
    cache = getattr(detector, "cache_path", None)
    if cache is None:
        return None
    try:
        import openvino as ov

        xml_path = next(Path(cache).glob("*.xml"))
        core = ov.Core()
        model = core.read_model(str(xml_path))
        config = {"NUM_STREAMS": int(candidate["num_streams"])}
        compiled = core.compile_model(model, "GPU", config)
        input_port = compiled.input(0)
        shape = tuple(int(value) for value in input_port.shape)
        tensor = np.zeros(shape, dtype=np.float32)
        jobs = max(1, int(candidate["num_requests"]))
        queue = ov.AsyncInferQueue(compiled, jobs)

        for _ in range(warmup):
            queue.start_async(tensor)
        queue.wait_all()

        started = time.perf_counter()
        for _ in range(iterations):
            queue.start_async(tensor)
        queue.wait_all()
        elapsed = time.perf_counter() - started
        return {
            "requests": iterations,
            "jobs": jobs,
            "elapsed_s": round(elapsed, 4),
            "throughput_fps": round(iterations / elapsed, 2) if elapsed > 0 else None,
            "execution_devices": [
                str(item) for item in compiled.get_property("EXECUTION_DEVICES")
            ],
        }
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _benchmark_models(config: Any, count: int, iterations: int, warmup: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        row: dict[str, Any] = {"candidate": dict(candidate)}
        detector = None
        try:
            detector = _prepare_candidate(
                candidate,
                confidence=config.person_detection.confidence_threshold,
                ram_budget_mb=config.hardware_optimization.max_process_ram_mb,
                cpu_threads=config.hardware_optimization.cpu_threads,
            )
            row["end_to_end"] = _end_to_end_benchmark(detector, iterations, warmup)
            row["raw_openvino"] = _raw_openvino_throughput(
                detector, candidate, iterations, warmup
            )
        except Exception as exc:
            row["unavailable"] = f"{type(exc).__name__}: {exc}"
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
        results.append(row)
    return results


def _recommend_model(results: list[dict[str, Any]], count: int) -> dict[str, Any]:
    target = count * adaptive_person_profile(count).inference_fps
    usable: list[tuple[dict[str, Any], float]] = []
    for row in results:
        raw = row.get("raw_openvino") or {}
        throughput = raw.get("throughput_fps") if isinstance(raw, dict) else None
        if isinstance(throughput, (int, float)):
            usable.append((row, float(throughput)))

    # Preserve the larger S model whenever it has 30% aggregate headroom.
    # Among adequate candidates choose the fewest streams: this limits queueing,
    # intermediate buffers and shared-memory pressure on an integrated GPU.
    adequate = [item for item in usable if item[1] >= target * 1.30]
    s_rows = [
        item
        for item in adequate
        if item[0]["candidate"]["model"].endswith("yolo26s.pt")
    ]
    preferred = s_rows or adequate
    if preferred:
        preferred.sort(
            key=lambda item: (
                int(item[0]["candidate"]["num_streams"]),
                -item[1],
            )
        )
        chosen, throughput = preferred[0]
    elif usable:
        chosen, throughput = max(usable, key=lambda item: item[1])
    else:
        return {
            "source": "policy_only",
            "profile": asdict(adaptive_person_profile(count)),
            "reason": "no real OpenVINO candidate produced throughput measurements",
        }
    return {
        "source": "measured",
        "candidate": chosen["candidate"],
        "measured_raw_throughput_fps": throughput,
        "target_aggregate_fps": target,
        "headroom_ratio": round(throughput / target, 2) if target > 0 else None,
    }


def _recommend_decode(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose hardware decode only when it is actually used and does not regress FPS."""

    usable = [
        row
        for row in results
        if "unavailable" not in row
        and isinstance(row.get("decoded_fps"), (int, float))
        and float(row.get("decoded_fps", 0.0)) > 0
    ]
    software = next((row for row in usable if row.get("requested") == "none"), None)
    hardware = [
        row
        for row in usable
        if row.get("requested") in {"mfx", "d3d11"}
        and row.get("actual") == row.get("requested")
    ]
    if not hardware:
        return {
            "requested": "none",
            "source": "fallback",
            "reason": "no hardware decoder was verified",
        }

    if software is None:
        chosen = min(
            hardware,
            key=lambda row: (
                float(row.get("process_cpu_percent", 1e9)),
                -float(row.get("decoded_fps", 0.0)),
            ),
        )
        return {"requested": chosen["requested"], "source": "measured", "row": chosen}

    baseline_fps = float(software.get("decoded_fps", 0.0))
    baseline_cpu = float(software.get("process_cpu_percent", 0.0))
    candidates = [
        row
        for row in hardware
        if float(row.get("decoded_fps", 0.0)) >= baseline_fps * 0.95
    ]
    if not candidates:
        return {
            "requested": "none",
            "source": "measured",
            "reason": "hardware decode reduced decoded FPS by more than 5%",
            "row": software,
        }

    chosen = min(
        candidates,
        key=lambda row: (
            float(row.get("process_cpu_percent", 1e9)),
            -float(row.get("decoded_fps", 0.0)),
        ),
    )
    chosen_cpu = float(chosen.get("process_cpu_percent", 0.0))
    chosen_fps = float(chosen.get("decoded_fps", 0.0))
    materially_better = (
        chosen_cpu <= baseline_cpu * 0.95
        or chosen_fps >= baseline_fps * 1.05
    )
    if not materially_better:
        return {
            "requested": "none",
            "source": "measured",
            "reason": "hardware decode produced no material CPU/FPS improvement",
            "row": software,
        }
    return {"requested": chosen["requested"], "source": "measured", "row": chosen}


def _apply_recommendation(
    config_path: Path,
    *,
    count: int,
    model_recommendation: dict[str, Any],
    decode_recommendation: dict[str, Any] | None,
) -> Path:
    candidate = model_recommendation.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(
            "--apply requires a measured OpenVINO candidate; no benchmark candidate was available"
        )

    source = config_path
    if not source.is_file():
        raise ValueError(f"configuration file does not exist: {source}")
    target = (
        source.with_name("config.local.yaml")
        if source.name == "config.example.yaml"
        else source
    )
    if target.is_file():
        source = target

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a YAML mapping")

    hardware = raw.setdefault("hardware_optimization", {})
    person = raw.setdefault("person_detection", {})
    inference = raw.setdefault("inference", {})
    video = raw.setdefault("video", {})
    if not all(isinstance(value, dict) for value in (hardware, person, inference, video)):
        raise ValueError("hardware/person/inference/video configuration sections must be mappings")

    adaptive = adaptive_person_profile(count)
    streams = int(candidate["num_streams"])
    requests = int(candidate["num_requests"])
    performance_mode = str(candidate["performance_mode"])
    hardware.update(
        {
            "enabled": True,
            "profile": "intel_iris_xe",
            # The benchmark result is now the explicit local policy. The user
            # can restore adaptive mode later by setting this back to true.
            "adaptive_person_detection": False,
            "gpu_performance_mode": performance_mode,
            "gpu_streams": streams,
            "gpu_num_requests": requests,
        }
    )
    person.update(
        {
            "backend": "openvino",
            "model": str(candidate["model"]),
            "precision": "fp16",
            "device": "gpu",
            "fallback_device": "cpu",
            "image_size": int(candidate["image_size"]),
            "openvino_performance_mode": performance_mode,
            "openvino_num_streams": streams,
            "openvino_num_requests": requests,
            "classes": ["person"],
            "prompts": ["person"],
            "show_masks": False,
        }
    )
    inference["person_detection_fps"] = adaptive.inference_fps

    if decode_recommendation is not None:
        decode = str(decode_recommendation.get("requested") or "auto")
        if decode not in {"auto", "none", "mfx", "d3d11"}:
            raise ValueError(f"invalid decode recommendation: {decode}")
        hardware["decode_acceleration"] = decode
        video["hardware_acceleration"] = decode

    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(
        raw,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _stream_url(config: Any, camera_id: str | None) -> str:
    if camera_id:
        camera = get_camera(config, camera_id)
    else:
        camera = next((item for item in config.cameras if item.enabled), None)
        if camera is None:
            raise ValueError("no enabled camera is available for decode benchmark")
    return validate_stream_url(camera.stream_url)


def _decode_once(config: Any, url: str, mode: str, seconds: float) -> dict[str, Any]:
    video = config.video.model_copy(update={"hardware_acceleration": mode})
    source = create_opencv_source(url, video=video)
    frames = 0
    corrupt = 0
    process = psutil.Process()
    try:
        source.open()
        # Do not charge backend startup/RTSP negotiation to steady-state decode.
        warmup_deadline = time.perf_counter() + min(1.0, max(0.2, seconds * 0.25))
        while time.perf_counter() < warmup_deadline:
            result = source.read(0.1)
            if result.packet is not None:
                break

        process.cpu_percent(None)
        started = time.perf_counter()
        deadline = started + seconds
        while time.perf_counter() < deadline:
            result = source.read(min(0.25, max(0.01, deadline - time.perf_counter())))
            if result.packet is not None:
                frames += 1
            elif str(result.status.value).lower() == "corrupt":
                corrupt += 1
        elapsed = max(1e-9, time.perf_counter() - started)
        return {
            "requested": mode,
            "actual": source.hardware_acceleration_used,
            "decoded_fps": round(frames / elapsed, 2),
            "frames": frames,
            "corrupt": corrupt,
            "process_cpu_percent": round(process.cpu_percent(None), 1),
        }
    except Exception as exc:
        return {"requested": mode, "unavailable": f"{type(exc).__name__}: {exc}"}
    finally:
        source.close()


def _benchmark_decode(config: Any, camera_id: str | None, seconds: float) -> list[dict[str, Any]]:
    url = _stream_url(config, camera_id)
    return [
        _decode_once(config, url, mode, seconds)
        for mode in ("none", "mfx", "d3d11")
    ]


def main() -> int:
    args = parser().parse_args()
    try:
        if args.apply and not args.run:
            raise ValueError("--apply requires --run")
        if args.iterations < 1:
            raise ValueError("--iterations must be at least 1")
        if args.warmup < 0:
            raise ValueError("--warmup cannot be negative")
        if args.decode_seconds <= 0:
            raise ValueError("--decode-seconds must be greater than zero")
        config_path, config = _load(args.config)
        count = _camera_count(config, args.camera_count)
        report: dict[str, Any] = {
            "config": str(config_path),
            "plan": _plan(config, count),
            "mode": "run" if args.run else "plan",
        }
        if args.run:
            report["model_benchmarks"] = _benchmark_models(
                config, count, args.iterations, args.warmup
            )
            report["recommendation"] = _recommend_model(
                report["model_benchmarks"], count
            )
            decode_recommendation = None
            if args.benchmark_decode:
                report["decode_benchmarks"] = _benchmark_decode(
                    config, args.camera_id, args.decode_seconds
                )
                decode_recommendation = _recommend_decode(report["decode_benchmarks"])
                report["decode_recommendation"] = decode_recommendation
            if args.apply:
                applied_path = _apply_recommendation(
                    config_path,
                    count=count,
                    model_recommendation=report["recommendation"],
                    decode_recommendation=decode_recommendation,
                )
                report["applied_config"] = str(applied_path)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"Intel profile mode: {report['mode']}")
            print(json.dumps(report["plan"], ensure_ascii=False, indent=2))
            if args.run:
                print(json.dumps(report.get("recommendation"), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"INTEL PROFILE BENCHMARK ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
