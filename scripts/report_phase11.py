"""Generate the evidence-based Phase 11 Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from app.benchmark import ScalabilityReport
from app.config import ConfigurationError, load_config
from app.logging_setup import redact_log_text
from scripts.benchmark_scalability import run_scalability
from scripts.check_environment import _check_hardening


def _safe(value: object) -> str:
    return redact_log_text(str(value)).replace("\n", " ").strip()


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence {path} must contain an object")
    return payload


def _format_number(value: object, digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/d"
    return f"{value:.{digits}f}"


def _normalise_benchmark(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if "levels" in payload:
        return payload
    report = payload.get("report")
    if isinstance(report, dict) and "levels" in report:
        return report
    raise ValueError("benchmark evidence does not contain scalability levels")


def _normalise_environment(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if "hardening" in payload:
        return payload
    if "checks" in payload:
        return {"checks": payload, "hardening": payload}
    report = payload.get("report")
    if isinstance(report, dict) and "hardening" in report:
        return report
    raise ValueError("environment evidence does not contain hardening results")


def _hardening_lines(environment: dict[str, Any] | None) -> list[str]:
    if environment is None:
        return ["- Verifica hardening: evidenza JSON non fornita."]
    hardening = environment.get("hardening")
    if not isinstance(hardening, dict):
        return ["- Verifica hardening: risultato non disponibile."]
    status = _safe(hardening.get("status", "n/d"))
    counts = hardening.get("counts")
    if isinstance(counts, dict):
        count_text = ", ".join(
            f"{key}={_safe(counts.get(key, 0))}"
            for key in ("PASS", "INFO", "DEFERRED", "FAIL")
        )
    else:
        count_text = "conteggi n/d"
    lines = [f"- Hardening locale: stato aggregato `{status}` ({count_text})."]
    checks = hardening.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = _safe(check.get("name", "check"))
            check_status = _safe(check.get("status", "n/d"))
            detail = _safe(check.get("detail", ""))
            lines.append(f"- `{name}`: `{check_status}` — {detail}.")
    return lines


def _level_limit_lines(benchmark: dict[str, Any] | None) -> list[str]:
    if benchmark is None:
        return ["- Nessun benchmark di scalabilità allegato; limiti numerici non disponibili."]
    execution = _safe(benchmark.get("execution", "n/d"))
    lines: list[str] = []
    levels = benchmark.get("levels")
    if not isinstance(levels, list):
        return ["- Il report benchmark non contiene livelli interpretabili."]
    for level in levels:
        if not isinstance(level, dict):
            continue
        count = _safe(level.get("camera_count", "n/d"))
        status = _safe(level.get("status", "n/d"))
        if status != "measured":
            reason = _safe(level.get("reason", "nessuna motivazione disponibile"))
            lines.append(f"- Livello {count}: `{status}` — {reason}.")
            continue
        cameras = level.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            lines.append(f"- Livello {count}: `measured`, metriche camera non disponibili.")
            continue
        camera_summaries: list[str] = []
        for camera in cameras:
            if not isinstance(camera, dict):
                continue
            metrics = camera.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            camera_summaries.append(
                f"{_safe(camera.get('camera_id', 'camera'))}: "
                f"sampled={_format_number(metrics.get('sampled_fps'))} FPS, "
                f"person={_format_number(metrics.get('person_detection_fps'))} FPS, "
                f"face={_format_number(metrics.get('face_detection_fps'))} FPS, "
                f"dropped={_safe(metrics.get('dropped_frames', 'n/d'))}, "
                f"queue_max={_safe(camera.get('queue_max', 'n/d'))}, "
                "face_concurrency=n/d"
            )
        lines.append(
            f"- Livello {count}: `measured` ({execution}; simulazione se execution=`simulated`): "
            + "; ".join(camera_summaries)
            + "."
        )
    return lines or ["- Nessun livello benchmark disponibile."]


def build_phase11_report(
    benchmark_payload: dict[str, Any] | None = None,
    environment_payload: dict[str, Any] | None = None,
    *,
    pytest_passed: int | None = None,
    compileall_passed: bool | None = None,
) -> str:
    """Build the required report using only supplied evidence."""

    benchmark = _normalise_benchmark(benchmark_payload)
    environment = _normalise_environment(environment_payload)
    execution = _safe(benchmark.get("execution")) if benchmark else None
    real_levels = []
    if benchmark and execution == "real":
        real_levels = [
            level
            for level in benchmark.get("levels", [])
            if isinstance(level, dict) and level.get("status") == "measured"
        ]

    automatic_lines: list[str] = []
    if pytest_passed is None:
        automatic_lines.append("- Pytest: conteggio non fornito dal comando di generazione.")
    else:
        automatic_lines.append(f"- Pytest: `{pytest_passed} passed`.")
    if compileall_passed is True:
        automatic_lines.append("- `compileall`: completato con esito positivo.")
    elif compileall_passed is False:
        automatic_lines.append("- `compileall`: eseguito con errori.")
    else:
        automatic_lines.append("- `compileall`: evidenza non fornita.")
    automatic_lines.extend(_hardening_lines(environment))

    if benchmark is not None:
        automatic_lines.append(
            f"- Benchmark scalabilità: `{execution}` con "
            f"{len(benchmark.get('levels', []))} livelli richiesti."
        )
    else:
        automatic_lines.append("- Benchmark scalabilità: evidenza JSON non fornita.")

    if real_levels:
        hardware_lines = [
            f"- Livelli reali misurati: {', '.join(str(level.get('camera_count')) for level in real_levels)}."
        ]
    else:
        hardware_lines = [
            "- Nessun livello `real` con stato `measured` è presente nelle evidenze fornite.",
            "- Il benchmark fake/simulato non costituisce prova di supporto hardware.",
        ]

    unavailable_lines = [
        (
            "- URL LAN/RTSP concreti e modelli ONNX reali non sono verificati da questo report."
            if not real_levels
            else "- La pipeline facciale live non è verificata da questo report."
        ),
        "- GPU/VRAM resta non disponibile quando il controllo ambiente non fornisce uno stato GPU `available`.",
        "- La concorrenza della face pipeline è `n/d`: il wiring live resta fuori scope.",
    ]

    if real_levels:
        known_problems = [
            "- Il benchmark real misura soltanto i livelli e la durata riportati; non implica supporto stabile oltre quei livelli.",
            "- La pipeline facciale live e la relativa concorrenza restano non implementate (`n/d`).",
        ]
    else:
        known_problems = [
            "- Le evidenze raccolte non attestano inferenza ONNX, connettività RTSP o supporto GPU reali.",
            "- La pipeline facciale live e la relativa concorrenza restano non implementate (`n/d`).",
        ]

    return "\n".join(
        [
            "# Report Fase 11",
            "",
            "## IMPLEMENTATO",
            "",
            "- Motion detection opt-in, indipendente per camera, fail-open e con reset dopo reconnect.",
            "- Runtime multi-camera con buffer bounded, tracking preservato su scena statica e metriche motion.",
            "- Benchmark di scalabilità fake/real con livelli sequenziali, risorse, code, reconnect e stato camera.",
            "- Calibrazione offline da CSV/JSONL con distribuzioni, FAR, FRR e soglie candidate.",
            "- Hardening locale strutturato con redazione segreti, probe storage, compatibilità embedding, isolamento e shutdown.",
            "- Nessuna modifica automatica a `config.local.yaml` e nessun wiring live face-analysis -> recognition -> eventi.",
            "",
            "## TESTATO AUTOMATICAMENTE",
            "",
            *automatic_lines,
            "",
            "## TESTATO SU HARDWARE REALE",
            "",
            *hardware_lines,
            "",
            "## NON TESTABILE NELL'AMBIENTE ATTUALE",
            "",
            *unavailable_lines,
            "",
            "## LIMITI MISURATI",
            "",
            *_level_limit_lines(benchmark),
            "- Non vengono applicate soglie universali per CPU, FPS o numero di camere.",
            "",
            "## CONFIGURAZIONE CONSIGLIATA",
            "",
            "- Mantenere `motion_detection.enabled: false` finché soglia e percentuale non sono validate sulla scena reale.",
            "- Usare URL e segreti soltanto in `config.local.yaml` o variabili d’ambiente ignorate da Git.",
            "- Abilitare person/face/recognition solo dopo aver installato runtime e modelli locali compatibili.",
            "- Rifare la calibrazione per ogni modello, camera e condizioni di luce; usare la soglia come suggerimento da validare.",
            "",
            "## PROBLEMI NOTI",
            "",
            *known_problems,
            "- Il benchmark fake misura soltanto la riproducibilità del runtime simulato.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Phase 11 evidence report.")
    parser.add_argument("--benchmark-json", type=Path, default=None)
    parser.add_argument("--environment-json", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=SCRIPT_ROOT / "config" / "config.example.yaml")
    parser.add_argument("--run-fake", action="store_true", help="Collect a fake scalability report first.")
    parser.add_argument("--run-hardening", action="store_true", help="Collect local hardening evidence first.")
    parser.add_argument("--max-cameras", type=int, default=6)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--scenario", choices=("none", "one_person", "two_persons"), default="none")
    parser.add_argument("--pytest-passed", type=int, default=None)
    parser.add_argument("--compileall-passed", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the collected evidence and Markdown as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        benchmark_payload = _load_json(args.benchmark_json)
        environment_payload = _load_json(args.environment_json)
        if args.run_fake:
            benchmark_payload = run_scalability(
                mode="fake",
                max_cameras=args.max_cameras,
                duration=args.duration,
                warmup=args.warmup,
                scenario=args.scenario,
            ).to_dict()
        if args.run_hardening:
            config = load_config(args.config)
            environment_payload = {"hardening": _check_hardening(config, emit=False).to_dict()}
        markdown = build_phase11_report(
            benchmark_payload,
            environment_payload,
            pytest_passed=args.pytest_passed,
            compileall_passed=True if args.compileall_passed else None,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown, encoding="utf-8")
        if args.json:
            print(
                json.dumps(
                    {
                        "benchmark": benchmark_payload,
                        "environment": environment_payload,
                        "markdown": markdown,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.output is None:
            print(markdown)
    except (ConfigurationError, OSError, ValueError, RuntimeError) as exc:
        print(f"PHASE 11 REPORT ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
