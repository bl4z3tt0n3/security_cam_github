from __future__ import annotations

from app.hardening import HardeningCheck, HardeningReport
from scripts.report_phase11 import build_phase11_report


def hardening_payload() -> dict[str, object]:
    return {
        "hardening": HardeningReport(
            (
                HardeningCheck("Secret redaction", "PASS", "credentials hidden"),
                HardeningCheck("Live camera isolation", "DEFERRED", "LAN required"),
            )
        ).to_dict()
    }


def test_report_with_benchmark_contains_required_sections_and_limits() -> None:
    report = build_phase11_report(
        {
            "execution": "simulated",
            "levels": [
                {
                    "camera_count": 1,
                    "status": "measured",
                    "cameras": [
                        {
                            "camera_id": "fake_1",
                            "status": "measured",
                            "metrics": {
                                "sampled_fps": 10.0,
                                "person_detection_fps": 9.0,
                                "face_detection_fps": 0.0,
                                "dropped_frames": 0,
                            },
                            "queue_max": 1,
                        }
                    ],
                }
            ],
        },
        hardening_payload(),
        pytest_passed=170,
        compileall_passed=True,
    )

    for heading in (
        "IMPLEMENTATO",
        "TESTATO AUTOMATICAMENTE",
        "TESTATO SU HARDWARE REALE",
        "NON TESTABILE NELL'AMBIENTE ATTUALE",
        "LIMITI MISURATI",
        "CONFIGURAZIONE CONSIGLIATA",
        "PROBLEMI NOTI",
    ):
        assert f"## {heading}" in report
    assert "161 passed" not in report
    assert "170 passed" in report
    assert "simulated" in report
    assert "face_concurrency=n/d" in report
    assert "plain-secret" not in report
    assert "token-secret" not in report


def test_report_without_evidence_does_not_invent_hardware_results() -> None:
    report = build_phase11_report()

    assert "Nessun livello `real` con stato `measured`" in report
    assert "evidenza JSON non fornita" in report
    assert "Non vengono applicate soglie universali" in report
    assert "hardware" in report.lower()


def test_report_real_measured_level_is_not_labeled_unavailable() -> None:
    report = build_phase11_report(
        {
            "execution": "real",
            "levels": [
                {
                    "camera_count": 2,
                    "status": "measured",
                    "cameras": [],
                }
            ],
        }
    )

    assert "Livelli reali misurati: 2" in report
    assert "Nessun livello `real` con stato `measured`" not in report
    assert "URL LAN/RTSP concreti e modelli ONNX reali non sono verificati" not in report
