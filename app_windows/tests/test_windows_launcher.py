from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "Avvia_Monitor_Windows.bat"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher requires cmd.exe")
def test_windows_bat_launcher_runs_python_entrypoint() -> None:
    cmd = shutil.which("cmd.exe")
    if cmd is None:
        pytest.skip("cmd.exe is unavailable")

    result = subprocess.run(
        [cmd, "/d", "/c", "call Avvia_Monitor_Windows.bat --help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Local Windows monitor" in result.stdout
    assert "--fake-cameras" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher requires cmd.exe")
def test_windows_bat_forwards_config_and_fake_arguments() -> None:
    cmd = shutil.which("cmd.exe")
    if cmd is None:
        pytest.skip("cmd.exe is unavailable")

    result = subprocess.run(
        [
            cmd,
            "/d",
            "/c",
            "call Avvia_Monitor_Windows.bat --config config\\config.example.yaml --fake-cameras --help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "--config CONFIG" in result.stdout
    assert "--fake-cameras" in result.stdout


def test_windows_bat_rebuilds_wpf_instead_of_running_cached_binary() -> None:
    content = LAUNCHER.read_text(encoding="utf-8")

    assert "-t:Rebuild" in content
    assert "if exist \"%WPF_EXE%\" goto run_wpf" not in content
    assert "nessun binario precedente verra' eseguito" in content
