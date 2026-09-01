from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import pytest

from app.config import VideoConfig
from app.video.base import ReadStatus, VideoSourceError, redact_url
from app.video.factory import create_opencv_source
from app.video.fake_source import FakeVideoSource
import app.video.opencv_source as opencv_module
from app.video.opencv_source import OpenCVVideoSource, cv2


def test_fake_source_reads_frames_and_reconnects() -> None:
    source = FakeVideoSource([np.zeros((4, 5, 3), dtype=np.uint8)], repeat=True)
    info = source.open()
    assert info.backend == "fake"

    result = source.read(0.1)
    assert result.status is ReadStatus.FRAME
    assert result.packet is not None
    assert result.packet.frame.shape == (4, 5, 3)

    source.close()
    assert source.read(0.01).status is ReadStatus.DISCONNECTED
    source.reconnect()
    assert source.reconnect_count == 1


def test_fake_source_timeout_is_bounded() -> None:
    source = FakeVideoSource([np.zeros((2, 2, 3), dtype=np.uint8)], read_delay_s=0.05)
    source.open()
    started = time.perf_counter()
    result = source.read(0.005)
    elapsed = time.perf_counter() - started
    assert result.status is ReadStatus.TIMEOUT
    assert elapsed < 0.1


class ScriptedCapture:
    def __init__(self, values: list[tuple[bool, Any]], *, opened: bool = True) -> None:
        self.values = list(values)
        self.opened = opened
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, Any]:
        if self.values:
            return self.values.pop(0)
        time.sleep(0.005)
        return False, None

    def get(self, property_id: int) -> float:
        if cv2 is not None:
            values = {
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FOURCC: float(ord("M") | (ord("J") << 8) | (ord("P") << 16) | (ord("G") << 24)),
            }
            return values.get(property_id, self.properties.get(property_id, 0.0))
        return self.properties.get(property_id, 0.0)

    def set(self, property_id: int, value: float) -> bool:
        self.properties[property_id] = value
        return True

    def release(self) -> None:
        self.released = True


class StartupCapture(ScriptedCapture):
    def __init__(self, values: list[tuple[bool, Any]], frame: np.ndarray) -> None:
        super().__init__(values)
        self._tail_frame = frame

    def read(self) -> tuple[bool, Any]:
        if self.values:
            return self.values.pop(0)
        time.sleep(0.01)
        return True, self._tail_frame


class LateFrameCapture(ScriptedCapture):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__([])
        self._frame = frame
        self.read_started = threading.Event()
        self._allow_read = threading.Event()

    def read(self) -> tuple[bool, Any]:
        self.read_started.set()
        self._allow_read.wait(timeout=1.0)
        return True, self._frame

    def release(self) -> None:
        super().release()
        self._allow_read.set()


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_uses_stream_metadata_and_bounded_read() -> None:
    frame = np.ones((3, 4, 3), dtype=np.uint8)
    capture = ScriptedCapture([(True, frame), (True, frame), (False, None)])
    source = OpenCVVideoSource(
        "rtsp://user:secret@camera.local:8554/live",
        capture_factory=lambda _url, _backend: capture,
        max_buffer_frames=1,
    )
    info = source.open()
    assert info.url == "rtsp://user:***@camera.local:8554/live"
    assert info.width == 640
    assert info.height == 480
    assert info.declared_fps == 25
    assert info.codec == "MJPG"

    result = source.read(0.2)
    assert result.status in {ReadStatus.FRAME, ReadStatus.CORRUPT}
    source.close()


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_waits_for_first_decodable_frame() -> None:
    frame = np.full((3, 4, 3), 9, dtype=np.uint8)
    capture = StartupCapture([(False, None), (False, None), (True, frame)], frame)
    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        capture_factory=lambda _url, _backend: capture,
        read_timeout_s=0.05,
    )

    source.open()
    result = source.read(1.0)

    assert result.status is ReadStatus.FRAME
    assert result.packet is not None
    assert int(result.packet.frame[0, 0, 0]) == 9
    assert source.is_connected is True
    assert capture.released is False
    source.close()
    assert capture.released is True


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_reports_unreachable_capture() -> None:
    source = OpenCVVideoSource(
        "rtsp://camera.local/live",
        capture_factory=lambda _url, _backend: ScriptedCapture([], opened=False),
    )
    with pytest.raises(VideoSourceError, match="could not open"):
        source.open()


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_open_timeout_is_bounded() -> None:
    factory_finished = threading.Event()
    captures: list[ScriptedCapture] = []

    def stalled_factory(_url: str, _backend: str) -> ScriptedCapture:
        time.sleep(0.1)
        capture = ScriptedCapture([])
        captures.append(capture)
        factory_finished.set()
        return capture

    source = OpenCVVideoSource(
        "rtsp://camera.local/live",
        open_timeout_s=0.01,
        capture_factory=stalled_factory,
    )
    started = time.perf_counter()
    with pytest.raises(VideoSourceError, match="timed out"):
        source.open()
    assert time.perf_counter() - started < 0.08
    source.close()
    assert factory_finished.wait(timeout=1.0)
    assert captures[0].released is True


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_reuses_late_capture_after_open_timeout() -> None:
    frame = np.full((2, 2, 3), 8, dtype=np.uint8)
    open_started = threading.Event()
    allow_open = threading.Event()
    captures: list[StartupCapture] = []
    factory_calls = 0

    def delayed_factory(_url: str, _backend: str) -> StartupCapture:
        nonlocal factory_calls
        factory_calls += 1
        open_started.set()
        if not allow_open.wait(timeout=2.0):
            raise RuntimeError("test open gate was not released")
        capture = StartupCapture([(True, frame)], frame)
        captures.append(capture)
        return capture

    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        open_timeout_s=0.1,
        read_timeout_s=0.05,
        capture_factory=delayed_factory,
    )

    reconnect_errors: list[BaseException] = []
    reconnect_finished = threading.Event()

    try:
        with pytest.raises(VideoSourceError, match="timed out"):
            source.open()
        assert open_started.wait(timeout=1.0)

        def reconnect() -> None:
            try:
                source.reconnect()
            except BaseException as exc:  # captured for the assertion below
                reconnect_errors.append(exc)
            finally:
                reconnect_finished.set()

        reconnect_thread = threading.Thread(target=reconnect, daemon=True)
        reconnect_thread.start()
        time.sleep(0.02)
        assert reconnect_thread.is_alive()
        assert factory_calls == 1

        allow_open.set()
        assert reconnect_finished.wait(timeout=1.0)
        reconnect_thread.join(timeout=1.0)

        assert reconnect_errors == []
        assert factory_calls == 1
        result = source.read(1.0)
        assert result.status is ReadStatus.FRAME
        assert result.packet is not None
        assert int(result.packet.frame[0, 0, 0]) == 8
    finally:
        allow_open.set()
        source.close()

    assert len(captures) == 1
    assert captures[0].released is True


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_stop_cancels_and_releases_pending_open() -> None:
    open_started = threading.Event()
    allow_open = threading.Event()
    factory_finished = threading.Event()
    captures: list[ScriptedCapture] = []

    def delayed_factory(_url: str, _backend: str) -> ScriptedCapture:
        open_started.set()
        if not allow_open.wait(timeout=2.0):
            raise RuntimeError("test open gate was not released")
        capture = ScriptedCapture([])
        captures.append(capture)
        factory_finished.set()
        return capture

    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        open_timeout_s=0.01,
        capture_factory=delayed_factory,
    )

    with pytest.raises(VideoSourceError, match="timed out"):
        source.open()
    assert open_started.wait(timeout=1.0)

    source.close()
    allow_open.set()

    assert factory_finished.wait(timeout=1.0)
    assert len(captures) == 1
    assert captures[0].released is True
    assert source.is_connected is False


def test_url_credentials_are_redacted() -> None:
    assert redact_url("rtsp://admin:password@192.168.1.50:8554/live") == (
        "rtsp://admin:***@192.168.1.50:8554/live"
    )


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_rtsp_transport_override_is_scoped_and_restored(monkeypatch) -> None:
    option_name = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    monkeypatch.setenv(option_name, "video_codec;h264|rtsp_transport;udp")
    seen_options: list[str | None] = []
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def factory(_url: str, _backend: str) -> ScriptedCapture:
        seen_options.append(os.environ.get(option_name))
        return ScriptedCapture([(True, frame)])

    source = OpenCVVideoSource(
        "rtsp://camera.local/live",
        rtsp_transport="tcp",
        capture_factory=factory,
    )
    source.open()
    source.close()

    assert seen_options == ["video_codec;h264|rtsp_transport;tcp"]
    assert os.environ[option_name] == "video_codec;h264|rtsp_transport;udp"


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_rtsp_reconnect_releases_old_capture_and_reapplies_tcp(monkeypatch) -> None:
    option_name = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    monkeypatch.delenv(option_name, raising=False)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    captures: list[ScriptedCapture] = []
    seen_options: list[str | None] = []

    def factory(_url: str, _backend: str) -> ScriptedCapture:
        seen_options.append(os.environ.get(option_name))
        capture = ScriptedCapture([(True, frame), (False, None)])
        captures.append(capture)
        return capture

    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        rtsp_transport="tcp",
        capture_factory=factory,
    )
    source.open()
    source.reconnect()
    source.close()

    assert len(captures) == 2
    assert captures[0] is not captures[1]
    assert all(capture.released for capture in captures)
    assert seen_options == ["rtsp_transport;tcp", "rtsp_transport;tcp"]


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_current_generation_publishes_after_reopen() -> None:
    first_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
    second_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
    captures: list[StartupCapture] = []

    def factory(_url: str, _backend: str) -> StartupCapture:
        frame = first_frame if not captures else second_frame
        capture = StartupCapture([(True, frame)], frame)
        captures.append(capture)
        return capture

    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        capture_factory=factory,
        read_timeout_s=0.05,
    )

    source.open()
    first = source.read(1.0)
    source.close()
    source.open()
    second = source.read(1.0)
    source.close()

    assert first.status is ReadStatus.FRAME
    assert second.status is ReadStatus.FRAME
    assert first.packet is not None
    assert second.packet is not None
    assert int(first.packet.frame[0, 0, 0]) == 1
    assert int(second.packet.frame[0, 0, 0]) == 2
    assert len(captures) == 2
    assert all(capture.released for capture in captures)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_opencv_source_discards_late_frame_from_old_generation() -> None:
    old_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
    new_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
    old_capture = LateFrameCapture(old_frame)
    new_capture = StartupCapture([(True, new_frame)], new_frame)
    captures = [old_capture, new_capture]

    source = OpenCVVideoSource(
        "rtsp://camera.local:8554/live",
        capture_factory=lambda _url, _backend: captures.pop(0),
        read_timeout_s=0.05,
    )

    source.open()
    assert old_capture.read_started.wait(timeout=1.0)
    source.close()
    source.open()
    result = source.read(1.0)
    source.close()

    assert result.status is ReadStatus.FRAME
    assert result.packet is not None
    assert int(result.packet.frame[0, 0, 0]) == 2
    assert old_capture.released is True
    assert new_capture.released is True


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for the OpenCV source test")
def test_rtsp_transport_does_not_modify_non_rtsp_capture(monkeypatch) -> None:
    option_name = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    monkeypatch.setenv(option_name, "custom;value")
    seen_options: list[str | None] = []
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def factory(_url: str, _backend: str) -> ScriptedCapture:
        seen_options.append(os.environ.get(option_name))
        return ScriptedCapture([(True, frame)])

    source = OpenCVVideoSource(
        "http://camera.local/live",
        rtsp_transport="tcp",
        capture_factory=factory,
    )
    source.open()
    source.close()

    assert seen_options == ["custom;value"]
    assert os.environ[option_name] == "custom;value"

def test_video_factory_carries_hardware_acceleration_setting() -> None:
    source = create_opencv_source(
        "rtsp://camera.local/live",
        video=VideoConfig(hardware_acceleration="mfx"),
    )
    assert source.hardware_acceleration == "mfx"


def test_invalid_hardware_acceleration_is_rejected() -> None:
    with pytest.raises(ValueError, match="hardware_acceleration"):
        OpenCVVideoSource(
            "rtsp://camera.local/live",
            hardware_acceleration="invalid",
        )


def test_default_capture_factory_falls_back_from_mfx_to_software(monkeypatch) -> None:
    class Capture:
        def __init__(self, opened: bool) -> None:
            self.opened = opened
            self.released = False

        def isOpened(self) -> bool:
            return self.opened

        def release(self) -> None:
            self.released = True

    class FakeCv2:
        CAP_FFMPEG = 1900
        CAP_PROP_HW_ACCELERATION = 50
        VIDEO_ACCELERATION_NONE = 0
        VIDEO_ACCELERATION_ANY = 1
        VIDEO_ACCELERATION_D3D11 = 2
        VIDEO_ACCELERATION_MFX = 4
        error = RuntimeError

        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []
            self.hardware_capture = Capture(False)
            self.software_capture = Capture(True)

        def VideoCapture(self, url: str, *args: object) -> Capture:
            self.calls.append((url, *args))
            if len(args) == 2:
                return self.hardware_capture
            return self.software_capture

    fake = FakeCv2()
    monkeypatch.setattr(opencv_module, "cv2", fake)

    capture = opencv_module._default_capture_factory(
        "rtsp://camera.local/live",
        "auto",
        "mfx",
    )

    assert capture is fake.software_capture
    assert fake.hardware_capture.released is True
    assert fake.calls[0][1:] == (
        fake.CAP_FFMPEG,
        [fake.CAP_PROP_HW_ACCELERATION, fake.VIDEO_ACCELERATION_MFX],
    )
    assert fake.calls[1][1:] == (fake.CAP_FFMPEG,)

