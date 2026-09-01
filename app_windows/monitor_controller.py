"""Qt controller that polls six independent bounded frame providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
import threading
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot

from app.logging_setup import log_event, redact_log_text
from app.video.base import FramePacket
from app.video.worker import WorkerState

from app_windows.models.camera_view_state import (
    CameraSlot,
    CameraViewSnapshot,
    CameraViewStatus,
)
from app_windows.video.frame_provider import FrameProvider, ProviderSnapshot


ProviderFactory = Callable[[CameraSlot], FrameProvider]


class CameraMonitorController(QObject):
    """Keep acquisition out of the GUI thread while publishing latest snapshots."""

    snapshot_changed = Signal(str, object)
    camera_reconfigured = Signal(str, bool, str)
    _reconfigure_completed = Signal(str, int, object, object)

    def __init__(
        self,
        slots: Iterable[CameraSlot],
        provider_factory: ProviderFactory,
        *,
        display_fps: float,
        read_timeout_s: float,
        logger: logging.Logger | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if display_fps <= 0:
            raise ValueError("display_fps must be greater than zero")
        if read_timeout_s <= 0:
            raise ValueError("read_timeout_s must be greater than zero")

        normalized_slots = tuple(slots)
        if not normalized_slots:
            raise ValueError("at least one camera slot is required")
        ids = [slot.camera_id for slot in normalized_slots]
        if len(ids) != len(set(ids)):
            raise ValueError("camera slot ids must be unique")

        self._slots = normalized_slots
        self._provider_factory = provider_factory
        self._display_fps = float(display_fps)
        self._read_timeout_s = float(read_timeout_s)
        self._logger = logger or logging.getLogger(__name__)
        self._providers: dict[str, FrameProvider] = {}
        self._provider_errors: dict[str, str] = {}
        self._last_packets: dict[str, object] = {}
        self._last_packet_monotonic: dict[str, float] = {}
        self._last_status: dict[str, CameraViewStatus] = {}
        self._snapshots: dict[str, CameraViewSnapshot] = {}
        self._started = False
        self._reconfigure_generation: dict[str, int] = {}
        self._reconfigure_threads: dict[str, threading.Thread] = {}
        self._stop_timeout_s = max(1.0, self._read_timeout_s + 1.0)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.timeout.connect(self.poll)
        self._reconfigure_completed.connect(self._on_reconfigure_completed)

        for slot in self._slots:
            status, message = self._initial_status(slot)
            snapshot = CameraViewSnapshot(
                slot=slot,
                status=status,
                message=message,
                display_fps=self._display_fps,
            )
            self._snapshots[slot.camera_id] = snapshot
            self._last_status[slot.camera_id] = status

    @property
    def slots(self) -> tuple[CameraSlot, ...]:
        return self._slots

    @property
    def providers(self) -> dict[str, FrameProvider]:
        return dict(self._providers)

    def provider_for(self, camera_id: str) -> FrameProvider | None:
        """Return the existing provider for inference without creating one."""

        return self._providers.get(camera_id)

    @property
    def snapshots(self) -> dict[str, CameraViewSnapshot]:
        return dict(self._snapshots)

    @property
    def display_interval_ms(self) -> int:
        return max(1, round(1000.0 / self._display_fps))

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Start all configured providers; one failure never aborts the fleet."""

        if self._started:
            return
        self._started = True
        for slot in self._slots:
            if not slot.enabled or not slot.configured:
                continue
            try:
                provider = self._provider_factory(slot)
                self._providers[slot.camera_id] = provider
                provider.start()
            except Exception as exc:
                message = redact_log_text(exc)
                self._provider_errors[slot.camera_id] = message
                self._logger.error(
                    "camera=%s monitor provider start failed: %s",
                    slot.camera_id,
                    message,
                )

        self._timer.start(self.display_interval_ms)
        self.poll()

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop every provider independently and never skip later cameras."""

        self._timer.stop()
        providers = tuple(self._providers.items())
        self._providers.clear()

        def stop_one(camera_id: str, provider: FrameProvider) -> None:
            try:
                provider.stop(timeout_s=timeout_s)
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                self._logger.error(
                    "camera=%s monitor provider stop failed: %s",
                    camera_id,
                    redact_log_text(exc),
                )

        # A slow/offline source must not make the GUI wait once per camera.
        stop_threads = [
            threading.Thread(
                target=stop_one,
                args=(camera_id, provider),
                name=f"monitor-stop-{camera_id}",
                daemon=True,
            )
            for camera_id, provider in providers
        ]
        for thread in stop_threads:
            thread.start()

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for (camera_id, _), thread in zip(providers, stop_threads, strict=True):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                self._logger.warning(
                    "camera=%s monitor provider stop timed out",
                    camera_id,
                )

        reconfigure_threads = tuple(self._reconfigure_threads.items())
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for camera_id, thread in reconfigure_threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                self._logger.warning("camera=%s reconfiguration thread did not stop in time", camera_id)
            else:
                self._reconfigure_threads.pop(camera_id, None)
        self._started = False

    def apply_camera_slot(self, slot: CameraSlot, *, stop_timeout_s: float | None = None) -> None:
        """Apply one slot without stopping or recreating any other provider."""

        current_index = next(
            (index for index, current in enumerate(self._slots) if current.camera_id == slot.camera_id),
            None,
        )
        if current_index is None:
            raise ValueError(f"camera slot '{slot.camera_id}' is not known")
        if slot.camera_id in self._reconfigure_threads:
            raise RuntimeError(f"camera '{slot.camera_id}' is already being reconfigured")

        slots = list(self._slots)
        slots[current_index] = slot
        self._slots = tuple(slots)
        self._provider_errors.pop(slot.camera_id, None)
        self._last_packets.pop(slot.camera_id, None)
        self._last_packet_monotonic.pop(slot.camera_id, None)
        initial_status, initial_message = self._initial_status(slot)
        self._publish(
            CameraViewSnapshot(
                slot=slot,
                status=initial_status,
                message=initial_message,
                display_fps=self._display_fps,
            )
        )

        old_provider = self._providers.pop(slot.camera_id, None)
        if not self._started:
            if old_provider is not None:
                old_provider.stop(timeout_s=stop_timeout_s)
            return

        generation = self._reconfigure_generation.get(slot.camera_id, 0) + 1
        self._reconfigure_generation[slot.camera_id] = generation
        effective_stop_timeout = stop_timeout_s or self._stop_timeout_s

        def reconfigure() -> None:
            try:
                if old_provider is not None:
                    old_provider.stop(timeout_s=effective_stop_timeout)

                provider: FrameProvider | None = None
                if slot.enabled and slot.configured:
                    provider = self._provider_factory(slot)
                    provider.start()
                self._reconfigure_completed.emit(slot.camera_id, generation, provider, None)
            except Exception as exc:
                self._reconfigure_completed.emit(slot.camera_id, generation, None, exc)

        thread = threading.Thread(
            target=reconfigure,
            name=f"monitor-reconfigure-{slot.camera_id}",
            daemon=True,
        )
        self._reconfigure_threads[slot.camera_id] = thread
        thread.start()

    def probe_existing_camera(self, camera_id: str) -> tuple[bool, bool, str]:
        """Inspect an active provider without opening a second stream."""

        provider = self._providers.get(camera_id)
        if provider is None:
            return False, False, ""
        try:
            snapshot = provider.snapshot()
            packet = provider.latest_frame()
        except Exception as exc:
            return True, False, f"Errore provider: {redact_log_text(exc)}"

        worker = snapshot.worker
        if worker is not None and worker.state is WorkerState.RUNNING and packet is not None:
            age = max(0.0, time.monotonic() - packet.received_monotonic)
            if age <= max(2.0, self._read_timeout_s * 1.5):
                return True, True, "Connessione riuscita (provider attivo)"
        if worker is not None and worker.state in {WorkerState.STARTING, WorkerState.RECONNECTING}:
            return True, False, "Connessione in corso sul provider attivo"
        return True, False, redact_log_text(snapshot.last_error or "Stream non disponibile")

    def poll(self) -> None:
        """Consume at most one newest packet per provider on each UI tick."""

        now = time.monotonic()
        for slot in self._slots:
            provider = self._providers.get(slot.camera_id)
            if provider is None:
                if slot.camera_id in self._provider_errors:
                    self._publish(
                        CameraViewSnapshot(
                            slot=slot,
                            status=CameraViewStatus.ERROR,
                            message="Provider non disponibile",
                            display_fps=self._display_fps,
                        )
                    )
                else:
                    self._publish(self._snapshots[slot.camera_id])
                continue

            try:
                packet = provider.latest_frame()
                provider_snapshot = provider.snapshot()
            except Exception as exc:
                message = redact_log_text(exc)
                self._provider_errors[slot.camera_id] = message
                self._publish(
                    CameraViewSnapshot(
                        slot=slot,
                        status=CameraViewStatus.ERROR,
                        message="Errore lettura stream",
                        display_fps=self._display_fps,
                    )
                )
                continue

            if packet is not None:
                self._last_packets[slot.camera_id] = packet
                self._last_packet_monotonic[slot.camera_id] = packet.received_monotonic

            snapshot = self._build_snapshot(
                slot,
                provider_snapshot,
                now=now,
                packet=self._last_packets.get(slot.camera_id),
            )
            self._publish(snapshot)

    def snapshot_for(self, camera_id: str) -> CameraViewSnapshot | None:
        return self._snapshots.get(camera_id)

    @Slot(str, int, object, object)
    def _on_reconfigure_completed(
        self,
        camera_id: str,
        generation: int,
        provider: object,
        error: object,
    ) -> None:
        current_generation = self._reconfigure_generation.get(camera_id)
        self._schedule_reconfigure_cleanup(camera_id, generation)
        current_slot = next(
            (slot for slot in self._slots if slot.camera_id == camera_id),
            None,
        )
        if current_slot is None:
            return

        if current_generation != generation or not self._started:
            if provider is not None and hasattr(provider, "stop"):
                try:
                    provider.stop(timeout_s=self._stop_timeout_s)  # type: ignore[attr-defined]
                except Exception:
                    pass
            return

        if error is not None:
            message = redact_log_text(error)
            self._provider_errors[camera_id] = message
            self._publish(
                CameraViewSnapshot(
                    slot=current_slot,
                    status=CameraViewStatus.ERROR,
                    message="Provider non disponibile",
                    display_fps=self._display_fps,
                )
            )
            self.camera_reconfigured.emit(camera_id, False, message)
            return

        if provider is not None and hasattr(provider, "start"):
            self._providers[camera_id] = provider
        self._publish(
            CameraViewSnapshot(
                slot=current_slot,
                status=self._initial_status(current_slot)[0],
                message=self._initial_status(current_slot)[1],
                display_fps=self._display_fps,
            )
        )
        self.camera_reconfigured.emit(camera_id, True, "Configurazione applicata")

    def _schedule_reconfigure_cleanup(self, camera_id: str, generation: int) -> None:
        QTimer.singleShot(
            0,
            lambda: self._cleanup_reconfigure_thread(camera_id, generation),
        )

    def _cleanup_reconfigure_thread(self, camera_id: str, generation: int) -> None:
        if self._reconfigure_generation.get(camera_id) != generation:
            return
        thread = self._reconfigure_threads.get(camera_id)
        if thread is None:
            return
        if thread.is_alive():
            QTimer.singleShot(
                5,
                lambda: self._cleanup_reconfigure_thread(camera_id, generation),
            )
            return
        self._reconfigure_threads.pop(camera_id, None)

    def _build_snapshot(
        self,
        slot: CameraSlot,
        provider_snapshot: ProviderSnapshot,
        *,
        now: float,
        packet: object | None,
    ) -> CameraViewSnapshot:
        worker = provider_snapshot.worker
        typed_packet = packet if isinstance(packet, FramePacket) else None
        age: float | None = None
        if slot.camera_id in self._last_packet_monotonic:
            age = max(0.0, now - self._last_packet_monotonic[slot.camera_id])

        if worker is None:
            status = CameraViewStatus.ERROR
            message = "Provider senza worker"
        elif worker.state is WorkerState.STARTING:
            status = CameraViewStatus.CONNECTING
            message = "Connessione in corso..."
        elif worker.state is WorkerState.RECONNECTING:
            status = CameraViewStatus.RECONNECTING
            message = "Tentativo di riconnessione..."
        elif worker.state is WorkerState.RUNNING:
            stale_after = max(2.0, self._read_timeout_s * 1.5)
            if typed_packet is not None and (age is None or age <= stale_after):
                status = CameraViewStatus.LIVE
                message = "Stream attivo"
            elif age is not None and age > stale_after:
                status = CameraViewStatus.OFFLINE
                message = "Nessun frame recente"
            else:
                status = CameraViewStatus.CONNECTING
                message = "Attesa del primo frame..."
        elif worker.state is WorkerState.FAILED:
            status = (
                CameraViewStatus.OFFLINE
                if self._looks_offline(provider_snapshot.last_error)
                else CameraViewStatus.ERROR
            )
            message = "Stream offline" if status is CameraViewStatus.OFFLINE else "Errore stream"
        elif worker.state is WorkerState.STOPPING:
            status = CameraViewStatus.OFFLINE
            message = "Arresto stream..."
        else:
            status = CameraViewStatus.OFFLINE
            message = "Stream non attivo"

        return CameraViewSnapshot(
            slot=slot,
            status=status,
            message=message,
            frame=typed_packet,
            stream_info=provider_snapshot.stream_info,
            worker_snapshot=worker,
            last_frame_age_s=age,
            display_fps=self._display_fps,
            hardware_acceleration=provider_snapshot.hardware_acceleration,
        )

    def _publish(self, snapshot: CameraViewSnapshot) -> None:
        self._snapshots[snapshot.slot.camera_id] = snapshot
        previous = self._last_status.get(snapshot.slot.camera_id)
        if previous is not snapshot.status:
            self._last_status[snapshot.slot.camera_id] = snapshot.status
            if snapshot.status is CameraViewStatus.LIVE:
                log_event(
                    self._logger,
                    logging.INFO,
                    "ui_stream_live",
                    camera=snapshot.slot.camera_id,
                )
            elif snapshot.status is CameraViewStatus.RECONNECTING:
                log_event(
                    self._logger,
                    logging.WARNING,
                    "ui_stream_reconnecting",
                    camera=snapshot.slot.camera_id,
                )
            elif snapshot.status is CameraViewStatus.OFFLINE:
                log_event(
                    self._logger,
                    logging.WARNING,
                    "ui_stream_offline",
                    camera=snapshot.slot.camera_id,
                )
        self.snapshot_changed.emit(snapshot.slot.camera_id, snapshot)

    @staticmethod
    def _initial_status(slot: CameraSlot) -> tuple[CameraViewStatus, str]:
        if not slot.enabled:
            return CameraViewStatus.DISABLED, "Camera disabilitata"
        if not slot.configured:
            return CameraViewStatus.NOT_CONFIGURED, "Configura l'URL locale dello stream"
        return CameraViewStatus.CONNECTING, "Connessione in corso..."

    @staticmethod
    def _looks_offline(message: str | None) -> bool:
        normalized = (message or "").lower()
        return any(
            marker in normalized
            for marker in (
                "offline",
                "unreachable",
                "disconnected",
                "timed out",
                "timeout",
                "reconnect attempts exhausted",
                "could not open",
            )
        )
