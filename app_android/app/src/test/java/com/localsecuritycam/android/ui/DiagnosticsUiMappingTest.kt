package com.localsecuritycam.android.ui

import com.localsecuritycam.android.camera.CameraOrientationState
import com.localsecuritycam.android.camera.OrientationSource
import com.localsecuritycam.android.diagnostics.StreamMetricsSnapshot
import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.diagnostics.StreamSubsystemSnapshot
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.service.PreviewState
import com.localsecuritycam.android.service.ServiceSnapshot
import com.localsecuritycam.android.settings.AppSettings
import com.localsecuritycam.android.settings.CameraLens
import com.localsecuritycam.android.settings.Resolution
import com.localsecuritycam.android.settings.StreamSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DiagnosticsUiMappingTest {
    @Test
    fun mapsRuntimeSnapshotToAllLiveDiagnosticsFields() {
        val uiState = diagnosticsUiState(
            snapshot(
                state = StreamState.STREAMING,
                subsystems = StreamSubsystemSnapshot(
                    camera = SubsystemState.RUNNING,
                    encoder = SubsystemState.RUNNING,
                    rtspServer = SubsystemState.RUNNING,
                ),
                metrics = metrics(
                    cameraFrames = 120,
                    encodedFrames = 100,
                    cameraFps = 24.0,
                    encodedFps = 20.0,
                    currentBitrate = 2_000_000,
                    connectedClients = 2,
                ),
                lastError = "previous socket warning",
            ),
        )

        assertEquals(StreamState.STREAMING, uiState.streamState)
        assertEquals(SubsystemState.RUNNING, uiState.cameraState())
        assertEquals(SubsystemState.RUNNING, uiState.encoderState())
        assertEquals(SubsystemState.RUNNING, uiState.rtspServerState())
        assertEquals("192.168.1.20", uiState.localIp)
        assertEquals(8554, uiState.rtspPort)
        assertEquals("rtsp://192.168.1.20:8554/stream", uiState.rtspUrl)
        assertEquals(120L, uiState.cameraFrames)
        assertEquals(100L, uiState.encodedFrames)
        assertEquals(20.0, uiState.encodedFps, 0.0)
        assertEquals(2_000_000L, uiState.currentBitrate)
        assertEquals(2, uiState.connectedClients)
        assertEquals("previous socket warning", uiState.lastError)
    }

    @Test
    fun keepsZeroClientsAndZeroBitrateWithoutRtpTraffic() {
        val uiState = diagnosticsUiState(
            snapshot(
                state = StreamState.STREAMING,
                metrics = metrics(connectedClients = 0, currentBitrate = 0),
            ),
        )

        assertEquals(0, uiState.connectedClients)
        assertEquals(0L, uiState.currentBitrate)
    }

    @Test
    fun preservesNullErrorForTheScreenFallback() {
        val uiState = diagnosticsUiState(snapshot(lastError = null))

        assertNull(uiState.lastError)
    }

    @Test
    fun fallsBackToTheMetricErrorWhenSnapshotErrorIsMissing() {
        val uiState = diagnosticsUiState(
            snapshot(
                lastError = null,
                metrics = metrics().copy(lastError = "encoder warning"),
            ),
        )

        assertEquals("encoder warning", uiState.lastError)
    }

    @Test
    fun keepsPreviewErrorSeparateFromStreamError() {
        val uiState = diagnosticsUiState(
            snapshot(
                lastError = "RTSP bind failed",
                previewState = PreviewState.ERROR,
                previewError = "Preview surface lost",
            ),
        )

        assertEquals("RTSP bind failed", uiState.lastError)
        assertEquals("Preview surface lost", uiState.previewError)
    }

    @Test
    fun groupsAllCurrentDiagnosticsFieldsWithoutChangingTheirFormats() {
        val sections = diagnosticsSections(
            snapshot(
                state = StreamState.ERROR,
                wifiConnected = false,
                rtspUrl = "rtsp://admin:***@192.168.1.20:8554/stream/with/a/long/path",
                subsystems = StreamSubsystemSnapshot(
                    camera = SubsystemState.ERROR,
                    encoder = SubsystemState.IDLE,
                    rtspServer = SubsystemState.ERROR,
                ),
                metrics = metrics(
                    cameraFrames = 120,
                    encodedFrames = 100,
                    framesDropped = 3,
                    cameraFps = 24.0,
                    encodedFps = 20.0,
                    encoderLatencyMs = 18.5,
                    bytesSent = 42_000,
                    currentBitrate = 2_000_000,
                    connectedClients = 2,
                    reconnectCount = 4,
                    sessionRestartCount = 5,
                    streamUptimeMs = 6_000,
                ),
                lastError = "encoder thread did not stop",
                orientation = orientation(),
            ),
        )

        assertEquals(
            listOf(
                "Stream & camera state",
                "Connection",
                "Orientation & geometry",
                "Performance",
            ),
            sections.map { it.title },
        )
        assertEquals(
            listOf(
                listOf(
                    "Stream state",
                    "Preview state",
                    "Camera state",
                    "Encoder state",
                    "RTSP server state",
                    "Preview error",
                    "Last error",
                ),
                listOf("Wi-Fi status", "SSID", "Wi-Fi IP", "RTSP port", "RTSP URI"),
                listOf(
                    "Orientation source",
                    "Lens / sensor",
                    "Physical / display",
                    "Requested rotation",
                    "Camera buffer",
                    "Encoder output",
                    "Mirror preview / stream",
                ),
                listOf(
                    "Camera frames",
                    "Encoded frames",
                    "Camera FPS",
                    "Encoded FPS",
                    "Dropped frames",
                    "Encoder latency",
                    "Bytes sent",
                    "Bitrate",
                    "Connected clients",
                    "Reconnects",
                    "Session restarts",
                    "Uptime",
                ),
            ),
            sections.map { section -> section.rows.map { row -> row.label } },
        )
        assertEquals(31, sections.sumOf { it.rows.size })

        assertEquals("ERROR", sections.row(DiagnosticsField.STREAM_STATE).value)
        assertEquals(DiagnosticsValueTone.ERROR, sections.row(DiagnosticsField.STREAM_STATE).tone)
        assertEquals("IDLE", sections.row(DiagnosticsField.PREVIEW_STATE).value)
        assertEquals(DiagnosticsValueTone.ERROR, sections.row(DiagnosticsField.CAMERA_STATE).tone)
        assertEquals(DiagnosticsValueTone.ERROR, sections.row(DiagnosticsField.RTSP_SERVER_STATE).tone)
        assertEquals("none", sections.row(DiagnosticsField.PREVIEW_ERROR).value)
        assertEquals(DiagnosticsValueTone.NEUTRAL, sections.row(DiagnosticsField.PREVIEW_ERROR).tone)
        assertEquals("encoder thread did not stop", sections.row(DiagnosticsField.LAST_ERROR).value)
        assertEquals(DiagnosticsValueTone.ERROR, sections.row(DiagnosticsField.LAST_ERROR).tone)
        assertTrue(sections.row(DiagnosticsField.LAST_ERROR).allowsMultilineValue)
        assertEquals("disconnected", sections.row(DiagnosticsField.WIFI_STATUS).value)
        assertEquals(DiagnosticsValueTone.ERROR, sections.row(DiagnosticsField.WIFI_STATUS).tone)
        assertEquals("rtsp://admin:***@192.168.1.20:8554/stream/with/a/long/path", sections.row(DiagnosticsField.RTSP_URI).value)
        assertTrue(sections.row(DiagnosticsField.RTSP_URI).allowsMultilineValue)
        assertEquals("physical_sensor", sections.row(DiagnosticsField.ORIENTATION_SOURCE).value)
        assertEquals("back / 90 deg", sections.row(DiagnosticsField.LENS_SENSOR).value)
        assertEquals("0 deg / 0 deg", sections.row(DiagnosticsField.PHYSICAL_DISPLAY).value)
        assertEquals("90 deg", sections.row(DiagnosticsField.REQUESTED_ROTATION).value)
        assertEquals("1920x1080", sections.row(DiagnosticsField.CAMERA_BUFFER).value)
        assertEquals("1080x1920", sections.row(DiagnosticsField.ENCODER_OUTPUT).value)
        assertEquals("false / false", sections.row(DiagnosticsField.MIRROR_PREVIEW_STREAM).value)
        assertEquals("24.00", sections.row(DiagnosticsField.CAMERA_FPS).value)
        assertEquals("20.00", sections.row(DiagnosticsField.ENCODED_FPS).value)
        assertEquals("18.50 ms", sections.row(DiagnosticsField.ENCODER_LATENCY).value)
        assertEquals("2000 kbps", sections.row(DiagnosticsField.BITRATE).value)
        assertEquals("2", sections.row(DiagnosticsField.CONNECTED_CLIENTS).value)
        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.CONNECTED_CLIENTS).tone)
        assertEquals("6 s", sections.row(DiagnosticsField.UPTIME).value)
    }

    @Test
    fun marksRunningSubsystemsAndConnectedNetworkAsSuccess() {
        val sections = diagnosticsSections(
            snapshot(
                state = StreamState.STREAMING,
                subsystems = StreamSubsystemSnapshot(
                    camera = SubsystemState.RUNNING,
                    encoder = SubsystemState.RUNNING,
                    rtspServer = SubsystemState.RUNNING,
                ),
                metrics = metrics(connectedClients = 1),
            ),
        )

        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.STREAM_STATE).tone)
        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.CAMERA_STATE).tone)
        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.ENCODER_STATE).tone)
        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.RTSP_SERVER_STATE).tone)
        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.WIFI_STATUS).tone)
        assertEquals(DiagnosticsValueTone.SUCCESS, sections.row(DiagnosticsField.CONNECTED_CLIENTS).tone)
    }

    private fun snapshot(
        state: StreamState = StreamState.STOPPED,
        wifiConnected: Boolean = true,
        ssid: String? = "Test Wi-Fi",
        localIp: String? = "192.168.1.20",
        rtspUrl: String? = "rtsp://192.168.1.20:8554/stream",
        subsystems: StreamSubsystemSnapshot = StreamSubsystemSnapshot(),
        metrics: StreamMetricsSnapshot = metrics(),
        lastError: String? = null,
        previewState: PreviewState = PreviewState.IDLE,
        previewError: String? = null,
        orientation: CameraOrientationState? = null,
    ) = ServiceSnapshot(
        state = state,
        wifiConnected = wifiConnected,
        ssid = ssid,
        localIp = localIp,
        rtspUrl = rtspUrl,
        metrics = metrics,
        settings = AppSettings(stream = StreamSettings(port = 8554)),
        lastError = lastError,
        previewState = previewState,
        previewError = previewError,
        subsystems = subsystems,
        orientation = orientation,
    )

    private fun metrics(
        cameraFrames: Long = 0,
        encodedFrames: Long = 0,
        framesDropped: Long = 0,
        cameraFps: Double = 0.0,
        encodedFps: Double = 0.0,
        encoderLatencyMs: Double? = null,
        bytesSent: Long = 0,
        currentBitrate: Long = 0,
        connectedClients: Int = 0,
        reconnectCount: Long = 0,
        sessionRestartCount: Long = 0,
        streamUptimeMs: Long = 0,
    ) = StreamMetricsSnapshot(
        cameraFrames = cameraFrames,
        encodedFrames = encodedFrames,
        framesDropped = framesDropped,
        encodedFps = encodedFps,
        encoderLatencyMs = encoderLatencyMs,
        bytesSent = bytesSent,
        currentBitrate = currentBitrate,
        connectedClients = connectedClients,
        reconnectCount = reconnectCount,
        sessionRestartCount = sessionRestartCount,
        streamUptimeMs = streamUptimeMs,
        lastError = null,
        cameraFps = cameraFps,
    )

    private fun orientation() = CameraOrientationState(
        sensorOrientationDegrees = 90,
        lensFacing = CameraLens.BACK,
        physicalOrientationDegrees = 0,
        displayRotationDegrees = 0,
        targetSurfaceRotationDegrees = 0,
        requestedRotationDegrees = 90,
        mirrorPreview = false,
        mirrorStream = false,
        bufferResolution = Resolution(1920, 1080),
        outputResolution = Resolution(1080, 1920),
        source = OrientationSource.PHYSICAL_SENSOR,
    )

    private fun List<DiagnosticsSectionUiState>.row(field: DiagnosticsField): DiagnosticsRowUiState =
        flatMap { it.rows }.single { it.field == field }
}
