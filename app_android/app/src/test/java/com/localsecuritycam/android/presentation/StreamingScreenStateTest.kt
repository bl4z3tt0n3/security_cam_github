package com.localsecuritycam.android.presentation

import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamMetricsSnapshot
import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.diagnostics.StreamSubsystemSnapshot
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.service.PreviewState
import com.localsecuritycam.android.service.ServiceSnapshot
import com.localsecuritycam.android.settings.AppSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamingScreenStateTest {
    @Test
    fun stoppedEnablesStartAndDisablesNoServerClaim() {
        val state = StreamingScreenStateMapper.map(snapshot(StreamState.STOPPED))

        assertEquals(StreamAction.START, state.action)
        assertTrue(state.actionEnabled)
        assertEquals(StreamingVisualState.STOPPED, state.visualState)
        assertEquals(HeaderUiMode.IDLE, state.headerUiMode)
        assertTrue(state.headerUiMode.showsCameraSelector)
        assertEquals("LAN", state.headerLabel)
        assertEquals("Preview idle", state.previewLabel)
        assertFalse(state.serverReady)
        assertFalse(state.clientConnected)
    }

    @Test
    fun activePreviewWithoutStreamUsesTheExplicitPreviewBadge() {
        val state = StreamingScreenStateMapper.map(
            snapshot(
                StreamState.STOPPED,
                previewState = PreviewState.ACTIVE,
                subsystems = StreamSubsystemSnapshot(camera = SubsystemState.RUNNING),
            ),
        )

        assertEquals("Preview active · stream not started", state.previewLabel)
        assertTrue(state.cameraReady)
        assertFalse(state.encoderReady)
        assertFalse(state.serverReady)
    }

    @Test
    fun previewErrorIsDisplayedSeparatelyFromAStreamError() {
        val state = StreamingScreenStateMapper.map(
            snapshot(
                StreamState.STOPPED,
                previewState = PreviewState.ERROR,
                previewError = "Camera2 preview failed",
                previewErrorKind = StreamErrorKind.CAMERA,
            ),
        )

        assertEquals("Preview error", state.previewLabel)
        assertEquals(StreamUiErrorKind.CAMERA_ERROR, state.lastError?.kind)
    }

    @Test
    fun startingAndWaitingNetworkAllowCancellation() {
        val starting = StreamingScreenStateMapper.map(snapshot(StreamState.STARTING))
        val waiting = StreamingScreenStateMapper.map(snapshot(StreamState.WAITING_NETWORK))
        val stopping = StreamingScreenStateMapper.map(snapshot(StreamState.STOPPING))

        assertEquals(StreamAction.STOP, starting.action)
        assertTrue(starting.actionEnabled)
        assertEquals(StreamAction.STOP, waiting.action)
        assertEquals(StreamUiErrorKind.NETWORK_UNAVAILABLE, waiting.lastError?.kind)
        listOf(starting, waiting, stopping).forEach { state ->
            assertEquals(HeaderUiMode.ACTIVE, state.headerUiMode)
            assertFalse(state.headerUiMode.showsCameraSelector)
        }
    }

    @Test
    fun activeServerWithoutClientIsNotAnError() {
        val state = StreamingScreenStateMapper.map(
            snapshot(
                StreamState.STREAMING,
                subsystems = StreamSubsystemSnapshot(
                    camera = SubsystemState.RUNNING,
                    encoder = SubsystemState.RUNNING,
                    rtspServer = SubsystemState.RUNNING,
                ),
            ),
        )

        assertEquals(StreamingVisualState.WAITING_FOR_CLIENT, state.visualState)
        assertEquals(HeaderUiMode.ACTIVE, state.headerUiMode)
        assertFalse(state.headerUiMode.showsCameraSelector)
        assertEquals("SERVER ATTIVO", state.headerLabel)
        assertEquals("Streaming active", state.previewLabel)
        assertTrue(state.serverReady)
        assertFalse(state.clientConnected)
        assertNull(state.lastError)
    }

    @Test
    fun connectedClientChangesOnlyTheClientProjection() {
        val state = StreamingScreenStateMapper.map(
            snapshot(
                StreamState.STREAMING,
                clients = 1,
                subsystems = StreamSubsystemSnapshot(
                    camera = SubsystemState.RUNNING,
                    encoder = SubsystemState.RUNNING,
                    rtspServer = SubsystemState.RUNNING,
                ),
            ),
        )

        assertEquals(StreamingVisualState.CLIENT_CONNECTED, state.visualState)
        assertEquals(HeaderUiMode.ACTIVE, state.headerUiMode)
        assertFalse(state.headerUiMode.showsCameraSelector)
        assertEquals("CLIENT CONNESSO", state.headerLabel)
        assertEquals("Streaming active", state.previewLabel)
        assertTrue(state.clientConnected)
        assertTrue(state.serverReady)
    }

    @Test
    fun encoderFailureProducesSemanticUiErrorAndRetry() {
        val state = StreamingScreenStateMapper.map(
            snapshot(
                StreamState.ERROR,
                lastError = "Errore encoder H.264",
                lastErrorKind = StreamErrorKind.ENCODER,
            ),
        )

        assertEquals(StreamAction.START, state.action)
        assertEquals(StreamUiErrorKind.ENCODER_ERROR, state.lastError?.kind)
        assertEquals(HeaderUiMode.ERROR, state.headerUiMode)
        assertTrue(state.headerUiMode.showsCameraSelector)
        assertEquals("ERRORE", state.headerLabel)
        assertEquals("Preview active · stream not started", state.previewLabel)
        assertTrue(state.actionEnabled)
    }

    @Test
    fun streamErrorKeepsPreviewErrorBadgeReservedForPreviewFailures() {
        val state = StreamingScreenStateMapper.map(
            snapshot(
                StreamState.ERROR,
                previewState = PreviewState.ACTIVE,
                lastError = "RTSP bind failed",
                lastErrorKind = StreamErrorKind.RTSP_SERVER,
            ),
        )

        assertEquals("Preview active · stream not started", state.previewLabel)
        assertEquals(StreamUiErrorKind.SERVER_BIND_ERROR, state.lastError?.kind)
    }

    private fun snapshot(
        state: StreamState,
        clients: Int = 0,
        subsystems: StreamSubsystemSnapshot = StreamSubsystemSnapshot(),
        lastError: String? = null,
        lastErrorKind: StreamErrorKind? = null,
        previewState: PreviewState = if (state == StreamState.STOPPED) PreviewState.IDLE else PreviewState.ACTIVE,
        previewError: String? = null,
        previewErrorKind: StreamErrorKind? = null,
    ): ServiceSnapshot = ServiceSnapshot(
        state = state,
        wifiConnected = true,
        ssid = "local",
        localIp = "192.168.1.20",
        rtspUrl = "rtsp://camera:***@192.168.1.20:8554/stream",
        metrics = StreamMetricsSnapshot(
            cameraFrames = 0,
            encodedFrames = 0,
            framesDropped = 0,
            encodedFps = 0.0,
            encoderLatencyMs = null,
            bytesSent = 0,
            currentBitrate = 0,
            connectedClients = clients,
            reconnectCount = 0,
            sessionRestartCount = 0,
            streamUptimeMs = 0,
            lastError = lastError,
            cameraFps = 0.0,
        ),
        settings = AppSettings(),
        lastError = lastError,
        lastErrorKind = lastErrorKind,
        previewState = previewState,
        previewError = previewError,
        previewErrorKind = previewErrorKind,
        subsystems = subsystems,
    )
}
