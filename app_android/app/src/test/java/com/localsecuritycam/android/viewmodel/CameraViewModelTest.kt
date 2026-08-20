package com.localsecuritycam.android.viewmodel

import com.localsecuritycam.android.PlaceholderPanel
import com.localsecuritycam.android.diagnostics.StreamMetricsSnapshot
import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.service.ServiceSnapshot
import com.localsecuritycam.android.settings.AppSettings
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraViewModelTest {
    @Test
    fun startStopAndRetryEffectsFollowTheServiceProjection() = runBlocking {
        val viewModel = CameraViewModel()

        assertEquals(
            CameraUiEffect.StartStream,
            nextEffect(viewModel) {
                viewModel.onSnapshot(snapshot(StreamState.STOPPED))
                viewModel.onStreamActionClicked()
            },
        )
        assertEquals(
            CameraUiEffect.StopStream,
            nextEffect(viewModel) {
                viewModel.onSnapshot(snapshot(StreamState.STARTING))
                viewModel.onStreamActionClicked()
            },
        )
        assertEquals(
            CameraUiEffect.StartStream,
            nextEffect(viewModel) {
                viewModel.onSnapshot(snapshot(StreamState.ERROR))
                viewModel.onStreamActionClicked()
            },
        )
    }

    @Test
    fun navigationAndPanelStateRemainUiOnly() {
        val viewModel = CameraViewModel()

        viewModel.openDestination(CameraDestination.SETUP)
        assertEquals(CameraDestination.SETUP, viewModel.state.value.activeDestination)
        viewModel.toggleControlPanel()
        assertEquals(com.localsecuritycam.android.ControlPanelState.COLLAPSED, viewModel.state.value.local.controlPanelState)
        viewModel.openDestination(CameraDestination.DIAGNOSTICS)
        assertEquals(PlaceholderPanel.DIAGNOSTICS, viewModel.state.value.local.placeholderPanel)
        viewModel.closeDestination()
        assertEquals(CameraDestination.PREVIEW, viewModel.state.value.activeDestination)
        assertTrue(viewModel.state.value.diagnostics.isEmpty())
    }

    private suspend fun nextEffect(
        viewModel: CameraViewModel,
        trigger: () -> Unit,
    ): CameraUiEffect = coroutineScope {
        val deferred = async(start = CoroutineStart.UNDISPATCHED) {
            viewModel.effects.first()
        }
        trigger()
        withTimeout(1_000L) { deferred.await() }
    }

    private fun snapshot(state: StreamState) = ServiceSnapshot(
        state = state,
        wifiConnected = true,
        ssid = "local",
        localIp = "192.168.1.20",
        rtspUrl = "rtsp://192.168.1.20:8554/stream",
        metrics = StreamMetricsSnapshot(
            cameraFrames = 0,
            encodedFrames = 0,
            framesDropped = 0,
            encodedFps = 0.0,
            encoderLatencyMs = null,
            bytesSent = 0,
            currentBitrate = 0,
            connectedClients = 0,
            reconnectCount = 0,
            sessionRestartCount = 0,
            streamUptimeMs = 0,
            lastError = null,
            cameraFps = 0.0,
        ),
        settings = AppSettings(),
        lastError = null,
    )
}
