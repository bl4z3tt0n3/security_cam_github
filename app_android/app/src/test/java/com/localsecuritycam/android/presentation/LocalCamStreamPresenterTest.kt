package com.localsecuritycam.android.presentation

import com.localsecuritycam.android.diagnostics.StreamMetricsSnapshot
import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.service.ServiceSnapshot
import com.localsecuritycam.android.settings.AppSettings
import org.junit.Assert.assertEquals
import org.junit.Test

class LocalCamStreamPresenterTest {
    @Test
    fun startStopAndRetryIntentsFollowOnlyTheLatestServiceSnapshot() {
        var starts = 0
        var stops = 0
        val presenter = LocalCamStreamPresenter(
            onStartRequested = { starts++ },
            onStopRequested = { stops++ },
            onStateChanged = {},
        )

        presenter.onSnapshot(snapshot(StreamState.STOPPED))
        presenter.onStreamActionClicked()
        presenter.onSnapshot(snapshot(StreamState.STARTING))
        presenter.onStreamActionClicked()
        presenter.onSnapshot(snapshot(StreamState.STOPPING))
        presenter.onStreamActionClicked()
        presenter.onSnapshot(snapshot(StreamState.ERROR))
        presenter.onStreamActionClicked()

        assertEquals(2, starts)
        assertEquals(1, stops)
    }

    private fun snapshot(state: StreamState): ServiceSnapshot = ServiceSnapshot(
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
