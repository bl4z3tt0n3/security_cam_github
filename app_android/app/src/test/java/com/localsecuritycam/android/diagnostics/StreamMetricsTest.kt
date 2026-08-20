package com.localsecuritycam.android.diagnostics

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamMetricsTest {
    @Test
    fun computesRatesFromBoundedSnapshotWindow() {
        val metrics = StreamMetrics()
        metrics.start()
        metrics.recordCameraFrame()
        metrics.recordEncodedFrame()
        metrics.recordBytesSent(104)
        val snapshot = metrics.snapshot(System.nanoTime() + 1_000_000_000L)
        assertEquals(1L, snapshot.cameraFrames)
        assertEquals(1L, snapshot.encodedFrames)
        assertEquals(104L, snapshot.bytesSent)
        assertTrue(snapshot.encodedFps >= 0.0)
    }

    @Test
    fun currentBitrateIsZeroWhenNoRtpBytesWereSent() {
        val metrics = StreamMetrics()
        metrics.start()

        val snapshot = metrics.snapshot(System.nanoTime() + 1_000_000_000L)

        assertEquals(0L, snapshot.bytesSent)
        assertEquals(0L, snapshot.currentBitrate)
    }

    @Test
    fun currentBitrateUsesBytesActuallySentToRtpClients() {
        val metrics = StreamMetrics()
        metrics.start()
        metrics.recordBytesSent(1_000)

        val snapshot = metrics.snapshot(System.nanoTime() + 1_000_000_000L)

        assertTrue(snapshot.currentBitrate > 0L)
    }

    @Test
    fun preservesLastErrorAcrossMetricRefreshes() {
        val metrics = StreamMetrics()
        metrics.start()
        metrics.recordError("socket warning")

        val first = metrics.snapshot(System.nanoTime() + 1_000_000_000L)
        val second = metrics.snapshot(System.nanoTime() + 2_000_000_000L)

        assertEquals("socket warning", first.lastError)
        assertEquals("socket warning", second.lastError)
    }

    @Test
    fun countsDropsAndReconnects() {
        val metrics = StreamMetrics()
        metrics.recordDroppedFrame()
        metrics.recordReconnect()
        metrics.recordSessionRestart()
        val snapshot = metrics.snapshot()
        assertEquals(1L, snapshot.framesDropped)
        assertEquals(1L, snapshot.reconnectCount)
        assertEquals(1L, snapshot.sessionRestartCount)
    }

    @Test
    fun preservesCountersWhenAStreamPipelineIsRecreated() {
        val metrics = StreamMetrics(initialReconnectCount = 2, initialSessionRestartCount = 3)
        metrics.start()

        val snapshot = metrics.snapshot()

        assertEquals(2L, snapshot.reconnectCount)
        assertEquals(3L, snapshot.sessionRestartCount)
    }
}
