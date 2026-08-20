package com.localsecuritycam.android.diagnostics

import kotlin.math.max

enum class StreamState {
    STOPPED,
    WAITING_NETWORK,
    STARTING,
    STREAMING,
    STOPPING,
    ERROR,
}

data class StreamMetricsSnapshot(
    val cameraFrames: Long,
    val encodedFrames: Long,
    val framesDropped: Long,
    val encodedFps: Double,
    val encoderLatencyMs: Double?,
    val bytesSent: Long,
    val currentBitrate: Long,
    val connectedClients: Int,
    val reconnectCount: Long,
    val sessionRestartCount: Long,
    val streamUptimeMs: Long,
    val lastError: String?,
    val cameraFps: Double,
)

class StreamMetrics(
    initialReconnectCount: Long = 0L,
    initialSessionRestartCount: Long = 0L,
) {
    private var startedAtNs: Long? = null
    private var lastSnapshotNs = System.nanoTime()
    private var lastSnapshotFrames = 0L
    private var lastSnapshotCameraFrames = 0L
    private var lastSnapshotBytes = 0L
    private var cameraFrames = 0L
    private var encodedFrames = 0L
    private var framesDropped = 0L
    private var bytesSent = 0L
    private var connectedClients = 0
    private var reconnectCount = initialReconnectCount.coerceAtLeast(0L)
    private var sessionRestartCount = initialSessionRestartCount.coerceAtLeast(0L)
    private var encoderLatencyTotalMs = 0.0
    private var encoderLatencySamples = 0L
    private var lastError: String? = null

    @Synchronized
    fun start() {
        startedAtNs = System.nanoTime()
        lastSnapshotNs = System.nanoTime()
        lastSnapshotFrames = encodedFrames
        lastSnapshotCameraFrames = cameraFrames
        lastSnapshotBytes = bytesSent
        lastError = null
    }

    @Synchronized
    fun stop() {
        startedAtNs = null
        connectedClients = 0
    }

    @Synchronized
    fun recordCameraFrame() {
        cameraFrames++
    }

    @Synchronized
    fun recordEncodedFrame(latencyMs: Double? = null) {
        encodedFrames++
        if (latencyMs != null && latencyMs.isFinite() && latencyMs >= 0) {
            encoderLatencyTotalMs += latencyMs
            encoderLatencySamples++
        }
    }

    @Synchronized
    fun recordBytesSent(byteCount: Long) {
        bytesSent += max(0L, byteCount)
    }

    @Synchronized
    fun recordDroppedFrame() {
        framesDropped++
    }

    @Synchronized
    fun recordReconnect() {
        reconnectCount++
    }

    @Synchronized
    fun recordSessionRestart() {
        sessionRestartCount++
    }

    @Synchronized
    fun setConnectedClients(count: Int) {
        connectedClients = max(0, count)
    }

    @Synchronized
    fun recordError(message: String?) {
        lastError = message
    }

    @Synchronized
    fun snapshot(nowNs: Long = System.nanoTime()): StreamMetricsSnapshot {
        val elapsedNs = max(1L, nowNs - lastSnapshotNs)
        val framesDelta = encodedFrames - lastSnapshotFrames
        val cameraFramesDelta = cameraFrames - lastSnapshotCameraFrames
        val bytesDelta = bytesSent - lastSnapshotBytes
        val fps = framesDelta.toDouble() * 1_000_000_000.0 / elapsedNs.toDouble()
        val bitrate = bytesDelta.toLong() * 8_000_000_000L / elapsedNs
        lastSnapshotNs = nowNs
        lastSnapshotFrames = encodedFrames
        lastSnapshotCameraFrames = cameraFrames
        lastSnapshotBytes = bytesSent
        val started = startedAtNs
        val uptime = if (started == null) 0L else max(0L, (nowNs - started) / 1_000_000L)
        return StreamMetricsSnapshot(
            cameraFrames = cameraFrames,
            encodedFrames = encodedFrames,
            framesDropped = framesDropped,
            encodedFps = fps,
            encoderLatencyMs = if (encoderLatencySamples == 0L) null else encoderLatencyTotalMs / encoderLatencySamples,
            bytesSent = bytesSent,
            currentBitrate = bitrate,
            connectedClients = connectedClients,
            reconnectCount = reconnectCount,
            sessionRestartCount = sessionRestartCount,
            streamUptimeMs = uptime,
            lastError = lastError,
            cameraFps = cameraFramesDelta.toDouble() * 1_000_000_000.0 / elapsedNs.toDouble(),
        )
    }
}
