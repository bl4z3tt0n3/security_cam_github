package com.localsecuritycam.android.ui

import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.diagnostics.StreamSubsystemSnapshot
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.service.PreviewState
import com.localsecuritycam.android.service.ServiceSnapshot
import java.util.Locale

internal data class DiagnosticsUiState(
    val streamState: StreamState,
    val previewState: PreviewState,
    val subsystems: StreamSubsystemSnapshot,
    val wifiConnected: Boolean,
    val ssid: String?,
    val localIp: String?,
    val rtspPort: Int,
    val rtspUrl: String?,
    val cameraFrames: Long,
    val encodedFrames: Long,
    val cameraFps: Double,
    val encodedFps: Double,
    val framesDropped: Long,
    val encoderLatencyMs: Double?,
    val bytesSent: Long,
    val currentBitrate: Long,
    val connectedClients: Int,
    val reconnectCount: Long,
    val sessionRestartCount: Long,
    val streamUptimeMs: Long,
    val lastError: String?,
    val previewError: String?,
)

internal fun diagnosticsUiState(snapshot: ServiceSnapshot): DiagnosticsUiState {
    val metrics = snapshot.metrics
    return DiagnosticsUiState(
        streamState = snapshot.state,
        previewState = snapshot.previewState,
        subsystems = snapshot.subsystems,
        wifiConnected = snapshot.wifiConnected,
        ssid = snapshot.ssid,
        localIp = snapshot.localIp,
        rtspPort = snapshot.settings.stream.port,
        rtspUrl = snapshot.rtspUrl,
        cameraFrames = metrics.cameraFrames,
        encodedFrames = metrics.encodedFrames,
        cameraFps = metrics.cameraFps,
        encodedFps = metrics.encodedFps,
        framesDropped = metrics.framesDropped,
        encoderLatencyMs = metrics.encoderLatencyMs,
        bytesSent = metrics.bytesSent,
        currentBitrate = metrics.currentBitrate,
        connectedClients = metrics.connectedClients,
        reconnectCount = metrics.reconnectCount,
        sessionRestartCount = metrics.sessionRestartCount,
        streamUptimeMs = metrics.streamUptimeMs,
        lastError = snapshot.lastError ?: metrics.lastError,
        previewError = snapshot.previewError,
    )
}

internal fun DiagnosticsUiState.cameraState(): SubsystemState = subsystems.camera

internal fun DiagnosticsUiState.encoderState(): SubsystemState = subsystems.encoder

internal fun DiagnosticsUiState.rtspServerState(): SubsystemState = subsystems.rtspServer

internal enum class DiagnosticsSectionIcon {
    STREAM_CAMERA,
    CONNECTION,
    ORIENTATION,
    PERFORMANCE,
}

internal enum class DiagnosticsField {
    STREAM_STATE,
    PREVIEW_STATE,
    CAMERA_STATE,
    ENCODER_STATE,
    RTSP_SERVER_STATE,
    PREVIEW_ERROR,
    LAST_ERROR,
    WIFI_STATUS,
    SSID,
    WIFI_IP,
    RTSP_PORT,
    RTSP_URI,
    ORIENTATION_SOURCE,
    LENS_SENSOR,
    PHYSICAL_DISPLAY,
    REQUESTED_ROTATION,
    CAMERA_BUFFER,
    ENCODER_OUTPUT,
    MIRROR_PREVIEW_STREAM,
    CAMERA_FRAMES,
    ENCODED_FRAMES,
    CAMERA_FPS,
    ENCODED_FPS,
    DROPPED_FRAMES,
    ENCODER_LATENCY,
    BYTES_SENT,
    BITRATE,
    CONNECTED_CLIENTS,
    RECONNECTS,
    SESSION_RESTARTS,
    UPTIME,
}

internal enum class DiagnosticsValueTone {
    NEUTRAL,
    SUCCESS,
    ERROR,
}

internal data class DiagnosticsRowUiState(
    val field: DiagnosticsField,
    val label: String,
    val value: String,
    val tone: DiagnosticsValueTone = DiagnosticsValueTone.NEUTRAL,
    val allowsMultilineValue: Boolean = false,
)

internal data class DiagnosticsSectionUiState(
    val title: String,
    val icon: DiagnosticsSectionIcon,
    val rows: List<DiagnosticsRowUiState>,
)

/**
 * Presentation-only grouping for the Diagnostics sheet. It deliberately reads the same
 * [ServiceSnapshot] values used by [diagnosticsUiState] and adds no telemetry or service state.
 */
internal fun diagnosticsSections(snapshot: ServiceSnapshot): List<DiagnosticsSectionUiState> {
    val uiState = diagnosticsUiState(snapshot)
    val orientation = snapshot.orientation

    return listOf(
        DiagnosticsSectionUiState(
            title = "Stream & camera state",
            icon = DiagnosticsSectionIcon.STREAM_CAMERA,
            rows = listOf(
                DiagnosticsRowUiState(
                    field = DiagnosticsField.STREAM_STATE,
                    label = "Stream state",
                    value = uiState.streamState.name,
                    tone = uiState.streamState.diagnosticsTone(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.PREVIEW_STATE,
                    label = "Preview state",
                    value = uiState.previewState.name,
                    tone = uiState.previewState.diagnosticsTone(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.CAMERA_STATE,
                    label = "Camera state",
                    value = uiState.cameraState().name,
                    tone = uiState.cameraState().diagnosticsTone(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.ENCODER_STATE,
                    label = "Encoder state",
                    value = uiState.encoderState().name,
                    tone = uiState.encoderState().diagnosticsTone(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.RTSP_SERVER_STATE,
                    label = "RTSP server state",
                    value = uiState.rtspServerState().name,
                    tone = uiState.rtspServerState().diagnosticsTone(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.PREVIEW_ERROR,
                    label = "Preview error",
                    value = uiState.previewError ?: NO_ERROR,
                    tone = if (uiState.previewError == null) {
                        DiagnosticsValueTone.NEUTRAL
                    } else {
                        DiagnosticsValueTone.ERROR
                    },
                    allowsMultilineValue = true,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.LAST_ERROR,
                    label = "Last error",
                    value = uiState.lastError ?: NO_ERROR,
                    tone = if (uiState.lastError == null) {
                        DiagnosticsValueTone.NEUTRAL
                    } else {
                        DiagnosticsValueTone.ERROR
                    },
                    allowsMultilineValue = true,
                ),
            ),
        ),
        DiagnosticsSectionUiState(
            title = "Connection",
            icon = DiagnosticsSectionIcon.CONNECTION,
            rows = listOf(
                DiagnosticsRowUiState(
                    field = DiagnosticsField.WIFI_STATUS,
                    label = "Wi-Fi status",
                    value = if (uiState.wifiConnected) "connected" else "disconnected",
                    tone = if (uiState.wifiConnected) {
                        DiagnosticsValueTone.SUCCESS
                    } else {
                        DiagnosticsValueTone.ERROR
                    },
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.SSID,
                    label = "SSID",
                    value = uiState.ssid ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.WIFI_IP,
                    label = "Wi-Fi IP",
                    value = uiState.localIp ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.RTSP_PORT,
                    label = "RTSP port",
                    value = uiState.rtspPort.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.RTSP_URI,
                    label = "RTSP URI",
                    value = uiState.rtspUrl ?: NOT_AVAILABLE,
                    allowsMultilineValue = true,
                ),
            ),
        ),
        DiagnosticsSectionUiState(
            title = "Orientation & geometry",
            icon = DiagnosticsSectionIcon.ORIENTATION,
            rows = listOf(
                DiagnosticsRowUiState(
                    field = DiagnosticsField.ORIENTATION_SOURCE,
                    label = "Orientation source",
                    value = orientation?.source?.name?.lowercase() ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.LENS_SENSOR,
                    label = "Lens / sensor",
                    value = orientation?.let {
                        "${it.lensFacing.name.lowercase()} / ${it.sensorOrientationDegrees} deg"
                    } ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.PHYSICAL_DISPLAY,
                    label = "Physical / display",
                    value = orientation?.let {
                        "${it.physicalOrientationDegrees ?: "fallback"} deg / ${it.displayRotationDegrees} deg"
                    } ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.REQUESTED_ROTATION,
                    label = "Requested rotation",
                    value = orientation?.requestedRotationDegrees?.let { "$it deg" } ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.CAMERA_BUFFER,
                    label = "Camera buffer",
                    value = orientation?.bufferResolution?.toString() ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.ENCODER_OUTPUT,
                    label = "Encoder output",
                    value = orientation?.outputResolution?.toString() ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.MIRROR_PREVIEW_STREAM,
                    label = "Mirror preview / stream",
                    value = orientation?.let { "${it.mirrorPreview} / ${it.mirrorStream}" } ?: NOT_AVAILABLE,
                ),
            ),
        ),
        DiagnosticsSectionUiState(
            title = "Performance",
            icon = DiagnosticsSectionIcon.PERFORMANCE,
            rows = listOf(
                DiagnosticsRowUiState(
                    field = DiagnosticsField.CAMERA_FRAMES,
                    label = "Camera frames",
                    value = uiState.cameraFrames.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.ENCODED_FRAMES,
                    label = "Encoded frames",
                    value = uiState.encodedFrames.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.CAMERA_FPS,
                    label = "Camera FPS",
                    value = formatDiagnosticsDecimal(uiState.cameraFps),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.ENCODED_FPS,
                    label = "Encoded FPS",
                    value = formatDiagnosticsDecimal(uiState.encodedFps),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.DROPPED_FRAMES,
                    label = "Dropped frames",
                    value = uiState.framesDropped.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.ENCODER_LATENCY,
                    label = "Encoder latency",
                    value = uiState.encoderLatencyMs?.let { "${formatDiagnosticsDecimal(it)} ms" } ?: NOT_AVAILABLE,
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.BYTES_SENT,
                    label = "Bytes sent",
                    value = uiState.bytesSent.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.BITRATE,
                    label = "Bitrate",
                    value = "${uiState.currentBitrate / 1000} kbps",
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.CONNECTED_CLIENTS,
                    label = "Connected clients",
                    value = uiState.connectedClients.toString(),
                    tone = if (uiState.connectedClients > 0) {
                        DiagnosticsValueTone.SUCCESS
                    } else {
                        DiagnosticsValueTone.NEUTRAL
                    },
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.RECONNECTS,
                    label = "Reconnects",
                    value = uiState.reconnectCount.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.SESSION_RESTARTS,
                    label = "Session restarts",
                    value = uiState.sessionRestartCount.toString(),
                ),
                DiagnosticsRowUiState(
                    field = DiagnosticsField.UPTIME,
                    label = "Uptime",
                    value = "${uiState.streamUptimeMs / 1000} s",
                ),
            ),
        ),
    )
}

private fun StreamState.diagnosticsTone(): DiagnosticsValueTone = when (this) {
    StreamState.STREAMING -> DiagnosticsValueTone.SUCCESS
    StreamState.ERROR -> DiagnosticsValueTone.ERROR
    else -> DiagnosticsValueTone.NEUTRAL
}

private fun PreviewState.diagnosticsTone(): DiagnosticsValueTone = when (this) {
    PreviewState.ACTIVE -> DiagnosticsValueTone.SUCCESS
    PreviewState.ERROR -> DiagnosticsValueTone.ERROR
    else -> DiagnosticsValueTone.NEUTRAL
}

private fun SubsystemState.diagnosticsTone(): DiagnosticsValueTone = when (this) {
    SubsystemState.RUNNING -> DiagnosticsValueTone.SUCCESS
    SubsystemState.ERROR -> DiagnosticsValueTone.ERROR
    else -> DiagnosticsValueTone.NEUTRAL
}

private fun formatDiagnosticsDecimal(value: Double): String = String.format(Locale.US, "%.2f", value)

private const val NOT_AVAILABLE = "n/d"
private const val NO_ERROR = "none"
