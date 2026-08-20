package com.localsecuritycam.android.presentation

import com.localsecuritycam.android.diagnostics.StreamErrorKind
import com.localsecuritycam.android.diagnostics.StreamState
import com.localsecuritycam.android.diagnostics.SubsystemState
import com.localsecuritycam.android.service.PreviewState
import com.localsecuritycam.android.service.ServiceSnapshot

enum class StreamAction {
    START,
    STOP,
    NONE,
}

enum class StreamingVisualState {
    STOPPED,
    STARTING,
    WAITING_FOR_NETWORK,
    WAITING_FOR_CLIENT,
    CLIENT_CONNECTED,
    STOPPING,
    ERROR,
}

enum class HeaderUiMode(
    val showsCameraSelector: Boolean,
) {
    IDLE(showsCameraSelector = true),
    ACTIVE(showsCameraSelector = false),
    ERROR(showsCameraSelector = true),
}

enum class StreamUiErrorKind {
    CAMERA_ERROR,
    ENCODER_ERROR,
    SERVER_BIND_ERROR,
    AUTH_CONFIG_ERROR,
    NETWORK_UNAVAILABLE,
    WRITER_ERROR,
    STREAM_START_ERROR,
    STREAM_STOP_ERROR,
}

data class StreamUiError(
    val kind: StreamUiErrorKind,
    val message: String,
)

/** Immutable presentation projection of the service-owned stream and preview state. */
data class StreamingScreenState(
    val streamState: StreamState,
    val visualState: StreamingVisualState,
    val headerUiMode: HeaderUiMode,
    val action: StreamAction,
    val actionEnabled: Boolean,
    val headerLabel: String,
    val previewLabel: String,
    val panelMessage: String,
    val previewState: PreviewState,
    val cameraReady: Boolean,
    val encoderReady: Boolean,
    val serverReady: Boolean,
    val clientConnected: Boolean,
    val localAddress: String?,
    val rtspPort: Int,
    val lastError: StreamUiError? = null,
)

object StreamingScreenStateMapper {
    fun map(snapshot: ServiceSnapshot): StreamingScreenState {
        val clientConnected = snapshot.metrics.connectedClients > 0
        val visualState = when (snapshot.state) {
            StreamState.STOPPED -> StreamingVisualState.STOPPED
            StreamState.STARTING -> StreamingVisualState.STARTING
            StreamState.WAITING_NETWORK -> StreamingVisualState.WAITING_FOR_NETWORK
            StreamState.STREAMING -> {
                if (clientConnected) StreamingVisualState.CLIENT_CONNECTED
                else StreamingVisualState.WAITING_FOR_CLIENT
            }
            StreamState.STOPPING -> StreamingVisualState.STOPPING
            StreamState.ERROR -> StreamingVisualState.ERROR
        }
        val action = when (snapshot.state) {
            StreamState.STOPPED,
            StreamState.ERROR,
            -> StreamAction.START

            StreamState.STARTING,
            StreamState.WAITING_NETWORK,
            StreamState.STREAMING,
            -> StreamAction.STOP

            StreamState.STOPPING -> StreamAction.NONE
        }
        val error = mapError(snapshot)
        return StreamingScreenState(
            streamState = snapshot.state,
            visualState = visualState,
            headerUiMode = headerUiMode(visualState),
            action = action,
            actionEnabled = action != StreamAction.NONE,
            headerLabel = headerLabel(visualState),
            previewLabel = previewLabel(snapshot),
            panelMessage = panelMessage(snapshot, visualState, error),
            previewState = snapshot.previewState,
            cameraReady = snapshot.subsystems.camera == SubsystemState.RUNNING,
            encoderReady = snapshot.subsystems.encoder == SubsystemState.RUNNING,
            serverReady = snapshot.subsystems.rtspServer == SubsystemState.RUNNING,
            clientConnected = clientConnected,
            localAddress = snapshot.localIp,
            rtspPort = snapshot.settings.stream.port,
            lastError = error,
        )
    }

    private fun mapError(snapshot: ServiceSnapshot): StreamUiError? {
        if (snapshot.state == StreamState.WAITING_NETWORK) {
            return StreamUiError(
                StreamUiErrorKind.NETWORK_UNAVAILABLE,
                "Wi-Fi non disponibile. Lo stream riprenderà quando la LAN tornerà disponibile.",
            )
        }
        val previewFailure = snapshot.previewError?.trim().orEmpty()
        val streamFailure = snapshot.lastError?.trim().orEmpty()
        val usePreviewFailure = snapshot.previewState == PreviewState.ERROR && previewFailure.isNotEmpty()
        val message = when {
            usePreviewFailure -> previewFailure
            streamFailure.isNotEmpty() -> streamFailure
            else -> previewFailure
        }
        if (message.isEmpty()) return null
        val errorKind = when {
            usePreviewFailure -> snapshot.previewErrorKind
            streamFailure.isNotEmpty() -> snapshot.lastErrorKind
            else -> snapshot.previewErrorKind
        }
        val kind = when (errorKind) {
            StreamErrorKind.PERMISSION,
            StreamErrorKind.CAMERA,
            StreamErrorKind.CAPTURE_SESSION,
            StreamErrorKind.SURFACE,
            -> StreamUiErrorKind.CAMERA_ERROR

            StreamErrorKind.MEDIACODEC,
            StreamErrorKind.ENCODER,
            -> StreamUiErrorKind.ENCODER_ERROR

            StreamErrorKind.RTSP_SERVER,
            StreamErrorKind.PORT,
            -> StreamUiErrorKind.SERVER_BIND_ERROR

            StreamErrorKind.SOCKET -> StreamUiErrorKind.WRITER_ERROR
            StreamErrorKind.CONFIGURATION -> StreamUiErrorKind.AUTH_CONFIG_ERROR
            StreamErrorKind.THREAD,
            null,
            -> StreamUiErrorKind.STREAM_START_ERROR
        }
        return StreamUiError(kind, message)
    }

    private fun headerLabel(state: StreamingVisualState): String = when (state) {
        StreamingVisualState.STOPPED -> "LAN"
        StreamingVisualState.STARTING -> "AVVIO"
        StreamingVisualState.WAITING_FOR_NETWORK -> "RETE"
        StreamingVisualState.WAITING_FOR_CLIENT -> "SERVER ATTIVO"
        StreamingVisualState.CLIENT_CONNECTED -> "CLIENT CONNESSO"
        StreamingVisualState.STOPPING -> "ARRESTO"
        StreamingVisualState.ERROR -> "ERRORE"
    }

    private fun headerUiMode(state: StreamingVisualState): HeaderUiMode = when (state) {
        StreamingVisualState.STOPPED -> HeaderUiMode.IDLE
        StreamingVisualState.ERROR -> HeaderUiMode.ERROR
        StreamingVisualState.STARTING,
        StreamingVisualState.WAITING_FOR_NETWORK,
        StreamingVisualState.WAITING_FOR_CLIENT,
        StreamingVisualState.CLIENT_CONNECTED,
        StreamingVisualState.STOPPING,
        -> HeaderUiMode.ACTIVE
    }

    private fun previewLabel(snapshot: ServiceSnapshot): String = when {
        snapshot.previewState == PreviewState.ERROR -> "Preview error"
        snapshot.state == StreamState.STREAMING -> "Streaming active"
        snapshot.previewState == PreviewState.ACTIVE -> "Preview active · stream not started"
        snapshot.previewState == PreviewState.STARTING -> "Preview starting"
        else -> "Preview idle"
    }

    private fun panelMessage(
        snapshot: ServiceSnapshot,
        state: StreamingVisualState,
        error: StreamUiError?,
    ): String = when (state) {
        StreamingVisualState.STOPPED -> if (snapshot.previewState == PreviewState.ACTIVE) {
            "Preview camera ready · stream not started · ${snapshot.settings.stream.displayVideoLine()}"
        } else {
            "Camera pronta · ${snapshot.settings.stream.displayVideoLine()}"
        }
        StreamingVisualState.STARTING -> if (snapshot.previewState == PreviewState.ACTIVE) {
            "Avvio encoder H.264 e server RTSP..."
        } else {
            "Avvio preview Camera2..."
        }
        StreamingVisualState.WAITING_FOR_NETWORK -> if (snapshot.previewState == PreviewState.ACTIVE) {
            "Preview attiva · Wi-Fi non disponibile"
        } else {
            error?.message ?: "Wi-Fi non disponibile"
        }
        StreamingVisualState.WAITING_FOR_CLIENT ->
            "Server RTSP attivo · in attesa del client Windows"
        StreamingVisualState.CLIENT_CONNECTED ->
            "Client Windows connesso · ${snapshot.metrics.connectedClients} attivo"
        StreamingVisualState.STOPPING ->
            "Arresto della pipeline in corso..."
        StreamingVisualState.ERROR ->
            error?.message ?: "Errore stream. Premi Riprova."
    }
}
