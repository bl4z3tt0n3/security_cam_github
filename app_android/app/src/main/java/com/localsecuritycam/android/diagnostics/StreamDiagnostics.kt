package com.localsecuritycam.android.diagnostics

enum class StreamSubsystem {
    CAMERA,
    ENCODER,
    RTSP_SERVER,
}

enum class SubsystemState {
    IDLE,
    WAITING_NETWORK,
    STARTING,
    RUNNING,
    STOPPING,
    ERROR,
}

data class StreamSubsystemSnapshot(
    val camera: SubsystemState = SubsystemState.IDLE,
    val encoder: SubsystemState = SubsystemState.IDLE,
    val rtspServer: SubsystemState = SubsystemState.IDLE,
) {
    fun withState(subsystem: StreamSubsystem, next: SubsystemState): StreamSubsystemSnapshot = when (subsystem) {
        StreamSubsystem.CAMERA -> copy(camera = next)
        StreamSubsystem.ENCODER -> copy(encoder = next)
        StreamSubsystem.RTSP_SERVER -> copy(rtspServer = next)
    }

    fun withAll(next: SubsystemState): StreamSubsystemSnapshot = copy(
        camera = next,
        encoder = next,
        rtspServer = next,
    )
}

internal fun subsystemStatesForFailure(failure: StreamFailure): StreamSubsystemSnapshot {
    val failedSubsystem = when (failure.kind) {
        StreamErrorKind.PERMISSION,
        StreamErrorKind.CAMERA,
        StreamErrorKind.CAPTURE_SESSION,
        StreamErrorKind.SURFACE,
        -> StreamSubsystem.CAMERA

        StreamErrorKind.MEDIACODEC,
        StreamErrorKind.ENCODER,
        -> StreamSubsystem.ENCODER

        StreamErrorKind.RTSP_SERVER,
        StreamErrorKind.SOCKET,
        StreamErrorKind.PORT,
        -> StreamSubsystem.RTSP_SERVER

        StreamErrorKind.CONFIGURATION,
        StreamErrorKind.THREAD,
        -> null
    }
    return failedSubsystem?.let {
        StreamSubsystemSnapshot().withState(it, SubsystemState.ERROR)
    } ?: StreamSubsystemSnapshot()
}

internal fun subsystemStatesForCleanupFailure(cleanup: CleanupReport): StreamSubsystemSnapshot {
    var result = StreamSubsystemSnapshot()
    cleanup.failures.forEach { failure ->
        val resource = failure.resource.lowercase()
        val subsystem = when {
            "camera" in resource -> StreamSubsystem.CAMERA
            "encoder" in resource || "mediacodec" in resource -> StreamSubsystem.ENCODER
            "rtsp" in resource || "server" in resource -> StreamSubsystem.RTSP_SERVER
            else -> null
        }
        if (subsystem != null) result = result.withState(subsystem, SubsystemState.ERROR)
    }
    return result
}
