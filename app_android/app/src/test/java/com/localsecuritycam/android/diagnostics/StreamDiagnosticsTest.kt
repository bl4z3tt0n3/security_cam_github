package com.localsecuritycam.android.diagnostics

import org.junit.Assert.assertEquals
import org.junit.Test

class StreamDiagnosticsTest {
    @Test
    fun mapsResourceFailuresToTheCorrectSubsystem() {
        assertFailureSubsystem(StreamErrorKind.PERMISSION, StreamSubsystem.CAMERA)
        assertFailureSubsystem(StreamErrorKind.CAMERA, StreamSubsystem.CAMERA)
        assertFailureSubsystem(StreamErrorKind.CAPTURE_SESSION, StreamSubsystem.CAMERA)
        assertFailureSubsystem(StreamErrorKind.SURFACE, StreamSubsystem.CAMERA)
        assertFailureSubsystem(StreamErrorKind.MEDIACODEC, StreamSubsystem.ENCODER)
        assertFailureSubsystem(StreamErrorKind.ENCODER, StreamSubsystem.ENCODER)
        assertFailureSubsystem(StreamErrorKind.RTSP_SERVER, StreamSubsystem.RTSP_SERVER)
        assertFailureSubsystem(StreamErrorKind.SOCKET, StreamSubsystem.RTSP_SERVER)
        assertFailureSubsystem(StreamErrorKind.PORT, StreamSubsystem.RTSP_SERVER)
    }

    @Test
    fun nonResourceFailuresDoNotInventAComponentFailure() {
        val configuration = subsystemStatesForFailure(
            StreamFailure(StreamErrorKind.CONFIGURATION, "invalid configuration"),
        )
        val thread = subsystemStatesForFailure(
            StreamFailure(StreamErrorKind.THREAD, "control thread failed"),
        )

        assertEquals(StreamSubsystemSnapshot(), configuration)
        assertEquals(StreamSubsystemSnapshot(), thread)
    }

    @Test
    fun cleanupFailuresMarkOnlyTheAffectedSubsystems() {
        val states = subsystemStatesForCleanupFailure(
            CleanupReport(
                listOf(
                    CleanupFailure("RTSP server", "close failed"),
                    CleanupFailure("encoder", "release failed"),
                ),
            ),
        )

        assertEquals(SubsystemState.IDLE, states.camera)
        assertEquals(SubsystemState.ERROR, states.encoder)
        assertEquals(SubsystemState.ERROR, states.rtspServer)
    }

    private fun assertFailureSubsystem(kind: StreamErrorKind, expected: StreamSubsystem) {
        val states = subsystemStatesForFailure(StreamFailure(kind, "failure"))
        assertEquals(SubsystemState.ERROR, stateOf(states, expected))
        StreamSubsystem.entries
            .filter { it != expected }
            .forEach { subsystem -> assertEquals(SubsystemState.IDLE, stateOf(states, subsystem)) }
    }

    private fun stateOf(states: StreamSubsystemSnapshot, subsystem: StreamSubsystem): SubsystemState = when (subsystem) {
        StreamSubsystem.CAMERA -> states.camera
        StreamSubsystem.ENCODER -> states.encoder
        StreamSubsystem.RTSP_SERVER -> states.rtspServer
    }
}
