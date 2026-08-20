package com.localsecuritycam.android.service

import com.localsecuritycam.android.diagnostics.StreamState

/** Keeps automatic Wi-Fi recovery scoped to an already live stream. */
internal enum class NetworkResumeReason {
    ACTIVE_STREAM_RECOVERY,
}

internal object NetworkRecoveryPolicy {
    fun reasonForNetworkLoss(state: StreamState): NetworkResumeReason? =
        if (state == StreamState.STREAMING) NetworkResumeReason.ACTIVE_STREAM_RECOVERY else null

    fun shouldResume(state: StreamState, reason: NetworkResumeReason?): Boolean =
        state == StreamState.WAITING_NETWORK && reason == NetworkResumeReason.ACTIVE_STREAM_RECOVERY
}
