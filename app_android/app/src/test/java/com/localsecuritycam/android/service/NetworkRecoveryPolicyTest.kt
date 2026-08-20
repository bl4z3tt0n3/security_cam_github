package com.localsecuritycam.android.service

import com.localsecuritycam.android.diagnostics.StreamState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NetworkRecoveryPolicyTest {
    @Test
    fun onlyAPreviouslyLiveStreamCreatesAnAutomaticResumeReason() {
        assertEquals(
            NetworkResumeReason.ACTIVE_STREAM_RECOVERY,
            NetworkRecoveryPolicy.reasonForNetworkLoss(StreamState.STREAMING),
        )
        assertNull(NetworkRecoveryPolicy.reasonForNetworkLoss(StreamState.STARTING))
        assertNull(NetworkRecoveryPolicy.reasonForNetworkLoss(StreamState.STOPPED))
    }

    @Test
    fun onlyWaitingRecoveryFromALiveStreamMayRestartAutomatically() {
        assertTrue(
            NetworkRecoveryPolicy.shouldResume(
                StreamState.WAITING_NETWORK,
                NetworkResumeReason.ACTIVE_STREAM_RECOVERY,
            ),
        )
        assertFalse(NetworkRecoveryPolicy.shouldResume(StreamState.WAITING_NETWORK, null))
        assertFalse(
            NetworkRecoveryPolicy.shouldResume(
                StreamState.STREAMING,
                NetworkResumeReason.ACTIVE_STREAM_RECOVERY,
            ),
        )
    }
}
