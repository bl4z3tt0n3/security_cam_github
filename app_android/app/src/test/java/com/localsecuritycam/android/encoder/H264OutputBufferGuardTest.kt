package com.localsecuritycam.android.encoder

import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class H264OutputBufferGuardTest {
    @Test
    fun releasesTheOutputBufferWhenProcessingCallbackFails() {
        val primary = IllegalStateException("output callback failed")
        var released = false
        var releaseErrors = 0

        var thrown: Throwable? = null
        try {
            withH264OutputBuffer(
                process = { throw primary },
                release = { released = true },
                onReleaseError = { releaseErrors++ },
            )
        } catch (error: Throwable) {
            thrown = error
        }

        assertSame(primary, thrown)
        assertTrue(released)
        assertTrue(releaseErrors == 0)
    }

    @Test
    fun reportsReleaseFailureWithoutReplacingPrimaryFailure() {
        val primary = IllegalStateException("output callback failed")
        val releaseFailure = IllegalStateException("release failed")
        var reported = 0

        var thrown: Throwable? = null
        try {
            withH264OutputBuffer(
                process = { throw primary },
                release = { throw releaseFailure },
                onReleaseError = { reported++ },
            )
        } catch (error: Throwable) {
            thrown = error
        }

        assertSame(primary, thrown)
        assertTrue(reported == 1)
    }
}
